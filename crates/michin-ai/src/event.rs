//! Streaming events emitted by LLM providers during generation.

use serde::{Deserialize, Serialize};

use super::types::*;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AssistantMessageEvent {
    #[serde(rename = "start")]
    Start,
    #[serde(rename = "text_start")]
    TextStart,
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "text_end")]
    TextEnd,
    #[serde(rename = "thinking_start")]
    ThinkingStart,
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    #[serde(rename = "thinking_end")]
    ThinkingEnd,
    #[serde(rename = "tool_call_start")]
    ToolCallStart { id: String, name: String },
    #[serde(rename = "tool_call_delta")]
    ToolCallDelta { id: String, arguments: String },
    #[serde(rename = "tool_call_end")]
    ToolCallEnd { id: String },
    #[serde(rename = "usage")]
    Usage { usage: Usage },
    #[serde(rename = "done")]
    Done {
        stop_reason: StopReason,
        usage: Option<Usage>,
    },
    #[serde(rename = "error")]
    Error { code: String, message: String },
}

impl AssistantMessageEvent {
    pub fn text_delta(text: impl Into<String>) -> Self {
        Self::TextDelta { text: text.into() }
    }

    pub fn thinking_delta(thinking: impl Into<String>) -> Self {
        Self::ThinkingDelta {
            thinking: thinking.into(),
        }
    }

    pub fn tool_call_delta(id: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self::ToolCallDelta {
            id: id.into(),
            arguments: arguments.into(),
        }
    }
}

#[derive(Debug, Default)]
pub struct EventAccumulator {
    text_buffer: String,
    thinking_buffer: String,
    tool_calls: Vec<ToolCallAccumulator>,
    usage: Option<Usage>,
    stop_reason: Option<StopReason>,
    error_message: Option<String>,
    in_text: bool,
    in_thinking: bool,
    started: bool,
}

#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
    done: bool,
}

/// A tool call that was started but never completed (stream cut off mid-arguments).
#[derive(Debug, Clone)]
pub struct UnresolvedToolCall {
    pub id: String,
    pub name: String,
}

fn tool_call_id_matches(existing_id: &str, event_id: &str) -> bool {
    if existing_id == event_id {
        return true;
    }

    let existing_call_id = existing_id.split('|').next().unwrap_or(existing_id);
    let event_call_id = event_id.split('|').next().unwrap_or(event_id);
    existing_call_id == event_call_id
}

impl EventAccumulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn feed(&mut self, event: &AssistantMessageEvent) {
        match event {
            AssistantMessageEvent::Start => {
                self.started = true;
            }
            AssistantMessageEvent::TextStart => {
                self.in_text = true;
                self.text_buffer.clear();
            }
            AssistantMessageEvent::TextDelta { text } => {
                self.text_buffer.push_str(text);
            }
            AssistantMessageEvent::TextEnd => {
                self.in_text = false;
            }
            AssistantMessageEvent::ThinkingStart => {
                self.in_thinking = true;
                self.thinking_buffer.clear();
            }
            AssistantMessageEvent::ThinkingDelta { thinking } => {
                self.thinking_buffer.push_str(thinking);
            }
            AssistantMessageEvent::ThinkingEnd => {
                self.in_thinking = false;
            }
            AssistantMessageEvent::ToolCallStart { id, name } => {
                self.tool_calls.push(ToolCallAccumulator {
                    id: id.clone(),
                    name: name.clone(),
                    ..Default::default()
                });
            }
            AssistantMessageEvent::ToolCallDelta { id, arguments } => {
                if let Some(tc) = self
                    .tool_calls
                    .iter_mut()
                    .find(|tc| tool_call_id_matches(&tc.id, id))
                {
                    tc.arguments.push_str(arguments);
                }
            }
            AssistantMessageEvent::ToolCallEnd { id } => {
                if let Some(tc) = self
                    .tool_calls
                    .iter_mut()
                    .find(|tc| tool_call_id_matches(&tc.id, id))
                {
                    tc.done = true;
                }
            }
            AssistantMessageEvent::Usage { usage } => {
                self.usage = Some(usage.clone());
            }
            AssistantMessageEvent::Done { stop_reason, usage } => {
                self.stop_reason = Some(match (self.stop_reason, *stop_reason) {
                    (Some(StopReason::ToolUse), StopReason::Stop) => StopReason::ToolUse,
                    (_, next) => next,
                });
                if let Some(u) = usage {
                    self.usage = Some(u.clone());
                }
            }
            AssistantMessageEvent::Error { message, .. } => {
                self.error_message = Some(message.clone());
                self.stop_reason = Some(StopReason::Error);
            }
        }
    }

    pub fn content_blocks(&mut self) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();

        // Thinking block (if any reasoning was emitted).
        if !self.thinking_buffer.is_empty() {
            blocks.push(ContentBlock::Thinking {
                thinking: std::mem::take(&mut self.thinking_buffer),
                signature: None,
            });
        }

        // Text block (if any text was emitted).
        if !self.text_buffer.is_empty() {
            blocks.push(ContentBlock::text(std::mem::take(&mut self.text_buffer)));
        }

        // Tool call blocks (only completed ones with valid JSON).
        // Drain tool_calls so the accumulator is truly single-use:
        // all three buffers (text, thinking, tool_calls) are cleared.
        for tc in std::mem::take(&mut self.tool_calls) {
            if tc.done {
                // Try to parse JSON arguments; fall back to raw string.
                let arguments = serde_json::from_str::<serde_json::Value>(&tc.arguments)
                    .unwrap_or(serde_json::Value::String(tc.arguments));
                blocks.push(ContentBlock::ToolCall {
                    id: tc.id,
                    name: tc.name,
                    arguments,
                });
            }
        }

        blocks
    }

    pub fn usage(&self) -> Option<&Usage> {
        self.usage.as_ref()
    }

    pub fn stop_reason(&self) -> Option<StopReason> {
        self.stop_reason
    }

    pub fn error_message(&self) -> Option<&str> {
        self.error_message.as_deref()
    }

    pub fn unresolved_tool_call_count(&self) -> usize {
        self.tool_calls.iter().filter(|tc| !tc.done).count()
    }

    pub fn unresolved_tool_calls(&self) -> Vec<UnresolvedToolCall> {
        self.tool_calls
            .iter()
            .filter(|tc| !tc.done)
            .map(|tc| UnresolvedToolCall {
                id: tc.id.clone(),
                name: tc.name.clone(),
            })
            .collect()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
