//! Streaming events emitted by LLM providers during generation.

use serde::{Deserialize, Serialize};

use super::types::*;

/// Events emitted during a streaming LLM request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AssistantMessageEvent {
    /// Stream has started.
    #[serde(rename = "start")]
    Start,
    /// Text generation has started (transitioning from thinking to content).
    #[serde(rename = "text_start")]
    TextStart,
    /// A text delta chunk.
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    /// Text generation has ended.
    #[serde(rename = "text_end")]
    TextEnd,
    /// Thinking/reasoning generation has started.
    #[serde(rename = "thinking_start")]
    ThinkingStart,
    /// A thinking/reasoning delta chunk.
    #[serde(rename = "thinking_delta")]
    ThinkingDelta { thinking: String },
    /// Thinking/reasoning generation has ended.
    #[serde(rename = "thinking_end")]
    ThinkingEnd,
    /// A tool call has started.
    #[serde(rename = "tool_call_start")]
    ToolCallStart { id: String, name: String },
    /// A tool call arguments delta chunk.
    #[serde(rename = "tool_call_delta")]
    ToolCallDelta { id: String, arguments: String },
    /// A tool call has ended (all arguments received).
    #[serde(rename = "tool_call_end")]
    ToolCallEnd { id: String },
    /// Token usage information (may appear mid-stream or at end).
    #[serde(rename = "usage")]
    Usage { usage: Usage },
    /// Stream completed successfully.
    #[serde(rename = "done")]
    Done {
        stop_reason: StopReason,
        usage: Option<Usage>,
    },
    /// Stream ended with an error.
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

/// Accumulate streaming events into final content blocks and usage.
#[derive(Debug, Default)]
pub struct EventAccumulator {
    /// Accumulated text content.
    text_buffer: String,
    /// Accumulated thinking content.
    thinking_buffer: String,
    /// Active tool calls being accumulated.
    tool_calls: Vec<ToolCallAccumulator>,
    /// Latest usage info.
    usage: Option<Usage>,
    /// Stop reason from final event.
    stop_reason: Option<StopReason>,
    /// Error message if any.
    error_message: Option<String>,
    /// Whether we've started text generation.
    in_text: bool,
    /// Whether we've started thinking.
    in_thinking: bool,
    /// Whether stream has started.
    started: bool,
}

#[derive(Debug, Default)]
struct ToolCallAccumulator {
    id: String,
    name: String,
    arguments: String,
    done: bool,
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

    /// Process a single event, updating internal state.
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

    /// Build the final content blocks from accumulated deltas.
    /// Clears internal buffers after building (single-use).
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

    /// Reset for the next stream.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
