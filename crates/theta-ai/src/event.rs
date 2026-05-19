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
                if let Some(tc) = self.tool_calls.iter_mut().find(|tc| &tc.id == id) {
                    tc.arguments.push_str(arguments);
                }
            }
            AssistantMessageEvent::ToolCallEnd { id } => {
                if let Some(tc) = self.tool_calls.iter_mut().find(|tc| &tc.id == id) {
                    tc.done = true;
                }
            }
            AssistantMessageEvent::Usage { usage } => {
                self.usage = Some(usage.clone());
            }
            AssistantMessageEvent::Done { stop_reason, usage } => {
                self.stop_reason = Some(*stop_reason);
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
    pub fn content_blocks(&self) -> Vec<ContentBlock> {
        let mut blocks = Vec::new();

        // Thinking block (if any reasoning was emitted).
        if !self.thinking_buffer.is_empty() {
            blocks.push(ContentBlock::Thinking {
                thinking: self.thinking_buffer.clone(),
                signature: None,
            });
        }

        // Text block (if any text was emitted).
        if !self.text_buffer.is_empty() {
            blocks.push(ContentBlock::text(std::mem::take(
                &mut self.text_buffer.clone(),
            )));
        }

        // Tool call blocks (only completed ones with valid JSON).
        for tc in &self.tool_calls {
            if tc.done {
                // Try to parse JSON arguments; fall back to raw string.
                let arguments = serde_json::from_str::<serde_json::Value>(&tc.arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(tc.arguments.clone()));
                blocks.push(ContentBlock::ToolCall {
                    id: tc.id.clone(),
                    name: tc.name.clone(),
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

    /// Reset for the next stream.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_text_only() {
        let mut acc = EventAccumulator::new();
        acc.feed(&AssistantMessageEvent::Start);
        acc.feed(&AssistantMessageEvent::TextStart);
        acc.feed(&AssistantMessageEvent::text_delta("Hello"));
        acc.feed(&AssistantMessageEvent::text_delta(" world"));
        acc.feed(&AssistantMessageEvent::TextEnd);
        acc.feed(&AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        });

        let blocks = acc.content_blocks();
        assert_eq!(blocks.len(), 1);
        if let ContentBlock::Text { text } = &blocks[0] {
            assert_eq!(text, "Hello world");
        } else {
            panic!("expected Text block");
        }
        assert_eq!(acc.stop_reason(), Some(StopReason::Stop));
    }

    #[test]
    fn test_accumulator_with_thinking() {
        let mut acc = EventAccumulator::new();
        acc.feed(&AssistantMessageEvent::Start);
        acc.feed(&AssistantMessageEvent::ThinkingStart);
        acc.feed(&AssistantMessageEvent::thinking_delta("Let me think..."));
        acc.feed(&AssistantMessageEvent::ThinkingEnd);
        acc.feed(&AssistantMessageEvent::TextStart);
        acc.feed(&AssistantMessageEvent::text_delta("Done."));
        acc.feed(&AssistantMessageEvent::TextEnd);
        acc.feed(&AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        });

        let blocks = acc.content_blocks();
        assert_eq!(blocks.len(), 2);
        assert!(matches!(blocks[0], ContentBlock::Thinking { .. }));
        assert!(matches!(blocks[1], ContentBlock::Text { .. }));
    }

    #[test]
    fn test_accumulator_reset() {
        let mut acc = EventAccumulator::new();
        acc.feed(&AssistantMessageEvent::text_delta("data"));
        acc.reset();
        assert!(acc.content_blocks().is_empty());
        assert!(acc.stop_reason().is_none());
    }
}
