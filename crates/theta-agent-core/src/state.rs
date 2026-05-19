//! Agent state: the mutable transcript and configuration.

use std::sync::Arc;

use theta_ai::{ContentBlock, Message, Model, ThinkingLevel};

use crate::types::AgentTool;

/// The mutable state of an agent run.
#[derive(Clone)]
pub struct AgentState {
    /// System prompt content blocks.
    pub system_prompt: Vec<ContentBlock>,

    /// The currently active model.
    pub model: Model,

    /// Available tools.
    pub tools: Vec<Arc<dyn AgentTool>>,

    /// The conversation transcript (all messages).
    pub messages: Vec<Message>,

    /// Whether the agent is currently streaming an LLM response.
    pub is_streaming: bool,

    /// Current thinking/reasoning level.
    pub thinking_level: ThinkingLevel,
}

impl AgentState {
    pub fn new(model: Model) -> Self {
        Self {
            system_prompt: Vec::new(),
            model,
            tools: Vec::new(),
            messages: Vec::new(),
            is_streaming: false,
            thinking_level: ThinkingLevel::Off,
        }
    }

    /// Add a user message to the transcript.
    pub fn add_user_message(&mut self, content: Vec<ContentBlock>, timestamp: u64) {
        self.messages.push(Message::User { content, timestamp });
    }

    /// Add an assistant message to the transcript.
    pub fn add_assistant_message(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// Add a tool result message to the transcript.
    pub fn add_tool_result(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// Get only the messages that should be sent to the LLM.
    /// Filters out ModelChange and ThinkingLevelChange events.
    pub fn llm_messages(&self) -> Vec<&Message> {
        self.messages
            .iter()
            .filter(|m| {
                matches!(
                    m,
                    Message::User { .. } | Message::Assistant { .. } | Message::ToolResult { .. }
                )
            })
            .collect()
    }

    /// Approximate total token count across all messages.
    pub fn token_count(&self) -> u32 {
        let msg_tokens: u32 = self.messages.iter().map(|m| m.token_count()).sum();
        let sys_tokens: u32 = self
            .system_prompt
            .iter()
            .map(|b| {
                theta_ai::approximate_token_count(&serde_json::to_string(b).unwrap_or_default())
            })
            .sum();
        msg_tokens + sys_tokens
    }
}
