//! Agent state: the mutable transcript and configuration.

use std::collections::HashSet;
use std::sync::Arc;

use theta_ai::{ContentBlock, Message, Model, ThinkingLevel, Tool};

use crate::types::AgentTool;
use crate::types::{RunReport, RunReportEvent, TurnEndReason};

/// The mutable state of an agent run.
#[derive(Clone)]
pub struct AgentState {
    /// System prompt content blocks.
    pub system_prompt: Vec<ContentBlock>,

    /// The currently active model.
    pub model: Model,

    /// Available tools.
    pub tools: Vec<Arc<dyn AgentTool>>,

    /// Resource context (skills, extensions).
    /// Injected at the start of the conversation, not in the system prompt.
    pub resource_context: Option<Vec<ContentBlock>>,

    /// The conversation transcript (all messages).
    pub messages: Vec<Message>,

    /// Whether the agent is currently streaming an LLM response.
    pub is_streaming: bool,

    /// Current thinking/reasoning level.
    pub thinking_level: ThinkingLevel,
    /// Available models from catalog for fallback resolution.
    pub available_models: Vec<Model>,
    /// Last explicit turn terminal reason.
    pub last_turn_end_reason: Option<TurnEndReason>,
    /// In-progress report for current run.
    pub current_run_report: Option<RunReport>,
    /// Last completed run report.
    pub last_run_report: Option<RunReport>,
    /// Current run lineage ID.
    pub current_run_id: Option<String>,
    /// Current turn lineage ID.
    pub current_turn_id: Option<String>,
    /// Tool-call IDs already executed in current turn.
    pub executed_tool_call_ids_in_turn: HashSet<String>,
    /// Cached approximate token count of the system prompt (computed once on set).
    pub(crate) system_prompt_tokens: u32,
    /// Cached approximate token count of the resource context (computed once on set).
    pub(crate) resource_context_tokens: u32,
    /// Cached theta_ai::Tool list built from tools Vec (cheap clone per turn).
    pub(crate) theta_ai_tools: Vec<Tool>,
}

impl AgentState {
    pub fn new(model: Model, available_models: Vec<Model>) -> Self {
        Self {
            system_prompt: Vec::new(),
            model,
            tools: Vec::new(),
            resource_context: None,
            messages: Vec::new(),
            is_streaming: false,
            thinking_level: ThinkingLevel::Off,
            available_models,
            last_turn_end_reason: None,
            current_run_report: None,
            last_run_report: None,
            current_run_id: None,
            current_turn_id: None,
            executed_tool_call_ids_in_turn: HashSet::new(),
            system_prompt_tokens: 0,
            resource_context_tokens: 0,
            theta_ai_tools: Vec::new(),
        }
    }

    pub fn push_run_event(
        &mut self,
        kind: &str,
        fields: impl IntoIterator<Item = (String, String)>,
    ) {
        if let Some(report) = self.current_run_report.as_mut() {
            let mut map = std::collections::BTreeMap::new();
            if let Some(run_id) = &self.current_run_id {
                map.insert("run_id".to_string(), run_id.clone());
            }
            if let Some(turn_id) = &self.current_turn_id {
                map.insert("turn_id".to_string(), turn_id.clone());
            }
            map.insert("model".to_string(), self.model.id.clone());
            map.insert("provider".to_string(), format!("{:?}", self.model.provider));
            for (k, v) in fields {
                map.insert(k.clone(), redact_field(&k, &v));
            }
            report.events.push(RunReportEvent {
                ts_ms: now_ms(),
                kind: kind.to_string(),
                fields: map,
            });
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
    /// Does NOT include resource_context — callers must prepend it separately.
    pub fn llm_messages(&self) -> Vec<Message> {
        self.messages
            .iter()
            .filter(|m| {
                matches!(
                    m,
                    Message::User { .. } | Message::Assistant { .. } | Message::ToolResult { .. }
                )
            })
            .cloned()
            .collect()
    }

    /// Approximate token count of the resource context blocks.
    /// Uses the cached value computed when the resource context was set.
    pub fn resource_context_tokens(&self) -> u32 {
        self.resource_context_tokens
    }

    /// Recompute and cache system prompt and resource context token counts.
    /// Called by Agent when these fields are set.
    pub fn update_cached_tokens(&mut self) {
        self.system_prompt_tokens = approximate_tokens_for_blocks(&self.system_prompt);
        self.resource_context_tokens = self
            .resource_context
            .as_deref()
            .map(approximate_tokens_for_blocks)
            .unwrap_or(0);
    }

    /// Rebuild the cached theta_ai::Tool list from self.tools.
    /// Called after add_tool().
    pub fn rebuild_theta_ai_tools(&mut self) {
        self.theta_ai_tools = self
            .tools
            .iter()
            .map(|t| Tool {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters(),
            })
            .collect();
    }

    /// Load past messages from a session (for continue/resume).
    /// Preserves system prompt, tools, and model; only replaces the transcript.
    pub fn load_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    /// Find the model ID from the last assistant message in the transcript, if any.
    pub fn last_model_id(&self) -> Option<&str> {
        self.messages.iter().rev().find_map(|m| match m {
            Message::Assistant { model, .. } => model.as_deref(),
            _ => None,
        })
    }

    /// Approximate total token count across all messages.
    pub fn token_count(&self) -> u32 {
        let msg_tokens: u32 = self.messages.iter().map(|m| m.token_count()).sum();
        msg_tokens + self.system_prompt_tokens + self.resource_context_tokens
    }

    /// The last API-reported input token count (real, from the most recent
    /// assistant message's usage). This is the actual prompt token count as
    /// counted by the provider.
    pub fn last_real_input_tokens(&self) -> Option<u32> {
        self.messages.iter().rev().find_map(|m| match m {
            Message::Assistant { usage, .. } => usage.as_ref().map(|u| u.input_tokens),
            _ => None,
        })
    }

    /// Best-effort context consumption: API-reported input tokens if available,
    /// otherwise the approximate token count.
    pub fn context_tokens(&self) -> u32 {
        self.last_real_input_tokens()
            .unwrap_or_else(|| self.token_count())
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Approximate token count for a slice of content blocks by serializing to JSON.
fn approximate_tokens_for_blocks(blocks: &[ContentBlock]) -> u32 {
    blocks
        .iter()
        .map(|b| theta_ai::approximate_token_count(&serde_json::to_string(b).unwrap_or_default()))
        .sum()
}

fn redact_field(key: &str, value: &str) -> String {
    let key_lower = key.to_ascii_lowercase();
    let looks_sensitive_key = [
        "token",
        "secret",
        "password",
        "authorization",
        "cookie",
        "api_key",
        "apikey",
        "access_key",
        "refresh",
    ]
    .iter()
    .any(|p| key_lower.contains(p));
    let value_lower = value.to_ascii_lowercase();
    let looks_sensitive_value = value.starts_with("sk-")
        || value_lower.contains("bearer ")
        || value_lower.contains("authorization:")
        || value_lower.contains("api_key=")
        || value_lower.contains("token=");

    if looks_sensitive_key || looks_sensitive_value {
        "[REDACTED]".to_string()
    } else {
        value.to_string()
    }
}
