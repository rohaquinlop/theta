//! Agent-level types: tools, config, and execution primitives.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::error::AgentError;
use theta_ai::ContentBlock;

/// Execution mode for tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolExecutionMode {
    /// Run independently in parallel with other parallel tools.
    Parallel,
    /// Run after all parallel tools in a batch complete.
    Sequential,
}

/// Result of a single tool execution.
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ContentBlock>,
    pub details: Option<serde_json::Value>,
    pub is_error: bool,
}

/// Progress update emitted during tool execution.
#[derive(Debug, Clone)]
pub struct ToolUpdate {
    pub tool_call_id: String,
    pub tool_name: String,
    pub status: ToolUpdateStatus,
    pub output: Option<String>,
}

#[derive(Debug, Clone)]
pub enum ToolUpdateStatus {
    Running,
    Progress,
    Completed,
    Error,
}

/// Sender for tool progress updates.
pub type ToolUpdateSender = Arc<dyn Fn(ToolUpdate) + Send + Sync>;

/// A tool that the agent can execute. Implement this trait for built-in
/// and custom tools.
#[async_trait::async_trait]
pub trait AgentTool: Send + Sync {
    /// Unique tool name, e.g. "read", "bash".
    fn name(&self) -> &str;

    /// Human-readable description for the LLM.
    fn description(&self) -> &str;

    /// Short label for display in the TUI.
    fn label(&self) -> &str;

    /// JSON Schema for the tool's parameters.
    fn parameters(&self) -> serde_json::Value;

    /// Execution mode: parallel (default) or sequential.
    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Parallel
    }

    /// Execute the tool with the given arguments.
    /// The `signal` token is set when the user aborts.
    /// `on_update` can be called to send progress updates.
    async fn execute(
        &self,
        tool_call_id: &str,
        args: serde_json::Value,
        signal: Option<CancellationToken>,
        on_update: Option<ToolUpdateSender>,
    ) -> Result<ToolResult, AgentError>;
}

/// Configuration for the agent loop.
#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    /// Maximum iterations of the inner (tool-calling) loop per turn.
    pub max_tool_rounds: Option<u32>,
    /// Maximum output tokens for each LLM call.
    pub max_tokens: Option<u32>,
    /// Temperature for LLM sampling.
    pub temperature: Option<f64>,
    /// Whether to request usage info in streams.
    pub include_usage: bool,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: Some(20),
            max_tokens: None,
            temperature: None,
            include_usage: false,
        }
    }
}

/// An assembled tool call extracted from an assistant message.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    /// Extract tool calls from an assistant message's content blocks.
    pub fn from_message(msg: &theta_ai::Message) -> Vec<Self> {
        match msg {
            theta_ai::Message::Assistant { content, .. } => content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } => Some(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: arguments.clone(),
                    }),
                    _ => None,
                })
                .collect(),
            _ => vec![],
        }
    }
}
