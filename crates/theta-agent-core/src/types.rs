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
    /// Context compaction settings.
    pub compaction: CompactionConfig,
    /// Provider retry settings.
    pub retry: RetryConfig,
    /// Provider request timeout in milliseconds.
    pub provider_timeout_ms: Option<u64>,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: Some(20),
            max_tokens: None,
            temperature: None,
            include_usage: false,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
            provider_timeout_ms: Some(120_000),
        }
    }
}

/// Context compaction settings.
#[derive(Debug, Clone)]
pub struct CompactionConfig {
    /// Whether automatic compaction is enabled.
    pub enabled: bool,
    /// Tokens to reserve for the model's response.
    pub reserve_tokens: u32,
    /// Whether to ask the model to summarize trimmed context.
    pub summarize_with_llm: bool,
    /// Maximum output tokens for compaction summaries.
    pub summary_max_tokens: u32,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 4096,
            summarize_with_llm: true,
            summary_max_tokens: 512,
        }
    }
}

/// Provider retry settings.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum retry attempts (0 = no retry).
    pub max_retries: u32,
    /// Base delay in milliseconds before first retry.
    pub base_delay_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 2,
            base_delay_ms: 1000,
        }
    }
}

impl RetryConfig {
    /// Whether this error is retryable (429, 5xx).
    pub fn is_retryable(&self, error_msg: &str) -> bool {
        let lower = error_msg.to_lowercase();
        lower.contains("429")
            || lower.contains("rate limit")
            || lower.contains("too many requests")
            || lower.contains("500")
            || lower.contains("502")
            || lower.contains("503")
            || lower.contains("504")
            || lower.contains("server error")
            || lower.contains("timeout")
            || lower.contains("connection")
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
