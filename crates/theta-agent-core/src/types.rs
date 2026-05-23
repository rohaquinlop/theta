//! Agent-level types: tools, config, and execution primitives.

use std::collections::BTreeMap;
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

/// A status-bar row specified by extensions (Rhai scripts).
/// Each row has three text slots: left, center, right.
#[derive(Debug, Clone, Default)]
pub struct ExtensionStatusRow {
    pub left: Vec<String>,
    pub center: Vec<String>,
    pub right: Vec<String>,
}

impl ExtensionStatusRow {
    pub fn is_empty(&self) -> bool {
        self.left.is_empty() && self.center.is_empty() && self.right.is_empty()
    }
}

/// Configuration for the agent loop.
#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    /// Named hardening profile for deterministic runtime behavior.
    pub runtime_profile: RuntimeProfile,
    /// Optional hard safety cap for inner-loop iterations per turn.
    /// `None` disables this cap.
    pub max_tool_rounds: Option<u32>,
    /// Maximum repeats allowed for the same tool call signature
    /// (same tool name + same arguments) within a turn.
    pub max_same_tool_call_repeats: Option<u32>,
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
    /// Tool execution watchdog policy.
    pub tool_watchdog: ToolWatchdogConfig,
    /// Model fallback chain (model IDs) used when provider calls fail.
    pub provider_fallback_chain: Vec<String>,
    /// Circuit breaker policy for provider/model reliability.
    pub provider_circuit_breaker: CircuitBreakerConfig,
    /// Whether command safety policy should run in strict mode.
    pub command_policy_strict: bool,
}

impl Default for AgentLoopConfig {
    fn default() -> Self {
        Self {
            runtime_profile: RuntimeProfile::Safe,
            max_tool_rounds: None,
            max_same_tool_call_repeats: Some(6),
            max_tokens: None,
            temperature: None,
            include_usage: false,
            compaction: CompactionConfig::default(),
            retry: RetryConfig::default(),
            provider_timeout_ms: Some(120_000),
            tool_watchdog: ToolWatchdogConfig::default(),
            provider_fallback_chain: Vec::new(),
            provider_circuit_breaker: CircuitBreakerConfig::default(),
            command_policy_strict: true,
        }
    }
}

/// Named runtime hardening profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeProfile {
    Dev,
    #[default]
    Safe,
    Prod,
}

/// Tool watchdog policy.
#[derive(Debug, Clone)]
pub struct ToolWatchdogConfig {
    /// Warn when a tool has no progress for this many milliseconds.
    pub stall_warning_ms: u64,
    /// Hard timeout for an individual tool call.
    pub hard_timeout_ms: u64,
}

impl Default for ToolWatchdogConfig {
    fn default() -> Self {
        Self {
            stall_warning_ms: 8_000,
            hard_timeout_ms: 60_000,
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
    /// Strategy for handling trimmed context.
    pub strategy: CompactionStrategy,
    /// Maximum output tokens for compaction summaries.
    pub summary_max_tokens: u32,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 4096,
            strategy: CompactionStrategy::Llm,
            summary_max_tokens: 512,
        }
    }
}

/// Compaction summary strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    None,
    Textual,
    Llm,
}

/// High-level intent for the current turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentIntent {
    Execute,
    PlanOnly,
    Inspect,
    AnalyzeOnly,
    Clarify,
    Default,
}

/// Deterministic execution mode for a turn.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnMode {
    Execute,
    Inspect,
    AnalyzeOnly,
    PlanOnly,
    Clarify,
}

impl From<AgentIntent> for TurnMode {
    fn from(value: AgentIntent) -> Self {
        match value {
            AgentIntent::Execute => Self::Execute,
            AgentIntent::Inspect => Self::Inspect,
            AgentIntent::AnalyzeOnly => Self::AnalyzeOnly,
            AgentIntent::PlanOnly => Self::PlanOnly,
            AgentIntent::Clarify | AgentIntent::Default => Self::Clarify,
        }
    }
}

/// Canonical reason for turn termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TurnEndReason {
    Completed,
    BlockedMissingInfo,
    BlockedPermission,
    BlockedRuntimeConstraint,
    ProviderFailure,
    ToolFailure,
    MaxToolRounds,
    NoopAfterRetry,
    AbortedByUser,
    SafetyRejected,
}

/// Structured safety outcome for command/tool policy checks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyDecisionKind {
    Allowed,
    Rejected,
}

/// Structured timeline event for run-report export.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunReportEvent {
    pub ts_ms: u64,
    pub kind: String,
    pub fields: BTreeMap<String, String>,
}

/// Structured run report for post-incident diagnostics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunReport {
    pub run_id: String,
    pub started_at_ms: u64,
    pub finished_at_ms: Option<u64>,
    pub outcome: Option<TurnEndReason>,
    pub events: Vec<RunReportEvent>,
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
    /// Whether this provider error is retryable.
    pub fn is_retryable(&self, error: &theta_ai::ThetaError) -> bool {
        matches!(error.class(), theta_ai::ErrorClass::Transient)
    }
}

/// Provider/model circuit-breaker policy.
#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    /// Open the circuit after this many consecutive transient failures.
    pub failure_threshold: u32,
    /// Time to keep the breaker open before allowing half-open retry.
    pub open_cooldown_ms: u64,
}

impl Default for CircuitBreakerConfig {
    fn default() -> Self {
        Self {
            failure_threshold: 3,
            open_cooldown_ms: 30_000,
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
