//! Agent-level types: tools, config, and execution primitives.

use std::collections::BTreeMap;
use std::sync::Arc;

use tokio_util::sync::CancellationToken;

use crate::error::AgentError;
use theta_ai::ContentBlock;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolExecutionMode {
    Parallel,
    Sequential,
}

#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_call_id: String,
    pub tool_name: String,
    pub content: Vec<ContentBlock>,
    pub details: Option<serde_json::Value>,
    pub is_error: bool,
}

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

pub type ToolUpdateSender = Arc<dyn Fn(ToolUpdate) + Send + Sync>;

#[async_trait::async_trait]
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn label(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;
    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Parallel
    }

    /// Execute the tool with the given arguments.
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

#[derive(Debug, Clone)]
pub struct AgentLoopConfig {
    pub runtime_profile: RuntimeProfile,
    pub max_tool_rounds: Option<u32>,
    pub max_same_tool_call_repeats: Option<u32>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub include_usage: bool,
    pub compaction: CompactionConfig,
    pub retry: RetryConfig,
    pub provider_timeout_ms: Option<u64>,
    pub tool_watchdog: ToolWatchdogConfig,
    pub provider_fallback_chain: Vec<String>,
    pub provider_circuit_breaker: CircuitBreakerConfig,
    pub command_policy_strict: bool,
    /// Hard cap on context window tokens.
    /// `None` = model's full window. `Some(n)` = `min(model.ctx, n)`.
    pub max_context_window: Option<u32>,
}

impl AgentLoopConfig {
    pub fn effective_context_window(&self, model_context_window: u32) -> u32 {
        self.max_context_window
            .map(|max| model_context_window.min(max))
            .unwrap_or(model_context_window)
    }
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
            max_context_window: None,
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

#[derive(Debug, Clone)]
pub struct ToolWatchdogConfig {
    pub stall_warning_ms: u64,
}

impl Default for ToolWatchdogConfig {
    fn default() -> Self {
        Self {
            stall_warning_ms: 8_000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompactionConfig {
    pub enabled: bool,
    pub reserve_tokens: u32,
    /// How many tokens of recent conversation to preserve during compaction.
    /// Older messages are summarized.
    pub keep_recent_tokens: u32,
    pub strategy: CompactionStrategy,
    pub summary_max_tokens: u32,
    /// Number of consecutive compactions before auto-pausing.
    /// When the kept tail alone overflows the context trigger, compacting
    /// every turn craters the prefix cache. This threshold detects that
    /// condition and pauses until a turn fits naturally. Set to `u32::MAX`
    /// to never auto-pause.
    pub auto_pause_threshold: u32,
}

impl Default for CompactionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 4096,
            keep_recent_tokens: 20_000,
            strategy: CompactionStrategy::Llm,
            summary_max_tokens: 512,
            auto_pause_threshold: 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    None,
    Textual,
    Llm,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentIntent {
    Execute,
    PlanOnly,
    Inspect,
    AnalyzeOnly,
    Clarify,
    Default,
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyDecisionKind {
    Allowed,
    Rejected,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunReportEvent {
    pub ts_ms: u64,
    pub kind: String,
    pub fields: BTreeMap<String, String>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RunReport {
    pub run_id: String,
    pub started_at_ms: u64,
    pub finished_at_ms: Option<u64>,
    pub outcome: Option<TurnEndReason>,
    pub events: Vec<RunReportEvent>,
}

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
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
    pub fn is_retryable(&self, error: &theta_ai::ThetaError) -> bool {
        matches!(error.class(), theta_ai::ErrorClass::Transient)
    }
}

#[derive(Debug, Clone)]
pub struct CircuitBreakerConfig {
    pub failure_threshold: u32,
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

#[derive(Debug, Clone)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
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
