//! Agent runtime for Theta.
//!
//! Provides the Agent struct, tool execution, lifecycle hooks,
//! event emission, and the nested prompt/continue loop.

pub mod agent;
pub mod command_policy;
pub mod compact;
pub mod error;
pub mod events;
pub mod hooks;
pub mod loop_mod;
pub mod state;
pub mod tools;
pub mod types;

pub use agent::Agent;
pub use error::AgentError;
pub use events::AgentEvent;
pub use hooks::{Hooks, NoopHooks};
pub use state::AgentState;
pub use types::{
    AgentIntent, AgentLoopConfig, AgentTool, CircuitBreakerConfig, CompactionConfig,
    CompactionStrategy, RetryConfig, RunReport, RunReportEvent, RuntimeProfile, SafetyDecisionKind,
    ToolCall, ToolExecutionMode, ToolResult, ToolUpdate, ToolUpdateSender, ToolUpdateStatus,
    ToolWatchdogConfig, TurnEndReason, TurnMode,
};
