//! Agent-level error types.

use thiserror::Error;

/// Errors that can occur during agent execution.
#[derive(Debug, Error)]
pub enum AgentError {
    /// Agent is already running a prompt/continue.
    #[error("agent is already running")]
    AlreadyRunning,

    /// Agent is not currently running (e.g. abort called when idle).
    #[error("agent is not running")]
    NotRunning,

    /// Underlying LLM error.
    #[error("LLM error: {0}")]
    Llm(#[from] theta_ai::ThetaError),

    /// A tool failed during execution.
    #[error("tool '{tool_name}' execution error: {message}")]
    ToolExecution { tool_name: String, message: String },

    /// Requested tool not found in registry.
    #[error("tool not found: '{tool_name}'")]
    ToolNotFound { tool_name: String },

    /// Request was aborted by user or signal.
    #[error("request aborted")]
    Aborted,

    /// Catch-all for other agent errors.
    #[error("{0}")]
    Other(String),
}
