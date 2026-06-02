use thiserror::Error;

#[derive(Debug, Error)]
pub enum AgentError {
    #[error("agent is already running")]
    AlreadyRunning,
    #[error("agent is not running")]
    NotRunning,
    #[error("LLM error: {0}")]
    Llm(#[from] theta_ai::ThetaError),
    #[error("tool '{tool_name}' execution error: {message}")]
    ToolExecution { tool_name: String, message: String },
    #[error("tool not found: '{tool_name}'")]
    ToolNotFound { tool_name: String },
    #[error("request aborted")]
    Aborted,
    #[error("{0}")]
    Other(String),
}
