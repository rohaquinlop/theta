//! Lifecycle hooks for agent customization.

use async_trait::async_trait;

use crate::state::AgentState;
use crate::types::{ToolCall, ToolResult};
use theta_ai::Message;

/// Hooks allow extensions to intercept and modify agent behavior.
/// Default implementations are no-ops.
#[async_trait]
pub trait Hooks: Send + Sync {
    /// Called before a tool call is executed.
    /// Return `Err` to block execution.
    async fn before_tool_call(
        &self,
        _state: &AgentState,
        _tool_call: &ToolCall,
    ) -> Result<(), crate::error::AgentError> {
        Ok(())
    }

    /// Called after a tool call completes.
    async fn after_tool_call(
        &self,
        _state: &AgentState,
        _result: &ToolResult,
    ) -> Result<(), crate::error::AgentError> {
        Ok(())
    }

    /// Called after each turn to decide whether the agent should stop.
    /// Return `true` to prevent follow-up turns.
    async fn should_stop_after_turn(&self, _state: &AgentState) -> bool {
        false
    }

    /// Called before the next turn to allow injecting additional messages
    /// (e.g., context, reminders, project rules).
    async fn prepare_next_turn(&self, _state: &AgentState) -> Vec<Message> {
        vec![]
    }
}

/// A no-op hooks implementation.
pub struct NoopHooks;

#[async_trait]
impl Hooks for NoopHooks {}
