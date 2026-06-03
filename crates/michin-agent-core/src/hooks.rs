use async_trait::async_trait;

use crate::state::AgentState;
use crate::types::{ExtensionStatusRow, ToolCall, ToolResult};
use michin_ai::Message;

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
    /// `tool_call` is the original tool call (with its arguments).
    async fn after_tool_call(
        &self,
        _state: &AgentState,
        _tool_call: &ToolCall,
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

    /// Return TUI status bar rows from extensions (Rhai scripts).
    /// Rows[0] maps to the primary status bar, rows[1..] to extra rows above.
    /// Each row has left/center/right text slots.
    fn tui_status_rows(&self) -> Vec<ExtensionStatusRow> {
        vec![]
    }

    /// Return TUI status line entries from extensions.
    /// Each entry is (key, text) — rendered near the status bar.
    /// DEPRECATED: use tui_status_rows() for full row control.
    fn tui_status_lines(&self) -> Vec<(String, String)> {
        vec![]
    }
}

pub struct NoopHooks;

#[async_trait]
impl Hooks for NoopHooks {}
