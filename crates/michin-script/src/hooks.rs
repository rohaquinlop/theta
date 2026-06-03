//! Bridge from Rhai scripts to MichiN's `Hooks` trait.
//!
//! Implements `michin_agent_core::hooks::Hooks` by delegating to the
//! `ScriptEngine` for `before_tool_call` and `after_tool_call`.

use std::sync::Arc;

use async_trait::async_trait;
use michin_agent_core::hooks::Hooks;
use michin_agent_core::state::AgentState;
use michin_agent_core::types::{ExtensionStatusRow, ToolCall, ToolResult};
use tokio::sync::Notify;

use crate::engine::{BeforeHookResult, ScriptEngine};

/// Hooks implementation backed by Rhai scripts.
pub struct ScriptHooks {
    engine: Arc<ScriptEngine>,
    /// Signaled after every `after_tool_call` evaluation so the TUI
    /// can refresh extension status rows on demand instead of polling.
    status_notify: Arc<Notify>,
}

impl ScriptHooks {
    /// Create hooks from a loaded script engine.
    pub fn new(engine: Arc<ScriptEngine>, status_notify: Arc<Notify>) -> Self {
        Self {
            engine,
            status_notify,
        }
    }
}

#[async_trait]
impl Hooks for ScriptHooks {
    async fn before_tool_call(
        &self,
        _state: &AgentState,
        tool_call: &ToolCall,
    ) -> Result<(), michin_agent_core::error::AgentError> {
        match self
            .engine
            .eval_before(&tool_call.name, &tool_call.arguments)
        {
            Ok(BeforeHookResult::Allow) => Ok(()),
            Ok(BeforeHookResult::Block { reason }) => {
                Err(michin_agent_core::error::AgentError::ToolExecution {
                    tool_name: tool_call.name.clone(),
                    message: reason,
                })
            }
            Err(e) => {
                tracing::warn!(
                    tool = %tool_call.name,
                    error = %e,
                    "script before_tool error"
                );
                // Script errors never block — let the tool proceed.
                Ok(())
            }
        }
    }

    async fn after_tool_call(
        &self,
        _state: &AgentState,
        tool_call: &ToolCall,
        result: &ToolResult,
    ) -> Result<(), michin_agent_core::error::AgentError> {
        // Build a summary string for the result content.
        let result_str = result
            .content
            .iter()
            .filter_map(|b| match b {
                michin_ai::ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        let args = &tool_call.arguments;
        if let Err(e) = self.engine.eval_after(&result.tool_name, args, &result_str) {
            tracing::warn!(
                tool = %result.tool_name,
                error = %e,
                "script after_tool error"
            );
        }

        // Wake the TUI poller so it can refresh extension status rows.
        self.status_notify.notify_one();

        Ok(())
    }

    fn tui_status_lines(&self) -> Vec<(String, String)> {
        self.engine.eval_tui_statuses()
    }

    fn tui_status_rows(&self) -> Vec<ExtensionStatusRow> {
        self.engine.eval_tui_rows()
    }
}
