//! Bridge from Rhai scripts to Theta's `Hooks` trait.
//!
//! Implements `theta_agent_core::hooks::Hooks` by delegating to the
//! `ScriptEngine` for `before_tool_call` and `after_tool_call`.

use std::sync::Arc;

use async_trait::async_trait;
use theta_agent_core::hooks::Hooks;
use theta_agent_core::state::AgentState;
use theta_agent_core::types::{ToolCall, ToolResult};

use crate::engine::{BeforeHookResult, ScriptEngine};

/// Hooks implementation backed by Rhai scripts.
pub struct ScriptHooks {
    engine: Arc<ScriptEngine>,
}

impl ScriptHooks {
    /// Create hooks from a loaded script engine.
    pub fn new(engine: Arc<ScriptEngine>) -> Self {
        Self { engine }
    }
}

#[async_trait]
impl Hooks for ScriptHooks {
    async fn before_tool_call(
        &self,
        _state: &AgentState,
        tool_call: &ToolCall,
    ) -> Result<(), theta_agent_core::error::AgentError> {
        match self
            .engine
            .eval_before(&tool_call.name, &tool_call.arguments)
        {
            Ok(BeforeHookResult::Allow) => Ok(()),
            Ok(BeforeHookResult::Block { reason }) => {
                Err(theta_agent_core::error::AgentError::ToolExecution {
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
        result: &ToolResult,
    ) -> Result<(), theta_agent_core::error::AgentError> {
        // Build a summary string for the result content.
        let result_str = result
            .content
            .iter()
            .filter_map(|b| match b {
                theta_ai::ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        // We need the original args — not available in ToolResult.
        // Pass empty args for after hooks (they can't block anyway).
        let args = serde_json::json!({});
        if let Err(e) = self
            .engine
            .eval_after(&result.tool_name, &args, &result_str)
        {
            tracing::warn!(
                tool = %result.tool_name,
                error = %e,
                "script after_tool error"
            );
        }
        Ok(())
    }
}
