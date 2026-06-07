//! Bridge from Rhai-registered custom tools to `michin_agent_core::AgentTool`.
//!
//! Wraps a `RegisteredToolDef` so the agent runtime can invoke it like any
//! built-in tool. The Rhai `Engine` is sync and uses interior mutability,
//! so execution is dispatched via `tokio::task::spawn_blocking`.

use std::sync::Arc;

use async_trait::async_trait;
use michin_agent_core::error::AgentError;
use michin_agent_core::types::{AgentTool, ToolExecutionMode, ToolResult, ToolUpdateSender};
use michin_ai::ContentBlock;
use tokio_util::sync::CancellationToken;

use crate::engine::{RegisteredToolDef, ScriptEngine};

/// An `AgentTool` backed by a Rhai script's `execute()` function.
pub struct RhaiCustomTool {
    def: RegisteredToolDef,
    engine: Arc<ScriptEngine>,
}

impl RhaiCustomTool {
    pub fn new(def: RegisteredToolDef, engine: Arc<ScriptEngine>) -> Self {
        Self { def, engine }
    }
}

#[async_trait]
impl AgentTool for RhaiCustomTool {
    fn name(&self) -> &str {
        &self.def.name
    }

    fn description(&self) -> &str {
        &self.def.description
    }

    fn label(&self) -> &str {
        &self.def.name
    }

    fn parameters(&self) -> serde_json::Value {
        self.def.parameters.clone()
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        self.def.execution_mode
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        args: serde_json::Value,
        signal: Option<CancellationToken>,
        _on_update: Option<ToolUpdateSender>,
    ) -> Result<ToolResult, AgentError> {
        let engine = Arc::clone(&self.engine);
        let def = self.def.clone();

        let result = tokio::task::spawn_blocking(move || engine.eval_tool_execute(&def, &args))
            .await
            .map_err(|e| AgentError::ToolExecution {
                tool_name: self.def.name.clone(),
                message: format!("custom tool task panicked: {e}"),
            })?
            .map_err(|e| AgentError::ToolExecution {
                tool_name: self.def.name.clone(),
                message: e,
            })?;

        // If cancellation was requested while spawn_blocking ran, discard result.
        if let Some(ref sig) = signal
            && sig.is_cancelled()
        {
            return Err(AgentError::Aborted);
        }

        Ok(ToolResult {
            tool_call_id: tool_call_id.to_string(),
            tool_name: self.def.name.clone(),
            content: vec![ContentBlock::text(result.content)],
            details: None,
            is_error: result.is_error,
        })
    }
}
