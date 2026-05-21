//! ls tool: lists directory contents.

use async_trait::async_trait;
use theta_agent_core::error::AgentError;
use theta_agent_core::types::{AgentTool, ToolExecutionMode, ToolResult, ToolUpdateSender};
use theta_ai::ContentBlock;
use tokio_util::sync::CancellationToken;

use super::{ToolContext, TruncationLimits, format_path_io_error, resolve_path, truncate_output};

pub struct LsTool {
    ctx: ToolContext,
}

impl LsTool {
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl AgentTool for LsTool {
    fn name(&self) -> &str {
        "ls"
    }

    fn description(&self) -> &str {
        "List the contents of a directory."
    }

    fn label(&self) -> &str {
        "ls"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": [],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path to list (defaults to working dir)"
                }
            }
        })
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Parallel
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        args: serde_json::Value,
        _signal: Option<CancellationToken>,
        _on_update: Option<ToolUpdateSender>,
    ) -> Result<ToolResult, AgentError> {
        let dir_path = args["path"]
            .as_str()
            .map(|p| resolve_path(&self.ctx, p))
            .unwrap_or_else(|| self.ctx.working_dir.clone());

        let mut entries =
            tokio::fs::read_dir(&dir_path)
                .await
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "ls".into(),
                    message: format_path_io_error("list directory", &dir_path, &e),
                })?;

        let mut results = Vec::new();
        let mut count = 0u64;
        let max_entries = 1000u64;

        while let Ok(Some(entry)) = entries.next_entry().await {
            if count >= max_entries {
                results.push(format!("\n... truncated (max {max_entries} entries)"));
                break;
            }
            let name = entry.file_name();
            let name_str = name.to_string_lossy().to_string();
            let file_type = entry
                .file_type()
                .await
                .ok()
                .map(|ft| {
                    if ft.is_dir() {
                        "/"
                    } else if ft.is_symlink() {
                        "@"
                    } else {
                        ""
                    }
                })
                .unwrap_or("");
            results.push(format!("{name_str}{file_type}"));
            count += 1;
        }

        let output = if results.is_empty() {
            "(empty directory)".to_string()
        } else {
            results.join("\n")
        };

        let mut result = ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: "ls".into(),
            content: vec![ContentBlock::Text { text: output }],
            details: Some(serde_json::json!({
                "entry_count": count,
                "path": dir_path.to_string_lossy().to_string()
            })),
            is_error: false,
        };

        truncate_output(&mut result, &TruncationLimits::default());

        Ok(result)
    }
}
