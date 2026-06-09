//! write tool: creates or overwrites files.

use async_trait::async_trait;
use michin_agent_core::error::AgentError;
use michin_agent_core::types::{AgentTool, ToolExecutionMode, ToolResult, ToolUpdateSender};
use michin_ai::ContentBlock;
use tokio_util::sync::CancellationToken;

use super::{ToolContext, format_path_io_error, resolve_path, shorten_path};

pub struct WriteTool {
    ctx: ToolContext,
}

impl WriteTool {
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl AgentTool for WriteTool {
    fn name(&self) -> &str {
        "write"
    }

    fn description(&self) -> &str {
        "Write content to a file. Creates the file if it doesn't exist, overwrites if it does. \
         Automatically creates parent directories."
    }

    fn label(&self) -> &str {
        "write"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["path", "content"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (relative or absolute)"
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file"
                }
            }
        })
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        // Write could be parallel but is sequential to avoid race on same file.
        ToolExecutionMode::Sequential
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        args: serde_json::Value,
        _signal: Option<CancellationToken>,
        _on_update: Option<ToolUpdateSender>,
    ) -> Result<ToolResult, AgentError> {
        let path = args["path"]
            .as_str()
            .ok_or_else(|| AgentError::ToolExecution {
                tool_name: "write".into(),
                message: "missing required 'path' parameter".into(),
            })?;
        let content = args["content"]
            .as_str()
            .ok_or_else(|| AgentError::ToolExecution {
                tool_name: "write".into(),
                message: "missing required 'content' parameter".into(),
            })?;

        let file_path = resolve_path(&self.ctx, path);

        // Create parent directories.
        if let Some(parent) = file_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "write".into(),
                    message: format_path_io_error("create parent directories", parent, &e),
                })?;
        }

        tokio::fs::write(&file_path, content)
            .await
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "write".into(),
                message: format_path_io_error("write file", &file_path, &e),
            })?;

        super::touch_fff_frecency(&self.ctx, &file_path);

        Ok(ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: "write".into(),
            content: vec![ContentBlock::Text {
                text: format!(
                    "Successfully wrote {} bytes to {}",
                    content.len(),
                    shorten_path(&file_path)
                ),
            }],
            details: Some(serde_json::json!({
                "bytes_written": content.len(),
                "path": file_path.to_string_lossy().to_string(),
            })),
            is_error: false,
        })
    }
}
