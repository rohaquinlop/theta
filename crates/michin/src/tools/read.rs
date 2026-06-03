//! read tool: reads file contents with line/byte limits and truncation.

use async_trait::async_trait;
use base64::Engine as _;
use michin_agent_core::error::AgentError;
use michin_agent_core::types::{AgentTool, ToolExecutionMode, ToolResult, ToolUpdateSender};
use michin_ai::ContentBlock;
use tokio_util::sync::CancellationToken;

use super::{ToolContext, TruncationLimits, format_path_io_error, resolve_path, truncate_output};

pub struct ReadTool {
    ctx: ToolContext,
}

impl ReadTool {
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl AgentTool for ReadTool {
    fn name(&self) -> &str {
        "read"
    }

    fn description(&self) -> &str {
        "Read the contents of a file. Supports text files and images (jpg, png, gif, webp). \
         Images are sent as attachments. For text files, output is truncated to 2000 lines or \
         50KB (whichever is hit first). Use offset/limit for large files. When you need the \
         full file, continue with offset until complete."
    }

    fn label(&self) -> &str {
        "read"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["path"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute)"
                },
                "offset": {
                    "type": "number",
                    "description": "Line number to start reading from (1-indexed)"
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to read"
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
        let path = args["path"]
            .as_str()
            .ok_or_else(|| AgentError::ToolExecution {
                tool_name: "read".into(),
                message: "missing required 'path' parameter".into(),
            })?;
        let offset = args["offset"].as_u64().unwrap_or(1);
        let limit = args["limit"].as_u64();

        let file_path = resolve_path(&self.ctx, path);

        // Attempt to read as image first.
        let ext = file_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "gif" | "webp") {
            let data =
                tokio::fs::read(&file_path)
                    .await
                    .map_err(|e| AgentError::ToolExecution {
                        tool_name: "read".into(),
                        message: format_path_io_error("read image file", &file_path, &e),
                    })?;
            let mime = match ext.as_str() {
                "jpg" | "jpeg" => "image/jpeg",
                "png" => "image/png",
                "gif" => "image/gif",
                "webp" => "image/webp",
                _ => unreachable!(),
            };
            return Ok(ToolResult {
                tool_call_id: tool_call_id.into(),
                tool_name: "read".into(),
                content: vec![ContentBlock::Image {
                    data: base64::engine::general_purpose::STANDARD.encode(&data),
                    media_type: mime.into(),
                }],
                details: Some(serde_json::json!({
                    "path": file_path.to_string_lossy().to_string(),
                })),
                is_error: false,
            });
        }

        // Read as text.
        let content =
            tokio::fs::read_to_string(&file_path)
                .await
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "read".into(),
                    message: format_path_io_error("read file", &file_path, &e),
                })?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();

        // Offset is 1-indexed; convert to 0-indexed.
        let start = if offset > 0 { (offset - 1) as usize } else { 0 };
        let end = limit.map_or(total_lines, |l| {
            std::cmp::min(start + l as usize, total_lines)
        });

        let selected: String = if start >= total_lines {
            String::new()
        } else {
            lines[start..end].join("\n")
        };

        let mut result = ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: "read".into(),
            content: vec![ContentBlock::Text { text: selected }],
            details: Some(serde_json::json!({
                "total_lines": total_lines,
                "offset": offset,
                "lines_read": end.saturating_sub(start),
                "path": file_path.to_string_lossy().to_string(),
            })),
            is_error: false,
        };

        truncate_output(&mut result, &TruncationLimits::default());

        Ok(result)
    }
}
