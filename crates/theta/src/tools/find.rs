//! find tool: search for files by name pattern.

use async_trait::async_trait;
use theta_agent_core::error::AgentError;
use theta_agent_core::types::{AgentTool, ToolExecutionMode, ToolResult, ToolUpdateSender};
use theta_ai::ContentBlock;
use tokio_util::sync::CancellationToken;

use super::{ToolContext, TruncationLimits, format_path_io_error, resolve_path, truncate_output};

pub struct FindTool {
    ctx: ToolContext,
}

impl FindTool {
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl AgentTool for FindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Find files matching a name pattern. Supports glob-style patterns."
    }

    fn label(&self) -> &str {
        "find"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "File name pattern (glob, e.g. '*.rs')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (defaults to working dir)"
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
        let pattern = args["pattern"]
            .as_str()
            .ok_or_else(|| AgentError::ToolExecution {
                tool_name: "find".into(),
                message: "missing required 'pattern' parameter".into(),
            })?;
        let search_path = args["path"]
            .as_str()
            .map(|p| resolve_path(&self.ctx, p))
            .unwrap_or_else(|| self.ctx.working_dir.clone());

        let meta =
            tokio::fs::metadata(&search_path)
                .await
                .map_err(|e| AgentError::ToolExecution {
                    tool_name: "find".into(),
                    message: format_path_io_error("inspect search path", &search_path, &e),
                })?;
        if !meta.is_dir() {
            return Err(AgentError::ToolExecution {
                tool_name: "find".into(),
                message: format!(
                    "inspect search path failed (invalid path) at '{}': not a directory",
                    search_path.to_string_lossy()
                ),
            });
        }

        let glob_pattern = format!("{}/**/{pattern}", search_path.display());

        let walker = glob::glob(&glob_pattern).map_err(|e| AgentError::ToolExecution {
            tool_name: "find".into(),
            message: format!("invalid glob pattern: {e}"),
        })?;

        let mut results = Vec::new();
        let mut count = 0u64;
        let max_results = 500u64;

        for entry in walker {
            if count >= max_results {
                results.push(format!("\n... truncated (max {max_results} results)"));
                break;
            }
            match entry {
                Ok(path) => {
                    let display = if let Ok(rel) = path.strip_prefix(&self.ctx.working_dir) {
                        rel.display().to_string()
                    } else {
                        path.display().to_string()
                    };
                    results.push(display);
                    count += 1;
                }
                Err(_) => continue,
            }
        }

        let output = if results.is_empty() {
            "no files found".to_string()
        } else {
            results.join("\n")
        };

        let mut result = ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: "find".into(),
            content: vec![ContentBlock::Text { text: output }],
            details: Some(serde_json::json!({
                "match_count": count,
                "pattern": pattern,
                "search_path": search_path.to_string_lossy().to_string()
            })),
            is_error: false,
        };

        truncate_output(&mut result, &TruncationLimits::default());

        Ok(result)
    }
}
