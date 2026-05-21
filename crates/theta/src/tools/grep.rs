//! grep tool: regex search in files.

use async_trait::async_trait;
use regex::Regex;
use theta_agent_core::error::AgentError;
use theta_agent_core::types::{AgentTool, ToolExecutionMode, ToolResult, ToolUpdateSender};
use theta_ai::ContentBlock;
use tokio_util::sync::CancellationToken;

use super::{ToolContext, TruncationLimits, format_path_io_error, resolve_path, truncate_output};

pub struct GrepTool {
    ctx: ToolContext,
}

impl GrepTool {
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl AgentTool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search for regex patterns in files. Returns matching lines with file paths."
    }

    fn label(&self) -> &str {
        "grep"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["pattern"],
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for"
                },
                "include": {
                    "type": "string",
                    "description": "Glob pattern for files to search (e.g. '*.rs')"
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file path to search (defaults to working dir)"
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
                tool_name: "grep".into(),
                message: "missing required 'pattern' parameter".into(),
            })?;
        let include = args["include"].as_str().unwrap_or("*");
        let search_path = args["path"]
            .as_str()
            .map(|p| resolve_path(&self.ctx, p))
            .unwrap_or_else(|| self.ctx.working_dir.clone());

        tokio::fs::metadata(&search_path)
            .await
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "grep".into(),
                message: format_path_io_error("inspect search path", &search_path, &e),
            })?;

        let re = Regex::new(pattern).map_err(|e| AgentError::ToolExecution {
            tool_name: "grep".into(),
            message: format!("invalid regex pattern: {e}"),
        })?;

        let glob_pattern = if search_path.is_dir() {
            format!("{}/**/{include}", search_path.display())
        } else {
            search_path.to_string_lossy().to_string()
        };

        let mut results = Vec::new();
        let mut match_count = 0u64;
        let max_matches = 500u64;

        let walker = glob::glob(&glob_pattern).map_err(|e| AgentError::ToolExecution {
            tool_name: "grep".into(),
            message: format!("invalid glob pattern: {e}"),
        })?;

        for entry in walker {
            if match_count >= max_matches {
                results.push(format!("\n... truncated (max {max_matches} matches)"));
                break;
            }

            let path = match entry {
                Ok(p) => p,
                Err(_) => continue,
            };

            if !path.is_file() {
                continue;
            }

            let content = match tokio::fs::read_to_string(&path).await {
                Ok(c) => c,
                Err(_) => continue,
            };

            for (line_idx, line) in content.lines().enumerate() {
                if match_count >= max_matches {
                    break;
                }
                if re.is_match(line) {
                    let file_path = path.strip_prefix(&self.ctx.working_dir).unwrap_or(&path);
                    results.push(format!("{}:{}:{}", file_path.display(), line_idx + 1, line));
                    match_count += 1;
                }
            }
        }

        let output = if results.is_empty() {
            "no matches found".to_string()
        } else {
            results.join("\n")
        };

        let mut result = ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: "grep".into(),
            content: vec![ContentBlock::Text { text: output }],
            details: Some(serde_json::json!({
                "match_count": match_count,
                "pattern": pattern,
                "include": include
            })),
            is_error: false,
        };

        truncate_output(&mut result, &TruncationLimits::default());

        Ok(result)
    }
}
