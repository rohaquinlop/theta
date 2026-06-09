//! fff_grep tool — content search powered by fff.
//!
//! Uses the FFF in-process grep index for fast content search with
//! typo-resistance and frecency context. Falls back to model-initiated
//! bash calls when FFF is unavailable.

use async_trait::async_trait;
use michin_agent_core::error::AgentError;
use michin_agent_core::types::{AgentTool, ToolExecutionMode, ToolResult, ToolUpdateSender};
use michin_ai::ContentBlock;
use tokio_util::sync::CancellationToken;

use crate::fff;

pub struct FffGrepTool {
    handle: fff::FffHandleRef,
}

impl FffGrepTool {
    pub fn new(handle: fff::FffHandleRef) -> Self {
        Self { handle }
    }
}

#[async_trait]
impl AgentTool for FffGrepTool {
    fn name(&self) -> &str {
        "grep"
    }

    fn description(&self) -> &str {
        "Search file contents in indexed files. Plain text by default — write code tokens \
         exactly as they appear: `parse_expr(` finds literal `parse_expr(`. Set `regex: true` \
         for patterns like `fn\\s+\\w+`. Use `path` to scope by directory (`src/`) or glob \
         (`*.rs`). This tool is FAST (in-process index) — prefer it over bash rg/grep."
    }

    fn label(&self) -> &str {
        "grep"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["query", "path"],
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory or file scope. Examples: 'src/', '*.rs', 'compiler/**/*.rs', '.' (all files). REQUIRED."
                },
                "query": {
                    "type": "string",
                    "description": "Search pattern. Plain text by default — write code tokens as-is: `parse_expr(`, `.parse_expr(`. No escaping needed. Set `regex: true` for regex patterns like `fn\\s+\\w+`."
                },
                "regex": {
                    "type": "boolean",
                    "description": "Enable regex mode (default: false). Only set true for actual regex patterns."
                },
                "case_sensitive": {
                    "type": "boolean",
                    "description": "Case-sensitive matching (default: false — smart-case)"
                },
                "max_results": {
                    "type": "number",
                    "description": "Maximum results to return (default 30, max 100)"
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
        let query = args["query"]
            .as_str()
            .ok_or_else(|| AgentError::ToolExecution {
                tool_name: "grep".into(),
                message: "missing required 'query' parameter".into(),
            })?;
        let path_constraint = match args["path"].as_str() {
            Some(p) if !p.is_empty() && p != "." => Some(p),
            _ => None,
        };
        let regex = args["regex"].as_bool().unwrap_or(false);
        let case_sensitive = args["case_sensitive"].as_bool().unwrap_or(false);
        let max_results = args["max_results"].as_u64().unwrap_or(30).min(100) as usize;

        let handle = match self.handle.lock() {
            Ok(guard) if guard.is_some() => guard,
            Ok(_) => {
                return Ok(ToolResult {
                    tool_call_id: tool_call_id.into(),
                    tool_name: "grep".into(),
                    content: vec![ContentBlock::Text {
                        text: "fff index not initialized for this project. Use bash with rg/grep instead."
                            .into(),
                    }],
                    details: None,
                    is_error: true,
                });
            }
            Err(e) => {
                return Ok(ToolResult {
                    tool_call_id: tool_call_id.into(),
                    tool_name: "grep".into(),
                    content: vec![ContentBlock::Text {
                        text: format!("fff handle lock error: {e}"),
                    }],
                    details: None,
                    is_error: true,
                });
            }
        };

        let handle = handle.as_ref().unwrap();
        let matches = fff::grep(
            handle,
            query,
            path_constraint,
            max_results,
            case_sensitive,
            regex,
        );

        match matches {
            None => {
                let scope_info = match path_constraint {
                    Some(p) => format!(" in files matching '{p}'"),
                    None => String::new(),
                };
                let hint = if path_constraint.is_some() {
                    "\nHint: try path=\".\" to search all indexed files, or use `find` to verify the path exists."
                } else {
                    ""
                };
                Ok(ToolResult {
                    tool_call_id: tool_call_id.into(),
                    tool_name: "grep".into(),
                    content: vec![ContentBlock::Text {
                        text: format!("No matches found for '{query}'{scope_info}{hint}"),
                    }],
                    details: Some(serde_json::json!({
                        "query": query,
                        "path": path_constraint.unwrap_or(""),
                        "results_count": 0,
                    })),
                    is_error: false,
                })
            }
            Some(matches) => {
                let count = matches.len();
                let text = matches
                    .iter()
                    .map(|m| {
                        format!(
                            "{}:{}:{}  {}",
                            m.path,
                            m.line_number,
                            m.column,
                            m.line_content.trim()
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n");

                let scope_info = match path_constraint {
                    Some(p) => format!(" in files matching '{p}'"),
                    None => " in ALL indexed files (no path filter)".to_string(),
                };

                Ok(ToolResult {
                    tool_call_id: tool_call_id.into(),
                    tool_name: "grep".into(),
                    content: vec![ContentBlock::Text {
                        text: format!(
                            "Found {count} match(es) for '{query}'{scope_info}:\n\n{text}"
                        ),
                    }],
                    details: Some(serde_json::json!({
                        "query": query,
                        "path": path_constraint.unwrap_or(""),
                        "results_count": count,
                        "matches": matches.iter().map(|m| serde_json::json!({
                            "path": m.path,
                            "line": m.line_number,
                            "col": m.column,
                            "content": m.line_content.trim(),
                        })).collect::<Vec<_>>(),
                    })),
                    is_error: false,
                })
            }
        }
    }
}
