//! fff_find tool — fuzzy file search powered by fff.
//!
//! Uses the FFF in-process index for frecency-ranked file search.
//! Falls back to model-initiated bash calls when FFF is unavailable.

use async_trait::async_trait;
use michin_agent_core::error::AgentError;
use michin_agent_core::types::{AgentTool, ToolExecutionMode, ToolResult, ToolUpdateSender};
use michin_ai::ContentBlock;
use tokio_util::sync::CancellationToken;

use crate::fff;

pub struct FffFindTool {
    handle: fff::FffHandleRef,
}

impl FffFindTool {
    pub fn new(handle: fff::FffHandleRef) -> Self {
        Self { handle }
    }
}

#[async_trait]
impl AgentTool for FffFindTool {
    fn name(&self) -> &str {
        "find"
    }

    fn description(&self) -> &str {
        "Search for files by name/path pattern. Results ranked by frecency \
         (recent/frequent files first). Searches file NAMES only, not contents. \
         Use for locating files before reading/editing. This tool is FAST — \
         prefer it over bash find/fd/ls."
    }

    fn label(&self) -> &str {
        "find"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Filename or path pattern to search for. Examples: 'lib.rs' (by name), 'src/tools' (by directory), '*.md' (by extension), 'src/**/*.rs !test/' (path + exclusion)"
                },
                "max_results": {
                    "type": "number",
                    "description": "Maximum number of results to return (default 20)"
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
                tool_name: "find".into(),
                message: "missing required 'query' parameter".into(),
            })?;
        let max_results = args["max_results"].as_u64().unwrap_or(20).min(100) as usize;

        let handle = match self.handle.lock() {
            Ok(guard) if guard.is_some() => guard,
            Ok(_) => {
                return Ok(ToolResult {
                    tool_call_id: tool_call_id.into(),
                    tool_name: "find".into(),
                    content: vec![ContentBlock::Text {
                        text: "fff index not initialized for this project. Use bash with find/fd instead."
                            .into(),
                    }],
                    details: None,
                    is_error: true,
                });
            }
            Err(e) => {
                return Ok(ToolResult {
                    tool_call_id: tool_call_id.into(),
                    tool_name: "find".into(),
                    content: vec![ContentBlock::Text {
                        text: format!("fff handle lock error: {e}"),
                    }],
                    details: None,
                    is_error: true,
                });
            }
        };

        let handle = handle.as_ref().unwrap();
        let results = fff::fuzzy_find(handle, query, max_results);

        if results.is_empty() {
            return Ok(ToolResult {
                tool_call_id: tool_call_id.into(),
                tool_name: "find".into(),
                content: vec![ContentBlock::Text {
                    text: format!("No files found matching '{query}'"),
                }],
                details: Some(serde_json::json!({
                    "query": query,
                    "results_count": 0,
                    "results": [],
                })),
                is_error: false,
            });
        }

        let count = results.len();
        let text = results
            .iter()
            .enumerate()
            .map(|(i, path)| format!("{}. {}", i + 1, path))
            .collect::<Vec<_>>()
            .join("\n");

        Ok(ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: "find".into(),
            content: vec![ContentBlock::Text {
                text: format!("Found {count} file(s) matching '{query}':\n\n{text}"),
            }],
            details: Some(serde_json::json!({
                "query": query,
                "results_count": count,
                "results": results,
            })),
            is_error: false,
        })
    }
}
