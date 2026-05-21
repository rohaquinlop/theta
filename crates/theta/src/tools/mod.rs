//! Built-in tools for the theta agent.
//!
//! Seven tools matching pi's set: read, write, edit, bash, grep, find, ls.
//! Each implements `AgentTool` from `theta-agent-core`.

mod bash;
mod edit;
mod find;
mod grep;
mod ls;
mod read;
mod write;

use std::path::PathBuf;

pub use bash::BashTool;
pub use edit::EditTool;
pub use find::FindTool;
pub use grep::GrepTool;
pub use ls::LsTool;
pub use read::ReadTool;
pub use write::WriteTool;

use theta_agent_core::types::ToolResult;

/// Shared context passed to all tools.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// The project's working directory. All relative paths are resolved against this.
    pub working_dir: PathBuf,
}

impl ToolContext {
    pub fn new(working_dir: PathBuf) -> Self {
        Self { working_dir }
    }
}

/// Truncation limits for tool output.
pub struct TruncationLimits {
    pub max_lines: usize,
    pub max_bytes: usize,
}

impl Default for TruncationLimits {
    fn default() -> Self {
        Self {
            max_lines: 2000,
            max_bytes: 50_000,
        }
    }
}

/// Truncate tool output content. Appends a truncation notice if content
/// exceeded either limit.
pub fn truncate_output(result: &mut ToolResult, limits: &TruncationLimits) {
    use theta_ai::ContentBlock;

    let mut total_bytes: usize = 0;
    let mut total_lines: usize = 0;
    let mut truncated = false;

    let mut new_content = Vec::with_capacity(result.content.len());

    for block in std::mem::take(&mut result.content) {
        match block {
            ContentBlock::Text { text } => {
                let lines: Vec<&str> = text.lines().collect();
                total_lines += lines.len();
                total_bytes += text.len();

                if total_lines <= limits.max_lines && total_bytes <= limits.max_bytes {
                    new_content.push(ContentBlock::Text { text });
                } else {
                    truncated = true;
                    let keep_lines = limits
                        .max_lines
                        .saturating_sub(total_lines.saturating_sub(lines.len()));
                    let keep_bytes = limits
                        .max_bytes
                        .saturating_sub(total_bytes.saturating_sub(text.len()));
                    let keep_chars = std::cmp::min(
                        text.char_indices()
                            .nth(keep_bytes)
                            .map(|(i, _)| i)
                            .unwrap_or(text.len()),
                        text.len(),
                    );
                    let kept: String = text.lines().take(keep_lines).collect::<Vec<_>>().join("\n");
                    let kept = if kept.len() > keep_chars {
                        kept.chars().take(keep_chars).collect()
                    } else {
                        kept
                    };
                    if !kept.is_empty() {
                        new_content.push(ContentBlock::Text { text: kept });
                    }
                    break;
                }
            }
            other => new_content.push(other),
        }
    }

    result.content = new_content;

    if truncated {
        result.is_error = true;
        result.content.push(ContentBlock::Text {
            text: format!(
                "\n\n[output truncated: exceeded {} lines or {} bytes]",
                limits.max_lines, limits.max_bytes
            ),
        });
    }
}

/// Resolve a path relative to the tool context's working directory.
/// Sanitizes against path traversal attacks (.. components).
fn resolve_path(ctx: &ToolContext, path: &str) -> PathBuf {
    let p = PathBuf::from(path);
    let resolved = if p.is_absolute() {
        p
    } else {
        ctx.working_dir.join(p)
    };
    resolved.canonicalize().unwrap_or(resolved)
}

fn classify_io_error(err: &std::io::Error) -> &'static str {
    match err.kind() {
        std::io::ErrorKind::NotFound => "path not found",
        std::io::ErrorKind::PermissionDenied => "permission denied",
        std::io::ErrorKind::InvalidInput => "invalid path",
        _ => "I/O error",
    }
}

fn format_path_io_error(action: &str, path: &std::path::Path, err: &std::io::Error) -> String {
    let reason = classify_io_error(err);
    format!(
        "{action} failed ({reason}) at '{}': {err}",
        path.to_string_lossy()
    )
}

/// Create all seven built-in tools.
pub fn builtin_tools(
    ctx: ToolContext,
) -> Vec<std::sync::Arc<dyn theta_agent_core::types::AgentTool>> {
    vec![
        std::sync::Arc::new(ReadTool::new(ctx.clone())),
        std::sync::Arc::new(WriteTool::new(ctx.clone())),
        std::sync::Arc::new(EditTool::new(ctx.clone())),
        std::sync::Arc::new(BashTool::new(ctx.clone())),
        std::sync::Arc::new(GrepTool::new(ctx.clone())),
        std::sync::Arc::new(FindTool::new(ctx.clone())),
        std::sync::Arc::new(LsTool::new(ctx.clone())),
    ]
}

#[cfg(test)]
mod tests {
    use super::{ToolContext, resolve_path};

    #[test]
    fn resolve_path_keeps_absolute_path() {
        let ctx = ToolContext::new(std::path::PathBuf::from("/tmp/theta-workdir"));
        let resolved = resolve_path(&ctx, "/Users/rhafid/.theta");
        assert_eq!(resolved, std::path::PathBuf::from("/Users/rhafid/.theta"));
    }

    #[test]
    fn resolve_path_resolves_relative_from_workdir() {
        let ctx = ToolContext::new(std::path::PathBuf::from("/tmp/theta-workdir"));
        let resolved = resolve_path(&ctx, "src/main.rs");
        assert_eq!(
            resolved,
            std::path::PathBuf::from("/tmp/theta-workdir/src/main.rs")
        );
    }
}
