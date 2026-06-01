//! Built-in tools for the theta agent.
//!
//! Four core tools: read, edit, write, bash. The agent uses bash for grep,
//! find, ls, sed, and other operations via shell commands (rg, find, ls, sed, etc.).

pub mod bash;
pub mod edit;
pub mod read;
pub mod write;

use std::path::{Component, Path, PathBuf};

pub use bash::BashTool;
pub use edit::EditTool;
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
        result.content.push(ContentBlock::Text {
            text: format!(
                "\n\n[output truncated: exceeded {} lines or {} bytes]",
                limits.max_lines, limits.max_bytes
            ),
        });
    }
}

/// Resolve a tool path against the working directory.
///
/// - Absolute paths are honored directly (not clamped to working dir).
/// - Relative paths are resolved against `working_dir`.
/// - `..` components are collapsed manually. Symlinks are NOT followed,
///   so a symlink inside the project pointing to /etc/passwd resolves
///   as `working_dir/symlink_name`, not the target.
/// - If `..` traversal escapes `working_dir`, the path is clamped.
pub fn resolve_path(ctx: &ToolContext, path: &str) -> PathBuf {
    let p = Path::new(path);

    if p.is_absolute() {
        // Absolute paths are honored directly — no clamping.
        return manually_resolve(p);
    }

    // Resolve relative path against working_dir.
    let resolved = manually_resolve(&ctx.working_dir.join(p));

    // Security: if .. traversal escaped working_dir, clamp it.
    if resolved.starts_with(&ctx.working_dir) {
        resolved
    } else {
        ctx.working_dir.clone()
    }
}

/// Resolve `.` and `..` components manually without following symlinks.
fn manually_resolve(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

fn classify_io_error(err: &std::io::Error) -> &'static str {
    match err.kind() {
        std::io::ErrorKind::NotFound => "path not found",
        std::io::ErrorKind::PermissionDenied => "permission denied",
        std::io::ErrorKind::InvalidInput => "invalid path",
        _ => "I/O error",
    }
}

/// Shorten an absolute path for display by replacing the home directory
/// prefix with `~`. If the path is not under the home directory, returns
/// the path unchanged.
pub fn shorten_path(path: &Path) -> String {
    if let Some(home) = dirs::home_dir()
        && let Ok(rest) = path.strip_prefix(&home)
    {
        return format!("~/{}", rest.display());
    }
    path.display().to_string()
}

fn format_path_io_error(action: &str, path: &Path, err: &std::io::Error) -> String {
    let reason = classify_io_error(err);
    format!(
        "{action} failed ({reason}) at '{}': {err}",
        shorten_path(path)
    )
}

/// Create all seven built-in tools.
pub fn builtin_tools(
    ctx: ToolContext,
) -> Vec<std::sync::Arc<dyn theta_agent_core::types::AgentTool>> {
    vec![
        std::sync::Arc::new(ReadTool::new(ctx.clone())),
        std::sync::Arc::new(EditTool::new(ctx.clone())),
        std::sync::Arc::new(WriteTool::new(ctx.clone())),
        std::sync::Arc::new(BashTool::new(ctx.clone())),
    ]
}
