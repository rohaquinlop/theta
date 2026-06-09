//! Built-in tools for the michin agent.
//!
//! Six core tools: read, edit, write, bash, fff_find, fff_grep.
//! Fff_find and fff_grep use the FFF in-process search index for
//! frecency-ranked file and content search.

pub mod bash;
pub mod edit;
pub mod fff_find;
pub mod fff_grep;
pub mod read;
pub mod write;

use std::path::{Component, Path, PathBuf};

pub use bash::BashTool;
pub use edit::EditTool;
pub use fff_find::FffFindTool;
pub use fff_grep::FffGrepTool;
pub use read::ReadTool;
pub use write::WriteTool;

use michin_agent_core::types::ToolResult;

/// Shared context passed to all tools.
#[derive(Debug, Clone)]
pub struct ToolContext {
    /// The project's working directory. All relative paths are resolved against this.
    pub working_dir: PathBuf,
    /// Optional FFF handle for frecency updates on file ops.
    pub fff_handle: Option<crate::fff::FffHandleRef>,
}

impl ToolContext {
    pub fn new(working_dir: PathBuf) -> Self {
        Self {
            working_dir,
            fff_handle: None,
        }
    }

    pub fn with_fff(working_dir: PathBuf, fff_handle: crate::fff::FffHandleRef) -> Self {
        Self {
            working_dir,
            fff_handle: Some(fff_handle),
        }
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
    use michin_ai::ContentBlock;

    // Collect all text content into one combined string. Non-text blocks
    // (images, etc.) are preserved in their original position.
    let mut combined_text = String::new();
    let mut new_content = Vec::with_capacity(result.content.len());

    for block in std::mem::take(&mut result.content) {
        match block {
            ContentBlock::Text { text } => {
                if !combined_text.is_empty() {
                    combined_text.push('\n');
                }
                combined_text.push_str(&text);
            }
            other => {
                // Flush accumulated text before the non-text block.
                if !combined_text.is_empty() {
                    new_content.push(ContentBlock::Text {
                        text: std::mem::take(&mut combined_text),
                    });
                }
                new_content.push(other);
            }
        }
    }

    // Flush any remaining text.
    if !combined_text.is_empty() {
        new_content.push(ContentBlock::Text {
            text: combined_text,
        });
    }

    // Now truncate from the head: keep only the last N lines / last ~50KB.
    let total_lines = new_content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.lines().count()),
            _ => None,
        })
        .sum::<usize>();
    let total_bytes: usize = new_content
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.len()),
            _ => None,
        })
        .sum();

    if total_lines <= limits.max_lines && total_bytes <= limits.max_bytes {
        result.content = new_content;
        return;
    }

    // Truncate: drop leading text blocks until we fit within limits, then
    // trim the first remaining text block from the head to hit the exact limit.
    let mut truncated = new_content;
    let mut lines_to_drop = total_lines.saturating_sub(limits.max_lines);
    let mut bytes_to_drop = total_bytes.saturating_sub(limits.max_bytes);

    // Remove leading text blocks that fall entirely within the drop budget.
    while let Some(first) = truncated.first() {
        if lines_to_drop == 0 && bytes_to_drop == 0 {
            break;
        }
        match first {
            ContentBlock::Text { text } => {
                let block_lines = text.lines().count();
                let block_bytes = text.len();
                if block_lines <= lines_to_drop && block_bytes <= bytes_to_drop {
                    lines_to_drop -= block_lines;
                    bytes_to_drop -= block_bytes;
                    truncated.remove(0);
                } else {
                    break;
                }
            }
            _ => break, // non-text block — stop, keep it
        }
    }

    // Trim the first remaining text block from the head.
    #[allow(clippy::collapsible_if)]
    if lines_to_drop > 0 || bytes_to_drop > 0 {
        if let Some(ContentBlock::Text { text }) = truncated.first_mut() {
            let mut lines: Vec<&str> = text.lines().collect();
            let drop_lines = lines_to_drop.min(lines.len());
            lines.drain(..drop_lines);

            let mut remaining: String = lines.join("\n");
            // Recalculate byte excess after line-draining, then trim from
            // the head to hit the exact byte limit. Walk forward to the
            // next char boundary so the slice is always valid UTF-8.
            let byte_excess = remaining.len().saturating_sub(limits.max_bytes);
            if byte_excess > 0 {
                let mut start = byte_excess.min(remaining.len());
                while start < remaining.len() && !remaining.is_char_boundary(start) {
                    start += 1;
                }
                remaining = remaining[start..].to_string();
            }
            *text = remaining;
        }
        // Remove text blocks that became empty after trimming.
        truncated.retain(|b| !matches!(b, ContentBlock::Text { text } if text.is_empty()));
    }

    result.content = truncated;
    result.content.push(ContentBlock::Text {
        text: format!(
            "\n\n[output truncated to last {} lines or {} bytes; {} lines / {} bytes dropped from head]",
            limits.max_lines, limits.max_bytes, total_lines.saturating_sub(limits.max_lines), total_bytes.saturating_sub(limits.max_bytes)
        ),
    });
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

/// Touch a file's frecency score in the FFF index after a file operation.
pub(crate) fn touch_fff_frecency(ctx: &ToolContext, resolved_path: &Path) {
    if let Some(ref handle_opt) = ctx.fff_handle
        && let Ok(guard) = handle_opt.lock()
        && let Some(ref handle) = *guard
        && let Ok(relative) = resolved_path.strip_prefix(&ctx.working_dir)
    {
        crate::fff::touch_frecency(handle, relative);
    }
}

/// Create all built-in tools.
pub fn builtin_tools(
    ctx: ToolContext,
    fff_handle: Option<crate::fff::FffHandleRef>,
) -> Vec<std::sync::Arc<dyn michin_agent_core::types::AgentTool>> {
    let mut tools: Vec<std::sync::Arc<dyn michin_agent_core::types::AgentTool>> = vec![
        std::sync::Arc::new(ReadTool::new(ctx.clone())),
        std::sync::Arc::new(EditTool::new(ctx.clone())),
        std::sync::Arc::new(WriteTool::new(ctx.clone())),
        std::sync::Arc::new(BashTool::new(ctx.clone())),
    ];

    if let Some(handle) = fff_handle {
        tools.push(std::sync::Arc::new(FffFindTool::new(handle.clone())));
        tools.push(std::sync::Arc::new(FffGrepTool::new(handle)));
    }

    tools
}
