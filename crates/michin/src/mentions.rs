//! File mention expansion for user text.

use std::path::{Component, Path, PathBuf};

use michin_ai::ContentBlock;

const MAX_FILES: usize = 16;
const MAX_FILE_BYTES: usize = 40 * 1024;
const MAX_TOTAL_BYTES: usize = 160 * 1024;

pub async fn expand_file_mentions(working_dir: &Path, text: &str) -> Vec<ContentBlock> {
    let mentions = extract_mentions(text);
    if mentions.is_empty() {
        return vec![ContentBlock::text(text)];
    }

    let mut sections = Vec::new();
    let mut total_bytes = 0usize;

    for mention in mentions.into_iter().take(MAX_FILES) {
        let Some(path) = resolve_mention(working_dir, &mention) else {
            continue;
        };
        let Ok(metadata) = tokio::fs::metadata(&path).await else {
            continue;
        };
        if !metadata.is_file() {
            continue;
        }
        let Ok(bytes) = tokio::fs::read(&path).await else {
            continue;
        };
        if total_bytes >= MAX_TOTAL_BYTES {
            break;
        }

        let available = MAX_TOTAL_BYTES.saturating_sub(total_bytes);
        let max_bytes = MAX_FILE_BYTES.min(available);
        let truncated = bytes.len() > max_bytes;
        let slice = &bytes[..bytes.len().min(max_bytes)];
        let content = String::from_utf8_lossy(slice);
        total_bytes += slice.len();

        let relative = path
            .strip_prefix(working_dir)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let suffix = if truncated {
            format!("\n[truncated at {max_bytes} bytes]")
        } else {
            String::new()
        };
        sections.push(format!(
            "## @{relative}\n```text\n{}{suffix}\n```",
            content.trim_end()
        ));
    }

    if sections.is_empty() {
        return vec![ContentBlock::text(text)];
    }

    vec![ContentBlock::text(format!(
        "{text}\n\n# Referenced files\n{}",
        sections.join("\n\n")
    ))]
}

fn resolve_mention(working_dir: &Path, mention: &str) -> Option<PathBuf> {
    let mention = mention.trim();
    if mention.is_empty() {
        return None;
    }
    let path = Path::new(mention);
    if path.is_absolute() {
        return safe_existing_path(path);
    }
    safe_existing_path(&working_dir.join(path))
}

fn safe_existing_path(path: &Path) -> Option<PathBuf> {
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return None;
    }
    path.exists().then(|| path.to_path_buf())
}

pub fn extract_mentions(text: &str) -> Vec<String> {
    let chars = text.char_indices().collect::<Vec<_>>();
    let mut out = Vec::new();
    let mut idx = 0usize;

    while idx < chars.len() {
        let (byte_idx, ch) = chars[idx];
        if ch != '@' || (byte_idx > 0 && is_path_char(text[..byte_idx].chars().last().unwrap())) {
            idx += 1;
            continue;
        }
        idx += 1;
        if idx >= chars.len() {
            break;
        }

        let (_, next) = chars[idx];
        if next == '"' || next == '\'' {
            let quote = next;
            idx += 1;
            let start = chars.get(idx).map(|(pos, _)| *pos).unwrap_or(text.len());
            while idx < chars.len() && chars[idx].1 != quote {
                idx += 1;
            }
            let end = chars.get(idx).map(|(pos, _)| *pos).unwrap_or(text.len());
            if end > start {
                out.push(text[start..end].to_string());
            }
            idx += 1;
            continue;
        }

        let start = chars[idx].0;
        while idx < chars.len() && is_path_char(chars[idx].1) {
            idx += 1;
        }
        let end = chars.get(idx).map(|(pos, _)| *pos).unwrap_or(text.len());
        if end > start {
            out.push(text[start..end].to_string());
        }
    }

    out.sort();
    out.dedup();
    out
}

fn is_path_char(ch: char) -> bool {
    ch.is_alphanumeric() || matches!(ch, '/' | '.' | '_' | '-' | ':' | '+' | '=')
}
