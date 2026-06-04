//! System prompt construction.
//!
//! Two outputs:
//! - `build_system_prompt()` — core operational instructions (project context,
//!   tools, runtime, response contract). Set via `agent.set_system_prompt()`.
//! - `build_resource_context()` — available resources (skills, extensions, auto-loading directive). Set via `agent.set_resource_context()`.
//!
//! Both are sent as `system` role: the resource context is appended to the
//! system prompt before each LLM call.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use michin_ai::ContentBlock;

use crate::scripts;
use crate::skills;
use crate::tools::{ToolContext, builtin_tools};

/// Build the core system prompt: project context + tools + runtime + response contract.
/// Does NOT include skills or extensions.
pub async fn build_system_prompt(
    working_dir: &Path,
    model_id: &str,
    thinking_level: Option<&str>,
    max_context_window: Option<u32>,
) -> Vec<ContentBlock> {
    let mut parts: Vec<String> = Vec::new();

    if let Some(ctx) = load_project_context(working_dir).await {
        parts.push(ctx);
    }

    let tools_prompt = build_tools_prompt(working_dir);
    if !tools_prompt.is_empty() {
        parts.push(tools_prompt);
    }

    parts.push(build_runtime_context(
        working_dir,
        model_id,
        thinking_level,
        max_context_window,
    ));
    parts.push(RESPONSE_CONTRACT.to_string());

    let text = parts.join("\n\n");

    let michin_dir = crate::config::michin_dir();
    let text = apply_system_prompt_overrides(&michin_dir, text).await;

    vec![ContentBlock::Text { text }]
}

/// Build the resource context: skills + extensions + auto-loading.
/// This gets appended to the system prompt before each LLM call.
pub async fn build_resource_context(working_dir: &Path) -> Vec<ContentBlock> {
    let mut parts: Vec<String> = Vec::new();

    // Available skills.
    let discovered = skills::discover_skills(working_dir).await;

    if let Some(skills_block) = skills::build_skills_prompt_block(&discovered) {
        parts.push(skills_block);
        parts.push(
            "## Skills

\
             When a user message matches a skill's description, you MUST read the \
             skill file before responding. Announce the skill name when loading it."
                .to_string(),
        );
    }

    // Available extension scripts — slim: name + location only, no source.
    let ext_scripts = scripts::discover_scripts(working_dir).await;
    if let Some(ext_block) = scripts::build_extensions_slim_block(&ext_scripts) {
        parts.push(ext_block);
    }

    // Extension-creation guardrails. Always present so the model knows when
    // to create one — even if no extensions are installed yet.
    parts.push(EXTENSION_CREATION_GUARDRAILS.to_string());

    if parts.is_empty() {
        return Vec::new();
    }

    let text = parts.join("\n\n");
    vec![ContentBlock::Text { text }]
}

/// Compatibility wrapper — builds both system prompt and resource context.
/// Used by call sites that previously called `build_system_prompt_with_skills`.
/// Returns (system_prompt, resource_context).
pub async fn build_system_prompt_with_skills(
    working_dir: &Path,
    model_id: &str,
    thinking_level: Option<&str>,
    max_context_window: Option<u32>,
) -> (Vec<ContentBlock>, Vec<ContentBlock>) {
    let system =
        build_system_prompt(working_dir, model_id, thinking_level, max_context_window).await;
    let resource = build_resource_context(working_dir).await;
    (system, resource)
}

// ── Resource context builders ──────────────────────────────────────

/// Extension-creation guardrails injected into the resource context.
/// Tells the model when it should (and should NOT) create Rhai scripts.
pub const EXTENSION_CREATION_GUARDRAILS: &str = r#"## Extensions

Do not create an extension when the user is working on their own project.
Only create one when the user explicitly asks to extend MichiN's behavior.

For ambiguous requests like "extend michin", ask whether the user means a
skill, extension, or code change."#;

// ── Project context discovery ──────────────────────────────────────

async fn load_project_context(working_dir: &Path) -> Option<String> {
    let agents_path = find_context_file(working_dir, "AGENTS.md").await?;
    let agents = tokio::fs::read_to_string(&agents_path).await.ok()?;
    let project_root = agents_path.parent().unwrap_or(working_dir);

    let mut context = format!(
        "# Project Context

{agents}"
    );

    let nested = discover_nested_agents(project_root).await;
    for (relative_path, content) in &nested {
        context.push_str(&format!(
            "

# Crate Context: {relative_path}

{content}"
        ));
    }

    // CLAUDE.md (only if different from root AGENTS.md)
    if let Some(claude_path) = find_context_file(working_dir, "CLAUDE.md").await
        && claude_path != agents_path
        && let Ok(claude) = tokio::fs::read_to_string(&claude_path).await
    {
        context.push_str(&format!(
            "

# Additional Context

{claude}"
        ));
    }

    // MichiN context file
    let ctx_path = working_dir.join(".michin").join("context.md");
    if ctx_path.exists()
        && let Ok(ctx) = tokio::fs::read_to_string(&ctx_path).await
    {
        context.push_str(&format!(
            "

# MichiN Context

{ctx}"
        ));
    }

    Some(context)
}

pub async fn discover_nested_agents(root: &Path) -> Vec<(String, String)> {
    let mut results = Vec::new();
    let mut dirs: Vec<PathBuf> = vec![root.to_path_buf()];

    while let Some(current) = dirs.pop() {
        if current != root {
            let candidate = current.join("AGENTS.md");
            if candidate.exists()
                && let Ok(content) = tokio::fs::read_to_string(&candidate).await
                && let Ok(relative) = candidate.strip_prefix(root)
            {
                let rel_path = relative
                    .parent()
                    .map(|p| p.to_string_lossy().to_string())
                    .unwrap_or_default();
                if !rel_path.is_empty() {
                    results.push((rel_path, content));
                }
            }
        }

        let mut entries = match tokio::fs::read_dir(&current).await {
            Ok(entries) => entries,
            Err(_) => continue,
        };

        while let Ok(Some(entry)) = entries.next_entry().await {
            let name = entry.file_name().to_string_lossy().to_string();

            if entry.file_type().await.is_ok_and(|ft| ft.is_dir()) {
                if is_ignorable_dir(&name) {
                    continue;
                }
                dirs.push(entry.path());
            }
        }
    }

    results.sort_by(|a, b| a.0.cmp(&b.0));
    results
}

pub fn is_ignorable_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | "target"
            | "node_modules"
            | ".michin"
            | ".github"
            | ".vscode"
            | ".idea"
            | ".DS_Store"
    ) || name.starts_with('.')
}

pub async fn find_context_file(start: &Path, filename: &str) -> Option<PathBuf> {
    let mut current = if start.is_absolute() {
        start.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(start)
    };
    loop {
        let candidate = current.join(filename);
        if candidate.exists() {
            return Some(candidate);
        }
        if !current.pop() {
            break;
        }
    }
    None
}

// ── Tools prompt ───────────────────────────────────────────────────

pub fn build_tools_prompt(working_dir: &Path) -> String {
    let ctx = ToolContext::new(working_dir.to_path_buf());
    let tools = builtin_tools(ctx);
    if tools.is_empty() {
        return String::new();
    }

    let mut p = String::from(
        "# Available Tools

",
    );
    p.push_str(
        "You have access to these tools via native function-calling. Invoke tools directly, not by writing XML or pseudo-calls in text.\n\n",
    );
    p.push_str(
        "**CRITICAL — Tool Discipline:** Use `read`, `write`, and `edit` for ALL file operations (reading, searching within files, editing, creating). Use `bash` ONLY for shell commands that these tools cannot handle: running tests, building, git operations, package managers, etc. Never use `bash` with `cat`, `sed`, `python3`, `grep` on a known file, or similar to read or manipulate files when the `read`, `write`, or `edit` tools can do the job.\n\n",
    );

    for tool in &tools {
        p.push_str(&format!(
            "## {}
",
            tool.name()
        ));
        p.push_str(&format!(
            "{}
",
            tool.description()
        ));
        p.push_str(&format!(
            "Parameters: {}

",
            serde_json::to_string_pretty(&tool.parameters()).unwrap_or_default()
        ));
    }

    p
}

// ── Runtime context ────────────────────────────────────────────────

fn build_runtime_context(
    working_dir: &Path,
    model_id: &str,
    thinking_level: Option<&str>,
    max_context_window: Option<u32>,
) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let date = chrono_now(now);
    let cwd_display = working_dir.display();
    let shell = std::env::var("SHELL").unwrap_or_else(|_| "unknown".into());
    let os = std::env::consts::OS;

    let ctx_cap = match max_context_window {
        Some(n) => {
            let formatted = format_number(n);
            format!("\n         Context window cap: {formatted} tokens (set in settings)")
        }
        None => String::new(),
    };

    format!(
        "# Runtime Context

         Current date: {date}
         Working directory: {cwd_display}
         Shell: {shell}
         OS: {os}
         Model: {model_id}
         Thinking level: {}{ctx_cap}",
        thinking_level.unwrap_or("default")
    )
}

fn chrono_now(ts: u64) -> String {
    let days_since_epoch = ts / 86400;
    let mut year = 1970i64;
    let mut remaining = days_since_epoch as i64;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }

    let days_in_month = if is_leap_year(year) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 0;
    for (i, &dim) in days_in_month.iter().enumerate() {
        if remaining < dim as i64 {
            month = i + 1;
            break;
        }
        remaining -= dim as i64;
    }

    let day = remaining + 1;
    format!("{:04}-{:02}-{:02}", year, month, day)
}

fn is_leap_year(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

// ── Response Contract ──────────────────────────────────────────────
//
// Slimmed down: only core behavioral directives. Skill auto-loading,
// extension creation docs live in the resource context, not here.

pub const RESPONSE_CONTRACT: &str = r#"# Response Contract

You are a coding agent. Do not narrate your actions — just do them.

## Turn Completion

When asked to implement or change code, finish the full cycle in one turn:
apply edits, validate (run tests/lint when available), report results.
Do not stop at a plan or promise.

When the user asks a question or requests analysis, do not implement changes.
Summarize findings and ask before modifying code.

## Tool Discipline

- **CRITICAL:** Use `read`, `write`, and `edit` for ALL file operations (reading, searching within files, editing, creating). Use `bash` ONLY for shell commands these tools cannot handle: running tests, builds, git, package managers, etc. Never use `bash` with `cat`, `sed`, `python3`, `grep` on a known file, or similar to read or manipulate files when the dedicated file tools can do the job.
- Read files before editing them.
- When a tool call fails, attempt to fix the issue and retry once before reporting an error.
- Do not repeat identical tool calls in a loop.

## Resources

Skills and extensions are listed in the conversation context. When a message
matches a skill's trigger, read its file and follow its instructions."#;

// ── System prompt overrides ────────────────────────────────────────

pub async fn apply_system_prompt_overrides(michin_dir: &Path, mut text: String) -> String {
    let sys_prompt_path = michin_dir.join("SYSTEM_PROMPT.md");
    let append_path = michin_dir.join("APPEND_SYSTEM_PROMPT.md");

    if sys_prompt_path.exists() {
        if let Ok(content) = tokio::fs::read_to_string(&sys_prompt_path).await {
            text = content;
        }
    } else if append_path.exists()
        && let Ok(content) = tokio::fs::read_to_string(&append_path).await
        && !content.is_empty()
    {
        text.push_str("\n\n");
        text.push_str(&content);
    }

    text
}

/// Format a u32 with thousands separators.
fn format_number(n: u32) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push('_');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
