//! System prompt construction.
//!
//! Builds the system prompt from project context files, tool descriptions,
//! available skills, and runtime context.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use theta_ai::ContentBlock;

use crate::scripts;
use crate::skills;
use crate::tools::{ToolContext, builtin_tools};

pub async fn build_system_prompt(
    working_dir: &Path,
    model_id: &str,
    thinking_level: Option<&str>,
    _latest_user_input: Option<&str>,
) -> Vec<ContentBlock> {
    let mut parts: Vec<String> = Vec::new();

    if let Some(ctx) = load_project_context(working_dir).await {
        parts.push(ctx);
    }

    // Available skills.
    let discovered = skills::discover_skills(working_dir).await;
    if let Some(skills_block) = skills::build_skills_prompt_block(&discovered) {
        parts.push(skills_block);
    }

    // Available extension scripts.
    let ext_scripts = scripts::discover_scripts(working_dir).await;
    if let Some(ext_block) = scripts::build_extensions_prompt_block(&ext_scripts) {
        parts.push(ext_block);
    }

    let tools_prompt = build_tools_prompt(working_dir);
    if !tools_prompt.is_empty() {
        parts.push(tools_prompt);
    }

    parts.push(build_runtime_context(working_dir, model_id, thinking_level));
    parts.push(RESPONSE_CONTRACT.to_string());

    vec![ContentBlock::Text {
        text: parts.join(
            "

",
        ),
    }]
}

async fn load_project_context(working_dir: &Path) -> Option<String> {
    let agents_path = find_context_file(working_dir, "AGENTS.md").await?;
    let agents = tokio::fs::read_to_string(&agents_path).await.ok()?;
    let mut context = format!(
        "# Project Context

{agents}"
    );

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

    let theta_ctx = working_dir.join(".theta").join("context.md");
    if theta_ctx.exists()
        && let Ok(ctx) = tokio::fs::read_to_string(&theta_ctx).await
    {
        context.push_str(&format!(
            "

# Theta Context

{ctx}"
        ));
    }

    Some(context)
}

async fn find_context_file(start: &Path, filename: &str) -> Option<PathBuf> {
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

fn build_tools_prompt(working_dir: &Path) -> String {
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

fn build_runtime_context(
    working_dir: &Path,
    model_id: &str,
    thinking_level: Option<&str>,
) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let date = chrono_now(now);
    let cwd_display = working_dir.display();

    let shell = std::env::var("SHELL").unwrap_or_else(|_| "unknown".into());
    let os = std::env::consts::OS;

    format!(
        "# Runtime Context

         Current date: {date}
         Working directory: {cwd_display}
         Shell: {shell}
         OS: {os}
         Model: {model_id}
         Thinking level: {}",
        thinking_level.unwrap_or("default")
    )
}

// Simple date formatting without chrono dependency.
fn chrono_now(ts: u64) -> String {
    // Approximate: seconds since epoch -> YYYY-MM-DD.
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

/// Pi-style Response Contract — a universal set of rules the model
/// follows on every turn. No heuristic intent detection, no mode gating.
/// The contract drives all behavior: analysis vs execution, tool discipline,
/// and completion protocol.
const RESPONSE_CONTRACT: &str = r#"# Response Contract

You are a coding agent inside Theta, a Rust-based terminal harness. Your
behavior is governed by the rules below. Follow them on every turn.

## Execution Continuity

When the user asks you to implement, fix, change, or write code, do the full
implementation in this turn: run tools, apply edits, and validate. Do not
stop at a plan, status update, or promise to do something later.

Every reply must be exactly one of these states:
- DONE: changes applied, validations passed, results reported.
- BLOCKED: cannot continue without user input/decision/permission.
- FAILED: tool/runtime failure with error output and retry step.

No premature turn end. No "I'll do X" without doing X.

## Analysis vs Execution

When the user asks a question, describes a problem, asks "how to fix"
something, or requests analysis/investigation — ANALYZE AND REPORT.
Do not implement. Use read tools selectively (3-5 files max) to verify
your understanding. Then summarize findings and ask whether the user
wants implementation.

When the user explicitly asks you to implement, fix, change, refactor,
or write code — IMPLEMENT. Do not stop at a plan.

## Tool Use Discipline

- Read files before editing them.
- Be selective — read only relevant files, not the entire codebase.
- After reading 3-5 files, stop and report findings.
- Do not enter recursive exploration loops. If you find yourself reading the
  same file twice, you are in a loop — stop and report.
- Invoke tools using function-calling, not prose, plans, or XML-like text.
- For changes, use edit (exact text replacement) for partial edits. Use write
  for creating new files or full-file rewrites. Use bash with cat <<'EOF'
  when shell operations are needed.
- If blocked by missing info/permission, ask one precise question and stop.
- Report what you changed and validation results after completing tool calls.

The following skills provide specialized instructions for specific tasks.
Use the read tool to load a skill's SKILL.md file when the task matches its
description. When a skill file references a relative path, resolve it against
the skill directory.

## Theta Extensions

Theta supports Rhai script extensions in `~/.theta/extensions/*.rhai` (global)
and `./.theta/extensions/*.rhai` (project-local). Extensions take effect on
the next session (loaded at agent startup).

### Tool-Call Hooks

Scripts use `tool.before(name, callback)` and `tool.after(name, callback)`
to intercept tool calls at runtime.

### TUI Status Lines

Scripts use `tui.status(key, callback)` to display a status line in the TUI
status bar. The callback must return a string.

CRITICAL — Only create an extension when the user uses one of these EXACT
phrases:
- "create an extension" / "write an extension" / "make an extension"
- "add a tool hook" / "add a before hook" / "add an after hook"
- "add a status line" / "add a TUI status" / "add an extension status"
- "install an extension"
- "I want to extend theta" / "how do I extend theta"

Do NOT create an extension from general task language.

When the user says "modify theta" or "extend theta" without specifying how,
ask: 1) A skill (Markdown file), 2) An extension (Rhai script), or 3) A Rust
change (fork + recompile)."#;

#[cfg(test)]
mod tests {
    use super::{RESPONSE_CONTRACT, build_tools_prompt};
    use std::path::Path;

    #[test]
    fn tools_prompt_uses_function_calling_not_xml() {
        let s = build_tools_prompt(Path::new("."));
        assert!(s.contains("native function-calling"));
        assert!(!s.contains("XML invocation format"));
    }

    #[test]
    fn response_contract_contains_execution_continuity() {
        assert!(RESPONSE_CONTRACT.contains("DONE"));
        assert!(RESPONSE_CONTRACT.contains("BLOCKED"));
        assert!(RESPONSE_CONTRACT.contains("FAILED"));
    }

    #[test]
    fn response_contract_contains_analysis_vs_execution() {
        assert!(RESPONSE_CONTRACT.contains("ANALYZE AND REPORT"));
        assert!(RESPONSE_CONTRACT.contains("IMPLEMENT"));
    }

    #[test]
    fn response_contract_contains_tool_discipline() {
        assert!(RESPONSE_CONTRACT.contains("3-5 files"));
        assert!(RESPONSE_CONTRACT.contains("function-calling"));
    }
}
