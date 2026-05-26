//! System prompt construction.
//!
//! Builds the system prompt from project context files, tool descriptions,
//! available skills, and runtime context.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use theta_ai::ContentBlock;

use crate::scripts;
use crate::skills;
use crate::skills::Skill;
use crate::tools::{ToolContext, builtin_tools};

pub async fn build_system_prompt(
    working_dir: &Path,
    model_id: &str,
    thinking_level: Option<&str>,
    _latest_user_input: Option<&str>,
) -> Vec<ContentBlock> {
    build_system_prompt_with_skills(working_dir, model_id, thinking_level, &[]).await
}

/// Build system prompt with active startup skills injected at system-prompt authority.
pub async fn build_system_prompt_with_skills(
    working_dir: &Path,
    model_id: &str,
    thinking_level: Option<&str>,
    startup_skills: &[String],
) -> Vec<ContentBlock> {
    let mut parts: Vec<String> = Vec::new();

    if let Some(ctx) = load_project_context(working_dir).await {
        parts.push(ctx);
    }

    // Available skills.
    let discovered = skills::discover_skills(working_dir).await;

    // Active startup skills — inject their full bodies at system-prompt level.
    if !startup_skills.is_empty()
        && let Some(active_block) = build_active_skills_block(&discovered, startup_skills)
    {
        parts.push(active_block);
    }

    if let Some(skills_block) = skills::build_skills_prompt_block(&discovered) {
        parts.push(skills_block);
        // Immediate directive: placed right after available skills so the model
        // sees the skills listing and the instructions as one unit.
        parts.push(
            "## Skill Auto-Loading (MANDATORY)\n\n\
             For EVERY user message, scan the `<available_skills>` list above and check if\n\
             the user's request matches any skill's `<description>` field.\n\
             The `<description>` tells you exactly when that skill applies — trust it.\n\
             When matched: READ the skill file (`<location>`) → Apply instructions → Respond.\n\
             This is mandatory for every turn."
                .to_string(),
        );
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

/// Build an `<active_skills>` block for skills activated at session start.
/// These are injected with system-prompt-level authority, not as user messages.
fn build_active_skills_block(discovered: &[Skill], startup_skills: &[String]) -> Option<String> {
    let mut block = String::from("\n<active_skills>\n");
    block.push_str("The following skills are ACTIVE for this entire session. Their instructions\n");
    block.push_str("carry the same authority as system instructions. Follow them on every turn\n");
    block.push_str("until you are told to stop.\n\n");

    let mut found_any = false;
    for invocation in startup_skills {
        // Parse "skill-name level" or just "skill-name"
        let (skill_name, level) = match invocation.find(' ') {
            Some(idx) => (&invocation[..idx], invocation[idx + 1..].trim()),
            None => (invocation.as_str(), ""),
        };

        if let Some(skill) = discovered.iter().find(|s| s.name == skill_name) {
            found_any = true;
            block.push_str(&format!("## Active Skill: {}\n", skill.name));
            if !level.is_empty() {
                block.push_str(&format!("Level: {}\n", level));
            }
            block.push_str(&format!("Location: {}\n\n", skill.location.display()));
            block.push_str(skill.body.trim());
            block.push_str("\n\n");
            block.push_str(
                "--- RESPONSE DIRECTIVE: Apply the skill instructions above to EVERY response \
                 you produce. Re-read this directive before generating each reply. ---\n\n",
            );
        }
    }

    if !found_any {
        return None;
    }
    block.push_str("</active_skills>");
    Some(block)
}

async fn load_project_context(working_dir: &Path) -> Option<String> {
    // 1. Find root AGENTS.md (walk up from working dir)
    let agents_path = find_context_file(working_dir, "AGENTS.md").await?;
    let agents = tokio::fs::read_to_string(&agents_path).await.ok()?;
    let project_root = agents_path.parent().unwrap_or(working_dir);

    let mut context = format!(
        "# Project Context

{agents}"
    );

    // 2. Discover nested AGENTS.md files (walk down from project root)
    let nested = discover_nested_agents(project_root).await;
    for (relative_path, content) in &nested {
        context.push_str(&format!(
            "

# Crate Context: {relative_path}

{content}"
        ));
    }

    // 3. CLAUDE.md (only if different from root AGENTS.md)
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

    // 4. Theta context file
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

/// Discover all AGENTS.md files nested under `root`, excluding ignorable directories.
/// Returns a sorted Vec of (relative_path, content) pairs.
/// Does NOT include the root AGENTS.md (handled separately).
async fn discover_nested_agents(root: &Path) -> Vec<(String, String)> {
    let mut results = Vec::new();
    // Manual stack to avoid async_recursion dependency
    let mut dirs: Vec<PathBuf> = vec![root.to_path_buf()];

    while let Some(current) = dirs.pop() {
        // Check for AGENTS.md in this directory (skip root level)
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

        // Enumerate subdirectories
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

    // Sort by relative path for deterministic ordering
    results.sort_by(|a, b| a.0.cmp(&b.0));
    results
}

/// Directories to skip when walking for nested AGENTS.md files.
fn is_ignorable_dir(name: &str) -> bool {
    matches!(
        name,
        ".git"
            | "target"
            | "node_modules"
            | ".theta"
            | ".github"
            | ".vscode"
            | ".idea"
            | ".DS_Store"
    ) || name.starts_with('.')
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

## Skill Auto-Loading

Re-read the Skill Auto-Loading instructions above (next to `<available_skills>`).
Apply them on this turn.

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
change (fork + recompile).

## Startup Skills

Theta can auto-invoke skills at session start via config.toml:

```toml
[startup]
skills = ["caveman ultra", "other-skill lite"]
```

Each entry is `"<skill-name> <level>"`. Levels optional if skill doesn't use them.
When a user asks "auto-load X at start" or "run X every session", write this config.

Do NOT edit config.toml without explicit user request or clear intent."#;

#[cfg(test)]
mod tests {
    use super::{RESPONSE_CONTRACT, build_tools_prompt, is_ignorable_dir};
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
    fn response_contract_contains_skill_auto_loading_reminder() {
        assert!(
            RESPONSE_CONTRACT.contains("Skill Auto-Loading"),
            "RESPONSE_CONTRACT must have skill auto-loading reminder"
        );
        assert!(
            RESPONSE_CONTRACT.contains("Re-read the Skill Auto-Loading instructions above"),
            "must reference the main directive above"
        );
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

    #[test]
    fn ignorable_dirs_reject_vcs_and_build() {
        assert!(is_ignorable_dir(".git"));
        assert!(is_ignorable_dir("target"));
        assert!(is_ignorable_dir("node_modules"));
        assert!(is_ignorable_dir(".theta"));
        assert!(is_ignorable_dir(".github"));
        assert!(is_ignorable_dir(".vscode"));
        assert!(is_ignorable_dir(".idea"));
        assert!(is_ignorable_dir(".DS_Store"));
    }

    #[test]
    fn ignorable_dirs_reject_dot_prefix() {
        assert!(is_ignorable_dir(".cache"));
        assert!(is_ignorable_dir(".config"));
        assert!(is_ignorable_dir(".local"));
    }

    #[test]
    fn ignorable_dirs_accept_normal_names() {
        assert!(!is_ignorable_dir("src"));
        assert!(!is_ignorable_dir("crates"));
        assert!(!is_ignorable_dir("docs"));
        assert!(!is_ignorable_dir("tests"));
    }

    #[tokio::test]
    async fn discover_nested_agents_finds_crate_files() {
        // CARGO_MANIFEST_DIR is crates/theta, parent is crates/
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let nested = super::discover_nested_agents(root).await;

        let found_crates: Vec<&str> = nested.iter().map(|(p, _)| p.as_str()).collect();

        // Per-crate AGENTS.md files (relative to crates/)
        assert!(
            found_crates.contains(&"theta-ai"),
            "missing theta-ai: {:?}",
            found_crates
        );
        assert!(
            found_crates.contains(&"theta-agent-core"),
            "missing theta-agent-core: {:?}",
            found_crates
        );
        assert!(
            found_crates.contains(&"theta-tui"),
            "missing theta-tui: {:?}",
            found_crates
        );
        assert!(
            found_crates.contains(&"theta-models"),
            "missing theta-models: {:?}",
            found_crates
        );
        assert!(
            found_crates.contains(&"theta-script"),
            "missing theta-script: {:?}",
            found_crates
        );

        // Each should have non-empty content
        for (_path, content) in &nested {
            assert!(!content.is_empty(), "empty AGENTS.md found");
        }

        // Should NOT include root AGENTS.md
        assert!(
            !found_crates.contains(&""),
            "root AGENTS.md should not be in nested results"
        );
    }
}
