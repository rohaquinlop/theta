//! System prompt construction.
//!
//! Two outputs:
//! - `build_system_prompt()` — core operational instructions (project context,
//!   tools, runtime, response contract). Set via `agent.set_system_prompt()`.
//! - `build_resource_context()` — available resources (skills, extensions,
//!   startup skills, auto-loading directive). Set via `agent.set_resource_context()`.
//!
//! This split keeps system instructions lean and moves resource listings
//! into the conversation where the model sees them as context, not mandates.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use theta_ai::ContentBlock;

use crate::scripts;
use crate::skills;
use crate::skills::Skill;
use crate::tools::{ToolContext, builtin_tools};

/// Build the core system prompt: project context + tools + runtime + response contract.
/// Does NOT include skills, extensions, or startup skills.
pub async fn build_system_prompt(
    working_dir: &Path,
    model_id: &str,
    thinking_level: Option<&str>,
) -> Vec<ContentBlock> {
    let mut parts: Vec<String> = Vec::new();

    if let Some(ctx) = load_project_context(working_dir).await {
        parts.push(ctx);
    }

    let tools_prompt = build_tools_prompt(working_dir);
    if !tools_prompt.is_empty() {
        parts.push(tools_prompt);
    }

    parts.push(build_runtime_context(working_dir, model_id, thinking_level));
    parts.push(RESPONSE_CONTRACT.to_string());

    let text = parts.join("\n\n");

    let theta_dir = crate::config::theta_dir();
    let text = apply_system_prompt_overrides(&theta_dir, text).await;

    vec![ContentBlock::Text { text }]
}

/// Build the resource context: skills + extensions + auto-loading + startup skills.
/// This gets injected as a synthetic user message, NOT the system prompt.
pub async fn build_resource_context(
    working_dir: &Path,
    startup_skills: &[String],
) -> Vec<ContentBlock> {
    let mut parts: Vec<String> = Vec::new();

    // Available skills.
    let discovered = skills::discover_skills(working_dir).await;

    // Active startup skills — inject their full bodies.
    if !startup_skills.is_empty()
        && let Some(active_block) = build_active_skills_block(&discovered, startup_skills)
    {
        parts.push(active_block);
    }

    if let Some(skills_block) = skills::build_skills_prompt_block(&discovered) {
        parts.push(skills_block);
        // Concise skill auto-loading directive — kept with the skills it references.
        parts.push(
            "## Skill Auto-Loading\n\n\
             For each user message, scan `<available_skills>` descriptions. \
             When a skill's description matches: read the file at `<location>` \
             and apply its instructions."
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
    startup_skills: &[String],
) -> (Vec<ContentBlock>, Vec<ContentBlock>) {
    let system = build_system_prompt(working_dir, model_id, thinking_level).await;
    let resource = build_resource_context(working_dir, startup_skills).await;
    (system, resource)
}

// ── Resource context builders ──────────────────────────────────────

/// Extension-creation guardrails injected into the resource context.
/// Tells the model when it should (and should NOT) create Rhai scripts.
const EXTENSION_CREATION_GUARDRAILS: &str = r#"## Theta Extensions

Extensions are Rhai scripts at `~/.theta/extensions/*.rhai` (global) or
`./.theta/extensions/*.rhai` (project-local). They take effect on the next
session (loaded at agent startup).

For the Rhai API reference, read `crates/theta-script/AGENTS.md`.

CRITICAL — Only create an extension when the user uses one of these EXACT
trigger phrases:
- "create an extension" / "write an extension" / "make an extension"
- "add a tool hook" / "add a before hook" / "add an after hook"
- "add a status line" / "add a TUI status" / "add an extension status"
- "install an extension"
- "I want to extend theta" / "how do I extend theta"

Do NOT create an extension from general task language.
For "modify/extend theta" without specifics, ask: 1) Skill, 2) Extension, 3) Rust change."#;

/// Build an `<active_skills>` block for skills activated at session start.
fn build_active_skills_block(discovered: &[Skill], startup_skills: &[String]) -> Option<String> {
    let mut block = String::from("\n<active_skills>\n");
    block.push_str("These skills are active for this session. Follow their instructions.\n\n");

    let mut found_any = false;
    for invocation in startup_skills {
        let (skill_name, _level) = match invocation.find(' ') {
            Some(idx) => (&invocation[..idx], invocation[idx + 1..].trim()),
            None => (invocation.as_str(), ""),
        };

        if let Some(skill) = discovered.iter().find(|s| s.name == skill_name) {
            found_any = true;
            block.push_str(&format!("## Active Skill: {}\n", skill.name));
            block.push_str(&format!("Location: {}\n\n", skill.location.display()));
            block.push_str(skill.body.trim());
            block.push_str("\n\n");
        }
    }

    if !found_any {
        return None;
    }
    block.push_str("</active_skills>");
    Some(block)
}

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

    // Theta context file
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

async fn discover_nested_agents(root: &Path) -> Vec<(String, String)> {
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

// ── Tools prompt ───────────────────────────────────────────────────

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

// ── Runtime context ────────────────────────────────────────────────

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
// extension creation, and startup skills documentation live in the
// resource context, not here.

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

## Resources

Available skills and extensions are listed in the conversation context.
Consult them before responding to messages that match their descriptions.
Use available skills and extensions to guide your behavior."#;

// ── System prompt overrides ────────────────────────────────────────

async fn apply_system_prompt_overrides(theta_dir: &Path, mut text: String) -> String {
    let sys_prompt_path = theta_dir.join("SYSTEM_PROMPT.md");
    let append_path = theta_dir.join("APPEND_SYSTEM_PROMPT.md");

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

#[cfg(test)]
mod tests {
    use super::{
        EXTENSION_CREATION_GUARDRAILS, RESPONSE_CONTRACT, build_tools_prompt, is_ignorable_dir,
    };
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

    #[test]
    fn response_contract_no_longer_has_skill_auto_loading_section() {
        // The old "Skill Auto-Loading" section is removed from the contract
        // and lives in the resource context instead.
        assert!(
            !RESPONSE_CONTRACT.contains("## Skill Auto-Loading"),
            "Skill Auto-Loading section must not be in response contract"
        );
    }

    #[test]
    fn response_contract_no_longer_has_theta_extensions_docs() {
        assert!(
            !RESPONSE_CONTRACT.contains("tool.before(name, callback)"),
            "Rhai extension docs must not be in response contract"
        );
        assert!(
            !RESPONSE_CONTRACT.contains("create an extension"),
            "Extension creation docs must not be in response contract"
        );
    }

    #[test]
    fn response_contract_no_longer_has_startup_skills_config() {
        assert!(
            !RESPONSE_CONTRACT.contains("[startup]"),
            "Startup skills config docs must not be in response contract"
        );
    }

    #[test]
    fn response_contract_has_resource_section() {
        assert!(
            RESPONSE_CONTRACT.contains("## Resources"),
            "Response contract must reference resources"
        );
    }

    #[test]
    fn extension_guardrails_has_trigger_phrases() {
        assert!(EXTENSION_CREATION_GUARDRAILS.contains("create an extension"));
        assert!(EXTENSION_CREATION_GUARDRAILS.contains("write an extension"));
        assert!(EXTENSION_CREATION_GUARDRAILS.contains("add a tool hook"));
        assert!(EXTENSION_CREATION_GUARDRAILS.contains("add a TUI status"));
    }

    #[test]
    fn extension_guardrails_rejects_general_language() {
        assert!(
            EXTENSION_CREATION_GUARDRAILS
                .contains("Do NOT create an extension from general task language")
        );
    }

    #[test]
    fn extension_guardrails_references_rhai_api_docs() {
        assert!(EXTENSION_CREATION_GUARDRAILS.contains("crates/theta-script/AGENTS.md"));
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
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let nested = super::discover_nested_agents(root).await;

        let found_crates: Vec<&str> = nested.iter().map(|(p, _)| p.as_str()).collect();

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

        for (_path, content) in &nested {
            assert!(!content.is_empty(), "empty AGENTS.md found");
        }

        assert!(
            !found_crates.contains(&""),
            "root AGENTS.md should not be in nested results"
        );
    }

    #[tokio::test]
    async fn apply_overrides_no_files_returns_original() {
        let dir = tempfile::tempdir().unwrap();
        let original = "base prompt content".to_string();
        let result = super::apply_system_prompt_overrides(dir.path(), original.clone()).await;
        assert_eq!(result, original);
    }

    #[tokio::test]
    async fn apply_overrides_system_prompt_replaces() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(dir.path().join("SYSTEM_PROMPT.md"), "replacement prompt")
            .await
            .unwrap();
        let original = "base prompt content".to_string();
        let result = super::apply_system_prompt_overrides(dir.path(), original).await;
        assert_eq!(result, "replacement prompt");
    }

    #[tokio::test]
    async fn apply_overrides_append_adds_to_original() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(
            dir.path().join("APPEND_SYSTEM_PROMPT.md"),
            "extra instructions",
        )
        .await
        .unwrap();
        let original = "base prompt".to_string();
        let result = super::apply_system_prompt_overrides(dir.path(), original).await;
        assert!(result.contains("base prompt"));
        assert!(result.contains("extra instructions"));
        assert!(result.contains("\n\n"));
    }

    #[tokio::test]
    async fn apply_overrides_both_files_system_prompt_wins() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(dir.path().join("SYSTEM_PROMPT.md"), "system wins")
            .await
            .unwrap();
        tokio::fs::write(dir.path().join("APPEND_SYSTEM_PROMPT.md"), "append ignored")
            .await
            .unwrap();
        let original = "base".to_string();
        let result = super::apply_system_prompt_overrides(dir.path(), original).await;
        assert_eq!(result, "system wins");
        assert!(!result.contains("append ignored"));
    }

    #[tokio::test]
    async fn apply_overrides_append_ignored_when_system_present() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(dir.path().join("SYSTEM_PROMPT.md"), "only system here")
            .await
            .unwrap();
        tokio::fs::write(
            dir.path().join("APPEND_SYSTEM_PROMPT.md"),
            "should not appear",
        )
        .await
        .unwrap();
        let original = "base".to_string();
        let result = super::apply_system_prompt_overrides(dir.path(), original).await;
        assert_eq!(result, "only system here");
        assert!(!result.contains("should not appear"));
        assert!(!result.contains("base"));
    }

    #[tokio::test]
    async fn apply_overrides_append_empty_file_no_op() {
        let dir = tempfile::tempdir().unwrap();
        tokio::fs::write(dir.path().join("APPEND_SYSTEM_PROMPT.md"), "")
            .await
            .unwrap();
        let original = "just the base".to_string();
        let result = super::apply_system_prompt_overrides(dir.path(), original).await;
        assert_eq!(result, "just the base");
    }

    #[tokio::test]
    async fn apply_overrides_system_prompt_handles_missing_file_gracefully() {
        let dir = tempfile::tempdir().unwrap();
        let original = "stays intact".to_string();
        let result = super::apply_system_prompt_overrides(dir.path(), original).await;
        assert_eq!(result, "stays intact");
    }
}
