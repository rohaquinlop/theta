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
use std::sync::LazyLock;
use std::time::{SystemTime, UNIX_EPOCH};

use michin_ai::ContentBlock;

/// Compute the date string once, frozen at process start.
/// Prevents spurious DeepSeek prefix cache misses when a session spans
/// midnight — the date in the system prompt stays byte-identical.
static SESSION_DATE: LazyLock<String> = LazyLock::new(|| {
    chrono_now(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs(),
    )
});

use crate::scripts;
use crate::skills;
use crate::tools::{ToolContext, builtin_tools};

/// Configuration for building the system prompt.
///
/// Groups together the parameters that influence prompt content so new mode
/// toggles don't keep adding positional parameters to `build_system_prompt`.
pub struct SystemPromptConfig<'a> {
    pub model_id: &'a str,
    pub thinking_level: Option<&'a str>,
    pub max_context_window: Option<u32>,
}

/// Build the core system prompt: project context + tools + runtime context.
/// Does NOT include skills, extensions, RESPONSE_CONTRACT, plan mode, or caveman mode.
/// Those all live in volatile_overlays to keep the system prompt byte-stable.
pub async fn build_system_prompt(
    working_dir: &Path,
    config: &SystemPromptConfig<'_>,
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
        &SESSION_DATE,
        config.model_id,
        config.thinking_level,
        config.max_context_window,
    ));

    // NO mode contracts here — RESPONSE_CONTRACT, plan, caveman all live in
    // volatile_overlays set via agent.set_volatile_overlays().

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
/// Returns (system_prompt, resource_context).
pub async fn build_system_prompt_with_skills(
    working_dir: &Path,
    config: &SystemPromptConfig<'_>,
) -> (Vec<ContentBlock>, Vec<ContentBlock>) {
    let system = build_system_prompt(working_dir, config).await;
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
skill, extension, custom tool, or code change.

## Custom Tools

Custom tools are Rhai scripts in `~/.michin/tools/*.rhai` that register
new tools the LLM can invoke. A custom tool script must:

1. Call `tool.register(name, schema)` at the top level to register the tool.
2. Define an `execute(args)` function that returns a string or a map with
   `content` (string) and `is_error` (bool).

Available built-in functions: `exec(cmd, args)`, `read_file(path)`,
`write_file(path, content)`, `set_state(key, value)`, `get_state(key)`,
`cwd()`, `home_dir()`.

The schema map must include `description` and `parameters` (JSON Schema).
Optional: `execution_mode` ("parallel" or "sequential", default "parallel")."#;

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
    let tools = builtin_tools(ctx, None);
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

    // Tool selection guide — positive framing, before tool details.
    p.push_str("## Tool Selection\n\n");
    p.push_str(
        "- **File operations:** `read` (view), `write` (create/overwrite), `edit` (targeted replacement).\n",
    );
    p.push_str(
        "- **Code search:** `find` (by filename/path), `grep` (by content). Both use an in-process\n  indexed search that is faster than shell tools and respects .gitignore. Always prefer\n  these over bash equivalents.\n",
    );
    p.push_str(
        "- **Shell:** `bash` for commands that have no dedicated tool: running tests, building,\n  git operations, package managers, project setup. Do NOT use bash to search files —\n  `find` and `grep` cover that domain.\n\n",
    );

    // Present tools grouped by category, not alphabetically.
    p.push_str("## File Operations\n\n");
    for tool in &tools {
        if is_file_tool(tool.name()) {
            push_tool_section(&mut p, tool);
        }
    }

    p.push_str("## Code Search\n\n");
    for tool in &tools {
        if is_search_tool(tool.name()) {
            push_tool_section(&mut p, tool);
        }
    }

    p.push_str("## Shell\n\n");
    for tool in &tools {
        if tool.name() == "bash" {
            push_tool_section(&mut p, tool);
        }
    }

    p
}

fn is_file_tool(name: &str) -> bool {
    matches!(name, "read" | "write" | "edit")
}

fn is_search_tool(name: &str) -> bool {
    matches!(name, "find" | "grep")
}

fn push_tool_section(
    p: &mut String,
    tool: &std::sync::Arc<dyn michin_agent_core::types::AgentTool>,
) {
    p.push_str(&format!("### {}\n", tool.name()));
    p.push_str(&format!("{}\n", tool.description()));
    p.push_str(&format!(
        "Parameters: {}\n\n",
        serde_json::to_string_pretty(&tool.parameters()).unwrap_or_default()
    ));
}

// ── Runtime context ────────────────────────────────────────────────

fn build_runtime_context(
    working_dir: &Path,
    date: &str,
    model_id: &str,
    thinking_level: Option<&str>,
    max_context_window: Option<u32>,
) -> String {
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

    // Byte-stable model ID for DeepSeek — prevents cache bust on flash↔pro switch.
    // Non-DeepSeek models show their actual ID.
    let model_display = if model_id.starts_with("deepseek-") {
        "deepseek-reasoning"
    } else {
        model_id
    };

    format!(
        "# Runtime Context

         Current date: {date}
         Working directory: {cwd_display}
         Shell: {shell}
         OS: {os}
         Model: {model_display}
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

## Tool Selection

- **File operations:** `read`, `write`, `edit` for viewing, creating, and modifying files.
- **Code search:** `find` and `grep` for locating files and searching content. These use an in-process indexed search that is faster than shell tools and respects .gitignore. Always prefer these over bash equivalents.
- **Shell:** `bash` for commands with no dedicated tool: running tests, building, git operations, package managers. Do NOT use bash for `rg`, `grep`, `find`, `fd`, or `ripgrep` — `find` and `grep` are purpose-built for those tasks.
- **Grep is plain text by default.** Write code tokens exactly as they appear: `parse_expr(` finds literal `parse_expr(`. Set `regex: true` only for actual regex patterns like `fn\s+\w+`. Path parameter scopes results — `path: "compiler/rustc_parse/"` searches only files in that directory.
- If grep returns 0 results, try broadening: use `path: "."` (all files), shorten the query, or remove path constraint. Do NOT fall back to bash rg/grep.
- Read files before editing them.
- When a tool call fails, attempt to fix the issue and retry once before reporting an error.
- Do not repeat identical tool calls in a loop.

## Resources

Skills and extensions are listed in the conversation context. When a message
matches a skill's trigger, read its file and follow its instructions."#;

pub const CAVEMAN_CONTRACT: &str = r#"# Caveman Mode

Caveman mode ACTIVE. All output compressed per rules below.

## Rules

- Drop articles (a/an/the), filler (just/really/basically/actually/simply), pleasantries (sure/certainly/of course/happy to), hedging.
- Fragments OK. Use short synonyms (big not extensive, fix not \"implement a solution for\").
- Technical terms must remain exact. Code blocks remain unchanged. Errors quoted exact.
- Pattern: `[thing] [action] [reason]. [next step].`

## Intensity

| Level | Behavior |
|-------|----------|
| **lite** | No filler/hedging. Keep articles + full sentences. Professional but tight. |
| **full** | Drop articles, fragments OK, short synonyms. Classic caveman. |
| **ultra** | Abbreviate prose words (DB/auth/config/req/res/fn/impl), strip conjunctions, arrows for causality (X → Y), one word when one word enough. Never abbreviate code symbols, function names, API names, error strings. |
| **wenyan-lite** | Semi-classical Chinese register. Drop filler/hedging but keep grammar structure. |
| **wenyan-full** | Maximum classical Chinese terseness. 80-90% character reduction. Classical sentence patterns, subjects often omitted, classical particles (之/乃/為/其). |
| **wenyan-ultra** | Extreme abbreviation while keeping classical Chinese feel. Maximum compression. |

## Auto-Clarity Override

Temporarily drop caveman when:
- Security warnings
- Irreversible action confirmations
- Multi-step sequences where fragment order risks misread
- Compression itself creates technical ambiguity
- User asks to clarify or repeats a question

Resume caveman after the clear part is done."#;

/// Plan mode contract — injected as a volatile overlay at request time.
/// Overrides Turn Completion to guide the model toward plan-only exploration.
pub const PLAN_MODE_CONTRACT: &str = r#"# Plan Mode

You are operating in **plan mode**: explore, analyze, and plan before implementing.
Your job is to investigate the codebase, collect data, and prepare a clear
step-by-step implementation plan.

## Plan Mode Rules

- **Explore only — no source mutation:** You CANNOT execute `edit` or any
  bash command that mutates files (mv, rm, sed -i, cargo add, git push, etc.).
  These are blocked at the tool level — attempts will be rejected automatically.
- **Read freely, write plan files on request:** Use `read`, `bash` (for
  read-only commands: grep, git log, cargo check, etc.), and
  `web_search`/`web_fetch` to investigate. `write` is available ONLY to save
  a plan file when explicitly asked — never for source files.
- **Do NOT create files unless asked:** Do not save a plan to a file unless
  the user explicitly requests it ("save to file", "write that down", etc.).
  Until asked, present plans inline in your response.
- **When asked to save:** Use `write` to save the plan file — `write` is
  allowed in plan mode for this purpose only. Do not `write` anything else.
  Save to the working directory (shown in Runtime Context above) unless
  the user specifies a different path. Default to Markdown (`.md`) — the
  user may request another format.
- **Exit:** The user toggles plan mode with `/plan`. Do not try to switch
  modes yourself.

When asked to implement or change code while in plan mode, explain what you
would do in the plan instead — do not try to modify source files."#;

/// Build the plan mode volatile overlay.
pub fn build_plan_mode_overlay() -> Vec<ContentBlock> {
    vec![ContentBlock::text(PLAN_MODE_CONTRACT)]
}

/// Build the caveman mode volatile overlay.
/// Overrides RESPONSE_CONTRACT (also in overlays) — both positioned
/// together so the model sees them as a single cohesive block.
pub fn build_caveman_mode_overlay(level: &str) -> Vec<ContentBlock> {
    let mut p = String::new();
    p.push_str("# Caveman Mode\n\n");
    p.push_str(&format!("Caveman mode ACTIVE at level: {level}.\n\n"));
    p.push_str(
        "Override: ignore any earlier output-style instructions (\"be concise\", \
         \"use technical prose\"). The following rules are the ONLY formatting authority.\n\n",
    );
    p.push_str(CAVEMAN_CONTRACT);
    vec![ContentBlock::text(p)]
}

/// Build all active volatile overlays from current mode state.
/// RESPONSE_CONTRACT is always included as the base behavioral contract.
pub fn build_active_overlays(plan_mode: bool, caveman_level: Option<&str>) -> Vec<ContentBlock> {
    let mut overlays = vec![ContentBlock::text(RESPONSE_CONTRACT)];
    if plan_mode {
        overlays.extend(build_plan_mode_overlay());
    }
    if let Some(level) = caveman_level {
        overlays.extend(build_caveman_mode_overlay(level));
    }
    overlays
}

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
