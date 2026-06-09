# MichiN — Agent Rules

> Rules for coding agents working on MichiN.

## Conversational Style

- Short, concise answers.
- No emojis in commits, code, or docs.
- No fluff. Technical prose only.
- Answer first, then implement.

## Project Philosophy

MichiN is a terminal coding-agent harness.

**Extend without forking internals:** custom tools via Rust traits, skills via Markdown, prompt templates, Rhai scripts, themes. No sub-agents, no plan mode in core.

### The model decides, not the code

MichiN is an agent **harness** — it provides structure (tools, prompts, events) and gets out of the way.
The LLM drives all behavioral decisions: when to use a skill, what mode to operate in,
how to interpret user intent. Do not replace this with heuristic Rust code.

Wrong: write a `find_matching_skills()` that scores keywords to decide which skill to load.
Right: list `<available_skills>` with name+description in the system prompt, tell the model
to read the `<description>` field and decide for itself.

Wrong: write a classifier to detect "plan mode" or "action mode".
Right: put instructions in the system prompt, let the model follow them.

If you find yourself writing a scoring function, keyword matcher, or intent classifier
in Rust to drive agent behavior — stop. The system prompt is the right place for that logic.

## Architecture

Six crates in Cargo workspace (`edition = "2024"`, `resolver = "3"`):

```
crates/michin              — CLI + TUI + sessions + built-in tools + skills + themes + scripts + RPC
crates/michin-agent-core   — agent runtime: Agent, loop, tool execution, compaction, events, hooks
crates/michin-ai           — unified LLM API: types, provider trait, streaming, replay, two providers
crates/michin-tui          — terminal UI (ratatui + crossterm): chat, editor, fuzzy, logins, selectors, status bar
crates/michin-models       — built-in model catalog (compile-time definitions + runtime OpenCode fetch)
crates/michin-script       — Rhai-powered hooks: before/after tool calls, TUI status rows
```

**Dependency order:** `michin-ai` ← `michin-agent-core` ← `michin` (+ `michin-tui`, `michin-models`, `michin-script`)

Each crate has its own `AGENTS.md` with crate-specific conventions. When working in a crate's code, load that crate's `AGENTS.md` file for detailed rules.

## Rust Conventions

- **Edition 2024** across all crates.
- **`tokio`** (full features) for async. No `async-std` or `smol`.
- **`serde` + `serde_json`** for serialization. `serde_yaml` only for skill frontmatter.
- **`tracing`** for logging, not `log` or `println!`.
- **`anyhow`** for app errors (binary + tui), **`thiserror`** for library errors (ai, agent-core, settings, config).
- No `unwrap()` in library code. Use `?` or proper error handling. `expect()` only with clear message.
- No `unsafe` unless necessary, documented with safety comment.
- No panic in library code. Libraries return `Result`, never abort.
- Traits over inheritance. Extension points are `#[async_trait]` traits.
- `tokio::sync::RwLock` over `std::sync::RwLock` for state held across `.await`.
- Never hold `agent.state().await` guards across awaited calls that may take a write lock. Read needed fields, `drop(state)`, then await.
- `std::sync::Mutex` for short-lived locks never crossing await.
- `Arc<Mutex<Vec<T>>>` for shared queues between agent and loop.
- Single-line helpers with one call site: inline them.
- Read files in full before wide-ranging changes. Don't rely only on `grep` snippets.
- Dependencies in `Cargo.toml` use workspace references. New deps go in `[workspace.dependencies]`.

## Comment Style

Before writing any comment, ask: **"Can a competent reader infer this from the code alone?"** If yes, delete the comment. Do not rewrite it — remove it.

- Comments explain WHY, not WHAT. The code says what; comments say why.
- No field-level docs that restate the field name: `/// The text buffer.` for `pub text: String` is noise — remove or add semantic context.
- No narrative inline comments that walk through code line-by-line. Each comment must add info not visible in the code.
- Module-level docs (`//!`): one line describing purpose. Multi-line only when the module has non-obvious invariants or safety requirements.
- Struct/enum docs (`///`): one line. Multi-line only when invariants, lifecycle, or safety constraints are non-obvious.
- Section separator comments (`// ── Section ──`): keep lightweight. One per logical section, no box-drawing.
- Good comments: why this approach, why not the obvious alternative, invariants, safety, non-obvious side effects, bug references.
- LLM-facing text (tool descriptions, prompt templates): keep as-is — these are not just comments, they're data.

### Anti-patterns (do NOT write these)

```rust
// BAD: restates the field name
/// Input tokens consumed.
pub input_tokens: u32,

// BAD: narrates what the next line does
// Call the LLM.
let response = provider.stream(&model, &context).await?;

// BAD: restates the function name
/// Set lifecycle hooks.
pub fn set_hooks(&mut self, hooks: Arc<dyn Hooks>) { ... }

// BAD: restates the module name
//! Agent-level error types.

// BAD: restates the struct purpose from its name + fields
/// The agent: holds all state and orchestrates LLM interaction.
pub struct Agent { ... }
```

```rust
// GOOD: explains a non-obvious design choice
// Write could be parallel but is sequential to avoid race on same file.

// GOOD: explains an invariant that would be lost without the comment
// Prefix-preserving: keeps earliest messages byte-stable to preserve
// DeepSeek/MiMo prefix cache across compaction.

// GOOD: explains why this code exists (not what it does)
// Don't persist error/aborted responses — they cause the replay
// sanitizer to fire on every subsequent context build.
```

## Commands

```bash
# Build all crates
cargo build

# Run all tests (no LLM calls)
cargo test

# Run with integration tests (requires API keys)
cargo test --features integration-tests

# Check formatting
cargo fmt --check

# Lint
cargo clippy -- -D warnings

# Full check before commit
cargo fmt --check && cargo clippy -- -D warnings && cargo test

# Run michin from source
cargo run -- <args>
```

After code changes (not docs): run `cargo fmt && cargo clippy -- -D warnings && cargo test` before committing. Fix all warnings and errors.

## Git Rules

- Never commit unless user explicitly asks.
- Never push, pull, or interact with remotes. User does remote ops.
- Stage only changed files: `git add <specific-files>`. Never `git add -A` or `git add .`.
- Check `git status` before every commit.
- No `git reset --hard`, `git checkout .`, `git clean -fd`, `git stash`. These destroy work.
- Rebase, don't merge. `git pull --rebase` when needed.
- If rebase conflict in file you didn't touch, abort and ask user.

## Tool System

Six built-in tools: `read`, `write`, `edit`, `bash`, `find`, `grep`. Each implements `michin_agent_core::AgentTool`.

- Absolute paths honored directly (not clamped to working dir).
- Output truncation at 2000 lines / 50KB.

## Extension Model

Three tiers:

1. **Skills** (`SKILL.md` files) — Markdown with YAML frontmatter, discovered from `~/.michin/skills/` and `./.michin/skills/`.
2. **Rhai Scripts** (`~/.michin/extensions/*.rhai`, `./.michin/extensions/*.rhai`) — Runtime hooks.
3. **Rust Traits** — `AgentTool`, `Hooks`, `LlmProvider`. Fork MichiN, implement traits.

When user says "modify/extend michin" without specifics: ask whether they want skill, script, or Rust change.

## Non-Goals

- Anthropic, Google, Mistral, or Bedrock providers
- Slack bot, web UI, or vLLM infrastructure
- Dynamic WASM extension loading
- Windows-specific workarounds
- GitHub Actions / CI integration
- Session sharing / telemetry / analytics
- Sub-agents or plan mode in core
