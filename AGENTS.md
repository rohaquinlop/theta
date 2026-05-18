# Theta — Agent Rules

> Rules for both humans and coding agents working on Theta.

## Conversational Style

- Keep answers short and concise.
- No emojis in commits, code, or docs.
- No fluff or cheerful filler. Technical prose only.
- Answer questions first, then implement.

## Project Philosophy

Theta is a minimal terminal coding harness in Rust, inspired by [pi](https://github.com/earendil-works/pi).

> **Adapt theta to your workflows, not the other way around.**

Users should extend Theta without forking internals: custom tools via Rust traits, skills via Markdown files, prompt templates, and themes. No sub-agents, no plan mode in core. Users build or install what they want.

## Architecture

Three layers, mirroring pi:

```
theta (binary)          — CLI + TUI + sessions + built-in tools + skills + themes
theta-agent-core (lib)  — agent runtime: loop, tool calling, events, state
theta-ai (lib)          — unified LLM API: types, provider trait, streaming
theta-tui (lib)         — terminal UI (ratatui + crossterm)
theta-models (lib)      — built-in model catalog (compile-time)
```

**Dependency order:** `theta-ai` ← `theta-agent-core` ← `theta` (+ `theta-tui`, `theta-models`)

See `PLAN.md` for the full implementation plan and phase breakdown.

## Rust Conventions

- **Edition 2024** across all crates.
- **`tokio`** for all async. No `async-std` or `smol`.
- **`serde` + `serde_json`** for serialization. Avoid `serde_yaml` unless parsing YAML frontmatter.
- **`tracing`** for logging, not `log` or `println!`.
- **`anyhow`** for application errors, **`thiserror`** for library errors.
- **No `unwrap()` in library code.** Use `?` or proper error handling. `expect()` only with a clear message.
- **No `unsafe`** unless absolutely necessary and documented with a safety comment.
- **No panic in library code paths.** Libraries return `Result`, never abort.
- **Traits over inheritance.** Extension points are `#[async_trait]` traits.
- **`Arc<RwLock<T>>`** for shared mutable state in the Agent. Avoid `Mutex` for hot paths.
- **Single-line helpers with one call site are forbidden.** Inline them.
- **Read files in full** before wide-ranging changes. Don't rely only on `grep` snippets.

## Provider Strategy

**One provider to rule them all.** OpenAI, DeepSeek, and OpenCode all speak OpenAI's `/v1/chat/completions` API. A single `OpenAiCompatProvider` handles all three with per-model compatibility flags:

| Flag | Purpose |
|------|---------|
| `thinking_format` | `"openai"` (reasoning_effort) vs `"deepseek"` (thinking: { type }) |
| `supports_developer_role` | o-series models need `developer` instead of `system` |
| `requires_reasoning_content_on_assistant` | DeepSeek needs empty `reasoning_content` on replayed messages |
| `max_tokens_field` | `max_completion_tokens` vs `max_tokens` |

**Current models** (from `theta-models`):
- **OpenAI**: `gpt-5.5`, `gpt-5.5-instant`, `o4`, `o4-mini`
- **DeepSeek**: `deepseek-v4-pro` (1M ctx), `deepseek-v4-flash` (1M ctx)
- **OpenCode**: OpenAI-compatible, user-configured base URL

**API keys:** Read from env vars (`OPENAI_API_KEY`, `DEEPSEEK_API_KEY`, `OPENCODE_API_KEY`) and from config file.

**No Anthropic, no Google, no Mistral in MVP.** Those are deferred.

## Session Format

**Pi-compatible JSONL.** Theta reads and writes the same session format as pi. This means:

- Users can switch between Pi and Theta on the same project.
- Sessions are portable.
- The format is a JSONL file with entries like `{"type":"user",...}`, `{"type":"assistant",...}`, `{"type":"toolResult",...}`, `{"type":"model_change",...}`, etc.

**Do not invent a new format.** Copy pi's entry types exactly. See pi's `SessionManager` for the contract.

## Tool System

Seven built-in tools (same set as pi):
- `read` — file reading with line/byte limits and truncation
- `write` — create/overwrite files
- `edit` — exact string replacement (pi's `edit` semantics)
- `bash` — shell command execution with timeout
- `grep` — regex search in files
- `find` — file search by name
- `ls` — directory listing

**Trait:**
```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn label(&self) -> &str;
    fn parameters(&self) -> serde_json::Value; // JSON Schema
    fn execution_mode(&self) -> ToolExecutionMode { ToolExecutionMode::Parallel }
    async fn execute(&self, tool_call_id: &str, args: serde_json::Value,
                     signal: Option<CancellationToken>,
                     on_update: Option<ToolUpdateSender>)
        -> Result<ToolResult>;
}
```

## Extension Model (MVP)

**Traits, not dynamic loading.** Extensions are compiled into the binary. Users who want custom tools fork Theta, implement the `Tool` trait, and build their own binary.

This is the Rust way and aligns with "adapt theta to your workflows." Later phases can add WASM component model for dynamic loading.

## Non-Goals for MVP

These are intentionally out of scope. Do not implement them:
- Anthropic, Google, Mistral, or Bedrock providers
- Slack bot, web UI, or vLLM infrastructure
- Dynamic WASM extension loading
- Windows-specific workarounds
- GitHub Actions / CI integration
- Session sharing / telemetry / analytics
- Sub-agents or plan mode in core

## Implementation Order

Follow the phases in `PLAN.md`. Build bottom-up:
1. `theta-ai` → types + provider trait + OpenAI-compat provider
2. `theta-models` → built-in model definitions
3. `theta-agent-core` → Agent + loop + tool execution
4. `theta-tui` → terminal UI components
5. `theta` → CLI + sessions + built-in tools + TUI integration

**Do not skip ahead.** `theta-agent-core` cannot work before `theta-ai` is functional. The TUI cannot work before the agent loop emits events.

## Testing

- **Unit tests:** `#[cfg(test)]` modules in each crate, `cargo test`
- **Integration tests:** in `tests/` directory at workspace root
- **LLM-dependent tests:** behind `#[cfg(feature = "integration-tests")]` with real API keys
- **Faux provider:** create a mock `theta-ai` provider that returns canned responses for testing the agent loop without hitting real APIs
- **No paid API keys in CI.** Integration tests are local-only.

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

# Run theta from source
cargo run -- <args>
```

**After code changes (not docs):** Run `cargo fmt --check && cargo clippy -- -D warnings && cargo test` before committing. Fix all warnings and errors.

## Git Rules

- **Never commit unless the user explicitly asks.**
- **Stage only files you changed:** `git add <specific-files>`. Never `git add -A` or `git add .`.
- **Check `git status`** before every commit.
- **No `git reset --hard`, `git checkout .`, `git clean -fd`, `git stash`.** These destroy work.
- **Rebase, don't merge.** `git pull --rebase` when needed.
- **If rebase conflict is in a file you didn't touch,** abort and ask the user.

## Adding a New LLM Provider (Future)

When a new provider is needed beyond the first three:
1. Add provider name to the `Provider` enum in `theta-ai/src/types.rs`
2. If it's OpenAI-compatible: add compat flags to the `Model` struct, update `OpenAiCompatProvider`
3. If it needs a new API: implement the `Provider` trait in `theta-ai/src/providers/`
4. Add models to `theta-models/src/<provider>.rs`
5. Add env var detection in `theta-ai/src/env_keys.rs`
6. Add default model to `theta/src/models.rs`
7. Update `PLAN.md` and this file

## The PLAN.md

`PLAN.md` is the canonical implementation plan. It is NOT auto-generated documentation — it is a living design document. Update it when:
- Architecture decisions change
- New non-goals are agreed on
- Phase estimates shift significantly
- New crates are added or merged

Keep it concise. It is a guide for both humans and agents.
