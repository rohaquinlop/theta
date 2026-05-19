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
theta (binary)          — CLI + TUI + sessions + built-in tools + skills + themes  [Phase 3+]
theta-agent-core (lib)  — agent runtime: loop, tool calling, events, state          [Phase 2 done]
theta-ai (lib)          — unified LLM API: types, provider trait, streaming          [Phase 1 done]
theta-tui (lib)         — terminal UI (ratatui + crossterm)                          [Phase 4+]
theta-models (lib)      — built-in model catalog (compile-time)                      [Phase 1 done]
```

**Dependency order:** `theta-ai` ← `theta-agent-core` ← `theta` (+ `theta-tui`, `theta-models`)

See `PLAN.md` for the full implementation plan and phase breakdown.

## Phase Completion Status

| Phase | Status | Deliverable |
|-------|--------|-------------|
| 1. Foundation | Done | `theta-ai` + `theta-models` |
| 2. Agent Runtime | Done | `theta-agent-core` |
| 3. CLI + Tools | Next | `theta` binary with built-in tools |
| 4. TUI | Pending | `theta-tui` + interactive mode |
| 5. Extensibility | Pending | Skills, templates, extension traits |
| 6. Polish | Pending | Compaction, docs, releases |

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
- **`tokio::sync::RwLock` over `std::sync::RwLock`** for any state held across `.await` points. The std variant makes futures `!Send`.
- **`std::sync::Mutex` for short-lived locks** that never cross await. `tokio::sync::Mutex` only when the lock must be held across `.await`.
- **`Arc<Mutex<Vec<T>>>` for shared queues** between agent and loop — steer/follow-up push from external threads while the loop drains.
- **Single-line helpers with one call site are forbidden.** Inline them.
- **Read files in full** before wide-ranging changes. Don't rely only on `grep` snippets.

## Provider Strategy

**One provider to rule them all.** OpenAI, DeepSeek, and OpenCode all speak OpenAI's `/v1/chat/completions` API. A single `OpenAiCompatProvider` handles all three with per-model compatibility flags. Codex (ChatGPT Plus) has its own `OpenAiCodexProvider` targeting `chatgpt.com/backend-api` with session-token auth.

| Flag | Purpose |
|------|---------|
| `thinking_format` | `"openai"` (reasoning_effort) vs `"deepseek"` (thinking: { type }) |
| `supports_developer_role` | o-series models need `developer` instead of `system` |
| `requires_reasoning_content_on_assistant` | DeepSeek needs empty `reasoning_content` on replayed messages |
| `max_tokens_field` | `max_completion_tokens` vs `max_tokens` |

**Current models** (from `theta-models`):
- **OpenAI**: `gpt-5.5`, `gpt-5.5-instant`, `o4`, `o4-mini`
  — auth via `OPENAI_API_KEY`
- **OpenAI Codex**: `gpt-5.5`, `gpt-5.5-instant`, `o4`, `o4-mini`
  — auth via `OPENAI_CODEX_TOKEN` (ChatGPT Plus session token),
  targets `https://chatgpt.com/backend-api`, no API key needed
- **DeepSeek**: `deepseek-v4-pro` (1M ctx), `deepseek-v4-flash` (1M ctx)
- **OpenCode**: OpenAI-compatible, user-configured base URL

**API keys:** Read from env vars (`OPENAI_API_KEY`, `OPENAI_CODEX_TOKEN`,
`DEEPSEEK_API_KEY`, `OPENCODE_API_KEY`) and from config file.

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

**Agent-level trait** (`theta-agent-core::AgentTool`):
```rust
#[async_trait]
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn label(&self) -> &str;
    fn parameters(&self) -> serde_json::Value; // JSON Schema
    fn execution_mode(&self) -> ToolExecutionMode { ToolExecutionMode::Parallel }
    async fn execute(&self, tool_call_id: &str, args: serde_json::Value,
                     signal: Option<CancellationToken>,
                     on_update: Option<ToolUpdateSender>)
        -> Result<ToolResult, AgentError>;
}
```

**LLM-level definition** (`theta_ai::Tool`) — separate struct for the JSON schema sent to the model, built from `AgentTool` at context-construction time.

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

**Do not skip ahead.** `theta-agent-core` cannot work before `theta-ai` is functional. The TUI cannot work before the agent loop emits events. Phases 3 and 4 can be worked in either order since the CLI binary depends on both `theta-agent-core` and `theta-tui`.

## Agent Loop Design

The agent loop uses a nested pattern:

- **Outer loop** (follow-up turns): after each turn, checks hooks and follow-up/steering queues; drains them into state for the next turn.
- **Inner loop** (tool calling): LLM call → accumulate stream events → if tool calls, execute tools → add results → call LLM again.

**Steering vs Follow-up:**
- `steer()`: injects message MID-TURN. Uses `AtomicBool` per-stream abort flag (not the permanent `CancellationToken`). After abort, inner loop drains steering queue and continues.
- `follow_up()`: queues message for AFTER current turn completes. Outer loop picks it up.

**Event flow:** `broadcast::channel(256)` — consumers subscribe via `agent.subscribe()`. `AgentEnd` is always emitted (even on error).

**Hooks** (`beforeToolCall`, `afterToolCall`, `shouldStopAfterTurn`, `prepareNextTurn`) — all `#[async_trait]` with default no-ops.

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

**After code changes (not docs):** Run `cargo fmt && cargo clippy -- -D warnings && cargo test` before committing. Fix all warnings and errors.

## Git Rules

- **Never commit unless the user explicitly asks.**
- **Stage only files you changed:** `git add <specific-files>`. Never `git add -A` or `git add .`.
- **Check `git status`** before every commit.
- **No `git reset --hard`, `git checkout .`, `git clean -fd`, `git stash`.** These destroy work.
- **Rebase, don't merge.** `git pull --rebase` when needed.
- **If rebase conflict is in a file you didn't touch,** abort and ask the user.

## Adding a New LLM Provider (Future)

When a new provider is needed beyond the first four:
1. Add provider name to the `Provider` enum in `theta-ai/src/types.rs`
2. If it's OpenAI-compatible: add compat flags to the `Model` struct, update `OpenAiCompatProvider`
3. If it needs a new API or auth flow (like Codex): implement the `Provider` trait in `theta-ai/src/providers/`
4. Add models to `theta-models/src/<provider>.rs`
5. Add env var or auth token detection in the provider implementation
6. Add default model to `theta/src/models.rs`
7. Update `PLAN.md` and this file

## The PLAN.md

`PLAN.md` is the canonical implementation plan. It is NOT auto-generated documentation — it is a living design document. Update it when:
- Architecture decisions change
- New non-goals are agreed on
- Phase estimates shift significantly
- New crates are added or merged

Keep it concise. It is a guide for both humans and agents.
