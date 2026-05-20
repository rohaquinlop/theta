# Theta — Implementation Plan

> Living design doc for coding agents.

## Architecture

```
theta (binary)          — CLI + TUI + sessions + tools + skills + templates  [Done]
theta-agent-core (lib)   — agent runtime: loop, compaction, events, hooks     [Done]
theta-ai (lib)           — unified LLM API: types, provider trait, streaming  [Done]
theta-tui (lib)          — terminal UI (ratatui + crossterm)                  [Done]
theta-models (lib)       — built-in model catalog (compile-time)              [Done]
```

**Dependency order:** `theta-ai` ← `theta-agent-core` ← `theta` (+ `theta-tui`, `theta-models`)

Extensions live in `theta/src/extensions/` (compiled-in traits, no separate crate).

## Phase Completion

| Phase | Status | Key Deliverables |
|-------|--------|-----------------|
| 1. Foundation | Done | theta-ai types, OpenAI/DeepSeek/OpenCode/Codex providers, theta-models |
| 2. Agent Runtime | Done | Agent, nested loops, tool execution, events, hooks, steering |
| 3. CLI + Tools | Done | CLI (clap), auth, sessions, 7 built-in tools, print mode, system prompt |
| 4. TUI | Done | theta-tui (ratatui), editor, chat, status bar, themes, event bridge |
| 5. Extensibility | Done | Skills, templates, continue/resume, slash commands, login flow, extensions |
| 6. Polish | In Progress | Compaction summary, retry+backoff+retry-after, session picker, OAuth PKCE flow, Codex WebSocket+SSE, service tiers, settings persistence, RPC, CI/docs |

## Remaining Work (Phase 6)

### Agent / Session
- [x] LLM-based summarization for compacted context
- [x] Deterministic fallback summary for compacted context
- [x] Session summary on resume — brief recap of prior conversation
- [x] Branch metadata for session navigation
- [x] Session metadata (project, branch, message count, last activity, token usage)

### CLI / Print Mode
- [x] RPC mode (JSON lines over stdin/stdout)
- [x] Print mode `--continue` flag, exit codes for tool errors

### Providers
- [ ] System prompt image warning for non-vision models
- [ ] Token proactive refresh + expiry detection (401 detection only for now)
- [x] Provider timeout configuration
- [ ] Graceful degradation when provider is unavailable

### TUI / UX
- [x] Theme cycling keybinding
- [x] Tool result snippets in chat transcript

### Project
- [x] README / CONTRIBUTING
- [x] CI workflow
- [ ] Version management, release artifacts

## Key Design Decisions

**1. Async:** `tokio` everywhere. No `async-std`/`smol`.

**2. Providers:** One `OpenAiCompatProvider` for OpenAI/DeepSeek/OpenCode + separate `OpenAiCodexProvider`. Per-model compat flags (`thinking_format`, `supports_developer_role`, `max_tokens_field`).

**3. Sessions:** Pi-compatible JSONL. Stored at `~/.theta/sessions/` (centralized). Index at `~/.theta/sessions/index.json`. Tests use `SessionManager::with_dir()`.

**4. Session entries:** `model_change` and `thinking_level_change` entries auto-emitted by `Agent::set_model()` / `set_thinking_level()`.

**5. Extensions:** Trait-based, compiled-in. No dynamic loading. Users fork theta and add their own `Extension` impls.

**6. Compaction:** Truncation (oldest pairs first). Config: `compaction.enabled`, `compaction.reserve_tokens`. Module: `theta-agent-core/src/compact.rs`. LLM summarization deferred.

**7. Retry:** Exponential backoff with `retry-after-ms` header support. Config: `retry.max_retries`, `retry.base_delay_ms`. Module: `theta-ai/src/providers/mod.rs` (registry-level retry loop).

**8. Config:** `~/.theta/config.toml` with `[compaction]`, `[retry]`, `theme` keys. Auth persistent across sessions via `~/.theta/auth.json`. Settings at `~/.theta/settings.json` (last model/thinking).

**9. Codex transport:** WebSocket (primary, lower latency) with SSE fallback. WebSocket sends JSON body as text frame, reads responses frame-by-frame. Both transports share the same `parse_codex_event()` parser.

**10. Slash commands:** `/model`, `/thinking`, `/clear`, `/session`, `/fork`, `/sessions`, `/login`, `/help`. Ctrl+P for model selector overlay.

## Non-Goals (MVP)

Anthropic/Google/Mistral providers, Slack bot, Web UI, vLLM, Windows-specific fixes, GH Actions, session sharing, dynamic WASM extensions, sub-agents, plan mode in core.

## Conventions

- Edition 2024. `tokio::sync::RwLock` across `.await`. `std::sync::Mutex` for short locks.
- `anyhow` for apps, `thiserror` for libs. No `unwrap()` in libs.
- Run `cargo fmt && cargo clippy -- -D warnings && cargo test` before committing non-doc changes.
- Never commit unprompted. Stage specific files.
- Read files in full before wide changes.
