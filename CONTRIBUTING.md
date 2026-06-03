# Contributing

Keep MichiN small and terminal-first.

## Rules

- Rust 2024 across all crates.
- `tokio` for async.
- `tracing` for logs.
- `anyhow` in the binary, `thiserror` in libraries.
- No `unwrap()` in library code.
- No dynamic provider or tool loading in MVP.

## Before Sending Changes

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo test -p michin-agent-core --test policy_scenario_matrix
```

Stage only files you changed. Do not commit generated or unrelated files.

## Project Architecture

MichiN is a minimal terminal coding-agent harness in Rust, inspired by [pi](https://github.com/earendil-works/pi). Six crates in a Cargo workspace (`edition = "2024"`, `resolver = "3"`):

```
crates/michin              — CLI + TUI + sessions + built-in tools + skills + themes + scripts + RPC
crates/michin-agent-core   — agent runtime: Agent, loop, tool execution, compaction, events, hooks
crates/michin-ai           — unified LLM API: types, provider trait, streaming, replay, two providers
crates/michin-tui          — terminal UI (ratatui + crossterm): chat, editor, fuzzy, logins, selectors, status bar
crates/michin-models       — built-in model catalog (compile-time definitions + runtime OpenCode fetch)
crates/michin-script       — Rhai-powered hooks: before/after tool calls, TUI status rows
```

**Dependency order:** `michin-ai` ← `michin-agent-core` ← `michin` (+ `michin-tui`, `michin-models`, `michin-script`)

## Phase Completion Status

All six phases complete. Active maintenance and polish.

| Phase            | Status | Key Deliverables                                                                                                                                |
| ---------------- | ------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| 1. Foundation    | Done   | `michin-ai` + `michin-models`                                                                                                                     |
| 2. Agent Runtime | Done   | `michin-agent-core`                                                                                                                              |
| 3. CLI + Tools   | Done   | `theta` binary with built-in tools                                                                                                              |
| 4. TUI           | Done   | `michin-tui` + interactive mode                                                                                                                  |
| 5. Extensibility | Done   | Skills, templates, continue/resume, slash commands, login flow, scripts                                                                         |
| 6. Polish        | Done   | Compaction (truncation + summary), retry (exponential backoff), session picker, tree selector, model selector, settings selector, theme cycling |

## Key Project Files

| File                                                   | Purpose                                                                                   |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| `Cargo.toml`                                           | Workspace root, shared dependencies                                                       |
| `README.md`                                            | User-facing install, usage, config, RPC docs                                              |
| `AGENTS.md`                                            | Agent guidance (root)                                                                     |
| `crates/*/AGENTS.md`                                   | Per-crate agent guidance                                                                  |
| `CONTRIBUTING.md`                                      | This file — dev setup, architecture, contributing rules                                   |
| `crates/michin-ai/src/lib.rs`                           | Public API: error, event, model, provider, providers/, replay, types                      |
| `crates/michin-ai/src/types.rs`                         | `ContentBlock`, `Message`, `Tool`, `Provider`, `Model`, `Context`, `StopReason`, etc.     |
| `crates/michin-ai/src/event.rs`                         | `EventAccumulator`, `AssistantMessageEvent` — streaming event types                       |
| `crates/michin-ai/src/providers/openai_compat.rs`       | `OpenAiCompatProvider` — handles OpenAI, DeepSeek, OpenCode                               |
| `crates/michin-ai/src/providers/openai_codex.rs`        | `OpenAiCodexProvider` — ChatGPT Plus session-token auth, WS+SSE                           |
| `crates/michin-agent-core/src/lib.rs`                   | Public API: `Agent`, `AgentError`, `AgentEvent`, `Hooks`, `AgentState`, tool/config types |
| `crates/michin-agent-core/src/agent.rs`                 | `Agent` struct: prompt, continue, steer, follow_up, subscribe, hooks                      |
| `crates/michin-agent-core/src/loop_mod.rs`              | Core loop: nested outer/inner, turn enforcement, steering drain, abort                    |
| `crates/michin-agent-core/src/compact.rs`               | Truncation compaction + inline text summary of trimmed messages                           |
| `crates/michin-agent-core/src/command_policy.rs`        | Centralized command safety policy engine                                                  |
| `crates/michin-agent-core/src/types.rs`                 | `AgentTool` trait, `ToolResult`, `ToolCall`, `AgentLoopConfig`, config types              |
| `crates/michin-agent-core/src/events.rs`                | `AgentEvent` enum                                                                         |
| `crates/michin-agent-core/src/hooks.rs`                 | `Hooks` trait                                                                             |
| `crates/michin-tui/src/app.rs`                          | `App` — top-level TUI state machine, event loop bridge                                    |
| `crates/michin-tui/src/components/mod.rs`               | `Component` trait, `Action` enum, re-exports                                              |
| `crates/michin-tui/src/components/chat.rs`              | Chat view with message rendering                                                          |
| `crates/michin-tui/src/components/editor.rs`            | Multi-line input editor with @-autocomplete                                               |
| `crates/michin-tui/src/components/fuzzy.rs`             | Fuzzy file path matching for @-autocomplete                                               |
| `crates/michin-tui/src/components/login_flow.rs`        | Interactive OAuth login flow for Codex                                                    |
| `crates/michin-tui/src/components/model_selector.rs`    | Ctrl+P model picker overlay                                                               |
| `crates/michin-tui/src/components/session_picker.rs`    | `/sessions` command session list                                                          |
| `crates/michin-tui/src/components/tree_selector.rs`     | `/tree` command branch/session tree with filters                                          |
| `crates/michin-tui/src/components/settings_selector.rs` | Settings overlay                                                                          |
| `crates/michin-tui/src/components/status.rs`            | Bottom status bar rendering                                                               |
| `crates/michin-tui/src/theme.rs`                        | `Theme` struct — `default` and `monokai` built-ins                                        |
| `crates/michin-tui/src/keybinding.rs`                   | Keybinding configuration                                                                  |
| `crates/michin-models/src/lib.rs`                       | `BuiltInCatalog` — implements `ModelCatalog` trait                                        |
| `crates/michin-models/src/openai.rs`                    | Static OpenAI model definitions                                                           |
| `crates/michin-models/src/deepseek.rs`                  | Static DeepSeek model definitions                                                         |
| `crates/michin-models/src/opencode.rs`                  | Dynamic OpenCode Zen model fetch + fallback, cost calculation                             |
| `crates/michin-models/src/codex.rs`                     | Static Codex model definitions                                                            |
| `crates/michin-script/src/lib.rs`                       | Public API: `ScriptEngine`, `ScriptHooks`, `ScriptLoader`                                 |
| `crates/michin-script/src/engine.rs`                    | Rhai engine setup                                                                         |
| `crates/michin-script/src/hooks.rs`                     | `ScriptHooks` — bridges Rhai callbacks to `Hooks` trait                                   |
| `crates/michin-script/src/loader.rs`                    | File discovery: `~/.michin/extensions/*.rhai` + `./.theta/extensions/*.rhai`               |
| `crates/michin/src/main.rs`                             | Entry point                                                                               |
| `crates/michin/src/cli.rs`                              | Clap argument parsing: prompt, continue, resume, fork, sessions, login, rpc, tui          |
| `crates/michin/src/config.rs`                           | `MichiNConfig` — config.toml parsing, `AuthConfig` — auth.json with env fallback           |
| `crates/michin/src/settings.rs`                         | Persistent settings.json (last model, thinking, steering mode, etc.)                      |
| `crates/michin/src/interactive.rs`                      | TUI mode glue: agent creation, model resolution, auth auto-switch                         |
| `crates/michin/src/system_prompt.rs`                    | System prompt builder: AGENTS.md (nested), CLAUDE.md, skills, extensions, tools           |
| `crates/michin/src/skills.rs`                           | Skill discovery (global + project-local), YAML frontmatter parsing                        |
| `crates/michin/src/scripts.rs`                          | Extension script discovery for system prompt injection                                    |
| `crates/michin/src/session.rs`                          | `SessionManager` — pi-compatible JSONL sessions in `~/.michin/sessions/`                   |
| `crates/michin/src/login.rs`                            | `theta login` — OAuth flow entry point                                                    |
| `crates/michin/src/oauth/codex.rs`                      | Codex OAuth token exchange and refresh                                                    |
| `crates/michin/src/rpc.rs`                              | JSON-RPC over stdin/stdout                                                                |
| `crates/michin/src/prompts.rs`                          | Print-mode prompt execution                                                               |
| `crates/michin/src/print_mode.rs`                       | Non-TUI streaming output formatter                                                        |
| `crates/michin/src/mentions.rs`                         | @-mention file content resolution                                                         |
| `crates/michin/src/tools/mod.rs`                        | Tool registry: builtin_tools(), ToolContext, truncation, path resolution                  |
| `crates/michin/src/tools/{bash,edit,read,write}.rs`     | Built-in tool implementations                                                             |
| `crates/michin/src/extensions/mod.rs`                   | TUI extension row rendering from Rhai scripts                                             |

## Provider Strategy

Four providers, two implementations:

1. **`OpenAiCompatProvider`** — handles OpenAI, DeepSeek, OpenCode. All speak OpenAI's `/v1/chat/completions`. Per-model compat flags handle differences.
2. **`OpenAiCodexProvider`** — ChatGPT Plus session-token auth targeting `chatgpt.com/backend-api`. WebSocket + SSE fallback.

### Compat Flags

| Flag                                      | Purpose                                                            |
| ----------------------------------------- | ------------------------------------------------------------------ |
| `thinking_format`                         | `"openai"` (reasoning_effort) vs `"deepseek"` (thinking: { type }) |
| `supports_developer_role`                 | o-series models need `developer` instead of `system`               |
| `requires_reasoning_content_on_assistant` | DeepSeek needs empty `reasoning_content` on replayed messages      |
| `max_tokens_field`                        | `max_completion_tokens` vs `max_tokens`                            |

### Codex Transport Notes

- WebSocket TLS via `tokio-tungstenite` with `rustls-tls-webpki-roots`.
- WS fails → fallback to SSE.
- Don't emit duplicate synthetic `Done(stop)` after parser already emitted `Done(toolUse)`.

### API Keys

Read from env vars and `~/.michin/auth.json`. OAuth tokens auto-refresh. `AuthConfig::merge_with_existing()` preserves unrelated provider credentials on save.

### Current Models

- **OpenAI**: `gpt-5.5`, `gpt-5.5-instant`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-chat-latest`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`, `o4`, `o4-mini`, `o3`, `o3-mini`, `o1`, `o1-mini` — auth via `OPENAI_API_KEY`
- **OpenAI Codex**: same model IDs as OpenAI — auth via `OPENAI_CODEX_TOKEN` env var or OAuth
- **DeepSeek**: `deepseek-v4-pro` (1M ctx), `deepseek-v4-flash` (1M ctx)
- **OpenCode Zen**: fetched from `opencode.ai/zen/v1/models` at runtime, static `opencode` fallback

## Session Format

Pi-compatible JSONL. MichiN reads/writes same format as pi. Sessions in `~/.michin/sessions/` with `index.json`. JSONL entries: `user`, `assistant`, `toolResult`, `model_change`, `thinking_level_change`.

## Tool System

Seven built-in tools, each implementing `theta_agent_core::AgentTool`:

| Tool    | File                              | Description                                       |
| ------- | --------------------------------- | ------------------------------------------------- |
| `read`  | `crates/michin/src/tools/read.rs`  | File reading with line/byte limits and truncation |
| `write` | `crates/michin/src/tools/write.rs` | Create/overwrite files                            |
| `edit`  | `crates/michin/src/tools/edit.rs`  | Exact string replacement (pi's edit semantics)    |
| `bash`  | `crates/michin/src/tools/bash.rs`  | Shell command execution with timeout              |

Path behavior: absolute paths honored directly. Output truncation at 2000 lines / 50KB.

## Extension Model

Three tiers:

1. **Skills** (`SKILL.md` files) — Markdown with YAML frontmatter, discovered from `~/.michin/skills/` and `./.theta/skills/`.
2. **Rhai Scripts** (`~/.michin/extensions/*.rhai`, `./.theta/extensions/*.rhai`) — Runtime hooks.
3. **Rust Traits** — `AgentTool`, `Hooks`, `LlmProvider`. Fork MichiN, implement traits.

## TUI Keybindings

| Key                 | Action                                           |
| ------------------- | ------------------------------------------------ |
| `Ctrl+C` / `Esc`    | Quit (Esc only when input empty)                 |
| `Ctrl+P`            | Open model selector                              |
| `Ctrl+T`            | Cycle themes (default ↔ monokai)                 |
| `Tab`               | Switch focus between input and chat              |
| `Enter`             | Send message (idle) / Queue steering (streaming) |
| `Alt+Enter`         | Queue follow-up (streaming)                      |
| `@` in editor       | File autocomplete (fuzzy, gitignore-aware)       |
| `/sessions`         | Open session picker                              |
| `/tree`             | Open branch/session tree selector                |
| `/new`              | Start fresh session                              |
| `/help`             | Show help                                        |
| `/model <id>`       | Switch model                                     |
| `/thinking <level>` | Set thinking level                               |
| `/settings`         | Open settings overlay                            |
| `/session`          | Show current session info                        |

## Config

Config: `~/.michin/config.toml`. Auth: `~/.michin/auth.json`. Settings: `~/.michin/settings.json`.

```toml
[model]
default = "deepseek-v4-flash"

[thinking]
default = "default"

[agent]
max_same_tool_call_repeats = 6
tool_stall_warning_ms = 8000
tool_timeout_ms = 60000
provider_fallback_chain = []
provider_failure_threshold = 3
provider_open_cooldown_ms = 30000

[compaction]
enabled = true
reserve_tokens = 4096

[retry]
max_retries = 2
base_delay_ms = 1000

[provider]
timeout_ms = 120000

[profile]
# "dev", "safe" (default), "prod"

[theme]
# "default" or "monokai"
```

## Agent Loop Design

Nested loop pattern: outer loop for follow-up turns, inner loop for LLM call → stream → tools → repeat. Turn modes: `Execute`, `Inspect`, `AnalyzeOnly`, `PlanOnly`, `Clarify`. Turn enforcement with intent flags and bounded one-shot retry. Circuit breaker, tool watchdog, provider fallback chain. Run reports with event timeline.

## Compaction

Truncation-based (oldest user/assistant pairs first) with inline text summary. System prompt and last user message never trimmed.

## Retry

Exponential backoff for 429, 5xx, connection/timeout errors. Configurable via `retry.max_retries` and `retry.base_delay_ms`.

## Testing

- Unit tests maintained across all crates.
- Integration tests behind `#[cfg(feature = "integration-tests")]`.
- Faux provider for testing agent loop without real APIs.
- Policy scenario matrix covers circuit breaker, watchdog, command policy, fallback chain, run reports.

## Adding a New LLM Provider

1. If OpenAI-compatible: add compat flags to `Model` struct, update `OpenAiCompatProvider`
2. If needs new API or auth: implement `Provider` trait in `michin-ai/src/providers/`
3. Add model definitions to `michin-models/src/<provider>.rs`
4. Register in `BuiltInCatalog::new()` in `michin-models/src/lib.rs`
5. Add env var in `config.rs::provider_env_var()` and `auth.rs::get_env_token()`
6. Update `Provider` enum in `michin-ai/src/types.rs`
7. Update `AGENTS.md` and `CONTRIBUTING.md`
