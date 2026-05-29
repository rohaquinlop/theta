# theta (CLI) — Agent Rules

> Rules for working on the theta binary crate: CLI, TUI glue, tools, sessions, config.

## Crate Purpose

The main binary crate. Clap CLI, TUI mode, built-in tools, session management, config/settings, login flows, RPC, system prompt construction.

## Key Files

| File                                  | Purpose                                                     |
| ------------------------------------- | ----------------------------------------------------------- |
| `src/main.rs`                         | Entry point                                                 |
| `src/cli.rs`                          | Clap argument parsing                                       |
| `src/config.rs`                       | `ThetaConfig`, `AuthConfig` with env fallback               |
| `src/settings.rs`                     | Persistent settings.json                                    |
| `src/interactive.rs`                  | TUI mode glue: agent ↔ TUI bridge                           |
| `src/system_prompt.rs`                | System prompt builder (AGENTS.md, CLAUDE.md, skills, tools) |
| `src/skills.rs`                       | Skill discovery, YAML frontmatter, XML generation           |
| `src/scripts.rs`                      | Extension script discovery                                  |
| `src/session.rs`                      | `SessionManager` — pi-compatible JSONL                      |
| `src/login.rs`                        | `theta login` OAuth entry point                             |
| `src/oauth/codex.rs`                  | Codex OAuth token exchange and refresh                      |
| `src/rpc.rs`                          | JSON-RPC over stdin/stdout                                  |
| `src/prompts.rs`                      | Print-mode prompt execution                                 |
| `src/print_mode.rs`                   | Non-TUI streaming output formatter                          |
| `src/mentions.rs`                     | @-mention file content resolution                           |
| `src/tools/mod.rs`                    | Tool registry, `ToolContext`, truncation                    |
| `src/tools/{bash,edit,read,write}.rs` | Built-in tool implementations                               |
| `src/extensions/mod.rs`               | TUI extension row rendering                                 |

## Tool System

Seven built-in tools in `src/tools/`: `read`, `write`, `edit`, `bash`.

- All implement `theta_agent_core::AgentTool`.
- `ToolContext` holds working directory — relative paths resolve against it.
- Output truncation: `max_lines: 2000`, `max_bytes: 50_000`.

## Session Format

Pi-compatible JSONL. Sessions in `~/.theta/sessions/` with `index.json`.
JSONL entries: `user`, `assistant`, `toolResult`, `model_change`, `thinking_level_change`.

## Config and Auth

- Config: `~/.theta/config.toml` (model default, thinking default, agent safety, compaction, retry, provider, profile, theme).
- Auth: `~/.theta/auth.json` with env var fallback. OAuth tokens auto-refresh.
- Supported providers: OpenAI, OpenAI Codex, DeepSeek, OpenCode, Xiaomi MiMo.

## Conventions

- Use `anyhow` for app errors.
- Config changes require explicit user request.
- System prompt composition follows nested AGENTS.md discovery (see system_prompt.rs).
- Comments follow root `AGENTS.md` Comment Style rules.
