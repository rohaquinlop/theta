# michin-script — Agent Rules

> Rules for working on the michin-script crate: Rhai script engine.

## Crate Purpose

Rhai-powered runtime hooks that bridge user scripts to `theta_agent_core::Hooks`. Scripts can intercept tool calls and render TUI status elements.

## Key Files

| File                | Purpose                                               |
| ------------------- | ----------------------------------------------------- |
| `src/lib.rs`        | Public API: `ScriptEngine`, `ScriptHooks`, `ScriptLoader` |
| `src/engine.rs`     | Rhai engine setup, `tool.before`/`tool.after`/`tui.status`/`tui.row` API |
| `src/hooks.rs`      | `ScriptHooks` — bridges Rhai callbacks to `Hooks` trait |
| `src/loader.rs`     | File discovery: `~/.theta/extensions/*.rhai` + `./.theta/extensions/*.rhai` |

## Script Loading

- Scripts auto-discovered on agent creation from global and project-local dirs.
- No `/reload` needed. Script errors never block tool execution.

## Rhai API

- `tool.before(name, callback)` — block/modify before execution
- `tool.after(name, callback)` — react after execution
- `tui.status(key, callback)` — add status bar line
- `tui.row(index, callback)` — add full TUI row
- `ctx.args` — tool arguments as object map
- `ctx.notify(msg)` — send notification to TUI
- `cwd()` — current working directory
- `home_dir()` — user home directory
- `get_state(key)` / `set_state(key, value)` — persistent string state

## Conventions

- `call` reserved in Rhai. Use `ctx` for callback parameter.
- Script callbacks must never panic — errors caught and logged.
- Dependencies: `michin-agent-core` + `michin-ai` for hook bridging.
- Comments follow root `AGENTS.md` Comment Style rules.
