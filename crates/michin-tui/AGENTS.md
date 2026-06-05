# michin-tui — Agent Rules

> Rules for working on the michin-tui crate: terminal UI.

## Crate Purpose

Terminal UI built with ratatui + crossterm. Chat view, multi-line editor with @-autocomplete, fuzzy file matching, login flows, model/session/tree/settings selectors, status bar.

## Key Files

| File                                         | Purpose                                |
| -------------------------------------------- | -------------------------------------- |
| `src/app.rs`                                 | Top-level TUI state machine            |
| `src/components/mod.rs`                      | `Component` trait, `Action` enum       |
| `src/components/chat.rs`                     | Chat view with message rendering       |
| `src/components/editor.rs`                   | Multi-line input with @-autocomplete   |
| `src/components/fuzzy.rs`                    | Fuzzy file path matching               |
| `src/components/login_flow.rs`               | OAuth login flow for Codex             |
| `src/components/model_selector.rs`           | Ctrl+P model picker                    |
| `src/components/session_picker.rs`           | `/sessions` command                    |
| `src/components/tree_selector.rs`            | `/tree` branch/session tree            |
| `src/components/settings_selector.rs`        | Settings overlay                       |
| `src/components/caveman_selector.rs`         | Caveman mode level picker              |
| `src/components/mimo_cluster.rs`             | MiMo cluster region picker             |
| `src/components/status.rs`                   | Bottom status bar                      |
| `src/theme.rs`                               | `Theme` — `default` and `monokai`      |
| `src/keybinding.rs`                          | Keybinding configuration               |

## Keybindings

| Key                 | Action                            |
| ------------------- | --------------------------------- |
| `Ctrl+C` / `Esc`    | Quit (Esc only when input empty)  |
| `Ctrl+P`            | Open model selector               |
| `Ctrl+T`            | Cycle themes                      |
| `Ctrl+U`            | Edit queued message               |
| `Tab`               | Switch focus input ↔ chat         |
| `Enter`             | Send / Queue steering             |
| `Shift+Enter`       | Insert newline                    |
| `Alt+Enter`         | Insert newline / Queue follow-up  |
| `Ctrl+Enter`        | Queue follow-up                   |
| `@` in editor       | File autocomplete (fuzzy)         |

## Conventions

- Use `anyhow` for app errors.
- Components implement the `Component` trait.
- Theme changes must work in both `default` and `monokai`.
- Config actions (`/model`, `/thinking`) are async with ack to avoid stale-state races.
- Comments follow root `AGENTS.md` Comment Style rules.

## Testing

- TUI tests use `ratatui`'s test backend where possible.
- Fuzzy matching tests cover gitignore-aware filtering.
