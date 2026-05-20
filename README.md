# Theta

Theta is a minimal terminal coding-agent harness in Rust, inspired by Pi.

## Install / Run

```bash
cargo run -- tui
cargo run -- prompt --new "inspect this repo"
cargo run -- prompt --continue "follow up"
cargo run -- continue "next task"
cargo run -- rpc
```

## Core Commands

- `theta` or `theta tui` starts a fresh TUI chat.
- `theta sessions` lists saved sessions.
- `theta resume <id>` resumes a session.
- `theta login <provider>` stores auth.
- `theta rpc` reads JSON requests from stdin and writes JSON responses to stdout.

## TUI

- `@` opens recursive file autocomplete. Query is a regex over relative paths.
- `/sessions` opens the session picker.
- `Ctrl+P` opens model selector.
- `Ctrl+T` cycles themes.
- `Tab` switches focus between input and chat.

## Config

Config lives at `~/.theta/config.toml`.

```toml
theme = "default"

[model]
default = "gpt-5.5"

[thinking]
default = "medium"

[compaction]
enabled = true
reserve_tokens = 4096
summarize_with_llm = true
summary_max_tokens = 512

[retry]
max_retries = 2
base_delay_ms = 1000

[provider]
timeout_ms = 120000
```

RPC examples:

```json
{"id":1,"method":"ping"}
{"id":2,"method":"sessions"}
{"id":3,"method":"prompt","params":{"text":"summarize this repo","model":"gpt-5.5"}}
```

## Checks

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
```
