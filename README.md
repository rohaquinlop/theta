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

- `@` opens Codex-style file autocomplete: gitignore-aware recursive paths, fuzzy-ranked.
- Sending `@path/to/file` appends that file's contents to the prompt context.
- `/sessions` opens the session picker.
- `/tree [default|no-tools|user-only|labeled-only|all]` opens branch/session tree picker.
- `Enter` sends normally when idle; while streaming it queues a steering message.
- `Alt+Enter` queues a follow-up message.
- `Ctrl+P` opens model selector.
- `Ctrl+T` cycles themes.
- `Tab` switches focus between input and chat.

## Config

Config lives at `~/.theta/config.toml`.

```toml
theme = "default"
working_dir = "/path/to/project"

[model]
default = "gpt-5.5"
providers = { openai = "gpt-5.5", deepseek = "deepseek-v4-pro", opencode = "gpt-5.5" }

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

[agent]
max_same_tool_call_repeats = 6
```

Available fields:

- `theme` (string, optional, default: unset): TUI theme name. Supported built-ins are `default` and `monokai`.
- `working_dir` (string/path, optional, default: unset): Working directory override in config. Note: current CLI behavior uses `--working-dir` (or current shell dir) and does not currently read this field.
- `[model].default` (string, optional, default: unset): default model ID when `--model` is not provided.
- `[model].providers` (map<string,string>, optional, default: `{}`): per-provider model defaults map (for example `openai`, `openai-codex`, `deepseek`, `opencode`).
- `[thinking].default` (string, optional, default: unset): default thinking level (commonly `off`, `low`, `medium`, `high`).
- `[compaction].enabled` (bool, default: `true`): enables automatic context compaction.
- `[compaction].reserve_tokens` (u32, default: `4096`): token budget reserved for model output.
- `[compaction].summarize_with_llm` (bool, default: `true`): summarize compacted content with the model.
- `[compaction].summary_max_tokens` (u32, default: `512`): max tokens used for compaction summaries.
- `[retry].max_retries` (u32, default: `2`): retry attempts for retryable provider errors.
- `[retry].base_delay_ms` (u64, default: `1000`): exponential backoff base delay in milliseconds.
- `[provider].timeout_ms` (u64, default: `120000`): provider request timeout in milliseconds.
- `[agent].max_same_tool_call_repeats` (u32, default: `6`): primary loop guard; maximum repeated identical tool-call signatures in one turn before aborting that loop.

Auth note:

- API keys and OAuth tokens are persisted in `~/.theta/auth.json` (and can also come from env vars like `OPENAI_API_KEY`, `OPENAI_CODEX_TOKEN`, `DEEPSEEK_API_KEY`, `OPENCODE_API_KEY`).
- The `[auth]` section is part of the internal config struct, but auth is loaded from `auth.json` at runtime.

## Settings File

Session-level runtime settings are stored in `~/.theta/settings.json` (not in `config.toml`).

Fields currently persisted there:

- `last_model` (string, optional): last model used in TUI.
- `last_thinking` (string, optional): last thinking level used in TUI.
- `steering_mode` (string, default: `"follow-up"`): Enter behavior while streaming.
- `follow_up_mode` (string, default: `"steer"`): Alt+Enter behavior while streaming.
- `transport_preference` (string, default: `"auto"`): transport hint (`auto`/`http`/`sse`).
- `show_thinking` (bool, default: `true`): show thinking text in UI.

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
