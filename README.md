# MichiN

```
┌─────────────────┐
│  /\_/\  ( o.o )  │
│  > ^ <          │
│  michin@dev:~$  │
└─────────────────┘
```

MichiN is a minimal terminal coding-agent harness in Rust, inspired by Pi.

## Install / Run

```bash
cargo run -- tui
cargo run -- prompt --new "inspect this repo"
cargo run -- prompt --continue "follow up"
cargo run -- continue "next task"
cargo run -- rpc
```

## Core Commands

- `michin` or `michin tui` starts a fresh TUI chat.
- `michin sessions` lists saved sessions.
- `michin resume <id>` resumes a session.
- `michin fork <id>` forks an existing session.
- `michin login <provider>` stores auth.
- `michin rpc` reads JSON requests from stdin and writes JSON responses to stdout.

## TUI

- `@` opens Codex-style file autocomplete: gitignore-aware recursive paths, fuzzy-ranked.
- Sending `@path/to/file` appends that file's contents to the prompt context.
- `/sessions` opens the session picker.
- `/tree [default|no-tools|user-only|labeled-only|all]` opens branch/session tree picker.
- `/themes` opens the theme picker with live color preview.
- `Enter` behavior is configurable via `settings.json` (`enter_behavior: "send" | "newline"`).
- With `enter_behavior = "send"` (default), `Enter` sends normally when idle; while streaming it queues a steering message.
- `Shift+Enter` inserts a newline in the editor.
- `Alt+Enter` inserts a newline (or queues follow-up depending on mode config).
- `Ctrl+Enter` queues a follow-up message.
- `Ctrl+P` opens model selector.
- `Ctrl+T` cycles themes (built-in + user themes).
- `Ctrl+U` edits the queued (steering/follow-up) message.
- `Tab` switches focus between input and chat.

## Themes

MichiN ships two built-in themes (`default` and `monokai`) and supports user-defined TOML theme files, inspired by Helix.

**Theme locations:**

- `~/.michin/themes/*.toml` — user themes (filename stem becomes the theme name)

**Switching themes:**

- `Ctrl+T` cycles through all available themes (built-in + user).
- `/themes` opens a picker with a live color preview of each theme before applying.
- Set `theme = "name"` in `~/.michin/config.toml` to persist a default across sessions. Selecting a theme via `/themes` does this automatically.

**Theme file format:**

```toml
# ~/.michin/themes/catppuccin_mocha.toml

# Base a new theme on a built-in ("default" or "monokai"). All fields optional.
inherits = "default"

# UI colors — hex, named ("red", "cyan", "dark_gray"), or rgb tuple.
accent   = "#cba6f7"
bg       = "#1e1e2e"
fg       = "#cdd6f4"
dim      = "#9399b2"
border   = "#313244"
highlight = "#45475a"
success  = "#a6e3a1"
error    = "#f38ba8"
warning  = "#f9e2af"

# Bubbles
user_bubble      = "#313244"
assistant_bubble = "#1e1e2e"

# Code blocks
code_fg = "#a6e3a1"
code_bg = "#181825"

# Markdown
md_heading_1    = "#f38ba8"
md_heading_2    = "#fab387"
md_list_marker  = "#94e2d5"
md_quote        = "#f5c2e7"
md_link         = "#89b4fa"
md_inline_code  = "#a6e3a1"
md_rule_border  = "#45475a"
md_table_header  = "#cba6f7"
md_task_marker   = "#a6e3a1"
```

**Color formats:**

- Named: `"red"`, `"green"`, `"cyan"`, `"dark_gray"`, `"reset"`, etc.
- Hex: `"#ff8800"` or `"ff8800"`
- RGB: `"rgb(255, 136, 0)"`

An example Dracula theme is shipped at `crates/michin/examples/dracula.toml`.

## Extensions (Rhai Scripts)

MichiN supports scriptable tool hooks via `.rhai` files — no fork, no recompile, no external runtime. The agent can write these for you when you ask.

**Ask the agent:**

- "Block any `git push --force` and ask me to confirm"
- "Warn me before editing `.env` files"
- "Don't allow `rm -rf` commands"

**Script locations:**

- `~/.michin/extensions/*.rhai` — global (all projects)
- `./.michin/extensions/*.rhai` — project-local

**Example script** (`~/.michin/extensions/guard.rhai`):

```rhai
// Block dangerous commands
tool.before("bash", |ctx| {
    if ctx.args.command.contains("rm -rf") {
        return #{ blocked: true, reason: "Blocked: rm -rf" };
    }
});

// Protect sensitive files
tool.before("write", |ctx| {
    if ctx.args.path.ends_with(".env") {
        return #{ blocked: true, reason: "no .env writes" };
    }
});
```

Scripts load automatically on next session. Script errors never block the tool they're guarding.

## Custom System Prompt

MichiN checks for two override files in `~/.michin/` at session start:

- **`~/.michin/SYSTEM_PROMPT.md`** — if present, replaces the entire system prompt (project context, skills, tools, and response contract). Use for a fully custom prompt.
- **`~/.michin/APPEND_SYSTEM_PROMPT.md`** — if present and `SYSTEM_PROMPT.md` does not exist, its content is appended to the normal system prompt. Use for adding extra instructions without rebuilding everything.

If both files exist, only `SYSTEM_PROMPT.md` is used.

**Example — appending a custom rule** (`~/.michin/APPEND_SYSTEM_PROMPT.md`):

```markdown
## Custom Rule

Always mention the estimated token cost of each operation before executing it.
```

**Example — full replacement** (`~/.michin/SYSTEM_PROMPT.md`):

```markdown
You are a helpful assistant with access to file tools.
Follow the user's instructions carefully.
```

No config changes needed — just drop the file in and start a new session.

## Config

Config lives at `~/.michin/config.toml`.

```toml
theme = "default"
working_dir = "/path/to/project"
profile = "safe"

[model]
default = "gpt-5.5"
providers = { openai = "gpt-5.5", deepseek = "deepseek-v4-pro", opencode = "gpt-5.5" }

[thinking]
default = "medium"

[compaction]
enabled = true
reserve_tokens = 4096
keep_recent_tokens = 20000
strategy = "llm"
summary_max_tokens = 512
auto_pause_threshold = 2

[retry]
max_retries = 2
base_delay_ms = 1000

[provider]
timeout_ms = 120000

[agent]
max_same_tool_call_repeats = 6
tool_stall_warning_ms = 8000
provider_fallback_chain = []
provider_failure_threshold = 3
provider_open_cooldown_ms = 30000

[profile_overrides]
# Override profile-specific defaults. All fields optional.
# max_retries = 3
# command_policy_strict = false
```

Available fields:

- `theme` (string, optional): TUI theme name. Built-ins: `default`, `monokai`. User themes placed in `~/.michin/themes/*.toml` are also available (see [Themes](#themes)).
- `working_dir` (string/path, optional): Working directory override in config. Note: current CLI behavior uses `--working-dir` (or current shell dir) and does not currently read this field.
- `profile` (string, default: `"safe"`): Runtime hardening profile. Options: `dev` (lenient, permissive), `safe` (default, balanced), `prod` (strict, aggressive limits).
- `[model].default` (string, optional): default model ID when `--model` is not provided.
- `[model].providers` (map<string,string>, default: `{}`): per-provider model defaults map (for example `openai`, `openai-codex`, `deepseek`, `opencode`).
- `[thinking].default` (string, optional): default thinking level (`off`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max`).
- `[compaction].enabled` (bool, default: `true`): enables automatic context compaction.
- `[compaction].reserve_tokens` (u32, default: `4096`): token budget reserved for model output.
- `[compaction].keep_recent_tokens` (u32, default: `20000`): tokens of recent conversation to preserve.
- `[compaction].strategy` (string, default: `"llm"`): compaction strategy. `"none"`, `"textual"`, or `"llm"`.
- `[compaction].summary_max_tokens` (u32, default: `512`): max tokens used for compaction summaries.
- `[compaction].auto_pause_threshold` (u32, default: `2`): consecutive compactions before auto-pausing. When the kept tail alone overflows the context trigger, compacting every turn degrades prefix cache. Set to `u32::MAX` to never auto-pause.
- `[retry].max_retries` (u32, default: `2`): retry attempts for retryable provider errors.
- `[retry].base_delay_ms` (u64, default: `1000`): exponential backoff base delay in milliseconds.
- `[provider].timeout_ms` (u64, default: `120000`): provider request timeout in milliseconds.
- `[agent].max_same_tool_call_repeats` (u32, default: `6`): primary loop guard; maximum repeated identical tool-call signatures in one turn before aborting that loop.
- `[agent].tool_stall_warning_ms` (u64, default: `8000`): warn if a tool execution stalls longer than this.
- `[agent].provider_fallback_chain` (string[], default: `[]`): optional fallback model IDs in preference order.
- `[agent].provider_failure_threshold` (u32, default: `3`): circuit breaker failure threshold.
- `[agent].provider_open_cooldown_ms` (u64, default: `30000`): circuit breaker open cooldown in ms.
- `[profile_overrides]` (table, optional): override individual profile defaults. All fields optional: `max_retries`, `base_delay_ms`, `provider_timeout_ms`, `tool_stall_warning_ms`, `provider_fallback_chain`, `provider_failure_threshold`, `provider_open_cooldown_ms`, `max_same_tool_call_repeats`, `command_policy_strict`.

Auth note:

- API keys and OAuth tokens are persisted in `~/.michin/auth.json` (and can also come from env vars like `OPENAI_API_KEY`, `OPENAI_CODEX_TOKEN`, `DEEPSEEK_API_KEY`, `OPENCODE_API_KEY`, `MIMO_API_KEY`).
- The `[auth]` section is part of the internal config struct, but auth is loaded from `auth.json` at runtime.

## Settings File

Session-level runtime settings are stored in `~/.michin/settings.json` (not in `config.toml`).

Fields currently persisted there:

- `last_session` (object, optional): last used provider+model pair (e.g. `{"provider": "openai", "model": "gpt-5.5"}`). Replaces old flat `last_model`/`last_thinking` fields.
- `model_thinking_map` (object, default: `{}`): per-provider, per-model thinking level map. Example: `{"openai": {"gpt-5.5": "high"}}`. Enables restoring thinking level when switching models across providers.
- `steering_mode` (string, default: `"follow-up"`): Enter behavior while streaming.
- `follow_up_mode` (string, default: `"steer"`): Ctrl+Enter behavior while streaming.
- `transport_preference` (string, default: `"auto"`): transport hint (`auto`/`http`/`sse`).
- `show_thinking` (bool, default: `true`): show thinking text in UI.
- `show_tool_diffs` (bool, default: `false`): show diffs in tool output (edit tool).
- `tool_progress_hz` (u64, default: `20`): tool progress update frequency in Hz.
- `enter_behavior` (string, default: `"send"`): editor Enter behavior (`"send"` or `"newline"`).
- `max_context_window` (u32 or null, default: `null`): hard cap on context window tokens. `null` disables the cap, using the model's full context window. Any number is clamped to `min(model.context_window, value)`. Most LLMs perform better below ~250K tokens, and this cap helps prevent hallucinations on long conversations. Change this or set to `null` to rely on the model's native context limit.
- `disabled_models` (string[], default: `[]`): model IDs to hide from the model selector.
- `favorite_models` (string[], default: `[]`): model IDs pinned at the top of the model selector.
- `mimo_cluster_url` (string or null, default: `null`): MiMo token-plan cluster base URL (region endpoint). Overrides `MIMO_BASE_URL` env var when set.

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
