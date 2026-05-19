# Theta — Rust Coding Agent Harness: Implementation Plan

## Architecture Overview

Mirrors Pi's 3-layer architecture but adapted for Rust's strengths:

```
theta (CLI binary)         — coding agent CLI + TUI
theta-agent-core (lib)     — agent runtime: loop, tool calling, events, state
theta-ai (lib)             — unified multi-provider LLM API
theta-tui (lib)            — terminal UI library (ratatui-based)
```

Plus two supporting crates:
```
theta-models (lib)         — built-in model catalog (separate from theta-ai for compile-time)
theta-extensions (lib)     — extension/tool registration traits (no dynamic loading initially)
```

---

## Crate Breakdown

### 1. `theta-ai` — Unified Multi-Provider LLM API

**Supported providers:**
- **OpenAI** — standard API key via `OPENAI_API_KEY`, `https://api.openai.com`
- **DeepSeek** — API key via `DEEPSEEK_API_KEY`, `https://api.deepseek.com`
- **OpenCode** — user-configured base URL, `https://api.opencode.ai`
- **OpenAI Codex** — ChatGPT Plus subscription token via `OPENAI_CODEX_TOKEN`,
  targets `https://chatgpt.com/backend-api`. No API key needed —
  users authenticate with their ChatGPT session token.

**What Pi's `pi-ai` does:**
- Type system: Message (User, Assistant, ToolResult), Tool (name, description, schema), Model
- Model registry with provider info, cost, context window, thinking support
- Stream-based LLM calls: `streamSimple(model, context, options)` → EventStream
- Provider API registry: pluggable providers registered at startup
- Event protocol: start, text_start/delta/end, thinking_start/delta/end, toolcall_start/delta/end, done, error
- Stop reasons: stop, length, toolUse, error, aborted
- Usage tracking with cost calculation

**Provider strategy:** All three target providers (OpenAI, DeepSeek, OpenCode) use
OpenAI-compatible completions API. DeepSeek adds `thinking: { type: "enabled" }` for
reasoning mode. OpenCode is a pure proxy. One provider handles all three, with
compatibility flags per model for the small differences.

```
theta-ai/
├── Cargo.toml
├── src/
│   ├── lib.rs              — re-exports
│   ├── types.rs            — Message, Tool, Model, Usage, Context, enums
│   ├── stream.rs           — AsyncStream of AssistantMessageEvent
│   ├── event.rs            — AssistantMessageEvent enum
│   ├── model.rs            — Model struct, model catalog trait
│   ├── provider.rs         — Provider trait (stream, stream_simple)
│   ├── providers/
│   │   ├── mod.rs          — registry (HashMap<Api, Box<dyn Provider>>)
│   │   └── openai_compat.rs — OpenAI-compatible: covers OpenAI, DeepSeek, OpenCode
│   ├── error.rs            — ThetaError enum
│   └── utils.rs            — token counting, JSON schema helpers
```

**Key types:**
```rust
enum Message {
    User { content: Vec<ContentBlock>, timestamp: u64 },
    Assistant { content: Vec<ContentBlock>, api: Api, provider: Provider, model: String,
                usage: Usage, stop_reason: StopReason, error_message: Option<String>, timestamp: u64 },
    ToolResult { tool_call_id: String, tool_name: String, content: Vec<ContentBlock>,
                 details: Option<serde_json::Value>, is_error: bool, timestamp: u64 },
}

struct Tool {
    name: String,
    description: String,
    parameters: serde_json::Value,  // JSON Schema
}

struct Model {
    id: String,
    name: String,
    api: Api,
    provider: Provider,
    base_url: String,
    reasoning: bool,
    thinking_level_map: HashMap<ThinkingLevel, String>,
    input_modalities: Vec<Modality>,
    cost: ModelCost,
    context_window: u32,
    max_tokens: u32,
}

#[async_trait]
trait Provider {
    async fn stream(&self, model: &Model, context: &Context, options: &StreamOptions)
        -> Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send + '_>>;

    async fn stream_simple(&self, model: &Model, context: &Context, options: &SimpleStreamOptions)
        -> Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send + '_>>;

    fn api() -> Api where Self: Sized;
}
```

**Dependencies:** `reqwest`, `tokio`, `serde`, `serde_json`, `futures`, `async-trait`, `tracing`

**Effort estimate:** ~1500-2000 LOC (single provider with compat flags beats 3 providers)

**OpenAI-compatible provider design:**

DeepSeek and OpenCode both speak OpenAI's `/v1/chat/completions` API. The differences
are small and model-specific:

| Feature | OpenAI | DeepSeek | OpenCode |
|---------|--------|----------|----------|
| Base URL | `api.openai.com` | `api.deepseek.com` | configurable |
| Reasoning param | `reasoning_effort` | `thinking: { type }` + `reasoning_effort` | `reasoning_effort` |
| Reasoning in stream | `reasoning_content` chunks | `reasoning_content` chunks | `reasoning_content` chunks |
| Replayed assistant msg | normal | must include empty `reasoning_content` | normal |
| Tool streaming | `tool_calls[].function.arguments` deltas | `tool_calls[].function.arguments` deltas | same |
| System prompt | `system` role (`developer` for o-series) | `system` role | `system` role |
| Usage in stream | `stream_options.include_usage` | `stream_options.include_usage` | `stream_options.include_usage` |

**Model compat struct:**
```rust
struct OpenAiCompat {
    /// How to send reasoning/thinking params.
    /// - "openai": `reasoning_effort` field
    /// - "deepseek": `thinking: { type: "enabled" }` + `reasoning_effort`
    thinking_format: ThinkingFormat,
    /// Whether to use `developer` role for system messages (o-series models)
    supports_developer_role: bool,
    /// Whether `stream_options: { include_usage: true }` works
    supports_usage_in_streaming: bool,
    /// Which field for max tokens
    max_tokens_field: MaxTokensField,
}
```

This is stored on each `Model` and applied by the single `OpenAiCompatProvider`.

---

### 2. `theta-agent-core` — Agent Runtime

**What Pi's `pi-agent-core` does:**
- Agent class: stateful, owns transcript, runs prompt/continue loops
- Agent loop: two nested loops (inner: tool+steering, outer: follow-ups)
- Tool execution: sequential and parallel modes
- Lifecycle events: agent_start, turn_start, message_start/update/end, tool_execution_*
- Queues: steering (interrupt current turn), follow-up (after stop)
- Hooks: beforeToolCall, afterToolCall, shouldStopAfterTurn, prepareNextTurn
- State: systemPrompt, tools, messages, isStreaming, pendingToolCalls
- Thinking level: off/minimal/low/medium/high/xhigh
- `AgentMessage` abstraction: LLM messages + custom app messages
- `convertToLlm`: AgentMessage[] → Message[] before each LLM call
- `transformContext`: pre-LLM context transform (compaction, pruning)
- `getApiKey`: dynamic API key resolution (for OAuth tokens)

**Rust implementation:**

```
theta-agent-core/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── agent.rs           — Agent struct (state, subscribe, prompt, continue, queues)
│   ├── loop.rs             — agent loop (run_prompt, run_continue, inner+outer loops)
│   ├── tools.rs            — Tool execution (sequential, parallel, prepare, finalize)
│   ├── events.rs           — AgentEvent enum
│   ├── state.rs            — AgentState
│   ├── types.rs            — AgentTool, AgentContext, AgentLoopConfig, thinking levels
│   ├── hooks.rs            — beforeToolCall, afterToolCall, shouldStopAfterTurn
│   └── error.rs
```

**Key design:**
```rust
pub struct Agent {
    state: RwLock<AgentState>,
    listeners: Vec<Box<dyn Fn(AgentEvent) -> Pin<Box<dyn Future<Output = ()> + Send>> + Send + Sync>>,
    steering_queue: Mutex<Vec<AgentMessage>>,
    follow_up_queue: Mutex<Vec<AgentMessage>>,
    active_run: Mutex<Option<ActiveRun>>,
    config: AgentLoopConfig,
}

impl Agent {
    pub async fn prompt(&self, messages: Vec<AgentMessage>) -> Result<()>;
    pub async fn continue_(&self) -> Result<()>;
    pub fn subscribe(&self, listener: ...) -> Subscription;
    pub fn steer(&self, msg: AgentMessage);
    pub fn follow_up(&self, msg: AgentMessage);
    pub fn abort(&self);
}
```

**Dependencies:** `theta-ai`, `tokio`, `futures`, `serde`, `tracing`

**Effort estimate:** ~2500-3500 LOC

---

### 3. `theta-tui` — Terminal UI Library

**What Pi's `pi-tui` does:**
- Differential rendering (ratatui equivalent)
- Component hierarchy with focus management
- Keybinding system
- Editor component (input area)
- Overlays, selectors, confirmations
- Theme support (colors)

**Rust implementation (ratatui-based):**

```
theta-tui/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── app.rs             — TUI main loop, component tree, rendering
│   ├── components/
│   │   ├── mod.rs
│   │   ├── editor.rs      — input text editor
│   │   ├── selector.rs    — list selector
│   │   ├── dialog.rs      — confirmation/text input dialogs
│   │   ├── chat.rs        — chat message display (scrollable)
│   │   ├── tool_output.rs — tool execution output
│   │   └── status.rs      — status bar / footer
│   ├── theme.rs           — color themes
│   ├── keybinding.rs      — keybinding manager
│   └── terminal.rs        — terminal raw mode setup
```

**Dependencies:** `ratatui`, `crossterm`, `tokio`

**Effort estimate:** ~2000-3000 LOC for functional TUI

---

### 4. `theta` — Coding Agent CLI (Binary)

**What Pi's `pi-coding-agent` does:**
- CLI argument parsing (clap equivalent)
- Configuration loading (settings, auth, models)
- Session management (JSONL files, branching, compaction)
- Extension system (tools, commands, event hooks)
- Built-in tools: read, write, edit, bash, grep, find, ls
- Skills: markdown files with YAML frontmatter
- Prompt templates: parameterized instructions
- Interactive mode: TUI with editor, shortcuts, model switching
- Print/RPC modes for programmatic use
- Model resolution and scoped model cycling

```
theta/
├── Cargo.toml
├── src/
│   ├── main.rs            — entry point, CLI setup
│   ├── cli.rs             — argument parsing (clap)
│   ├── config.rs          — configuration management
│   ├── session/
│   │   ├── mod.rs
│   │   ├── manager.rs     — session create/open/fork/resume
│   │   ├── repository.rs  — JSONL read/write
│   │   └── compaction.rs  — context compaction logic
│   ├── tools/
│   │   ├── mod.rs         — tool registry, Tool trait
│   │   ├── read.rs        — file reading with sizing/truncation
│   │   ├── write.rs       — file writing
│   │   ├── edit.rs        — precise text replacement
│   │   ├── bash.rs        — shell command execution
│   │   ├── grep.rs        — regex search
│   │   ├── find.rs        — file find
│   │   └── ls.rs          — directory listing
│   ├── extensions/
│   │   ├── mod.rs         — extension trait, registry
│   │   ├── system.rs      — system prompt builder
│   │   └── events.rs      — event bus for extensions
│   ├── skills.rs          — skill loading (markdown + frontmatter)
│   ├── prompts.rs         — prompt template loading
│   ├── auth.rs            — auth storage, credential management
│   ├── models.rs          — model registry, resolution, scoping
│   └── modes/
│       ├── mod.rs
│       ├── interactive.rs — TUI interactive mode
│       ├── print.rs       — non-interactive output mode
│       └── rpc.rs         — JSON-RPC mode
```

**Built-in tools:**
| Tool   | Pi equivalent | Description |
|--------|--------------|-------------|
| read   | read         | Read file with line/byte limits |
| write  | write        | Create/overwrite file |
| edit   | edit         | Exact string replacement in file |
| bash   | bash         | Shell command execution with timeout |
| grep   | grep         | Pattern search in files |
| find   | find         | File search by name/pattern |
| ls     | ls           | Directory listing |

**Dependencies:** `theta-ai`, `theta-agent-core`, `theta-tui`, `clap`, `serde`, `serde_json`, `serde_yaml`, `tokio`, `regex`, `glob`

**Effort estimate:** ~4000-6000 LOC

---

### 5. `theta-models` — Built-in Model Catalog

Separate crate for built-in model definitions so they can be compiled into the binary without runtime loading.

```
theta-models/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── openai.rs      — GPT-4o, GPT-4.1, o3, o4-mini, etc.
│   ├── anthropic.rs   — Claude Opus/Sonnet/Haiku
│   ├── google.rs      — Gemini models
│   ├── deepseek.rs    — DeepSeek models
│   └── openrouter.rs  — OpenRouter proxy models
```

**Effort estimate:** ~500-1000 LOC

---

## Implementation Phases

### Phase 1: Foundation (Core LLM API) — ~1-1.5 weeks

1. **`theta-ai` crate**
   - [x] Type system: Message, Tool, Model, ContentBlock, Context, Usage
   - [x] Event protocol: AssistantMessageEvent enum (text delta, thinking delta, tool call delta, done, error)
   - [x] Provider trait with async stream
   - [x] OpenAI-compatible provider (covers OpenAI, DeepSeek, OpenCode):
     - [x] Streaming completions via SSE
     - [x] Model-specific compat flags: `thinking_format` (openai vs deepseek), base URL
     - [x] Reasoning/thinking content extraction from `reasoning_content` chunks
     - [x] Tool call delta streaming
   - [x] Provider registry (single entry point dispatches by model.api)
   - [x] Token counting utilities (approximate char-based + optional tiktoken-rs)
   - [x] Codex provider (ChatGPT Plus subscription auth via OPENAI_CODEX_TOKEN)
   - [x] Integration tests with real API keys (behind `integration-tests` feature flag)

2. **`theta-models` crate**
   - [x] 11 models covering all 4 providers:
     - **OpenAI** (`api.openai.com`): `gpt-5.5`, `gpt-5.5-instant`, `o4`, `o4-mini`
     - **OpenAI Codex** (`chatgpt.com/backend-api`): `gpt-5.5`, `gpt-5.5-instant`,
       `o4`, `o4-mini` — ChatGPT Plus subscription, no API key
     - **DeepSeek** (`api.deepseek.com`): `deepseek-v4-pro` (1.6T/49B active, 1M ctx),
       `deepseek-v4-flash` (284B/13B active, 1M ctx)
     - **OpenCode** (`api.opencode.ai` or user-configured): OpenAI-compatible
   - [x] Model lookup by provider+id
   - [x] Model capabilities: reasoning, images, context window, thinking level map

### Phase 2: Agent Runtime — ~2 weeks

3. **`theta-agent-core` crate**
   - [ ] Agent struct with state management
   - [ ] Agent loop: prompt + continue with nested loops
   - [ ] Tool execution: sequential + parallel
   - [ ] Event emission: agent_start, agent_end, turn_start, turn_end, message_*
   - [ ] Steer/followUp queue system
   - [ ] Lifecycle hooks (beforeToolCall, afterToolCall, etc.)
   - [ ] Abort signal handling
   - [ ] Unit tests with mock LLM provider

### Phase 3: CLI + Tools — ~2 weeks

4. **`theta` binary**
   - [ ] CLI argument parsing with clap
   - [ ] Configuration: settings, auth, model defaults
   - [ ] Auth storage:
     - [ ] `~/.theta/auth.json` — provider tokens with expiry (codex, future providers)
     - [ ] Env var fallback: `OPENAI_CODEX_TOKEN`, `OPENAI_API_KEY`, etc.
   - [ ] `/login` command:
     - [ ] Opens browser to provider auth page (e.g. chatgpt.com for codex)
     - [ ] Runs local HTTP callback server to capture OAuth/session token
     - [ ] Stores token in `~/.theta/auth.json` with provider + expiry
     - [ ] Works for codex (ChatGPT Plus), future: github-copilot, anthropic OAuth
   - [ ] Model resolution and scoped models
   - [ ] Session management: create, open, fork, resume, continue
   - [ ] JSONL session format (Pi-compatible for interoperability)
   - [ ] Built-in tools: read, write, edit, bash, grep, find, ls
   - [ ] Tool output truncation and formatting
   - [ ] Print mode (non-interactive)
   - [ ] System prompt construction (project context, skills, tools)

### Phase 4: TUI — ~2-3 weeks

5. **`theta-tui` crate**
   - [ ] Terminal setup with crossterm raw mode
   - [ ] Component system with focus management
   - [ ] Editor component (multiline, autocomplete-ready)
   - [ ] Chat message display with scrolling
   - [ ] Tool execution display
   - [ ] Selectors and dialogs
   - [ ] Theme system (colors)
   - [ ] Keybinding manager

6. **Interactive mode integration**
   - [ ] TUI loop with agent event subscription
   - [ ] Streaming assistant message display
   - [ ] Tool execution progress
   - [ ] Model switching (Ctrl+P)
   - [ ] Session tree navigation
   - [ ] Slash commands (/login, /model, /session, etc.)

### Phase 5: Extensibility — ~2 weeks

7. **Skills system**
   - [ ] Markdown files with YAML frontmatter
   - [ ] Auto-discovery from ~/.theta/skills and project .theta/skills
   - [ ] Skill injection into system prompt
   - [ ] Compatible with Pi's SKILL.md format

8. **Prompt templates**
   - [ ] Template files with variable substitution
   - [ ] Auto-discovered from .theta/prompts

8. **Extension system (Trait-based)**
   - [ ] Extension trait: register tools, commands, event handlers
   - [ ] Extension registry loaded at startup
   - [ ] Custom tool registration API
   - [ ] Extension UI context (dialogs, selectors, status)
   - [ ] Note: No dynamic loading (WASM) in MVP — extensions are compiled-in via Cargo features

### Phase 6: Polish — ~2 weeks

9. **Polish + DX**
   - [ ] Compaction: automatic context window management
   - [ ] Branch summaries for session tree
   - [ ] RPC mode (JSON-RPC over stdin/stdout)
   - [ ] Image support (read images, send to vision models)
   - [ ] OAuth login support
   - [ ] Provider retry logic
   - [ ] Theme switching in TUI
   - [ ] Documentation (README, CONTRIBUTING)
   - [ ] Install script / release workflow

---

## Key Design Decisions

### 1. Async Runtime: Tokio
Pi is fully async. Rust -> `tokio`. All LLM calls, tool execution, and TUI run on tokio.

### 2. Schema Validation: serde_json::Value
Pi uses TypeBox (TypeScript) for tool parameter schemas. In Rust, we use raw `serde_json::Value` for JSON Schema, validated at runtime. This avoids compile-time schema generation for user-defined tools. We can later add `schemars` for built-in tool schemas.

### 3. Tool Execution: Trait-based
```rust
#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn label(&self) -> &str;
    fn parameters(&self) -> serde_json::Value;
    fn execution_mode(&self) -> ToolExecutionMode { ToolExecutionMode::Parallel }
    async fn execute(&self, tool_call_id: &str, args: serde_json::Value,
                     signal: Option<CancellationToken>,
                     on_update: Option<Box<dyn Fn(ToolUpdate) + Send>>)
        -> Result<ToolResult, ThetaError>;
}
```

### 4. JSONL Session Format: Pi-Compatible
Theta should use the same JSONL format as Pi for sessions. This enables:
- Switching between Pi and Theta on the same project
- Using Pi sessions in Theta and vice versa
- Interoperability with Pi's ecosystem

Session entry format (Pi-compatible):
```jsonl
{"type":"model_change","provider":"anthropic","modelId":"claude-sonnet-4-5","timestamp":1700000000000}
{"type":"thinking_level_change","level":"high","timestamp":1700000000001}
{"type":"user","content":[{"type":"text","text":"Hello"}],"timestamp":1700000000002}
{"type":"assistant","content":[...],"api":"anthropic-messages","provider":"anthropic","model":"claude-sonnet-4-5-20250929","usage":{...},"stopReason":"toolUse","timestamp":1700000000003}
{"type":"toolResult","toolCallId":"tool_001","toolName":"read","content":[...],"isError":false,"timestamp":1700000000004}
```

### 5. Extension Model: Traits over Dynamic Loading
Pi uses TypeScript module loading for extensions. In Rust:
- **MVP**: Extension trait implemented by compiled-in code (via Cargo features or workspace members). Users build their own theta binary with their extensions.
- **Future**: WASM component model for dynamic extension loading, matching Pi's "install and load" UX.
- **Future**: Embedded Lua/RhAI scripting for lighter-weight extensions.

This follows the philosophy of "adapt theta to your workflows" — users can fork and add their own compiled extensions, which is the Rust way.

### 6. System Prompt: Pi-Compatible Format
The system prompt should mirror Pi's structure:
- Project context files (AGENTS.md, CLAUDE.md, etc.)
- Available tools with descriptions
- Skills loaded from disk
- Guidelines
- Running context (OS, shell, date, directory)

### 7. Compaction
Pi compacts context by summarizing old messages. Theta should implement:
- Token counting (approximate, 4 chars/token heuristic initially)
- Cut point detection
- Summary generation using the LLM
- Branch summary for session tree navigation

---

## Cargo Workspace Structure

```
theta/                           # workspace root
├── Cargo.toml                   # workspace manifest
├── PLAN.md                      # this file
├── crates/
│   ├── theta-ai/               # unified LLM API
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── theta-agent-core/       # agent runtime
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── theta-tui/              # terminal UI
│   │   ├── Cargo.toml
│   │   └── src/
│   ├── theta-models/           # built-in model catalog
│   │   ├── Cargo.toml
│   │   └── src/
│   └── theta/                  # main binary
│       ├── Cargo.toml
│       └── src/
├── examples/                    # example extensions and integrations
├── tests/                       # integration tests
└── README.md
```

---

## Technology Choices

| Concern | Rust Crate | Pi Equivalent |
|---------|-----------|---------------|
| Async runtime | `tokio` | Node.js event loop |
| HTTP client | `reqwest` | native fetch |
| SSE streaming | `eventsource-stream` or hand-rolled | EventSource |
| Serialization | `serde` + `serde_json` | JSON |
| CLI parsing | `clap` | custom parser |
| TUI | `ratatui` + `crossterm` | custom TUI lib |
| Async streams | `futures::Stream` | AsyncIterator |
| Regex | `regex` | native RegExp |
| File glob | `glob` or `ignore` | fast-glob |
| YAML | `serde_yaml` | yaml |
| Token counting | `tiktoken-rs` (optional) | tiktoken |
| Diff | `similar` | diff |
| Terminal size | `crossterm::terminal::size` | terminal-size |
| Process spawning | `tokio::process::Command` | child_process |
| Key-value config | `toml` or `serde_json` | JSON files |

---

## Non-Goals for MVP

These are intentionally deferred to avoid scope creep:
- **Anthropic / Google / Mistral providers** — start with OpenAI-compatible only
- **vLLM/serving infrastructure** — pi's vLLM pod management is out of scope
- **Slack bot** — pi-chat equivalent
- **Web UI** — pi-web-ui equivalent (ratatui is terminal-only)
- **Browser automation tools** — puppeteer integration
- **Windows-specific quirks** — focus on macOS/Linux for MVP
- **GitHub Actions integration** — CI/CD tooling
- **Session sharing to HuggingFace** — analytics
- **Dynamic WASM extensions** — traits only initially
- **Google Vertex / AWS Bedrock** — focus on direct API providers first

---

## Total Effort Estimate

| Phase | Weeks | Core Deliverable |
|-------|-------|-----------------|
| 1. Foundation | 1-1.5 | theta-ai + theta-models |
| 2. Agent Runtime | 2 | theta-agent-core |
| 3. CLI + Tools | 2 | theta binary with all built-in tools |
| 4. TUI | 2-3 | theta-tui + interactive mode |
| 5. Extensibility | 2 | Skills, templates, extension traits |
| 6. Polish | 2 | Compaction, docs, releases |
| **Total** | **10-11 weeks** | Full coding agent |

**Rough LOC estimate:** ~15,000-20,000 lines of Rust across all crates.

---

## Immediate Next Steps

1. Set up Cargo workspace with the 5 crates
2. Implement `theta-ai` types and OpenAI provider
3. Implement `theta-ai` Anthropic provider
4. Add `theta-models` with 10-20 models
5. Build `theta-agent-core` with agent loop
6. Create `theta` binary with read/bash tools
7. Iterate from there

The implementation should start with Phase 1 immediately — I can begin scaffolding the workspace and implementing `theta-ai`.
