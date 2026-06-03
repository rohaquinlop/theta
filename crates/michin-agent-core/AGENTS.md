# michin-agent-core — Agent Rules

> Rules for working on the michin-agent-core crate: agent runtime.

## Crate Purpose

Agent runtime: `Agent` struct, nested loop, tool execution, compaction, command safety policy, retry, events, hooks. All agent behavior logic lives here.

## Public API

- `Agent` — prompt, continue, steer, follow_up, subscribe, hooks.
- `AgentEvent` enum — includes `TurnTerminated`, `TurnModeResolved`, `SafetyDecision`, `ToolWatchdogWarning`, `ProviderCircuitOpen`, `ProviderFallback`, `CompactionPaused`, `AgentEnd`.
- `Hooks` trait — `beforeToolCall`, `afterToolCall`, `shouldStopAfterTurn`, `prepareNextTurn`, `tui_status_lines()`, `tui_status_rows()`.
- `AgentTool` trait — `name()`, `description()`, `label()`, `parameters()`, `execution_mode()`, `execute()`.

## Key Files

| File                                 | Purpose                                         |
| ------------------------------------ | ----------------------------------------------- |
| `src/lib.rs`                         | Public API exports                              |
| `src/agent.rs`                       | `Agent` struct: prompt, continue, steer, hooks  |
| `src/loop_mod.rs`                    | Nested outer/inner loop, turn enforcement       |
| `src/compact.rs`                     | Truncation compaction + inline summary          |
| `src/command_policy.rs`              | Safety policy engine                            |
| `src/types.rs`                       | All config types, traits, enums                 |
| `src/events.rs`                      | AgentEvent enum                                 |
| `src/hooks.rs`                       | Hooks trait                                     |
| `src/state.rs`                       | Agent mutable state (RwLock-protected)          |
| `src/tools.rs`                       | AgentTool trait definition                      |
| `src/error.rs`                       | AgentError enum                                 |

## Agent Loop Design

**Nested loop:** outer loop (follow-up turns) → inner loop (LLM call → stream → tools → repeat).

**Turn modes:** `Execute`, `Inspect`, `AnalyzeOnly`, `PlanOnly`, `Clarify`.

**Turn enforcement (Pi-style):** intent flags with bounded one-shot retry per enforcement path.

**Loop guard:** `max_same_tool_call_repeats` (default 6) — aborts inner loop if same tool+args repeats.

**Tool watchdog:** warns on stall, hard timeout on tool execution.

**Provider circuit breaker:** opens after N consecutive transient failures, half-open after cooldown.

**Provider fallback chain:** on failure, fall through configured model IDs.

## Command Safety Policy

Centralized `evaluate_tool_call(mode, tool_call, strict)` engine. Classifies bash commands into `AuthorizationClass`: `FileMutation`, `VcsMutation`, `Commit`, `DependencyMutation`. Detects dangerous operations (git push/merge/rebase/reset, cargo add, npm install, etc.).

## Compaction

Prefix-preserving design: keeps the earliest messages (system prompt + prefix) byte-stable, summarizes the middle region, and keeps the recent tail verbatim. This preserves DeepSeek/MiMo prefix cache across compaction.

Auto-pause: when consecutive compactions reach `auto_pause_threshold` (default 2) and the kept tail alone exceeds the context trigger, auto-compaction pauses to prevent cache-cratering loops. Pauses resume once a turn fits naturally.

## Retry

Exponential backoff for 429, 5xx, connection/timeout. Non-retryable (4xx non-429) fail immediately. Mid-stream connection breaks (e.g. DeepSeek during thinking) are detected and retried up to `MAX_STREAM_BREAKS` (2) times before falling through to the fallback chain.

## Error Handling

- Use `thiserror` for library errors.
- No panics. Return `Result`.
- No `unwrap()` — use `?` or explicit error handling.
- `AgentEnd` event always emitted (even on error).
- Comments follow root `AGENTS.md` Comment Style rules.

## Testing

- Policy scenario matrix tests cover circuit breaker, watchdog, command policy, fallback chain, run reports.
- Critical loop regression tests cover action turns, inspection turns, commit-op turns, blockers, stop-reason handling.
