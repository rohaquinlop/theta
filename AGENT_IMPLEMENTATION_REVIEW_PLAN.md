# Theta Agent Implementation Review & Parity Plan (vs Pi)

## 1) Goal

Produce a complete, evidence-based review of Theta’s current agent behavior against Pi, identify root causes for execution gaps (e.g., planning/confirmation loops, missing follow-through, tool-call handling issues), and implement fixes required for a **fully functional minimal coding agent harness**.

This plan is designed for a fresh session and should be executed end-to-end.

---

## 2) Scope

### In Scope
- End-to-end agent loop behavior (prompt -> stream -> tool calls -> tool execution -> continuation -> final response).
- Skill invocation behavior (`/skill:*`) and execution continuity.
- System prompt construction and instruction quality.
- Provider stream parsing (especially Codex/OpenAI-compatible tool-call events).
- TUI ↔ agent bridge event handling and message delivery.
- Reliability guards (retry, abort/steer/follow-up interactions, error visibility).
- UX signals for state/intent (skills, tool activity, errors, idle transitions).

### Out of Scope
- New providers outside existing architecture.
- Non-essential product features unrelated to execution correctness.

---

## 3) Reference Baseline (Pi)

Primary comparison target:
- `/Users/rhafid/.bun/install/global/node_modules/@earendil-works`

Focus packages/files:
- `pi-coding-agent/dist/core/system-prompt.js`
- `pi-agent-core` runtime loop & tool orchestration
- `pi-ai` provider streaming/tool-call parsing
- Any extension hooks related to execution continuity

Objective: compare behavior contracts, not copy implementation blindly.

---

## 4) Expected Output of Review

1. **Findings Report**: prioritized list of implementation gaps with proof.
2. **Fix Plan**: concrete code changes mapped to each gap.
3. **Validation Matrix**: tests + manual scenarios proving behavior parity targets.
4. **Residual Risk List**: known limitations explicitly documented.

---

## 5) Review Methodology

## Phase A — Reproduce & Capture Failures

Create deterministic repro scripts for known issues:
1. `/skill:git-commit` -> confirmation loop -> user replies `2` -> no action.
2. “Implement X” -> model announces intent but does not execute tools.
3. Tool-call-only assistant turn returns idle without tool execution.

Artifacts to capture:
- Session transcript entries (user/assistant/toolResult/model changes).
- TUI events timeline (TextDelta, ToolStart/End, AgentEnd, Error).
- Provider raw stream events (tool-call deltas, done/stop reasons).

Deliverable: `docs/review/repro-cases.md`.

---

## Phase B — Behavior Contract Diff (Theta vs Pi)

For each layer, define **contract** and compare with Pi:

1. **System Prompt Contract**
   - Tool invocation style (function-calling vs XML).
   - Execution continuity language.
   - Ambiguity handling defaults.

2. **Loop Contract**
   - When to execute tools.
   - Stop conditions.
   - Handling missing/incorrect stop reason with tool calls present.

3. **Provider Parsing Contract**
   - How tool calls are reconstructed from streamed deltas.
   - Finish reason mapping.
   - Multi-call and partial-arguments edge cases.

4. **TUI/Bridge Contract**
   - Message send semantics.
   - Streaming state transitions.
   - Error/idle visibility and no-silent-failure guarantees.

Deliverable: `docs/review/theta-vs-pi-contract-diff.md`.

---

## Phase C — Root Cause Analysis

For each repro, produce:
- Symptom
- Trigger
- Failing contract
- Exact code location
- Why current logic fails
- Why Pi behavior avoids it

Deliverable: `docs/review/root-cause-analysis.md`.

---

## Phase D — Fix Implementation Plan (Prioritized)

### Priority 0 — Execution Correctness

1. **Tool execution trigger hardening**
   - In agent loop, execute tool calls if parsed calls exist, even when stop reason is absent/misreported.
   - Preserve existing behavior for normal stop reasons.

2. **Provider parser hardening (Codex/OpenAI compat)**
   - Ensure tool calls are emitted/assembled reliably across streamed fragments.
   - Add fallback mapping for provider-specific finish semantics.

3. **No silent drop on tool-call turns**
   - Emit explicit error/info event when assistant turn contains unresolved tool-call state.

### Priority 1 — Skill Execution Behavior

4. `/skill:name` no-arg should be action-first by default.
5. Add stronger “execute-now” instruction injection with anti-acknowledgement phrasing.
6. Ensure follow-up reply after skill prompt continues execution chain reliably.

### Priority 2 — UX & Debbugability

7. Add visible diagnostic breadcrumbs in UI for: tool-call detected, tool-call parsed count, tool round transitions.
8. Improve idle/error transitions to prevent “idle but nothing happened” ambiguity.

Deliverable: `docs/review/fix-plan.md`.

---

## 6) Test Strategy (Must Pass Before Done)

## Unit Tests
- Provider parser tests for streamed tool-call assembly (single/multi call, fragmented args, missing finish reason).
- Agent loop tests for fallback tool execution when stop reason not `ToolUse` but tool calls exist.
- Skill expansion tests for no-arg and arg flows.

## Integration Tests
- Mock provider scenario: confirmation reply should trigger tool execution path, not stall.
- Mock provider scenario: tool-call-only turn with inconsistent stop metadata still executes tools.

## Manual Scenarios
1. `/skill:git-commit` + minimal reply (`2`) proceeds to git inspection commands.
2. “Implement X” triggers real tools immediately.
3. On parser mismatch, user sees explicit error/diagnostic rather than silent idle.

Validation command:
```bash
cargo fmt && cargo clippy -- -D warnings && cargo test
```

Deliverable: `docs/review/validation-matrix.md`.

---

## 7) Concrete Work Breakdown (Fresh Session Checklist)

1. Read current Theta files fully:
   - `crates/theta-agent-core/src/loop_mod.rs`
   - `crates/theta-agent-core/src/tools.rs`
   - `crates/theta/src/interactive.rs`
   - `crates/theta/src/system_prompt.rs`
   - `crates/theta-ai/src/providers/openai_codex.rs`
   - `crates/theta-ai/src/providers/openai_compat.rs`

2. Read Pi counterparts under:
   - `/Users/rhafid/.bun/install/global/node_modules/@earendil-works`

3. Build contract diff docs.
4. Reproduce failures with instrumentation.
5. Implement P0 fixes first, then P1/P2.
6. Add/adjust tests.
7. Run full validation.
8. Summarize residual risks.

---

## 8) Acceptance Criteria

Implementation is acceptable when all are true:

- No known repro case ends in silent idle after user follow-up.
- Tool-call turns are executed robustly even under provider finish-reason inconsistencies.
- `/skill:*` feels action-first and does not default to acknowledgement-only behavior.
- Failures are explicit and diagnosable in UI.
- Full fmt/clippy/tests pass.
- Findings + fixes + evidence are documented.

---

## 9) Suggested Deliverable File Set

- `AGENT_IMPLEMENTATION_REVIEW_PLAN.md` (this file)
- `docs/review/repro-cases.md`
- `docs/review/theta-vs-pi-contract-diff.md`
- `docs/review/root-cause-analysis.md`
- `docs/review/fix-plan.md`
- `docs/review/validation-matrix.md`
- `docs/review/final-summary.md`

---

## 10) Notes for Next Session Agent

- Treat this as an implementation + verification mission, not just analysis.
- Prefer behavioral parity in contracts over code-level mimicry.
- Do not stop after proposing; apply fixes and validate.
- If a user-facing stall is possible, add explicit event/UI signaling.
