//! Agent lifecycle events emitted during execution.

use theta_ai::Message;

use crate::types::{SafetyDecisionKind, ToolResult, TurnEndReason};

/// Events emitted by the agent during execution.
/// Consumers (TUI, RPC, etc.) subscribe to these.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// A new agent run has started (prompt or continue).
    AgentStart,

    /// The agent run has completed.
    AgentEnd { aborted: bool },

    /// A turn (one LLM call + tool execution) is beginning.
    TurnStart { turn_index: u32 },

    /// A turn has completed.
    TurnEnd { turn_index: u32 },
    /// Canonical terminal reason for a turn.
    TurnTerminated {
        reason: TurnEndReason,
        details: String,
        turn: u32,
        round: u32,
    },

    /// An assistant message is beginning to stream.
    MessageStart,

    /// A streamed text delta from the assistant.
    TextDelta { text: String },

    /// A streamed thinking/reasoning delta from the assistant.
    ThinkingDelta { thinking: String },

    /// A tool call has started streaming.
    ToolCallStart { id: String, name: String },

    /// A streamed tool call arguments delta.
    ToolCallDelta { id: String, arguments: String },

    /// A streamed tool call has completed (all args received).
    ToolCallEnd { id: String },

    /// The assistant message is complete.
    MessageEnd { message: Message },

    /// A tool is about to be executed.
    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
    },

    /// Progress update during tool execution.
    ToolExecutionProgress {
        tool_call_id: String,
        output: String,
    },

    /// A tool execution completed.
    ToolExecutionEnd { result: ToolResult },

    /// An error occurred during execution.
    Error { message: String },

    /// Context was compacted (old messages trimmed to fit context window).
    ContextCompacted {
        /// Number of user/assistant/tool-result messages trimmed.
        trimmed_count: u32,
        /// Tokens before compaction.
        tokens_before: u32,
        /// Tokens after compaction.
        tokens_after: u32,
    },

    /// Retrying a failed provider call.
    Retrying {
        /// Current attempt number (1-based).
        attempt: u32,
        /// Delay in milliseconds before retry.
        delay_ms: u64,
    },
    /// Replay transcript was sanitized before provider call.
    ReplaySanitized {
        dropped_assistant_messages: u32,
        synthesized_tool_results: u32,
        normalized_tool_call_ids: u32,
        deduped_tool_results: u32,
    },
    /// Structured turn decision emitted by the runtime loop.
    TurnDecision {
        reason: TurnDecisionReason,
        details: String,
        turn: u32,
        round: u32,
    },
    /// Command/tool safety policy decision.
    SafetyDecision {
        decision: SafetyDecisionKind,
        tool_name: String,
        details: String,
    },
    /// Tool watchdog detected no progress for configured interval.
    ToolWatchdogWarning {
        tool_call_id: String,
        tool_name: String,
        stalled_ms: u64,
    },
    /// Circuit breaker prevented provider/model call.
    ProviderCircuitOpen { key: String, retry_in_ms: u64 },
    /// Fallback model/provider selected after failure.
    ProviderFallback {
        from_model: String,
        to_model: String,
        reason: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnDecisionReason {
    NoopRetry,
    BlockedNoop,
    MaxRounds,
    AnalyzeOnlyRejectedTool,
}
