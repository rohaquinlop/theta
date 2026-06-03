use michin_ai::Message;

use crate::types::{SafetyDecisionKind, ToolResult, TurnEndReason};

#[derive(Debug, Clone)]
pub enum AgentEvent {
    AgentStart,
    AgentEnd {
        aborted: bool,
    },
    TurnStart {
        turn_index: u32,
    },
    TurnEnd {
        turn_index: u32,
    },
    TurnTerminated {
        reason: TurnEndReason,
        details: String,
        turn: u32,
        round: u32,
    },

    MessageStart,
    TextDelta {
        text: String,
    },
    ThinkingDelta {
        thinking: String,
    },
    ThinkingStart,
    ThinkingEnd,
    ToolCallStart {
        id: String,
        name: String,
    },
    ToolCallDelta {
        id: String,
        arguments: String,
    },
    ToolCallEnd {
        id: String,
    },
    MessageEnd {
        message: Message,
    },

    ToolExecutionStart {
        tool_call_id: String,
        tool_name: String,
        arguments: Option<serde_json::Value>,
    },

    ToolExecutionProgress {
        tool_call_id: String,
        output: String,
    },

    ToolExecutionEnd {
        result: ToolResult,
    },

    Error {
        message: String,
    },

    ContextCompacted {
        /// Number of user/assistant/tool-result messages trimmed.
        trimmed_count: u32,
        /// Tokens before compaction.
        tokens_before: u32,
        /// Tokens after compaction.
        tokens_after: u32,
    },
    /// Auto-compaction paused: the kept tail alone exceeds the context trigger,
    /// so compacting every turn would crater the prefix cache.
    CompactionPaused {
        context_window: u32,
        reserve_tokens: u32,
    },

    Retrying {
        attempt: u32,
        delay_ms: u64,
    },
    ReplaySanitized {
        dropped_assistant_messages: u32,
        synthesized_tool_results: u32,
        normalized_tool_call_ids: u32,
        deduped_tool_results: u32,
    },
    TurnDecision {
        reason: TurnDecisionReason,
        details: String,
        turn: u32,
        round: u32,
    },
    SafetyDecision {
        decision: SafetyDecisionKind,
        tool_name: String,
        details: String,
    },
    ToolWatchdogWarning {
        tool_call_id: String,
        tool_name: String,
        stalled_ms: u64,
    },
    ProviderCircuitOpen {
        key: String,
        retry_in_ms: u64,
    },
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
