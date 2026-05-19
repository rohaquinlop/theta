//! Agent lifecycle events emitted during execution.

use theta_ai::Message;

use crate::types::ToolResult;

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
}
