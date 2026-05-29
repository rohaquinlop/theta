//! Core types for LLM messages, tools, models, and context.

use serde::{Deserialize, Serialize};

/// A block of content within a message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Plain text content.
    #[serde(rename = "text")]
    Text { text: String },
    /// A tool call the model wants to execute.
    #[serde(rename = "tool_call")]
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    /// An image (base64-encoded).
    #[serde(rename = "image")]
    Image { media_type: String, data: String },
    /// Thinking / reasoning content (o-series, DeepSeek R1-style).
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    /// A tool result (both successful and error).
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: Vec<ContentBlock>,
        details: Option<serde_json::Value>,
        is_error: bool,
    },
}

/// A message in the conversation transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Message {
    /// A user message.
    #[serde(rename = "user")]
    User {
        content: Vec<ContentBlock>,
        timestamp: u64,
    },
    /// An assistant (model) response.
    #[serde(rename = "assistant")]
    Assistant {
        content: Vec<ContentBlock>,
        api: Option<Api>,
        provider: Option<Provider>,
        model: Option<String>,
        usage: Option<Usage>,
        #[serde(alias = "stopReason")]
        stop_reason: Option<StopReason>,
        #[serde(alias = "errorMessage")]
        error_message: Option<String>,
        timestamp: u64,
    },
    /// A tool execution result.
    #[serde(rename = "tool_result", alias = "toolResult")]
    ToolResult {
        #[serde(alias = "toolCallId")]
        tool_call_id: String,
        #[serde(alias = "toolName")]
        tool_name: String,
        content: Vec<ContentBlock>,
        details: Option<serde_json::Value>,
        #[serde(alias = "isError")]
        is_error: bool,
        timestamp: u64,
    },
    /// A model change event (user switched model mid-conversation).
    #[serde(rename = "model_change")]
    ModelChange {
        provider: Option<Provider>,
        #[serde(alias = "modelId")]
        model_id: Option<String>,
        timestamp: u64,
    },
    /// A thinking level change event.
    #[serde(rename = "thinking_level_change")]
    ThinkingLevelChange {
        level: ThinkingLevel,
        timestamp: u64,
    },
}

/// Recognized LLM APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Api {
    #[serde(rename = "openai-completions")]
    OpenAiCompletions,
    #[serde(rename = "openai-responses")]
    OpenAiResponses,
    /// OpenAI Codex — ChatGPT Plus subscription token auth.
    /// Calls go to chatgpt.com/backend-api instead of api.openai.com.
    #[serde(rename = "openai-codex-responses")]
    OpenAiCodexResponses,
}

/// Recognized LLM providers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Provider {
    #[serde(rename = "openai")]
    OpenAI,
    /// OpenAI Codex — ChatGPT Plus subscription (no API key needed).
    #[serde(rename = "openai-codex")]
    OpenAiCodex,
    #[serde(rename = "deepseek")]
    DeepSeek,
    #[serde(rename = "opencode")]
    OpenCode,
    #[serde(rename = "opencode-go")]
    OpenCodeGo,
    /// Xiaomi MiMo — OpenAI-compatible provider.
    #[serde(rename = "xiaomi")]
    XiaomiMiMo,
}

/// Why the model stopped generating.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// Natural completion or stop sequence.
    #[serde(rename = "stop")]
    Stop,
    /// Hit max token limit.
    #[serde(rename = "length")]
    Length,
    /// Model called one or more tools.
    #[serde(rename = "toolUse")]
    ToolUse,
    /// An error occurred.
    #[serde(rename = "error")]
    Error,
    /// User aborted the request.
    #[serde(rename = "aborted")]
    Aborted,
}

/// Thinking / reasoning level for models that support it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThinkingLevel {
    #[serde(rename = "off")]
    Off,
    #[serde(rename = "minimal")]
    Minimal,
    #[serde(rename = "low")]
    Low,
    #[serde(rename = "medium")]
    Medium,
    #[serde(rename = "high")]
    High,
    #[serde(rename = "xhigh")]
    XHigh,
}

/// Modality / input type the model supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "image")]
    Image,
}

/// Token usage and cost.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    /// Input tokens consumed.
    #[serde(alias = "inputTokens")]
    pub input_tokens: u32,
    /// Output tokens generated.
    #[serde(alias = "outputTokens")]
    pub output_tokens: u32,
    /// Cache creation write tokens.
    #[serde(alias = "cacheWriteTokens", default)]
    pub cache_write_tokens: u32,
    /// Cache read tokens (prompt caching hit).
    #[serde(alias = "cacheReadTokens", default)]
    pub cache_read_tokens: u32,
}

/// A tool definition as sent to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    /// Unique tool name.
    pub name: String,
    /// Human-readable description for the model.
    pub description: String,
    /// JSON Schema for the tool's parameters.
    pub parameters: serde_json::Value,
}

/// Stream options passed to the provider.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StreamOptions {
    /// Maximum output tokens.
    pub max_tokens: Option<u32>,
    /// Temperature (0.0 - 2.0).
    pub temperature: Option<f64>,
    /// Top-p nucleus sampling.
    pub top_p: Option<f64>,
    /// Stop sequences.
    pub stop: Option<Vec<String>>,
    /// Thinking / reasoning level.
    pub thinking_level: Option<ThinkingLevel>,
    /// Whether to include usage info in stream.
    pub include_usage: bool,
    /// Whether the model should output in JSON mode.
    pub json_mode: bool,
    /// Seed for reproducible output.
    pub seed: Option<u64>,
    /// Service tier for Codex: "flex" (half cost), "default", or
    /// "priority" (higher throughput). Only applies to Codex provider.
    pub service_tier: Option<String>,
    /// Transport request timeout in milliseconds.
    pub timeout_ms: Option<u64>,
}

/// Simplified stream options for non-tool-calling requests.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimpleStreamOptions {
    /// Maximum output tokens.
    pub max_tokens: Option<u32>,
    /// Temperature.
    pub temperature: Option<f64>,
    /// System prompt override.
    pub system: Option<String>,
}

/// The full context sent to an LLM for a turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    /// System message (instructions).
    pub system: Option<Vec<ContentBlock>>,
    /// Conversation messages so far.
    pub messages: Vec<Message>,
    /// Available tools.
    pub tools: Vec<Tool>,
    /// Thinking level override.
    pub thinking_level: Option<ThinkingLevel>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            system: None,
            messages: Vec::new(),
            tools: Vec::new(),
            thinking_level: None,
        }
    }
}

impl Default for Context {
    fn default() -> Self {
        Self::new()
    }
}

/// Approximation: 4 chars per token for English text.
/// Use tiktoken-rs for precise counting when available.
pub fn approximate_token_count(text: &str) -> u32 {
    (text.chars().count() as f64 / 4.0).ceil() as u32
}

impl Message {
    /// Approximate token count of this message.
    pub fn token_count(&self) -> u32 {
        match self {
            Message::User { content, .. }
            | Message::Assistant { content, .. }
            | Message::ToolResult { content, .. } => content
                .iter()
                .map(|b| match b {
                    ContentBlock::Text { text } => approximate_token_count(text),
                    ContentBlock::Thinking { thinking, .. } => approximate_token_count(thinking),
                    ContentBlock::ToolResult { content, .. } => content
                        .iter()
                        .map(|b| match b {
                            ContentBlock::Text { text } => approximate_token_count(text),
                            _ => 1,
                        })
                        .sum(),
                    _ => 1,
                })
                .sum(),
            Message::ModelChange { .. } | Message::ThinkingLevelChange { .. } => 0,
        }
    }

    /// Get the timestamp of this message.
    pub fn timestamp(&self) -> u64 {
        match self {
            Message::User { timestamp, .. } => *timestamp,
            Message::Assistant { timestamp, .. } => *timestamp,
            Message::ToolResult { timestamp, .. } => *timestamp,
            Message::ModelChange { timestamp, .. } => *timestamp,
            Message::ThinkingLevelChange { timestamp, .. } => *timestamp,
        }
    }
}

impl ContentBlock {
    /// Create a text content block.
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text { text: text.into() }
    }

    /// Create a tool call content block.
    pub fn tool_call(
        id: impl Into<String>,
        name: impl Into<String>,
        arguments: serde_json::Value,
    ) -> Self {
        ContentBlock::ToolCall {
            id: id.into(),
            name: name.into(),
            arguments,
        }
    }

    /// Create a thinking content block.
    pub fn thinking(thinking: impl Into<String>) -> Self {
        ContentBlock::Thinking {
            thinking: thinking.into(),
            signature: None,
        }
    }
}
