//! Core types for LLM messages, tools, models, and context.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_call")]
    ToolCall {
        id: String,
        name: String,
        arguments: serde_json::Value,
    },
    #[serde(rename = "image")]
    Image { media_type: String, data: String },
    /// Thinking / reasoning content (o-series, DeepSeek R1-style).
    #[serde(rename = "thinking")]
    Thinking {
        thinking: String,
        signature: Option<String>,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_call_id: String,
        tool_name: String,
        content: Vec<ContentBlock>,
        details: Option<serde_json::Value>,
        is_error: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Message {
    #[serde(rename = "user")]
    User {
        content: Vec<ContentBlock>,
        timestamp: u64,
    },
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
    #[serde(rename = "model_change")]
    ModelChange {
        provider: Option<Provider>,
        #[serde(alias = "modelId")]
        model_id: Option<String>,
        timestamp: u64,
    },
    #[serde(rename = "thinking_level_change")]
    ThinkingLevelChange {
        level: ThinkingLevel,
        timestamp: u64,
    },
}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    #[serde(rename = "stop")]
    Stop,
    #[serde(rename = "length")]
    Length,
    #[serde(rename = "toolUse")]
    ToolUse,
    #[serde(rename = "error")]
    Error,
    #[serde(rename = "aborted")]
    Aborted,
}

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
    #[serde(rename = "max")]
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "image")]
    Image,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Usage {
    #[serde(alias = "inputTokens")]
    pub input_tokens: u32,
    #[serde(alias = "outputTokens")]
    pub output_tokens: u32,
    #[serde(alias = "cacheWriteTokens", default)]
    pub cache_write_tokens: u32,
    #[serde(alias = "cacheReadTokens", default)]
    pub cache_read_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StreamOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stop: Option<Vec<String>>,
    pub thinking_level: Option<ThinkingLevel>,
    pub include_usage: bool,
    pub json_mode: bool,
    pub seed: Option<u64>,
    /// Service tier for Codex: "flex" (half cost), "default", or
    /// "priority" (higher throughput). Only applies to Codex provider.
    pub service_tier: Option<String>,
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SimpleStreamOptions {
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub system: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Context {
    pub system: Option<Vec<ContentBlock>>,
    pub messages: Vec<Message>,
    pub tools: Vec<Tool>,
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

// 4 chars/token approximation. Use tiktoken-rs for precise counting.
pub fn approximate_token_count(text: &str) -> u32 {
    (text.chars().count() as f64 / 4.0).ceil() as u32
}

impl Message {
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
    pub fn text(text: impl Into<String>) -> Self {
        ContentBlock::Text { text: text.into() }
    }

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

    pub fn thinking(thinking: impl Into<String>) -> Self {
        ContentBlock::Thinking {
            thinking: thinking.into(),
            signature: None,
        }
    }
}
