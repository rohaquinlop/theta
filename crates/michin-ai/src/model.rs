//! Model definitions: struct, capabilities, and catalog traits.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use super::types::*;

/// How the provider encodes thinking/reasoning parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThinkingFormat {
    /// OpenAI-style: `reasoning_effort` field in the request.
    #[serde(rename = "openai")]
    OpenAI,
    /// DeepSeek-style: `thinking: { type: "enabled", reasoning_effort: ... }` block.
    #[serde(rename = "deepseek")]
    DeepSeek,
    /// Xiaomi MiMo-style: `thinking: { type: "enabled" }` — binary on/off only.
    #[serde(rename = "xiaomi")]
    XiaomiMiMo,
}

/// Which field to use for max tokens in the API request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaxTokensField {
    /// OpenAI: `max_completion_tokens`.
    #[serde(rename = "max_completion_tokens")]
    MaxCompletionTokens,
    /// Standard: `max_tokens`.
    #[serde(rename = "max_tokens")]
    MaxTokens,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCompat {
    #[serde(default)]
    pub thinking_format: Option<ThinkingFormat>,
    /// Whether to use `developer` role for system messages (o-series).
    #[serde(default, rename = "supportsDeveloperRole")]
    pub supports_developer_role: bool,
    #[serde(default, rename = "maxTokensField")]
    pub max_tokens_field: Option<MaxTokensField>,
    #[serde(default, rename = "supportsUsageInStreaming")]
    pub supports_usage_in_streaming: bool,
    /// Whether this provider supports eager tool-call streaming
    /// (content block deltas arriving before the full block).
    #[serde(default, rename = "supportsEagerToolInputStreaming")]
    pub supports_eager_tool_input_streaming: bool,
    /// Whether provider requires an assistant bridge message between
    /// tool results and subsequent user messages.
    #[serde(default, rename = "requiresAssistantAfterToolResult")]
    pub requires_assistant_after_tool_result: bool,
    /// Whether to use `api-key` header instead of `Authorization: Bearer`.
    /// Xiaomi MiMo uses this.
    #[serde(default, rename = "usesApiKeyHeader")]
    pub uses_api_key_header: bool,
}

impl ModelCompat {
    pub fn for_openai() -> Self {
        Self {
            thinking_format: Some(ThinkingFormat::OpenAI),
            supports_developer_role: true,
            max_tokens_field: Some(MaxTokensField::MaxCompletionTokens),
            supports_usage_in_streaming: true,
            ..Default::default()
        }
    }

    pub fn for_deepseek() -> Self {
        Self {
            thinking_format: Some(ThinkingFormat::DeepSeek),
            supports_usage_in_streaming: true,
            requires_assistant_after_tool_result: true,
            max_tokens_field: Some(MaxTokensField::MaxTokens),
            ..Default::default()
        }
    }

    pub fn for_opencode() -> Self {
        Self {
            thinking_format: Some(ThinkingFormat::OpenAI),
            supports_usage_in_streaming: true,
            // OpenCode endpoints are OpenAI-compatible adapters and commonly
            // expect the classic `max_tokens` field.
            max_tokens_field: Some(MaxTokensField::MaxTokens),
            ..Default::default()
        }
    }

    pub fn for_xiaomi() -> Self {
        Self {
            thinking_format: Some(ThinkingFormat::XiaomiMiMo),
            supports_usage_in_streaming: true,
            max_tokens_field: Some(MaxTokensField::MaxCompletionTokens),
            uses_api_key_header: true,
            ..Default::default()
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    pub name: String,
    pub api: Api,
    pub provider: Provider,
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    pub reasoning: bool,
    /// Mapping from thinking levels to provider-specific strings.
    /// e.g., `{"high": "high", "xhigh": "max"}`.
    #[serde(default, rename = "thinkingLevelMap")]
    pub thinking_level_map: HashMap<ThinkingLevel, Option<String>>,
    #[serde(default)]
    pub input: Vec<Modality>,
    #[serde(rename = "contextWindow")]
    pub context_window: u32,
    #[serde(rename = "maxTokens")]
    pub max_tokens: u32,
    #[serde(default)]
    pub compat: ModelCompat,
}

pub trait ModelCatalog: Send + Sync {
    fn find(&self, provider: Provider, model_id: &str) -> Option<&Model>;
    fn list(&self) -> Vec<&Model>;
    fn list_by_provider(&self, provider: Provider) -> Vec<&Model>;
}

impl Model {
    pub fn thinking_param(&self, level: ThinkingLevel) -> Option<String> {
        self.thinking_level_map.get(&level).and_then(|v| v.clone())
    }

    /// Whether this model needs reasoning content on replayed
    /// assistant messages. Derived from `thinking_format` — MiMo
    /// requires it (API returns 400 when thinking is enabled and
    /// tool calls are present without it). DeepSeek does NOT require
    /// it — Reasonix strips it entirely, and sending it prevents
    /// empty assistant message skipping.
    pub fn requires_reasoning_on_replay(&self) -> bool {
        matches!(
            self.compat.thinking_format,
            Some(ThinkingFormat::XiaomiMiMo)
        )
    }

    pub fn max_tokens_field_name(&self) -> &str {
        match self.compat.max_tokens_field {
            Some(MaxTokensField::MaxCompletionTokens) | None => "max_completion_tokens",
            Some(MaxTokensField::MaxTokens) => "max_tokens",
        }
    }

    pub fn system_role(&self) -> &str {
        if self.compat.supports_developer_role {
            "developer"
        } else {
            "system"
        }
    }
}
