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

/// Provider-specific compatibility flags.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelCompat {
    /// How thinking/reasoning params are sent.
    #[serde(default)]
    pub thinking_format: Option<ThinkingFormat>,
    /// Whether to use `developer` role for system messages (o-series).
    #[serde(default, rename = "supportsDeveloperRole")]
    pub supports_developer_role: bool,
    /// Whether this provider requires empty `reasoning_content` on
    /// replayed assistant messages (DeepSeek).
    #[serde(default, rename = "requiresReasoningContentOnAssistantMessages")]
    pub requires_reasoning_content_on_assistant: bool,
    /// Which field to use for max_tokens.
    #[serde(default, rename = "maxTokensField")]
    pub max_tokens_field: Option<MaxTokensField>,
    /// Whether `stream_options.include_usage` works on this provider.
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
            requires_reasoning_content_on_assistant: true,
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
            requires_reasoning_content_on_assistant: true,
            // MiMo supports usage in non-streaming responses but the
            // stream_options.include_usage field is OpenAI-specific.
            supports_usage_in_streaming: false,
            max_tokens_field: Some(MaxTokensField::MaxCompletionTokens),
            uses_api_key_header: true,
            ..Default::default()
        }
    }
}

/// A registered LLM model with all its metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    /// Unique model identifier (e.g. "gpt-5.5", "deepseek-v4-pro").
    pub id: String,
    /// Human-readable display name.
    pub name: String,
    /// Which API this model uses.
    pub api: Api,
    /// Which provider serves this model.
    pub provider: Provider,
    /// Base URL for API requests.
    #[serde(rename = "baseUrl")]
    pub base_url: String,
    /// Whether this model supports thinking/reasoning.
    pub reasoning: bool,
    /// Mapping from thinking levels to provider-specific strings.
    /// e.g., `{"high": "high", "xhigh": "max"}`.
    #[serde(default, rename = "thinkingLevelMap")]
    pub thinking_level_map: HashMap<ThinkingLevel, Option<String>>,
    /// Input modalities the model supports.
    #[serde(default)]
    pub input: Vec<Modality>,
    /// Context window size in tokens.
    #[serde(rename = "contextWindow")]
    pub context_window: u32,
    /// Maximum output tokens.
    #[serde(rename = "maxTokens")]
    pub max_tokens: u32,
    /// Provider-specific compatibility flags.
    #[serde(default)]
    pub compat: ModelCompat,
}

/// Trait for model catalogs. Implement this to provide model lookup.
pub trait ModelCatalog: Send + Sync {
    /// Find a model by provider and model ID.
    fn find(&self, provider: Provider, model_id: &str) -> Option<&Model>;

    /// List all models in the catalog.
    fn list(&self) -> Vec<&Model>;

    /// List models for a specific provider.
    fn list_by_provider(&self, provider: Provider) -> Vec<&Model>;
}

impl Model {
    /// Get the provider-specific thinking param value for a given level.
    /// Returns `None` if thinking is not supported or the level maps to None.
    pub fn thinking_param(&self, level: ThinkingLevel) -> Option<String> {
        self.thinking_level_map.get(&level).and_then(|v| v.clone())
    }

    /// Whether this model requires reasoning content on replayed assistant messages.
    pub fn requires_reasoning_on_replay(&self) -> bool {
        self.compat.requires_reasoning_content_on_assistant
    }

    /// The actual JSON field name for max_tokens in the API request body.
    pub fn max_tokens_field_name(&self) -> &str {
        match self.compat.max_tokens_field {
            Some(MaxTokensField::MaxCompletionTokens) | None => "max_completion_tokens",
            Some(MaxTokensField::MaxTokens) => "max_tokens",
        }
    }

    /// The system role name for this model ("system" or "developer").
    pub fn system_role(&self) -> &str {
        if self.compat.supports_developer_role {
            "developer"
        } else {
            "system"
        }
    }
}
