//! OpenAI Codex model definitions.
//!
//! Codex gives ChatGPT Plus subscribers API access to OpenAI models
//! without needing a separate API key. Authentication uses the
//! ChatGPT session token (set via `OPENAI_CODEX_TOKEN` env var).
//!
//! Models connect to `https://chatgpt.com/backend-api` instead of
//! `https://api.openai.com`.

use theta_ai::model::{Model, ModelCompat};
use theta_ai::types::{Api, Modality, ModelCost, Provider, ThinkingLevel};

/// Return all codex-enabled models.
/// ChatGPT Plus subscribers get access to these with their session token.
pub fn models() -> Vec<Model> {
    vec![
        codex_gpt_5_5(),
        codex_gpt_5_5_instant(),
        codex_o4(),
        codex_o4_mini(),
    ]
}

/// Codex GPT-5.5 — same model, authenticated via ChatGPT Plus token.
fn codex_gpt_5_5() -> Model {
    Model {
        id: "gpt-5.5".into(),
        name: "GPT-5.5 (Codex)".into(),
        api: Api::OpenAiCodexResponses,
        provider: Provider::OpenAiCodex,
        base_url: "https://chatgpt.com/backend-api".into(),
        reasoning: true,
        thinking_level_map: [
            (ThinkingLevel::Off, None),
            (ThinkingLevel::Minimal, Some("minimal".into())),
            (ThinkingLevel::Low, Some("low".into())),
            (ThinkingLevel::Medium, Some("medium".into())),
            (ThinkingLevel::High, Some("high".into())),
            (ThinkingLevel::XHigh, Some("max".into())),
        ]
        .into(),
        input: vec![Modality::Text],
        cost: ModelCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
        context_window: 272_000,
        max_tokens: 128_000,
        compat: ModelCompat::for_openai(),
    }
}

/// Codex GPT-5.5 Instant.
fn codex_gpt_5_5_instant() -> Model {
    Model {
        id: "gpt-5.5-instant".into(),
        name: "GPT-5.5 Instant (Codex)".into(),
        api: Api::OpenAiCodexResponses,
        provider: Provider::OpenAiCodex,
        base_url: "https://chatgpt.com/backend-api".into(),
        reasoning: true,
        thinking_level_map: [
            (ThinkingLevel::Off, None),
            (ThinkingLevel::Minimal, Some("minimal".into())),
            (ThinkingLevel::Low, Some("low".into())),
            (ThinkingLevel::Medium, Some("medium".into())),
            (ThinkingLevel::High, Some("high".into())),
            (ThinkingLevel::XHigh, Some("max".into())),
        ]
        .into(),
        input: vec![Modality::Text],
        cost: ModelCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
        context_window: 272_000,
        max_tokens: 128_000,
        compat: ModelCompat::for_openai(),
    }
}

/// Codex o4 — o-series reasoning via ChatGPT Plus.
fn codex_o4() -> Model {
    Model {
        id: "o4".into(),
        name: "o4 (Codex)".into(),
        api: Api::OpenAiCodexResponses,
        provider: Provider::OpenAiCodex,
        base_url: "https://chatgpt.com/backend-api".into(),
        reasoning: true,
        thinking_level_map: [
            (ThinkingLevel::Off, None),
            (ThinkingLevel::Minimal, Some("minimal".into())),
            (ThinkingLevel::Low, Some("low".into())),
            (ThinkingLevel::Medium, Some("medium".into())),
            (ThinkingLevel::High, Some("high".into())),
            (ThinkingLevel::XHigh, Some("max".into())),
        ]
        .into(),
        input: vec![Modality::Text],
        cost: ModelCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
        context_window: 200_000,
        max_tokens: 100_000,
        compat: {
            let mut c = ModelCompat::for_openai();
            c.supports_developer_role = true;
            c
        },
    }
}

/// Codex o4-mini.
fn codex_o4_mini() -> Model {
    Model {
        id: "o4-mini".into(),
        name: "o4-mini (Codex)".into(),
        api: Api::OpenAiCodexResponses,
        provider: Provider::OpenAiCodex,
        base_url: "https://chatgpt.com/backend-api".into(),
        reasoning: true,
        thinking_level_map: [
            (ThinkingLevel::Off, None),
            (ThinkingLevel::Minimal, Some("minimal".into())),
            (ThinkingLevel::Low, Some("low".into())),
            (ThinkingLevel::Medium, Some("medium".into())),
            (ThinkingLevel::High, Some("high".into())),
            (ThinkingLevel::XHigh, Some("max".into())),
        ]
        .into(),
        input: vec![Modality::Text],
        cost: ModelCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
        context_window: 200_000,
        max_tokens: 100_000,
        compat: {
            let mut c = ModelCompat::for_openai();
            c.supports_developer_role = true;
            c
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_models_valid() {
        for m in models() {
            assert!(!m.id.is_empty());
            assert_eq!(m.provider, Provider::OpenAiCodex);
            assert_eq!(m.api, Api::OpenAiCodexResponses);
            assert_eq!(m.base_url, "https://chatgpt.com/backend-api");
            assert!(m.context_window > 0);
            assert!(m.max_tokens > 0);
            // Codex is free with subscription — cost should be zero
            assert_eq!(m.cost.input, 0.0);
            assert_eq!(m.cost.output, 0.0);
        }
    }

    #[test]
    fn test_codex_model_count() {
        assert_eq!(models().len(), 4);
    }
}
