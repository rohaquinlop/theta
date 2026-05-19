//! OpenAI model definitions.

use theta_ai::model::{Model, ModelCompat};
use theta_ai::types::{Api, Modality, ModelCost, Provider, ThinkingLevel};

/// Return all OpenAI models.
pub fn models() -> Vec<Model> {
    vec![gpt_5_5(), gpt_5_5_instant(), o4(), o4_mini()]
}

/// GPT-5.5 "Spud" — latest flagship, released April 2026.
fn gpt_5_5() -> Model {
    Model {
        id: "gpt-5.5".into(),
        name: "GPT-5.5".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenAI,
        base_url: "https://api.openai.com".into(),
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
            input: 1.25,
            output: 10.0,
            cache_read: 0.625,
            cache_write: 0.0,
        },
        context_window: 272_000,
        max_tokens: 128_000,
        compat: ModelCompat::for_openai(),
    }
}

/// GPT-5.5 Instant — faster, cheaper variant.
fn gpt_5_5_instant() -> Model {
    Model {
        id: "gpt-5.5-instant".into(),
        name: "GPT-5.5 Instant".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenAI,
        base_url: "https://api.openai.com".into(),
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
            input: 0.35,
            output: 2.0,
            cache_read: 0.175,
            cache_write: 0.0,
        },
        context_window: 272_000,
        max_tokens: 128_000,
        compat: ModelCompat::for_openai(),
    }
}

/// o4 — o-series reasoning model.
fn o4() -> Model {
    Model {
        id: "o4".into(),
        name: "o4".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenAI,
        base_url: "https://api.openai.com".into(),
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
            input: 10.0,
            output: 40.0,
            cache_read: 2.5,
            cache_write: 0.0,
        },
        context_window: 200_000,
        max_tokens: 100_000,
        compat: {
            let mut c = ModelCompat::for_openai();
            // o-series uses developer role for system messages
            c.supports_developer_role = true;
            c
        },
    }
}

/// o4-mini — smaller o-series reasoning model.
fn o4_mini() -> Model {
    Model {
        id: "o4-mini".into(),
        name: "o4-mini".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenAI,
        base_url: "https://api.openai.com".into(),
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
            input: 1.1,
            output: 4.4,
            cache_read: 0.275,
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
            assert_eq!(m.provider, Provider::OpenAI);
            assert_eq!(m.base_url, "https://api.openai.com");
            assert!(m.context_window > 0);
            assert!(m.max_tokens > 0);
        }
    }
}
