//! DeepSeek model definitions.

use theta_ai::model::{Model, ModelCompat};
use theta_ai::types::{Api, Modality, ModelCost, Provider, ThinkingLevel};

/// Return all DeepSeek models.
pub fn models() -> Vec<Model> {
    vec![deepseek_v4_pro(), deepseek_v4_flash()]
}

/// DeepSeek V4 Pro — flagship, 1.6T total / 49B active params.
/// Released April 2026. 1M context window.
fn deepseek_v4_pro() -> Model {
    Model {
        id: "deepseek-v4-pro".into(),
        name: "DeepSeek V4 Pro".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::DeepSeek,
        base_url: "https://api.deepseek.com".into(),
        reasoning: true,
        thinking_level_map: [
            (ThinkingLevel::Off, None),
            (ThinkingLevel::Minimal, None),
            (ThinkingLevel::Low, None),
            (ThinkingLevel::Medium, None),
            (ThinkingLevel::High, Some("high".into())),
            (ThinkingLevel::XHigh, Some("max".into())),
        ]
        .into(),
        input: vec![Modality::Text],
        cost: ModelCost {
            input: 0.435,
            output: 0.87,
            cache_read: 0.003625,
            cache_write: 0.0,
        },
        context_window: 1_000_000,
        max_tokens: 384_000,
        compat: ModelCompat::for_deepseek(),
    }
}

/// DeepSeek V4 Flash — faster variant, 284B total / 13B active params.
/// Released April 2026. 1M context window.
fn deepseek_v4_flash() -> Model {
    Model {
        id: "deepseek-v4-flash".into(),
        name: "DeepSeek V4 Flash".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::DeepSeek,
        base_url: "https://api.deepseek.com".into(),
        reasoning: true,
        thinking_level_map: [
            (ThinkingLevel::Off, None),
            (ThinkingLevel::Minimal, None),
            (ThinkingLevel::Low, None),
            (ThinkingLevel::Medium, None),
            (ThinkingLevel::High, Some("high".into())),
            (ThinkingLevel::XHigh, Some("max".into())),
        ]
        .into(),
        input: vec![Modality::Text],
        cost: ModelCost {
            input: 0.14,
            output: 0.28,
            cache_read: 0.0028,
            cache_write: 0.0,
        },
        context_window: 1_000_000,
        max_tokens: 384_000,
        compat: ModelCompat::for_deepseek(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_models_valid() {
        for m in models() {
            assert!(!m.id.is_empty());
            assert_eq!(m.provider, Provider::DeepSeek);
            assert_eq!(m.base_url, "https://api.deepseek.com");
            assert!(m.context_window > 0);
            assert!(m.max_tokens > 0);
            // DeepSeek must have requires_reasoning_content_on_assistant
            assert!(
                m.compat.requires_reasoning_content_on_assistant,
                "DeepSeek models must require reasoning_content on replayed assistant messages"
            );
        }
    }

    #[test]
    fn test_all_have_thinking_map() {
        for m in models() {
            assert!(m.reasoning);
            assert!(
                m.thinking_param(ThinkingLevel::Off).is_none(),
                "Off should be None"
            );
            assert!(
                m.thinking_param(ThinkingLevel::High).is_some(),
                "High should have a value"
            );
        }
    }
}
