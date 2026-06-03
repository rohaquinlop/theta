//! DeepSeek model definitions.

use michin_ai::model::{Model, ModelCompat};
use michin_ai::types::{Api, Modality, Provider, ThinkingLevel};

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
        context_window: 1_000_000,
        max_tokens: 384_000,
        compat: ModelCompat::for_deepseek(),
    }
}
