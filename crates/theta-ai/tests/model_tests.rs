use std::collections::HashMap;
use theta_ai::model::{Model, ModelCompat};
use theta_ai::{Api, Modality, ModelCost, Provider, ThinkingLevel};

fn test_model() -> Model {
    Model {
        id: "test-model".into(),
        name: "Test Model".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenAI,
        base_url: "https://api.openai.com".into(),
        reasoning: true,
        thinking_level_map: HashMap::from([
            (ThinkingLevel::Off, None),
            (ThinkingLevel::High, Some("high".into())),
            (ThinkingLevel::XHigh, Some("max".into())),
        ]),
        input: vec![Modality::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 16_384,
        compat: ModelCompat::for_openai(),
    }
}

#[test]
fn test_thinking_param() {
    let m = test_model();
    assert_eq!(m.thinking_param(ThinkingLevel::Off), None);
    assert_eq!(
        m.thinking_param(ThinkingLevel::High),
        Some("high".to_string())
    );
    assert_eq!(m.thinking_param(ThinkingLevel::Low), None);
}

#[test]
fn test_max_tokens_field_name() {
    let m = test_model();
    assert_eq!(m.max_tokens_field_name(), "max_completion_tokens");
}

#[test]
fn test_requires_reasoning_on_replay() {
    let mut m = test_model();
    assert!(!m.requires_reasoning_on_replay());
    m.compat = ModelCompat::for_deepseek();
    assert!(m.requires_reasoning_on_replay());
}

#[test]
fn test_max_tokens_field_for_non_openai_compat() {
    let mut m = test_model();
    m.compat = ModelCompat::for_deepseek();
    assert_eq!(m.max_tokens_field_name(), "max_tokens");

    m.compat = ModelCompat::for_opencode();
    assert_eq!(m.max_tokens_field_name(), "max_tokens");
}
