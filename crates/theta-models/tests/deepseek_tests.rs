use theta_ai::{Provider, ThinkingLevel};
use theta_models::deepseek;

#[test]
fn test_all_models_valid() {
    for m in deepseek::models() {
        assert!(!m.id.is_empty());
        assert_eq!(m.provider, Provider::DeepSeek);
        assert_eq!(m.base_url, "https://api.deepseek.com");
        assert!(m.context_window > 0);
        assert!(m.max_tokens > 0);
        assert!(
            m.compat.requires_reasoning_content_on_assistant,
            "DeepSeek models must require reasoning_content on replayed assistant messages"
        );
    }
}

#[test]
fn test_all_have_thinking_map() {
    for m in deepseek::models() {
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
