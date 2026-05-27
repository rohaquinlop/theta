use theta_ai::{Api, Provider};
use theta_models::codex;

#[test]
fn test_all_models_valid() {
    for m in codex::models() {
        assert!(!m.id.is_empty());
        assert_eq!(m.provider, Provider::OpenAiCodex);
        assert_eq!(m.api, Api::OpenAiCodexResponses);
        assert_eq!(m.base_url, "https://chatgpt.com/backend-api");
        assert!(m.context_window > 0);
        assert!(m.max_tokens > 0);
    }
}
