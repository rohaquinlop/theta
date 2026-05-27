use theta_ai::Provider;
use theta_models::openai;

#[test]
fn test_all_models_valid() {
    for m in openai::models() {
        assert!(!m.id.is_empty());
        assert_eq!(m.provider, Provider::OpenAI);
        assert_eq!(m.base_url, "https://api.openai.com");
        assert!(m.context_window > 0);
        assert!(m.max_tokens > 0);
    }
}
