use theta_ai::Api;
use theta_ai::providers::ProviderRegistry;

#[test]
fn test_registry_creation() {
    let reg = ProviderRegistry::new();
    assert!(reg.get(&Api::OpenAiCompletions).is_none());
}
