use theta_ai::{ModelCatalog, Provider};
use theta_models::BuiltInCatalog;

#[test]
fn test_catalog_has_models() {
    let catalog = BuiltInCatalog::new();
    let all = catalog.list();
    assert!(!all.is_empty(), "Catalog should have models");

    let openai_models = catalog.list_by_provider(Provider::OpenAI);
    assert!(!openai_models.is_empty(), "Should have OpenAI models");

    let codex_models = catalog.list_by_provider(Provider::OpenAiCodex);
    assert!(!codex_models.is_empty(), "Should have Codex models");

    let deepseek_models = catalog.list_by_provider(Provider::DeepSeek);
    assert!(!deepseek_models.is_empty(), "Should have DeepSeek models");
}

#[test]
fn test_find_model() {
    let catalog = BuiltInCatalog::new();
    let gpt55 = catalog.find(Provider::OpenAI, "gpt-5.5");
    assert!(gpt55.is_some(), "gpt-5.5 should exist");
    assert_eq!(gpt55.unwrap().base_url, "https://api.openai.com");
}

#[test]
fn test_find_nonexistent() {
    let catalog = BuiltInCatalog::new();
    assert!(catalog.find(Provider::OpenAI, "nonexistent").is_none());
}
