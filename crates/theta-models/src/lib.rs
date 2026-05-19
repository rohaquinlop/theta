//! Built-in model catalog for Theta.
//!
//! Provides model definitions for all supported providers
//! compiled directly into the binary.

pub mod codex;
pub mod deepseek;
pub mod openai;
pub mod opencode;

use theta_ai::model::Model;
use theta_ai::types::Provider;

/// A static model catalog that holds all built-in models.
pub struct BuiltInCatalog {
    models: Vec<Model>,
}

impl BuiltInCatalog {
    pub fn new() -> Self {
        let mut models = Vec::new();
        models.extend(openai::models());
        models.extend(deepseek::models());
        models.extend(opencode::models());
        models.extend(codex::models());
        Self { models }
    }
}

impl Default for BuiltInCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl theta_ai::model::ModelCatalog for BuiltInCatalog {
    fn find(&self, provider: Provider, model_id: &str) -> Option<&Model> {
        self.models
            .iter()
            .find(|m| m.provider == provider && m.id == model_id)
    }

    fn list(&self) -> Vec<&Model> {
        self.models.iter().collect()
    }

    fn list_by_provider(&self, provider: Provider) -> Vec<&Model> {
        self.models
            .iter()
            .filter(|m| m.provider == provider)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use theta_ai::ModelCatalog;

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
}
