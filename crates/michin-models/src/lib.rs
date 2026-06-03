//! Built-in model catalog for MichiN.
//!
//! Provides model definitions for all supported providers
//! compiled directly into the binary.

pub mod codex;
pub mod deepseek;
pub mod openai;
pub mod opencode;
pub mod xiaomi;

use michin_ai::model::Model;
use michin_ai::types::Provider;

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
        models.extend(xiaomi::models());
        Self { models }
    }
}

impl Default for BuiltInCatalog {
    fn default() -> Self {
        Self::new()
    }
}

impl michin_ai::model::ModelCatalog for BuiltInCatalog {
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
