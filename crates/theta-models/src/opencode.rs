//! OpenCode model definitions.
//!
//! OpenCode is a coding agent that supports 75+ providers via the
//! Vercel AI SDK + Models.dev registry. Theta connects to
//! OpenCode's API as an OpenAI-compatible endpoint.
//!
//! By default, the base URL is user-configurable. Common defaults:
//! - `https://api.opencode.ai` (if that exists)
//! - Or self-hosted OpenCode instance

use theta_ai::model::{Model, ModelCompat};
use theta_ai::types::{Api, Modality, ModelCost, Provider, ThinkingLevel};

/// Return all OpenCode models.
pub fn models() -> Vec<Model> {
    // OpenCode is a proxy — models depend on which underlying
    // provider is configured. We register a single default
    // model that represents the OpenCode endpoint.
    vec![opencode_default()]
}

/// Default OpenCode model — user configures the actual underlying
/// model via their OpenCode setup.
fn opencode_default() -> Model {
    Model {
        id: "opencode".into(),
        name: "OpenCode".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenCode,
        // Configurable — users set this in their config.
        base_url: "https://api.opencode.ai".into(),
        reasoning: true,
        thinking_level_map: [
            (ThinkingLevel::Off, None),
            (ThinkingLevel::Minimal, Some("minimal".into())),
            (ThinkingLevel::Low, Some("low".into())),
            (ThinkingLevel::Medium, Some("medium".into())),
            (ThinkingLevel::High, Some("high".into())),
            (ThinkingLevel::XHigh, Some("max".into())),
        ]
        .into(),
        input: vec![Modality::Text],
        cost: ModelCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
        context_window: 200_000,
        max_tokens: 64_000,
        compat: ModelCompat::for_opencode(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_models_valid() {
        for m in models() {
            assert!(!m.id.is_empty());
            assert_eq!(m.provider, Provider::OpenCode);
            assert!(m.context_window > 0);
        }
    }
}
