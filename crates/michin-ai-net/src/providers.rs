//! Concrete provider implementations and registry helpers.
//!
//! ProviderRegistry lives in michin-ai. This module provides the
//! concrete providers and convenience functions for setting up a
//! registry with all built-in providers.

pub mod openai_codex;
pub mod openai_compat;

pub use openai_codex::OpenAiCodexProvider;
pub use openai_compat::OpenAiCompatProvider;

/// Build a default registry with the OpenAI-compatible provider
/// registered for `OpenAiCompletions` and the Codex provider for
/// `OpenAiCodexResponses`.
///
/// Also registers the MiMo cluster URL callback so that
/// `ProviderRegistry::set_mimo_base_url` works without knowing
/// about concrete provider types.
pub fn default_registry() -> michin_ai::providers::ProviderRegistry {
    let mut registry = michin_ai::providers::ProviderRegistry::new();

    // Create the OpenAI-compat provider and register the MiMo callback
    // before moving it into the registry.
    let compat = OpenAiCompatProvider::new();
    let mimo_sink = compat.mimo_url_sink();
    registry.register(michin_ai::types::Api::OpenAiCompletions, Box::new(compat));
    registry.register(
        michin_ai::types::Api::OpenAiCodexResponses,
        Box::new(OpenAiCodexProvider::new()),
    );
    registry.set_mimo_url_callback(move |url| mimo_sink(url));
    registry
}
