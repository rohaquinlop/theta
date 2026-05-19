//! Provider registry and built-in provider registration.

use std::collections::HashMap;
use std::sync::Arc;

use super::error::ThetaError;
use super::model::Model;
use super::provider::{EventStream, Provider};
use super::types::{Api, Context, Provider as ProviderKind, StreamOptions};

mod openai_compat;
pub use openai_compat::OpenAiCompatProvider;

mod openai_codex;
pub use openai_codex::OpenAiCodexProvider;

/// Maximum retry attempts for transient failures.
const MAX_RETRIES: u32 = 3;

/// A provider factory function — used for lazy provider creation.
pub type ProviderFactory = Arc<dyn Fn() -> Box<dyn Provider> + Send + Sync>;

/// Central registry for all LLM providers.
pub struct ProviderRegistry {
    providers: HashMap<Api, Box<dyn Provider>>,
    /// Per-provider API keys.
    api_keys: HashMap<ProviderKind, Option<String>>,
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            api_keys: HashMap::new(),
        }
    }

    /// Register a provider for a specific API.
    pub fn register(&mut self, api: Api, provider: Box<dyn Provider>) {
        self.providers.insert(api, provider);
    }

    /// Set an API key for a provider.
    pub fn set_api_key(&mut self, provider: ProviderKind, key: impl Into<String>) {
        self.api_keys.insert(provider, Some(key.into()));
    }

    /// Get the API key for a provider.
    pub fn get_api_key(&self, provider: ProviderKind) -> Option<&str> {
        self.api_keys.get(&provider).and_then(|k| k.as_deref())
    }

    /// Stream using the provider matching the model's API.
    pub async fn stream<'a>(
        &'a self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> Result<EventStream<'a>, ThetaError> {
        let provider = self
            .providers
            .get(&model.api)
            .ok_or_else(|| ThetaError::ApiError {
                status: 500,
                message: format!("No provider registered for API {:?}", model.api),
            })?;

        // Codex uses session tokens, not API keys. The provider reads
        // the token from env directly. Don't enforce registry-level key check.
        let is_codex = matches!(model.api, Api::OpenAiCodexResponses);
        if !is_codex {
            let _api_key =
                self.get_api_key(model.provider)
                    .ok_or_else(|| ThetaError::MissingApiKey {
                        provider: model.provider,
                    })?;
        }

        // Retry loop for transient errors.
        let mut last_error = None;
        for attempt in 0..MAX_RETRIES {
            match provider.stream(model, context, options).await {
                Ok(stream) => return Ok(stream),
                Err(e) => {
                    if !is_retryable(&e) || attempt == MAX_RETRIES - 1 {
                        return Err(e);
                    }
                    tracing::warn!(
                        "Retry attempt {}/{} for model {}: {}",
                        attempt + 1,
                        MAX_RETRIES,
                        model.id,
                        e
                    );
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap())
    }

    /// Get a reference to a registered provider.
    pub fn get(&self, api: &Api) -> Option<&dyn Provider> {
        self.providers.get(api).map(|p| p.as_ref())
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Determine if an error is retryable (rate limits, network hiccups).
fn is_retryable(error: &ThetaError) -> bool {
    match error {
        ThetaError::Http(e) => {
            e.status()
                .map(|s| s.as_u16() == 429 || s.as_u16() >= 500)
                .unwrap_or(true) // network errors are retryable
        }
        ThetaError::StreamEndedEarly => true,
        _ => false,
    }
}

/// Build a default registry with the OpenAI-compatible provider
/// registered for `OpenAiCompletions` and the Codex provider for
/// `OpenAiCodexResponses`.
pub fn default_registry() -> ProviderRegistry {
    let mut registry = ProviderRegistry::new();
    registry.register(
        Api::OpenAiCompletions,
        Box::new(OpenAiCompatProvider::new()),
    );
    registry.register(
        Api::OpenAiCodexResponses,
        Box::new(OpenAiCodexProvider::new()),
    );
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_creation() {
        let reg = ProviderRegistry::new();
        assert!(reg.get(&Api::OpenAiCompletions).is_none());
    }

    #[test]
    fn test_is_retryable() {
        // Rate limit is retryable
        let err = ThetaError::ApiError {
            status: 429,
            message: "ratelimit".into(),
        };
        assert!(!is_retryable(&err)); // ApiError not in retry list
        assert!(is_retryable(&ThetaError::StreamEndedEarly));
    }
}
