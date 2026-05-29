//! Provider registry and built-in provider registration.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use super::error::ThetaError;
use super::model::Model;
use super::provider::{EventStream, Provider};
use super::types::{Api, Context, Provider as ProviderKind, StreamOptions};

pub mod openai_compat;
pub use openai_compat::OpenAiCompatProvider;

pub mod openai_codex;
pub use openai_codex::OpenAiCodexProvider;

/// A provider factory function — used for lazy provider creation.
pub type ProviderFactory = Arc<dyn Fn() -> Box<dyn Provider> + Send + Sync>;

/// Central registry for all LLM providers.
pub struct ProviderRegistry {
    providers: HashMap<Api, Box<dyn Provider>>,
    /// Per-provider API keys.
    api_keys: RwLock<HashMap<ProviderKind, Option<String>>>,
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            api_keys: RwLock::new(HashMap::new()),
        }
    }

    /// Register a provider for a specific API.
    pub fn register(&mut self, api: Api, provider: Box<dyn Provider>) {
        self.providers.insert(api, provider);
    }

    /// Set an API key for a provider.
    /// Also passes the token to the registered provider via [`Provider::set_token`]
    /// if the provider stores it (e.g. Codex OAuth).
    pub fn set_api_key(&self, provider: ProviderKind, key: impl Into<String>) {
        let key: String = key.into();
        if let Ok(mut api_keys) = self.api_keys.write() {
            api_keys.insert(provider, Some(key.clone()));
        }
        // Forward token to the matching provider.
        if let Some(api) = provider_kind_to_api(provider)
            && let Some(p) = self.providers.get(&api)
        {
            p.set_token(&key);
        }
    }

    /// Get the API key for a provider.
    pub fn get_api_key(&self, provider: ProviderKind) -> Option<String> {
        self.api_keys
            .read()
            .ok()
            .and_then(|api_keys| api_keys.get(&provider).cloned().flatten())
    }

    /// Stream using the provider matching the model's API.
    /// No retry at this level — the agent loop handles retry with
    /// configurable backoff.
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
                retry_after_ms: None,
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

        provider.stream(model, context, options).await
    }

    /// Get a reference to a registered provider.
    pub fn get(&self, api: &Api) -> Option<&dyn Provider> {
        self.providers.get(api).map(|p| p.as_ref())
    }

    /// Set the MiMo cluster base URL on the OpenAI-compatible provider.
    /// Used after the user runs the latency test and selects a cluster.
    pub fn set_mimo_base_url(&self, url: &str) {
        if let Some(provider) = self
            .providers
            .get(&Api::OpenAiCompletions)
            .and_then(|p| p.as_any().downcast_ref::<OpenAiCompatProvider>())
        {
            provider.set_mimo_base_url(url);
        }
    }
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
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

/// Map a provider kind to the API it uses.
fn provider_kind_to_api(kind: ProviderKind) -> Option<Api> {
    match kind {
        ProviderKind::OpenAI => Some(Api::OpenAiCompletions),
        ProviderKind::OpenAiCodex => Some(Api::OpenAiCodexResponses),
        ProviderKind::DeepSeek => Some(Api::OpenAiCompletions),
        ProviderKind::OpenCode => Some(Api::OpenAiCompletions),
        ProviderKind::OpenCodeGo => Some(Api::OpenAiCompletions),
        ProviderKind::XiaomiMiMo => Some(Api::OpenAiCompletions),
    }
}
