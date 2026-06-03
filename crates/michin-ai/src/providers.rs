//! Provider registry — generic container for LLM providers.
//!
//! Concrete provider implementations (OpenAI-compat, Codex) live in
//! `michin-ai-net`. This module provides the registry that the agent
//! loop uses to dispatch requests by API type.

use std::collections::HashMap;
use std::sync::RwLock;

use super::error::MichiNError;
use super::model::Model;
use super::provider::{EventStream, Provider};
use super::types::{Api, Context, Provider as ProviderKind, StreamOptions};

/// Callback type for provider-specific configuration.
type ConfigCallback = Box<dyn Fn(&str) + Send + Sync>;

/// Central registry for all LLM providers.
pub struct ProviderRegistry {
    providers: HashMap<Api, Box<dyn Provider>>,
    /// Per-provider API keys.
    api_keys: RwLock<HashMap<ProviderKind, Option<String>>>,
    /// Provider-specific callbacks set by michin-ai-net during registry creation.
    mimo_url_setter: Option<ConfigCallback>,
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            api_keys: RwLock::new(HashMap::new()),
            mimo_url_setter: None,
        }
    }

    /// Register a provider for a specific API.
    pub fn register(&mut self, api: Api, provider: Box<dyn Provider>) {
        self.providers.insert(api, provider);
    }

    /// Register a callback for setting the MiMo cluster URL.
    /// Called by michin-ai-net's `default_registry()`.
    pub fn set_mimo_url_callback(&mut self, cb: impl Fn(&str) + Send + Sync + 'static) {
        self.mimo_url_setter = Some(Box::new(cb));
    }

    /// Set the MiMo cluster base URL (from latency test modal).
    pub fn set_mimo_base_url(&self, url: &str) {
        if let Some(ref setter) = self.mimo_url_setter {
            setter(url);
        }
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
    ) -> Result<EventStream<'a>, MichiNError> {
        let provider = self
            .providers
            .get(&model.api)
            .ok_or_else(|| MichiNError::ApiError {
                status: 500,
                message: format!("No provider registered for API {:?}", model.api),
                retry_after_ms: None,
            })?;

        // Codex uses session tokens, not API keys. The provider reads
        // the token from env directly. Don't enforce registry-level key check.
        let is_codex = matches!(model.api, Api::OpenAiCodexResponses);
        if !is_codex {
            let _api_key = self
                .get_api_key(model.provider)
                .ok_or(MichiNError::MissingApiKey {
                    provider: model.provider,
                })?;
        }

        provider.stream(model, context, options).await
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
