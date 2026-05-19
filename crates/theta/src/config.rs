//! Configuration and auth storage for theta.
//!
//! Loads from:
//! - `~/.theta/config.toml` — model defaults, thinking level, etc.
//! - `~/.theta/auth.json` — provider tokens with expiry
//! - Environment variables — API key fallback (OPENAI_API_KEY, etc.)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Full theta configuration.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThetaConfig {
    /// Default model selection.
    #[serde(default)]
    pub model: ModelDefaults,

    /// Thinking level default.
    #[serde(default)]
    pub thinking: ThinkingDefaults,

    /// Provider auth tokens.
    #[serde(default)]
    pub auth: AuthConfig,

    /// Working directory override.
    #[serde(default)]
    pub working_dir: Option<PathBuf>,

    /// TUI theme name ("default" or "monokai").
    #[serde(default)]
    pub theme: Option<String>,

    /// Context compaction settings.
    #[serde(default)]
    pub compaction: CompactionSettings,

    /// Provider retry settings.
    #[serde(default)]
    pub retry: RetrySettings,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelDefaults {
    /// Default model ID.
    pub default: Option<String>,

    /// Per-provider default models.
    #[serde(default)]
    pub providers: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThinkingDefaults {
    /// Default thinking level (off, low, medium, high).
    pub default: Option<String>,
}

/// Compaction settings loaded from config.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionSettings {
    /// Whether automatic compaction is enabled.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Tokens to reserve for the model's response.
    #[serde(default = "default_reserve_tokens")]
    pub reserve_tokens: u32,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 4096,
        }
    }
}

/// Retry settings loaded from config.toml.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrySettings {
    /// Maximum retry attempts (0 = no retry).
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    /// Base delay in milliseconds before first retry.
    #[serde(default = "default_base_delay")]
    pub base_delay_ms: u64,
}

impl Default for RetrySettings {
    fn default() -> Self {
        Self {
            max_retries: 2,
            base_delay_ms: 1000,
        }
    }
}

fn default_true() -> bool {
    true
}
fn default_reserve_tokens() -> u32 {
    4096
}
fn default_max_retries() -> u32 {
    2
}
fn default_base_delay() -> u64 {
    1000
}

/// Provider auth tokens loaded from ~/.theta/auth.json or env vars.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AuthConfig {
    /// API key token entries.
    #[serde(default)]
    pub tokens: Vec<ProviderToken>,

    /// OAuth token entries (subscription providers).
    #[serde(default)]
    pub oauth_tokens: Vec<ProviderOAuthToken>,
}

/// A stored API key / static token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderToken {
    /// Provider identifier: "openai", "openai-codex", "deepseek", "opencode".
    pub provider: String,

    /// Auth token / API key.
    pub token: String,

    /// Unix timestamp (ms) when this token expires.
    pub expires_at: Option<u64>,

    /// When the token was obtained.
    pub obtained_at: u64,
}

/// A stored OAuth credential (subscription providers like Codex).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderOAuthToken {
    /// Provider identifier.
    pub provider: String,

    /// OAuth access token (used as API key).
    pub access_token: String,

    /// OAuth refresh token (used to get new access tokens).
    pub refresh_token: String,

    /// Unix timestamp (ms) when the access token expires.
    pub expires_at: u64,

    /// When the token was obtained / refreshed.
    pub obtained_at: u64,

    /// Optional account identifier (e.g. ChatGPT account ID).
    pub account_id: Option<String>,
}

/// Build a user-friendly error message when no auth token is found.
pub fn auth_error_message(provider: &str) -> String {
    let env_var = provider_env_var(provider);
    format!(
        "no auth token for '{provider}'.\n\
         Set the {env_var} environment variable, or run `theta login {provider}`"
    )
}

/// Map a provider string to its environment variable name.
pub fn provider_env_var(provider: &str) -> &'static str {
    match provider {
        "openai" => "OPENAI_API_KEY",
        "openai-codex" => "OPENAI_CODEX_TOKEN",
        "deepseek" => "DEEPSEEK_API_KEY",
        "opencode" => "OPENCODE_API_KEY",
        _ => "",
    }
}

impl AuthConfig {
    /// Get a token for a specific provider. Checks stored tokens first,
    /// then OAuth tokens, then environment variables.
    ///
    /// For OAuth tokens, returns the access token even if expired —
    /// callers should use [`get_api_key`] for auto-refresh.
    pub fn get_token(&self, provider: &str) -> Option<String> {
        // Check stored API key tokens.
        for entry in &self.tokens {
            if entry.provider == provider {
                if let Some(expiry) = entry.expires_at {
                    let now = now_ms();
                    if now >= expiry {
                        continue;
                    }
                }
                return Some(entry.token.clone());
            }
        }

        // Check OAuth tokens.
        for entry in &self.oauth_tokens {
            if entry.provider == provider {
                return Some(entry.access_token.clone());
            }
        }

        // Check OAuth env var fallback (OPENAI_CODEX_TOKEN for codex).
        if let Some(env_key) = self.get_env_token(provider) {
            return Some(env_key);
        }

        None
    }

    /// Check if any token is configured for a given provider
    /// (including OAuth tokens, even if expired).
    pub fn has_token(&self, provider: &str) -> bool {
        if self.tokens.iter().any(|t| t.provider == provider) {
            return true;
        }
        if self.oauth_tokens.iter().any(|t| t.provider == provider) {
            return true;
        }
        self.get_env_token(provider).is_some()
    }

    /// Update or insert a stored API key token.
    pub fn set_token(&mut self, provider: &str, token: &str, expires_at: Option<u64>) {
        let now = now_ms();
        if let Some(existing) = self.tokens.iter_mut().find(|t| t.provider == provider) {
            existing.token = token.to_string();
            existing.expires_at = expires_at;
            existing.obtained_at = now;
        } else {
            // Also remove any OAuth token for this provider (migrate from OAuth to API key).
            self.oauth_tokens.retain(|t| t.provider != provider);
            self.tokens.push(ProviderToken {
                provider: provider.to_string(),
                token: token.to_string(),
                expires_at,
                obtained_at: now,
            });
        }
    }

    /// Store an OAuth credential (access + refresh token).
    pub fn set_oauth_token(
        &mut self,
        provider: &str,
        access_token: &str,
        refresh_token: &str,
        expires_at: u64,
    ) {
        let now = now_ms();
        if let Some(existing) = self
            .oauth_tokens
            .iter_mut()
            .find(|t| t.provider == provider)
        {
            existing.access_token = access_token.to_string();
            existing.refresh_token = refresh_token.to_string();
            existing.expires_at = expires_at;
            existing.obtained_at = now;
        } else {
            // Remove any static token for this provider.
            self.tokens.retain(|t| t.provider != provider);
            self.oauth_tokens.push(ProviderOAuthToken {
                provider: provider.to_string(),
                access_token: access_token.to_string(),
                refresh_token: refresh_token.to_string(),
                expires_at,
                obtained_at: now,
                account_id: None,
            });
        }
    }

    /// Get an API key with automatic OAuth token refresh if needed.
    ///
    /// Returns the unexpired access token, refreshing via the refresh
    /// token if the current one is expired. For static API keys, returns
    /// the key directly.
    pub async fn get_api_key(&mut self, provider: &str) -> Option<String> {
        // Check static API key tokens first.
        for entry in &self.tokens {
            if entry.provider == provider {
                if let Some(expiry) = entry.expires_at
                    && now_ms() >= expiry
                {
                    continue;
                }
                return Some(entry.token.clone());
            }
        }

        // Check OAuth tokens with auto-refresh.
        if let Some(pos) = self
            .oauth_tokens
            .iter()
            .position(|t| t.provider == provider)
        {
            let is_expired = now_ms() >= self.oauth_tokens[pos].expires_at;
            if is_expired {
                // Attempt refresh.
                let refresh_token = self.oauth_tokens[pos].refresh_token.clone();
                match crate::oauth::codex::refresh_codex_token(&refresh_token).await {
                    Ok(creds) => {
                        self.oauth_tokens[pos].access_token = creds.access_token.clone();
                        self.oauth_tokens[pos].refresh_token = creds.refresh_token;
                        self.oauth_tokens[pos].expires_at = creds.expires_at;
                        self.oauth_tokens[pos].obtained_at = now_ms();
                        // Persist the refreshed token.
                        if let Err(e) = save_auth(self, None).await {
                            tracing::warn!("Failed to save refreshed OAuth token: {e}");
                        }
                        return Some(creds.access_token);
                    }
                    Err(e) => {
                        tracing::warn!("Failed to refresh OAuth token for {provider}: {e}");
                        // Return the expired token anyway — let the caller decide.
                        return Some(self.oauth_tokens[pos].access_token.clone());
                    }
                }
            } else {
                return Some(self.oauth_tokens[pos].access_token.clone());
            }
        }

        // Fall back to env var.
        self.get_env_token(provider)
    }

    /// Check environment variables for a token.
    fn get_env_token(&self, provider: &str) -> Option<String> {
        let env_var = match provider {
            "openai" => "OPENAI_API_KEY",
            "openai-codex" => "OPENAI_CODEX_TOKEN",
            "deepseek" => "DEEPSEEK_API_KEY",
            "opencode" => "OPENCODE_API_KEY",
            _ => return None,
        };
        std::env::var(env_var).ok()
    }
}

/// Build an AgentLoopConfig from the Theta toml config.
pub fn to_agent_config(tc: &ThetaConfig) -> theta_agent_core::AgentLoopConfig {
    theta_agent_core::AgentLoopConfig {
        compaction: theta_agent_core::CompactionConfig {
            enabled: tc.compaction.enabled,
            reserve_tokens: tc.compaction.reserve_tokens,
        },
        retry: theta_agent_core::RetryConfig {
            max_retries: tc.retry.max_retries,
            base_delay_ms: tc.retry.base_delay_ms,
        },
        ..Default::default()
    }
}

/// Load or create the full config.
pub async fn load_config(config_path: Option<&Path>) -> Result<ThetaConfig, ConfigError> {
    let path = config_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_config_path);

    if path.exists() {
        let contents = tokio::fs::read_to_string(&path)
            .await
            .map_err(ConfigError::Read)?;
        let mut config: ThetaConfig =
            toml::from_str(&contents).map_err(|e| ConfigError::Parse {
                path: path.display().to_string(),
                error: e.to_string(),
            })?;

        // Load auth from auth.json separately.
        config.auth = load_auth(None).await?;

        Ok(config)
    } else {
        let config = ThetaConfig {
            auth: load_auth(None).await?,
            ..Default::default()
        };
        Ok(config)
    }
}

/// Save config to disk.
pub async fn save_config(
    config: &ThetaConfig,
    config_path: Option<&Path>,
) -> Result<(), ConfigError> {
    let path = config_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_config_path);

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(ConfigError::Write)?;
    }

    let contents = toml::to_string_pretty(config).map_err(|e| ConfigError::Parse {
        path: path.display().to_string(),
        error: e.to_string(),
    })?;
    tokio::fs::write(&path, contents)
        .await
        .map_err(ConfigError::Write)?;

    Ok(())
}

/// Load auth tokens from ~/.theta/auth.json.
pub async fn load_auth(auth_path: Option<&Path>) -> Result<AuthConfig, ConfigError> {
    let path = auth_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_auth_path);

    if path.exists() {
        let contents = tokio::fs::read_to_string(&path)
            .await
            .map_err(ConfigError::Read)?;
        let auth: AuthConfig = serde_json::from_str(&contents).map_err(|e| ConfigError::Parse {
            path: path.display().to_string(),
            error: e.to_string(),
        })?;
        Ok(auth)
    } else {
        Ok(AuthConfig::default())
    }
}

/// Save auth tokens to ~/.theta/auth.json.
pub async fn save_auth(auth: &AuthConfig, auth_path: Option<&Path>) -> Result<(), ConfigError> {
    let path = auth_path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(default_auth_path);

    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(ConfigError::Write)?;
    }

    let contents = serde_json::to_string_pretty(auth).map_err(|e| ConfigError::Parse {
        path: path.display().to_string(),
        error: e.to_string(),
    })?;
    tokio::fs::write(&path, contents)
        .await
        .map_err(ConfigError::Write)?;
    Ok(())
}

/// Default path: ~/.theta/config.toml
fn default_config_path() -> PathBuf {
    theta_dir().join("config.toml")
}

/// Default path: ~/.theta/auth.json
fn default_auth_path() -> PathBuf {
    theta_dir().join("auth.json")
}

/// ~/.theta directory.
fn theta_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".theta")
}

/// Current time in milliseconds.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Configuration errors.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("failed to read config: {0}")]
    Read(std::io::Error),

    #[error("failed to write config: {0}")]
    Write(std::io::Error),

    #[error("failed to parse {path}: {error}")]
    Parse { path: String, error: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_auth_token_storage() {
        let mut auth = AuthConfig::default();
        auth.set_token("openai", "sk-test-key", None);
        assert_eq!(auth.get_token("openai"), Some("sk-test-key".into()));

        auth.set_token("openai-codex", "codex-token", Some(now_ms() + 3600_000));
        assert!(auth.get_token("openai-codex").is_some());
    }

    #[tokio::test]
    async fn test_auth_env_fallback() {
        let auth = AuthConfig::default();
        // Without env vars, returns None for unknown provider.
        assert_eq!(auth.get_token("nonexistent"), None);
    }
}
