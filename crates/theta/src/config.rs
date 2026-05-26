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

    /// Provider transport settings.
    #[serde(default)]
    pub provider: ProviderSettings,

    /// Agent loop controls.
    #[serde(default)]
    pub agent: AgentSettings,
    /// Named runtime hardening profile.
    #[serde(default)]
    pub profile: RuntimeProfileSetting,
    /// Explicit per-project/runtime overrides applied on top of profile defaults.
    #[serde(default)]
    pub profile_overrides: ProfileOverrides,

    /// Skills to auto-invoke at session start (e.g. ["caveman ultra"]).
    #[serde(default)]
    pub startup_skills: Vec<String>,
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
    /// How many tokens of recent conversation to preserve during compaction.
    #[serde(default = "default_keep_recent_tokens")]
    pub keep_recent_tokens: u32,
    /// Strategy to preserve trimmed context.
    #[serde(default)]
    pub strategy: CompactionStrategySetting,
    /// Backward-compatible toggle; if present it overrides strategy.
    #[serde(default)]
    pub summarize_with_llm: Option<bool>,
    /// Maximum output tokens for compaction summaries.
    #[serde(default = "default_summary_max_tokens")]
    pub summary_max_tokens: u32,
}

impl Default for CompactionSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            reserve_tokens: 4096,
            keep_recent_tokens: 20_000,
            strategy: CompactionStrategySetting::Llm,
            summarize_with_llm: None,
            summary_max_tokens: 512,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompactionStrategySetting {
    None,
    Textual,
    #[default]
    Llm,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderSettings {
    /// Request timeout in milliseconds.
    #[serde(default = "default_timeout_ms")]
    pub timeout_ms: u64,
}

impl Default for ProviderSettings {
    fn default() -> Self {
        Self {
            timeout_ms: default_timeout_ms(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentSettings {
    /// Maximum same-signature tool-call repeats in one turn before aborting.
    #[serde(default = "default_max_same_tool_call_repeats")]
    pub max_same_tool_call_repeats: u32,
    /// Warn if tool stalls this long.
    #[serde(default = "default_tool_stall_warning_ms")]
    pub tool_stall_warning_ms: u64,
    /// Hard timeout for one tool call.
    #[serde(default = "default_tool_timeout_ms")]
    pub tool_timeout_ms: u64,
    /// Optional fallback model IDs in preference order.
    #[serde(default)]
    pub provider_fallback_chain: Vec<String>,
    /// Circuit breaker failure threshold.
    #[serde(default = "default_provider_failure_threshold")]
    pub provider_failure_threshold: u32,
    /// Circuit breaker open cooldown.
    #[serde(default = "default_provider_open_cooldown_ms")]
    pub provider_open_cooldown_ms: u64,
}

impl Default for AgentSettings {
    fn default() -> Self {
        Self {
            max_same_tool_call_repeats: default_max_same_tool_call_repeats(),
            tool_stall_warning_ms: default_tool_stall_warning_ms(),
            tool_timeout_ms: default_tool_timeout_ms(),
            provider_fallback_chain: Vec::new(),
            provider_failure_threshold: default_provider_failure_threshold(),
            provider_open_cooldown_ms: default_provider_open_cooldown_ms(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeProfileSetting {
    Dev,
    #[default]
    Safe,
    Prod,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProfileOverrides {
    #[serde(default)]
    pub max_retries: Option<u32>,
    #[serde(default)]
    pub base_delay_ms: Option<u64>,
    #[serde(default)]
    pub provider_timeout_ms: Option<u64>,
    #[serde(default)]
    pub tool_stall_warning_ms: Option<u64>,
    #[serde(default)]
    pub tool_timeout_ms: Option<u64>,
    #[serde(default)]
    pub provider_fallback_chain: Option<Vec<String>>,
    #[serde(default)]
    pub provider_failure_threshold: Option<u32>,
    #[serde(default)]
    pub provider_open_cooldown_ms: Option<u64>,
    #[serde(default)]
    pub max_same_tool_call_repeats: Option<u32>,
    #[serde(default)]
    pub command_policy_strict: Option<bool>,
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
fn default_keep_recent_tokens() -> u32 {
    20_000
}
fn default_summary_max_tokens() -> u32 {
    512
}
fn default_max_retries() -> u32 {
    2
}
fn default_base_delay() -> u64 {
    1000
}
fn default_timeout_ms() -> u64 {
    120_000
}
fn default_max_same_tool_call_repeats() -> u32 {
    6
}
fn default_tool_stall_warning_ms() -> u64 {
    8_000
}
fn default_tool_timeout_ms() -> u64 {
    60_000
}
fn default_provider_failure_threshold() -> u32 {
    3
}
fn default_provider_open_cooldown_ms() -> u64 {
    30_000
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
    /// Merge existing on-disk auth with a newer in-memory auth snapshot.
    ///
    /// New entries win for the same provider. Entries for unrelated providers
    /// are preserved so saving one login cannot delete another provider.
    fn merge_with_existing(mut self, existing: AuthConfig) -> Self {
        for token in existing.tokens {
            let replaced_by_token = self.tokens.iter().any(|t| t.provider == token.provider);
            let replaced_by_oauth = self
                .oauth_tokens
                .iter()
                .any(|t| t.provider == token.provider);
            if !replaced_by_token && !replaced_by_oauth {
                self.tokens.push(token);
            }
        }

        for oauth_token in existing.oauth_tokens {
            let replaced_by_oauth = self
                .oauth_tokens
                .iter()
                .any(|t| t.provider == oauth_token.provider);
            let replaced_by_token = self
                .tokens
                .iter()
                .any(|t| t.provider == oauth_token.provider);
            if !replaced_by_oauth && !replaced_by_token {
                self.oauth_tokens.push(oauth_token);
            }
        }

        self
    }

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
    let strategy = match tc.compaction.summarize_with_llm {
        Some(true) => theta_agent_core::CompactionStrategy::Llm,
        Some(false) => theta_agent_core::CompactionStrategy::Textual,
        None => match tc.compaction.strategy {
            CompactionStrategySetting::None => theta_agent_core::CompactionStrategy::None,
            CompactionStrategySetting::Textual => theta_agent_core::CompactionStrategy::Textual,
            CompactionStrategySetting::Llm => theta_agent_core::CompactionStrategy::Llm,
        },
    };
    let runtime_profile = match tc.profile {
        RuntimeProfileSetting::Dev => theta_agent_core::RuntimeProfile::Dev,
        RuntimeProfileSetting::Safe => theta_agent_core::RuntimeProfile::Safe,
        RuntimeProfileSetting::Prod => theta_agent_core::RuntimeProfile::Prod,
    };

    #[derive(Clone)]
    struct ProfileBase {
        max_retries: u32,
        base_delay_ms: u64,
        provider_timeout_ms: u64,
        tool_stall_warning_ms: u64,
        tool_timeout_ms: u64,
        provider_fallback_chain: Vec<String>,
        provider_failure_threshold: u32,
        provider_open_cooldown_ms: u64,
        max_same_tool_call_repeats: u32,
        command_policy_strict: bool,
    }

    let mut base = match tc.profile {
        RuntimeProfileSetting::Dev => ProfileBase {
            max_retries: 1,
            base_delay_ms: 250,
            provider_timeout_ms: 90_000,
            tool_stall_warning_ms: 15_000,
            tool_timeout_ms: 180_000,
            provider_fallback_chain: vec![],
            provider_failure_threshold: 6,
            provider_open_cooldown_ms: 5_000,
            max_same_tool_call_repeats: 10,
            command_policy_strict: false,
        },
        RuntimeProfileSetting::Safe => ProfileBase {
            max_retries: 2,
            base_delay_ms: 1_000,
            provider_timeout_ms: 120_000,
            tool_stall_warning_ms: 8_000,
            tool_timeout_ms: 60_000,
            provider_fallback_chain: vec![],
            provider_failure_threshold: 3,
            provider_open_cooldown_ms: 30_000,
            max_same_tool_call_repeats: 6,
            command_policy_strict: true,
        },
        RuntimeProfileSetting::Prod => ProfileBase {
            max_retries: 4,
            base_delay_ms: 1_500,
            provider_timeout_ms: 120_000,
            tool_stall_warning_ms: 5_000,
            tool_timeout_ms: 45_000,
            provider_fallback_chain: vec![],
            provider_failure_threshold: 2,
            provider_open_cooldown_ms: 60_000,
            max_same_tool_call_repeats: 6,
            command_policy_strict: true,
        },
    };

    // Backward-compatibility: existing top-level settings remain active for safe profile.
    if matches!(tc.profile, RuntimeProfileSetting::Safe) {
        base.max_retries = tc.retry.max_retries;
        base.base_delay_ms = tc.retry.base_delay_ms;
        base.provider_timeout_ms = tc.provider.timeout_ms;
        base.tool_stall_warning_ms = tc.agent.tool_stall_warning_ms;
        base.tool_timeout_ms = tc.agent.tool_timeout_ms;
        base.provider_fallback_chain = tc.agent.provider_fallback_chain.clone();
        base.provider_failure_threshold = tc.agent.provider_failure_threshold;
        base.provider_open_cooldown_ms = tc.agent.provider_open_cooldown_ms;
        base.max_same_tool_call_repeats = tc.agent.max_same_tool_call_repeats;
    }

    if let Some(v) = tc.profile_overrides.max_retries {
        base.max_retries = v;
    }
    if let Some(v) = tc.profile_overrides.base_delay_ms {
        base.base_delay_ms = v;
    }
    if let Some(v) = tc.profile_overrides.provider_timeout_ms {
        base.provider_timeout_ms = v;
    }
    if let Some(v) = tc.profile_overrides.tool_stall_warning_ms {
        base.tool_stall_warning_ms = v;
    }
    if let Some(v) = tc.profile_overrides.tool_timeout_ms {
        base.tool_timeout_ms = v;
    }
    if let Some(v) = &tc.profile_overrides.provider_fallback_chain {
        base.provider_fallback_chain = v.clone();
    }
    if let Some(v) = tc.profile_overrides.provider_failure_threshold {
        base.provider_failure_threshold = v;
    }
    if let Some(v) = tc.profile_overrides.provider_open_cooldown_ms {
        base.provider_open_cooldown_ms = v;
    }
    if let Some(v) = tc.profile_overrides.max_same_tool_call_repeats {
        base.max_same_tool_call_repeats = v;
    }
    if let Some(v) = tc.profile_overrides.command_policy_strict {
        base.command_policy_strict = v;
    }

    // Validation
    if base.tool_timeout_ms < 1_000 {
        tracing::warn!("tool_timeout_ms too low; clamping to 1000ms");
        base.tool_timeout_ms = 1_000;
    }
    if base.provider_timeout_ms < 5_000 {
        tracing::warn!("provider_timeout_ms too low; clamping to 5000ms");
        base.provider_timeout_ms = 5_000;
    }
    if base.provider_failure_threshold == 0 {
        tracing::warn!("provider_failure_threshold=0 is invalid; clamping to 1");
        base.provider_failure_threshold = 1;
    }

    theta_agent_core::AgentLoopConfig {
        runtime_profile,
        max_same_tool_call_repeats: Some(base.max_same_tool_call_repeats),
        compaction: theta_agent_core::CompactionConfig {
            enabled: tc.compaction.enabled,
            reserve_tokens: tc.compaction.reserve_tokens,
            keep_recent_tokens: tc.compaction.keep_recent_tokens,
            strategy,
            summary_max_tokens: tc.compaction.summary_max_tokens,
        },
        retry: theta_agent_core::RetryConfig {
            max_retries: base.max_retries,
            base_delay_ms: base.base_delay_ms,
        },
        provider_timeout_ms: Some(base.provider_timeout_ms),
        tool_watchdog: theta_agent_core::ToolWatchdogConfig {
            stall_warning_ms: base.tool_stall_warning_ms,
            hard_timeout_ms: base.tool_timeout_ms,
        },
        provider_fallback_chain: base.provider_fallback_chain,
        provider_circuit_breaker: theta_agent_core::CircuitBreakerConfig {
            failure_threshold: base.provider_failure_threshold,
            open_cooldown_ms: base.provider_open_cooldown_ms,
        },
        command_policy_strict: base.command_policy_strict,
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

    let auth = if path.exists() {
        let existing = load_auth(Some(&path)).await?;
        auth.clone().merge_with_existing(existing)
    } else {
        auth.clone()
    };

    let contents = serde_json::to_string_pretty(&auth).map_err(|e| ConfigError::Parse {
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
pub(crate) fn theta_dir() -> PathBuf {
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

    #[tokio::test]
    async fn test_save_auth_preserves_unrelated_provider_credentials() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let path = dir.path().join("auth.json");

        let mut existing = AuthConfig::default();
        existing.set_oauth_token(
            "openai-codex",
            "codex-access",
            "codex-refresh",
            now_ms() + 3600_000,
        );
        save_auth(&existing, Some(&path))
            .await
            .expect("initial auth should save");

        let mut incoming = AuthConfig::default();
        incoming.set_token("deepseek", "deepseek-key", None);
        save_auth(&incoming, Some(&path))
            .await
            .expect("merged auth should save");

        let saved = load_auth(Some(&path))
            .await
            .expect("merged auth should load");
        assert_eq!(saved.get_token("deepseek"), Some("deepseek-key".into()));
        assert_eq!(saved.get_token("openai-codex"), Some("codex-access".into()));
    }

    #[tokio::test]
    async fn test_save_auth_replaces_same_provider_across_auth_kinds() {
        let dir = tempfile::tempdir().expect("tempdir should be created");
        let path = dir.path().join("auth.json");

        let mut existing = AuthConfig::default();
        existing.set_oauth_token(
            "openai-codex",
            "old-access",
            "old-refresh",
            now_ms() + 3600_000,
        );
        save_auth(&existing, Some(&path))
            .await
            .expect("initial auth should save");

        let mut incoming = AuthConfig::default();
        incoming.set_token("openai-codex", "manual-token", None);
        save_auth(&incoming, Some(&path))
            .await
            .expect("replacement auth should save");

        let saved = load_auth(Some(&path))
            .await
            .expect("replacement auth should load");
        assert_eq!(saved.get_token("openai-codex"), Some("manual-token".into()));
        assert!(saved.oauth_tokens.is_empty());
    }

    #[test]
    fn test_profile_dev_defaults_are_applied() {
        let cfg = ThetaConfig {
            profile: RuntimeProfileSetting::Dev,
            ..Default::default()
        };
        let ac = to_agent_config(&cfg);
        assert_eq!(ac.retry.max_retries, 1);
        assert_eq!(ac.retry.base_delay_ms, 250);
        assert!(!ac.command_policy_strict);
        assert_eq!(ac.tool_watchdog.hard_timeout_ms, 180_000);
    }

    #[test]
    fn test_profile_prod_defaults_are_applied() {
        let cfg = ThetaConfig {
            profile: RuntimeProfileSetting::Prod,
            ..Default::default()
        };
        let ac = to_agent_config(&cfg);
        assert_eq!(ac.retry.max_retries, 4);
        assert!(ac.command_policy_strict);
        assert_eq!(ac.provider_circuit_breaker.failure_threshold, 2);
    }

    #[test]
    fn test_profile_overrides_take_precedence() {
        let cfg = ThetaConfig {
            profile: RuntimeProfileSetting::Prod,
            profile_overrides: ProfileOverrides {
                max_retries: Some(7),
                command_policy_strict: Some(false),
                tool_timeout_ms: Some(2_000),
                ..Default::default()
            },
            ..Default::default()
        };
        let ac = to_agent_config(&cfg);
        assert_eq!(ac.retry.max_retries, 7);
        assert!(!ac.command_policy_strict);
        assert_eq!(ac.tool_watchdog.hard_timeout_ms, 2_000);
    }
}
