//! Persistent session-level settings: last model, thinking level, etc.
//!
//! Stored in `~/.theta/settings.json`. Updated on model switch, thinking
//! level change, and agent creation. Read on startup to restore defaults.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Persistent settings stored across sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThetaSettings {
    /// Last used model ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_model: Option<String>,

    /// Per-provider, per-model thinking level map.
    /// Outer key: provider string ("openai", "deepseek", etc.),
    /// inner key: model_id, value: thinking level.
    /// Enables restoring the last used thinking level when switching models
    /// across different providers that may share model IDs.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub model_thinking_map: HashMap<String, HashMap<String, String>>,

    /// Last used thinking level (global fallback, kept for backward compat).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_thinking: Option<String>,

    /// Enter behavior while streaming: "steer" or "follow-up".
    #[serde(default = "default_steering_mode")]
    pub steering_mode: String,

    /// Alt+Enter behavior while streaming: "follow-up" or "steer".
    #[serde(default = "default_follow_up_mode")]
    pub follow_up_mode: String,

    /// Transport preference hint: "auto", "http", "sse".
    #[serde(default = "default_transport_preference")]
    pub transport_preference: String,

    /// Show thinking by default in UI.
    #[serde(default = "default_show_thinking")]
    pub show_thinking: bool,

    /// Tool progress update frequency in Hz.
    #[serde(default = "default_tool_progress_hz")]
    pub tool_progress_hz: u64,

    /// Enter behavior in editor: "send" or "newline".
    #[serde(default = "default_enter_behavior")]
    pub enter_behavior: String,

    /// Max context window in tokens. `None` disables the cap
    /// (uses the model's full context window). `Some(n)` caps at n tokens.
    /// Default is 250,000 — most LLMs perform better below this.
    #[serde(default = "default_max_context_window")]
    pub max_context_window: Option<u32>,

    /// Model IDs to hide from the model selector (e.g. models disabled
    /// in your Zen workspace). Add model IDs here to filter them out.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub disabled_models: Vec<String>,

    /// Favorite model IDs — shown in a pinned section at the top of the model selector.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub favorite_models: Vec<String>,

    /// MiMo cluster base URL selected by the user (for token-plan users).
    /// One of the regional endpoints: cn, sgp, ams.
    /// Overrides the MIMO_BASE_URL env var when set.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mimo_cluster_url: Option<String>,
}

fn default_steering_mode() -> String {
    "follow-up".to_string()
}

fn default_follow_up_mode() -> String {
    "steer".to_string()
}

fn default_transport_preference() -> String {
    "auto".to_string()
}

const fn default_show_thinking() -> bool {
    true
}

const fn default_tool_progress_hz() -> u64 {
    20
}

fn default_enter_behavior() -> String {
    "send".to_string()
}

const fn default_max_context_window() -> Option<u32> {
    Some(250_000)
}

impl ThetaSettings {
    /// Get the thinking level for a specific provider+model, falling back to `last_thinking`.
    pub fn thinking_for_model(&self, provider: &str, model_id: &str) -> Option<&str> {
        self.model_thinking_map
            .get(provider)
            .and_then(|map| map.get(model_id))
            .map(|s| s.as_str())
            .or(self.last_thinking.as_deref())
    }

    /// Record the thinking level used for a specific provider+model.
    pub fn set_model_thinking(&mut self, provider: &str, model_id: &str, thinking: &str) {
        self.model_thinking_map
            .entry(provider.to_string())
            .or_default()
            .insert(model_id.to_string(), thinking.to_string());
        // Keep last_thinking in sync as a global fallback.
        self.last_thinking = Some(thinking.to_string());
    }

    /// Check if a model is in the favorites list.
    pub fn is_favorite_model(&self, model_id: &str) -> bool {
        self.favorite_models.iter().any(|id| id == model_id)
    }

    /// Toggle a model in the favorites list. Returns true if now favorited.
    pub fn toggle_favorite_model(&mut self, model_id: &str) -> bool {
        if let Some(pos) = self.favorite_models.iter().position(|id| id == model_id) {
            self.favorite_models.remove(pos);
            false
        } else {
            self.favorite_models.push(model_id.to_string());
            true
        }
    }
}

impl Default for ThetaSettings {
    fn default() -> Self {
        Self {
            last_model: None,
            last_thinking: None,
            model_thinking_map: HashMap::new(),
            steering_mode: default_steering_mode(),
            follow_up_mode: default_follow_up_mode(),
            transport_preference: default_transport_preference(),
            show_thinking: default_show_thinking(),
            tool_progress_hz: default_tool_progress_hz(),
            enter_behavior: default_enter_behavior(),
            max_context_window: default_max_context_window(),
            disabled_models: Vec::new(),
            favorite_models: Vec::new(),
            mimo_cluster_url: None,
        }
    }
}

/// Load settings from `~/.theta/settings.json`.
/// Returns default if the file doesn't exist or can't be parsed.
pub async fn load_settings() -> ThetaSettings {
    let path = settings_path();
    match tokio::fs::read_to_string(&path).await {
        Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
        Err(_) => ThetaSettings::default(),
    }
}

/// Save settings to `~/.theta/settings.json`.
pub async fn save_settings(settings: &ThetaSettings) -> Result<(), SettingsError> {
    let path = settings_path();
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(SettingsError::Write)?;
    }
    let contents = serde_json::to_string_pretty(settings)
        .map_err(|e| SettingsError::Serialize(e.to_string()))?;
    tokio::fs::write(&path, contents)
        .await
        .map_err(SettingsError::Write)?;
    Ok(())
}

fn settings_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".theta")
        .join("settings.json")
}

#[derive(Debug, thiserror::Error)]
pub enum SettingsError {
    #[error("failed to write settings: {0}")]
    Write(std::io::Error),

    #[error("failed to serialize settings: {0}")]
    Serialize(String),
}
