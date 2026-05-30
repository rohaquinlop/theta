//! Persistent session-level settings: last model, thinking level, etc.
//!
//! Stored in `~/.theta/settings.json`. Updated on model switch, thinking
//! level change, and agent creation. Read on startup to restore defaults.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize};

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
    #[serde(
        default,
        skip_serializing_if = "HashMap::is_empty",
        deserialize_with = "deserialize_model_thinking_map"
    )]
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

/// Custom deserializer for `model_thinking_map` that accepts both:
/// - New format: `{"openai": {"gpt-5": "high"}}` (provider → model_id → thinking)
/// - Old format: `{"gpt-5": "high"}` (flat model_id → thinking)
///
/// The old format is silently discarded (we can't know the provider),
/// but critically this does NOT fail the entire struct parse — preserving
/// favorites, disabled models, and all other settings fields.
fn deserialize_model_thinking_map<'de, D>(
    deserializer: D,
) -> Result<HashMap<String, HashMap<String, String>>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    match value {
        serde_json::Value::Object(map) => {
            // If the first value is a string, this is the old flat format.
            // Silently return empty map rather than failing.
            if map.values().any(|v| v.is_string()) {
                tracing::warn!(
                    "settings.json contains old flat model_thinking_map; discarding (re-save will upgrade)"
                );
                return Ok(HashMap::new());
            }
            // Otherwise deserialize as the new nested format.
            HashMap::deserialize(serde_json::Value::Object(map)).map_err(serde::de::Error::custom)
        }
        _ => Ok(HashMap::new()),
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
        Ok(contents) => match serde_json::from_str(&contents) {
            Ok(settings) => settings,
            Err(e) => {
                tracing::warn!("Failed to parse settings.json, using defaults: {e}");
                ThetaSettings::default()
            }
        },
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn old_flat_model_thinking_map_does_not_crash() {
        // Old format (flat model_id → thinking) must not fail the entire struct parse.
        let json = r#"{
            "last_model": "gpt-5.5",
            "model_thinking_map": {
                "gpt-5.5": "high",
                "deepseek-v4-pro": "max"
            },
            "last_thinking": "medium",
            "favorite_models": ["gpt-5.5", "o3"],
            "disabled_models": ["gpt-5-nano"]
        }"#;
        let settings: ThetaSettings = serde_json::from_str(json).expect("old format should parse");
        // Old map is discarded (can't know provider), but nothing else is lost.
        assert!(settings.model_thinking_map.is_empty());
        assert_eq!(settings.last_model.as_deref(), Some("gpt-5.5"));
        assert_eq!(settings.last_thinking.as_deref(), Some("medium"));
        assert_eq!(settings.favorite_models, vec!["gpt-5.5", "o3"]);
        assert_eq!(settings.disabled_models, vec!["gpt-5-nano"]);
    }

    #[test]
    fn new_nested_model_thinking_map_parses_correctly() {
        let json = r#"{
            "model_thinking_map": {
                "openai": {"gpt-5.5": "high"},
                "deepseek": {"deepseek-v4-pro": "max"}
            },
            "favorite_models": ["gpt-5.5"]
        }"#;
        let settings: ThetaSettings = serde_json::from_str(json).expect("new format should parse");
        assert_eq!(
            settings
                .model_thinking_map
                .get("openai")
                .and_then(|m| m.get("gpt-5.5")),
            Some(&"high".to_string())
        );
        assert_eq!(
            settings
                .model_thinking_map
                .get("deepseek")
                .and_then(|m| m.get("deepseek-v4-pro")),
            Some(&"max".to_string())
        );
        assert_eq!(settings.favorite_models, vec!["gpt-5.5"]);
    }

    #[test]
    fn missing_model_thinking_map_defaults_to_empty() {
        let json = r#"{"last_model": "gpt-5.5", "favorite_models": ["o3"]}"#;
        let settings: ThetaSettings =
            serde_json::from_str(json).expect("missing field should parse");
        assert!(settings.model_thinking_map.is_empty());
        assert_eq!(settings.favorite_models, vec!["o3"]);
    }
}
