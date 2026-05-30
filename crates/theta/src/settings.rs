//! Persistent session-level settings: last model, thinking level, etc.
//!
//! Stored in `~/.theta/settings.json`. Updated on model switch, thinking
//! level change, and agent creation. Read on startup to restore defaults.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::{Deserialize, Deserializer, Serialize};

/// Last used provider + model pair.
///
/// Stored together so the thinking level can be inferred from
/// `model_thinking_map[provider][model]` without a separate field
/// that drifts out of sync.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LastSession {
    /// Provider identifier ("openai", "deepseek", "openai-codex", etc.).
    pub provider: String,
    /// Model ID ("gpt-5.5", "deepseek-v4-pro", etc.).
    pub model: String,
}

/// Persistent settings stored across sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThetaSettings {
    /// Last used provider + model (for session restore on startup).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_session: Option<LastSession>,

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
    /// The last used model ID, if any.
    pub fn last_model(&self) -> Option<&str> {
        self.last_session.as_ref().map(|s| s.model.as_str())
    }

    /// The thinking level for the last session, inferred from the per-model map.
    /// Returns `None` if there is no last session or no entry in the map.
    pub fn last_thinking(&self) -> Option<&str> {
        let session = self.last_session.as_ref()?;
        self.thinking_for_model(&session.provider, &session.model)
    }

    /// Get the thinking level for a specific provider+model, falling back to
    /// `last_thinking()` (inferred from the last session's map entry).
    pub fn thinking_for_model(&self, provider: &str, model_id: &str) -> Option<&str> {
        self.model_thinking_map
            .get(provider)
            .and_then(|map| map.get(model_id))
            .map(|s| s.as_str())
            .or_else(|| self.last_thinking())
    }

    /// Record the thinking level used for a specific provider+model.
    pub fn set_model_thinking(&mut self, provider: &str, model_id: &str, thinking: &str) {
        self.model_thinking_map
            .entry(provider.to_string())
            .or_default()
            .insert(model_id.to_string(), thinking.to_string());
        // Keep last_session in sync so last_thinking() stays coherent.
        self.set_last_session(provider, model_id);
    }

    /// Store the last used provider+model pair.
    pub fn set_last_session(&mut self, provider: &str, model: &str) {
        self.last_session = Some(LastSession {
            provider: provider.to_string(),
            model: model.to_string(),
        });
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
            last_session: None,
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
/// Handles backward compat: old `last_model`/`last_thinking` fields are
/// migrated into `last_session` when the per-model map contains a match.
pub async fn load_settings() -> ThetaSettings {
    let path = settings_path();
    let contents = match tokio::fs::read_to_string(&path).await {
        Ok(c) => c,
        Err(_) => return ThetaSettings::default(),
    };

    // Pre-process for backward compat: if the JSON has old top-level
    // `last_model` / `last_thinking` fields but no `last_session`, try
    // to inject a `last_session` by searching `model_thinking_map` for
    // a provider that contains the old model_id.
    let migrated = migrate_last_session_fields(&contents);

    let contents = migrated.as_deref().unwrap_or(&contents);
    match serde_json::from_str(contents) {
        Ok(settings) => settings,
        Err(e) => {
            tracing::warn!("Failed to parse settings.json, using defaults: {e}");
            ThetaSettings::default()
        }
    }
}

/// If the JSON contains old `last_model` but no `last_session`, search
/// `model_thinking_map` for a provider whose entry contains that model_id.
/// Returns a modified JSON string with `last_session` injected, or `None`
/// if no migration is needed / possible.
fn migrate_last_session_fields(json: &str) -> Option<String> {
    let mut value: serde_json::Value = serde_json::from_str(json).ok()?;
    let obj = value.as_object_mut()?;

    // Already migrated.
    if obj.contains_key("last_session") {
        return None;
    }

    let old_model = obj.get("last_model")?.as_str()?.to_string();

    // Search model_thinking_map for a provider containing this model.
    let provider = obj
        .get("model_thinking_map")
        .and_then(|m| m.as_object())
        .and_then(|map| {
            map.iter().find_map(|(prov, models)| {
                models
                    .as_object()
                    .and_then(|inner| inner.contains_key(&old_model).then_some(prov.as_str()))
            })
        });

    let prov = provider
        .map(|s| s.to_string())
        .unwrap_or_else(|| "unknown".to_string());

    // Inject last_session and remove old fields.
    obj.insert(
        "last_session".to_string(),
        serde_json::json!({"provider": &prov, "model": &old_model}),
    );
    obj.remove("last_model");
    obj.remove("last_thinking");

    tracing::info!("Migrated settings.json: last_model={old_model} → last_session.provider={prov}");

    serde_json::to_string(&value).ok()
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

    // ── Backward compat: old last_model / last_thinking fields ──

    #[test]
    fn old_last_model_migrates_to_last_session() {
        let json = r#"{
            "last_model": "gpt-5.5",
            "last_thinking": "high",
            "model_thinking_map": {
                "openai": {"gpt-5.5": "high"},
                "deepseek": {"deepseek-v4-pro": "max"}
            },
            "favorite_models": ["o3"]
        }"#;
        // Simulate what load_settings does internally.
        let migrated = migrate_last_session_fields(json).expect("should migrate");
        let settings: ThetaSettings =
            serde_json::from_str(&migrated).expect("migrated JSON should parse");
        let session = settings.last_session.as_ref().unwrap();
        assert_eq!(session.provider, "openai");
        assert_eq!(session.model, "gpt-5.5");
        assert_eq!(settings.last_thinking(), Some("high"));
        assert_eq!(settings.favorite_models, vec!["o3"]);
        // model_thinking_map preserved.
        assert_eq!(
            settings
                .model_thinking_map
                .get("openai")
                .and_then(|m| m.get("gpt-5.5")),
            Some(&"high".to_string())
        );
    }

    #[test]
    fn old_last_model_without_map_entry_uses_unknown_provider() {
        let json = r#"{
            "last_model": "some-gpu-model",
            "model_thinking_map": {},
            "favorite_models": ["o3"]
        }"#;
        let migrated = migrate_last_session_fields(json).expect("should migrate");
        let settings: ThetaSettings =
            serde_json::from_str(&migrated).expect("migrated JSON should parse");
        let session = settings.last_session.as_ref().unwrap();
        assert_eq!(session.provider, "unknown");
        assert_eq!(session.model, "some-gpu-model");
        // No map entry → last_thinking is None.
        assert_eq!(settings.last_thinking(), None);
    }

    #[test]
    fn already_migrated_settings_load_directly() {
        let json = r#"{
            "last_session": {"provider": "deepseek", "model": "deepseek-v4-pro"},
            "model_thinking_map": {
                "deepseek": {"deepseek-v4-pro": "max"}
            },
            "favorite_models": ["gpt-5.5"]
        }"#;
        // No migration needed — should be None.
        assert!(migrate_last_session_fields(json).is_none());
        let settings: ThetaSettings =
            serde_json::from_str(json).expect("new format should parse directly");
        assert_eq!(settings.last_model(), Some("deepseek-v4-pro"));
        assert_eq!(settings.last_thinking(), Some("max"));
        assert_eq!(settings.favorite_models, vec!["gpt-5.5"]);
    }

    #[test]
    fn no_last_model_does_not_migrate() {
        let json = r#"{"favorite_models": ["o3"]}"#;
        assert!(migrate_last_session_fields(json).is_none());
    }

    // ── model_thinking_map backward compat ──

    #[test]
    fn old_flat_model_thinking_map_does_not_crash() {
        let json = r#"{
            "last_session": {"provider": "openai", "model": "gpt-5.5"},
            "model_thinking_map": {
                "gpt-5.5": "high",
                "deepseek-v4-pro": "max"
            },
            "favorite_models": ["gpt-5.5", "o3"],
            "disabled_models": ["gpt-5-nano"]
        }"#;
        let settings: ThetaSettings = serde_json::from_str(json).expect("old format should parse");
        // Old flat map is discarded, but nothing else is lost.
        assert!(settings.model_thinking_map.is_empty());
        assert_eq!(settings.last_model(), Some("gpt-5.5"));
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
        let json = r#"{
            "last_session": {"provider": "openai", "model": "gpt-5.5"},
            "favorite_models": ["o3"]
        }"#;
        let settings: ThetaSettings =
            serde_json::from_str(json).expect("missing field should parse");
        assert!(settings.model_thinking_map.is_empty());
        assert_eq!(settings.favorite_models, vec!["o3"]);
    }

    // ── last_thinking inference ──

    #[test]
    fn last_thinking_inferred_from_map() {
        let mut settings = ThetaSettings::default();
        settings.set_model_thinking("openai", "gpt-5.5", "high");
        settings.set_model_thinking("deepseek", "deepseek-v4-pro", "max");
        // Last session should be set to the most recent set_model_thinking call.
        assert_eq!(settings.last_thinking(), Some("max"));
        // Check the per-model lookup.
        assert_eq!(
            settings.thinking_for_model("openai", "gpt-5.5"),
            Some("high")
        );
        assert_eq!(
            settings.thinking_for_model("deepseek", "deepseek-v4-pro"),
            Some("max")
        );
    }

    #[test]
    fn thinking_for_model_falls_back_to_last_thinking() {
        let mut settings = ThetaSettings::default();
        settings.set_model_thinking("openai", "gpt-5.5", "high");
        // Same provider, different model — no entry, falls back to last_thinking.
        assert_eq!(
            settings.thinking_for_model("openai", "gpt-5-mini"),
            Some("high")
        );
        // Different provider entirely — still falls back.
        assert_eq!(
            settings.thinking_for_model("deepseek", "deepseek-v4-pro"),
            Some("high")
        );
    }
}
