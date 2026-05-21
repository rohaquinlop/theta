//! Persistent session-level settings: last model, thinking level, etc.
//!
//! Stored in `~/.theta/settings.json`. Updated on model switch, thinking
//! level change, and agent creation. Read on startup to restore defaults.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Persistent settings stored across sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThetaSettings {
    /// Last used model ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_model: Option<String>,

    /// Last used thinking level.
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

impl Default for ThetaSettings {
    fn default() -> Self {
        Self {
            last_model: None,
            last_thinking: None,
            steering_mode: default_steering_mode(),
            follow_up_mode: default_follow_up_mode(),
            transport_preference: default_transport_preference(),
            show_thinking: default_show_thinking(),
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
