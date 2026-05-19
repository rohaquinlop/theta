//! Persistent session-level settings: last model, thinking level, etc.
//!
//! Stored in `~/.theta/settings.json`. Updated on model switch, thinking
//! level change, and agent creation. Read on startup to restore defaults.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Persistent settings stored across sessions.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThetaSettings {
    /// Last used model ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_model: Option<String>,

    /// Last used thinking level.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_thinking: Option<String>,
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
