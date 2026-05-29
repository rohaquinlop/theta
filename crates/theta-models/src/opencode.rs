//! OpenCode Zen model definitions.
//!
//! Models are fetched dynamically from the OpenCode Zen API
//! (https://opencode.ai/zen/v1/models) at catalog construction time.
//!
//! The Zen API is an OpenAI-compatible endpoint at:
//!   https://opencode.ai/zen/v1/chat/completions
//!
//! All models from the API are included, including free tier models.

use serde::Deserialize;
use std::collections::HashMap;
use theta_ai::model::{Model, ModelCompat};
use theta_ai::types::{Api, Modality, Provider, ThinkingLevel};

// ── Zen API response types ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ZenModelsList {
    data: Vec<ZenModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ZenModelEntry {
    id: String,
}

// ── Model capability detection ────────────────────────────────────────

/// Models from these provider families support reasoning/thinking.
/// All others do not and must not receive `reasoning_effort` in requests.
pub fn supports_reasoning(id: &str) -> bool {
    id.starts_with("gpt-")
        || id.starts_with("claude-")
        || id.starts_with("gemini-")
        || id.starts_with("deepseek-")
        || id.starts_with("mimo-")
        || id.starts_with("qwen")
}

fn no_reasoning_map() -> HashMap<ThinkingLevel, Option<String>> {
    [
        (ThinkingLevel::Off, None),
        (ThinkingLevel::Minimal, None),
        (ThinkingLevel::Low, None),
        (ThinkingLevel::Medium, None),
        (ThinkingLevel::High, None),
        (ThinkingLevel::XHigh, None),
    ]
    .into()
}

fn display_name(id: &str) -> String {
    // Standard kebab→Title Case conversion plus known overrides.
    id.split('-')
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn model_from_entry(entry: &ZenModelEntry) -> Model {
    let reasoning = supports_reasoning(&entry.id);
    let thinking_level_map = if reasoning {
        [
            (ThinkingLevel::Off, None),
            (ThinkingLevel::Minimal, Some("minimal".into())),
            (ThinkingLevel::Low, Some("low".into())),
            (ThinkingLevel::Medium, Some("medium".into())),
            (ThinkingLevel::High, Some("high".into())),
            (ThinkingLevel::XHigh, Some("max".into())),
        ]
        .into()
    } else {
        no_reasoning_map()
    };
    let mut compat = ModelCompat::for_opencode();
    if !reasoning {
        compat.thinking_format = None;
    }
    Model {
        id: entry.id.clone(),
        name: display_name(&entry.id),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenCode,
        base_url: "https://opencode.ai/zen".into(),
        reasoning,
        thinking_level_map,
        input: vec![Modality::Text],
        context_window: 200_000,
        max_tokens: 64_000,
        compat,
    }
}

/// Fetch the current list of OpenCode Zen models from the API.
///
/// When `api_key` is provided, Zen filters the response to only
/// models enabled in the user's workspace. Without a key, all
/// public models are returned.
pub async fn fetch_models(api_key: Option<&str>) -> Vec<Model> {
    let url = "https://opencode.ai/zen/v1/models";
    let mut req = reqwest::Client::new().get(url);
    if let Some(key) = api_key {
        req = req.header("Authorization", format!("Bearer {key}"));
    }
    match req.send().await {
        Ok(response) => match response.json::<ZenModelsList>().await {
            Ok(list) => list.data.iter().map(model_from_entry).collect(),
            Err(e) => {
                tracing::warn!("Failed to parse OpenCode Zen models: {e}");
                Vec::new()
            }
        },
        Err(e) => {
            tracing::warn!("Failed to fetch OpenCode Zen models: {e}");
            Vec::new()
        }
    }
}

/// Static fallback model (used when network is unavailable).
pub fn models() -> Vec<Model> {
    vec![Model {
        id: "opencode".into(),
        name: "OpenCode Zen".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenCode,
        base_url: "https://opencode.ai/zen".into(),
        reasoning: true,
        thinking_level_map: [
            (ThinkingLevel::Off, None),
            (ThinkingLevel::Minimal, Some("minimal".into())),
            (ThinkingLevel::Low, Some("low".into())),
            (ThinkingLevel::Medium, Some("medium".into())),
            (ThinkingLevel::High, Some("high".into())),
            (ThinkingLevel::XHigh, Some("max".into())),
        ]
        .into(),
        input: vec![Modality::Text],
        context_window: 200_000,
        max_tokens: 64_000,
        compat: ModelCompat::for_opencode(),
    }]
}
