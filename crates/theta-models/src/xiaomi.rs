//! Xiaomi MiMo model definitions.
//!
//! Models are fetched dynamically from the MiMo API
//! (https://api.xiaomimimo.com/v1/models) at runtime.
//!
//! The MiMo API is an OpenAI-compatible endpoint at:
//!   https://api.xiaomimimo.com/v1/chat/completions
//!
//! Two plan types with different endpoints:
//!   - Pay-as-you-go: api.xiaomimimo.com (keys start with sk-)
//!   - Token Plan:    token-plan-{region}.xiaomimimo.com (keys start with tp-)
//!
//! Static fallback models are provided when the network is unavailable.

use serde::Deserialize;
use std::collections::HashMap;
use theta_ai::model::{Model, ModelCompat};
use theta_ai::types::{Api, Modality, Provider, ThinkingLevel};

// ── MiMo API response types ──────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct MiMoModelsList {
    data: Vec<MiMoModelEntry>,
}

#[derive(Debug, Deserialize)]
struct MiMoModelEntry {
    id: String,
}

// ── Model capability detection ────────────────────────────────────────

/// All MiMo models support reasoning/thinking.
fn supports_reasoning(id: &str) -> bool {
    id.starts_with("mimo-v2")
}

/// MiMo only supports binary on/off thinking. All non-Off levels
/// enable thinking; there is no reasoning_effort granularity.
fn reasoning_map() -> HashMap<ThinkingLevel, Option<String>> {
    [
        (ThinkingLevel::Off, None),
        (ThinkingLevel::Minimal, Some("enabled".into())),
        (ThinkingLevel::Low, Some("enabled".into())),
        (ThinkingLevel::Medium, Some("enabled".into())),
        (ThinkingLevel::High, Some("enabled".into())),
        (ThinkingLevel::XHigh, Some("enabled".into())),
    ]
    .into()
}

fn display_name(id: &str) -> String {
    // "mimo-v2.5-pro" -> "MiMo V2.5 Pro"
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

/// Infer context window and max output from model ID.
fn model_limits(id: &str) -> (u32, u32) {
    // Pro/Omni series: 1M context, 128K output.
    // Flash series: 256K context, 64K output.
    // TTS models: 8K context, 8K output.
    if id.contains("flash") {
        (256_000, 64_000)
    } else if id.contains("tts") {
        (8_000, 8_000)
    } else {
        (1_000_000, 128_000)
    }
}

/// Infer input modalities from model ID.
fn model_modalities(id: &str) -> Vec<Modality> {
    if id.contains("omni") || id == "mimo-v2.5" {
        vec![Modality::Text, Modality::Image]
    } else {
        vec![Modality::Text]
    }
}

fn model_from_entry(entry: &MiMoModelEntry) -> Option<Model> {
    // Skip TTS models — theta only supports text generation.
    if entry.id.contains("tts") {
        return None;
    }

    let reasoning = supports_reasoning(&entry.id);
    let (context_window, max_tokens) = model_limits(&entry.id);
    let compat = ModelCompat::for_xiaomi();

    Some(Model {
        id: entry.id.clone(),
        name: display_name(&entry.id),
        api: Api::OpenAiCompletions,
        provider: Provider::XiaomiMiMo,
        base_url: "https://api.xiaomimimo.com".into(),
        reasoning,
        thinking_level_map: reasoning_map(),
        input: model_modalities(&entry.id),
        context_window,
        max_tokens,
        compat,
    })
}

/// Fetch the current list of MiMo models from the API.
///
/// Returns only text generation models (TTS excluded).
/// When the network is unavailable, returns an empty vec
/// and the caller falls back to static definitions.
pub async fn fetch_models(api_key: Option<&str>) -> Vec<Model> {
    // Detect base URL from key prefix:
    //   tp-* → token plan dedicated endpoint (with MIMO_BASE_URL override)
    //   sk-* or None → pay-as-you-go
    let base_url = if let Some(key) = api_key
        && key.starts_with("tp-")
    {
        if let Ok(env_url) = std::env::var("MIMO_BASE_URL") {
            env_url
        } else {
            "https://token-plan-sgp.xiaomimimo.com".to_string()
        }
    } else {
        "https://api.xiaomimimo.com".to_string()
    };
    let url = format!("{base_url}/v1/models");
    let mut req = reqwest::Client::new().get(url);
    if let Some(key) = api_key {
        // MiMo uses api-key header, not Authorization: Bearer.
        req = req.header("api-key", key);
    }
    match req.send().await {
        Ok(response) => match response.json::<MiMoModelsList>().await {
            Ok(list) => list.data.iter().filter_map(model_from_entry).collect(),
            Err(e) => {
                tracing::warn!("Failed to parse MiMo models: {e}");
                Vec::new()
            }
        },
        Err(e) => {
            tracing::warn!("Failed to fetch MiMo models: {e}");
            Vec::new()
        }
    }
}

/// Static fallback models (used when network is unavailable).
pub fn models() -> Vec<Model> {
    vec![
        mimo_v2_5_pro(),
        mimo_v2_pro(),
        mimo_v2_5(),
        mimo_v2_omni(),
        mimo_v2_flash(),
    ]
}

fn mimo_v2_5_pro() -> Model {
    Model {
        id: "mimo-v2.5-pro".into(),
        name: "MiMo V2.5 Pro".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::XiaomiMiMo,
        base_url: "https://api.xiaomimimo.com".into(),
        reasoning: true,
        thinking_level_map: reasoning_map(),
        input: vec![Modality::Text],
        context_window: 1_000_000,
        max_tokens: 128_000,
        compat: ModelCompat::for_xiaomi(),
    }
}

fn mimo_v2_pro() -> Model {
    Model {
        id: "mimo-v2-pro".into(),
        name: "MiMo V2 Pro".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::XiaomiMiMo,
        base_url: "https://api.xiaomimimo.com".into(),
        reasoning: true,
        thinking_level_map: reasoning_map(),
        input: vec![Modality::Text],
        context_window: 1_000_000,
        max_tokens: 128_000,
        compat: ModelCompat::for_xiaomi(),
    }
}

fn mimo_v2_5() -> Model {
    Model {
        id: "mimo-v2.5".into(),
        name: "MiMo V2.5".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::XiaomiMiMo,
        base_url: "https://api.xiaomimimo.com".into(),
        reasoning: true,
        thinking_level_map: reasoning_map(),
        input: vec![Modality::Text, Modality::Image],
        context_window: 1_000_000,
        max_tokens: 128_000,
        compat: ModelCompat::for_xiaomi(),
    }
}

fn mimo_v2_omni() -> Model {
    Model {
        id: "mimo-v2-omni".into(),
        name: "MiMo V2 Omni".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::XiaomiMiMo,
        base_url: "https://api.xiaomimimo.com".into(),
        reasoning: true,
        thinking_level_map: reasoning_map(),
        input: vec![Modality::Text, Modality::Image],
        context_window: 256_000,
        max_tokens: 128_000,
        compat: ModelCompat::for_xiaomi(),
    }
}

fn mimo_v2_flash() -> Model {
    Model {
        id: "mimo-v2-flash".into(),
        name: "MiMo V2 Flash".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::XiaomiMiMo,
        base_url: "https://api.xiaomimimo.com".into(),
        reasoning: true,
        thinking_level_map: reasoning_map(),
        input: vec![Modality::Text],
        context_window: 256_000,
        max_tokens: 64_000,
        compat: ModelCompat::for_xiaomi(),
    }
}
