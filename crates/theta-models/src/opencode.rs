//! OpenCode Zen model definitions.
//!
//! Models are fetched dynamically from the OpenCode Zen API
//! (https://opencode.ai/zen/v1/models) at catalog construction time.
//!
//! The Zen API is an OpenAI-compatible endpoint at:
//!   https://opencode.ai/zen/v1/chat/completions
//!
//! Free models are excluded because they are rate-limited for all
//! users (including paying Zen subscribers) and cause unnecessary
//! retry noise. Users should use paid Zen models instead.

use serde::Deserialize;
use theta_ai::model::{Model, ModelCompat};
use theta_ai::types::{Api, Modality, ModelCost, Provider, ThinkingLevel};

// ── Zen API response types ────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ZenModelsList {
    data: Vec<ZenModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ZenModelEntry {
    id: String,
}

// ── Free model IDs to exclude (rate-limited even for subscribers) ─────

const FREE_MODEL_IDS: &[&str] = &[
    "big-pickle",
    "deepseek-v4-flash-free",
    "nemotron-3-super-free",
    "qwen3.6-plus-free",
];

// ── Known paid model costs per 1M tokens (from Zen pricing page) ──

fn known_cost(id: &str) -> ModelCost {
    match id {
        "claude-opus-4-7" | "claude-opus-4-6" | "claude-opus-4-5" => ModelCost {
            input: 5.0,
            output: 25.0,
            cache_read: 0.5,
            cache_write: 6.25,
        },
        "claude-opus-4-1" => ModelCost {
            input: 15.0,
            output: 75.0,
            cache_read: 1.5,
            cache_write: 18.75,
        },
        "claude-sonnet-4-6" | "claude-sonnet-4-5" | "claude-sonnet-4" => ModelCost {
            input: 3.0,
            output: 15.0,
            cache_read: 0.3,
            cache_write: 3.75,
        },
        "claude-haiku-4-5" => ModelCost {
            input: 1.0,
            output: 5.0,
            cache_read: 0.1,
            cache_write: 1.25,
        },
        "gemini-3.5-flash" => ModelCost {
            input: 1.5,
            output: 9.0,
            cache_read: 0.15,
            cache_write: 0.0,
        },
        "gemini-3.1-pro" => ModelCost {
            input: 2.0,
            output: 12.0,
            cache_read: 0.2,
            cache_write: 0.0,
        },
        "gemini-3-flash" => ModelCost {
            input: 0.5,
            output: 3.0,
            cache_read: 0.05,
            cache_write: 0.0,
        },
        "gpt-5.5" | "gpt-5.4" => ModelCost {
            input: 2.5,
            output: 15.0,
            cache_read: 0.25,
            cache_write: 0.0,
        },
        "gpt-5.5-pro" | "gpt-5.4-pro" => ModelCost {
            input: 30.0,
            output: 180.0,
            cache_read: 30.0,
            cache_write: 0.0,
        },
        "gpt-5.4-mini" => ModelCost {
            input: 0.75,
            output: 4.5,
            cache_read: 0.075,
            cache_write: 0.0,
        },
        "gpt-5.4-nano" => ModelCost {
            input: 0.2,
            output: 1.25,
            cache_read: 0.02,
            cache_write: 0.0,
        },
        "gpt-5.3-codex-spark" | "gpt-5.3-codex" | "gpt-5.2" | "gpt-5.2-codex" => ModelCost {
            input: 1.75,
            output: 14.0,
            cache_read: 0.175,
            cache_write: 0.0,
        },
        "gpt-5.1" | "gpt-5.1-codex" | "gpt-5" | "gpt-5-codex" => ModelCost {
            input: 1.07,
            output: 8.5,
            cache_read: 0.107,
            cache_write: 0.0,
        },
        "gpt-5.1-codex-max" => ModelCost {
            input: 1.25,
            output: 10.0,
            cache_read: 0.125,
            cache_write: 0.0,
        },
        "gpt-5.1-codex-mini" => ModelCost {
            input: 0.25,
            output: 2.0,
            cache_read: 0.025,
            cache_write: 0.0,
        },
        "gpt-5-nano" => ModelCost {
            input: 0.05,
            output: 0.4,
            cache_read: 0.005,
            cache_write: 0.0,
        },
        "grok-build-0.1" => ModelCost {
            input: 1.0,
            output: 2.0,
            cache_read: 0.2,
            cache_write: 0.0,
        },
        "glm-5.1" => ModelCost {
            input: 1.4,
            output: 4.4,
            cache_read: 0.26,
            cache_write: 0.0,
        },
        "glm-5" => ModelCost {
            input: 1.0,
            output: 3.2,
            cache_read: 0.2,
            cache_write: 0.0,
        },
        "minimax-m2.7" | "minimax-m2.5" => ModelCost {
            input: 0.3,
            output: 1.2,
            cache_read: 0.06,
            cache_write: 0.375,
        },
        "kimi-k2.6" => ModelCost {
            input: 0.95,
            output: 4.0,
            cache_read: 0.16,
            cache_write: 0.0,
        },
        "kimi-k2.5" => ModelCost {
            input: 0.6,
            output: 3.0,
            cache_read: 0.1,
            cache_write: 0.0,
        },
        "qwen3.6-plus" => ModelCost {
            input: 0.5,
            output: 3.0,
            cache_read: 0.05,
            cache_write: 0.625,
        },
        "qwen3.5-plus" => ModelCost {
            input: 0.2,
            output: 1.2,
            cache_read: 0.02,
            cache_write: 0.25,
        },
        _ => ModelCost::default(),
    }
}

fn is_free(id: &str) -> bool {
    FREE_MODEL_IDS.contains(&id)
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
    Model {
        id: entry.id.clone(),
        name: display_name(&entry.id),
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
        cost: known_cost(&entry.id),
        context_window: 200_000,
        max_tokens: 64_000,
        compat: ModelCompat::for_opencode(),
    }
}

/// Fetch the current list of OpenCode Zen models from the API,
/// excluding free/rate-limited models.
pub async fn fetch_models() -> Vec<Model> {
    let url = "https://opencode.ai/zen/v1/models";
    match reqwest::get(url).await {
        Ok(response) => match response.json::<ZenModelsList>().await {
            Ok(list) => list
                .data
                .iter()
                .filter(|e| !is_free(&e.id))
                .map(model_from_entry)
                .collect(),
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
        cost: ModelCost::default(),
        context_window: 200_000,
        max_tokens: 64_000,
        compat: ModelCompat::for_opencode(),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_models_valid() {
        for m in models() {
            assert!(!m.id.is_empty());
            assert_eq!(m.provider, Provider::OpenCode);
            assert!(m.context_window > 0);
        }
    }

    #[test]
    fn test_free_models_are_excluded() {
        for id in FREE_MODEL_IDS {
            assert!(is_free(id), "free model {id} should be recognized");
        }
        assert!(!is_free("gpt-5.5"));
    }

    #[test]
    fn test_paid_models_have_cost() {
        let cost = known_cost("gpt-5.5");
        assert!(cost.input > 0.0);
        assert!(cost.output > 0.0);
    }
}
