//! OpenAI Codex model definitions.
//!
//! Codex gives ChatGPT Plus subscribers API access to OpenAI models
//! authenticated via session token.

use michin_ai::model::{Model, ModelCompat};
use michin_ai::types::{Api, Modality, Provider, ThinkingLevel};

/// Return all codex-enabled models.
pub fn models() -> Vec<Model> {
    vec![
        codex_model("gpt-5.5", "GPT-5.5 (Codex)", 272_000, 128_000, false),
        codex_model("gpt-5.3-codex", "GPT-5.3 Codex", 272_000, 128_000, false),
        codex_model(
            "gpt-5.5-instant",
            "GPT-5.5 Instant (Codex)",
            272_000,
            128_000,
            false,
        ),
        codex_model("gpt-5", "GPT-5 (Codex)", 272_000, 128_000, false),
        codex_model("gpt-5-mini", "GPT-5 Mini (Codex)", 272_000, 128_000, false),
        codex_model(
            "gpt-5-chat-latest",
            "GPT-5 Chat Latest (Codex)",
            272_000,
            128_000,
            false,
        ),
        codex_model("gpt-4.1", "GPT-4.1 (Codex)", 200_000, 100_000, false),
        codex_model(
            "gpt-4.1-mini",
            "GPT-4.1 Mini (Codex)",
            200_000,
            100_000,
            false,
        ),
        codex_model("gpt-4o", "GPT-4o (Codex)", 128_000, 16_384, false),
        codex_model("gpt-4o-mini", "GPT-4o Mini (Codex)", 128_000, 16_384, false),
        codex_model("o4", "o4 (Codex)", 200_000, 100_000, true),
        codex_model("o4-mini", "o4-mini (Codex)", 200_000, 100_000, true),
        codex_model("o3", "o3 (Codex)", 200_000, 100_000, true),
        codex_model("o3-mini", "o3-mini (Codex)", 200_000, 100_000, true),
        codex_model("o1", "o1 (Codex)", 200_000, 100_000, true),
        codex_model("o1-mini", "o1-mini (Codex)", 200_000, 100_000, true),
    ]
}

fn codex_model(
    id: &str,
    name: &str,
    context_window: u32,
    max_tokens: u32,
    o_series: bool,
) -> Model {
    Model {
        id: id.into(),
        name: name.into(),
        api: Api::OpenAiCodexResponses,
        provider: Provider::OpenAiCodex,
        base_url: "https://chatgpt.com/backend-api".into(),
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
        context_window,
        max_tokens,
        compat: {
            let mut c = ModelCompat::for_openai();
            if o_series {
                c.supports_developer_role = true;
            }
            c
        },
    }
}
