//! OpenAI model definitions.

use theta_ai::model::{Model, ModelCompat};
use theta_ai::types::{Api, Modality, ModelCost, Provider, ThinkingLevel};

/// Return all OpenAI models.
pub fn models() -> Vec<Model> {
    vec![
        // GPT-5 family
        openai_model("gpt-5.5", "GPT-5.5", 272_000, 128_000, false),
        openai_model(
            "gpt-5.5-instant",
            "GPT-5.5 Instant",
            272_000,
            128_000,
            false,
        ),
        openai_model("gpt-5", "GPT-5", 272_000, 128_000, false),
        openai_model("gpt-5-mini", "GPT-5 Mini", 272_000, 128_000, false),
        openai_model("gpt-5-nano", "GPT-5 Nano", 272_000, 128_000, false),
        openai_model(
            "gpt-5-chat-latest",
            "GPT-5 Chat Latest",
            272_000,
            128_000,
            false,
        ),
        // GPT-4.1 family
        openai_model("gpt-4.1", "GPT-4.1", 200_000, 100_000, false),
        openai_model("gpt-4.1-mini", "GPT-4.1 Mini", 200_000, 100_000, false),
        openai_model("gpt-4.1-nano", "GPT-4.1 Nano", 200_000, 100_000, false),
        // GPT-4o family
        openai_model("gpt-4o", "GPT-4o", 128_000, 16_384, false),
        openai_model("gpt-4o-mini", "GPT-4o Mini", 128_000, 16_384, false),
        // o-series
        openai_model("o4", "o4", 200_000, 100_000, true),
        openai_model("o4-mini", "o4-mini", 200_000, 100_000, true),
        openai_model("o3", "o3", 200_000, 100_000, true),
        openai_model("o3-mini", "o3-mini", 200_000, 100_000, true),
        openai_model("o1", "o1", 200_000, 100_000, true),
        openai_model("o1-mini", "o1-mini", 200_000, 100_000, true),
    ]
}

fn openai_model(
    id: &str,
    name: &str,
    context_window: u32,
    max_tokens: u32,
    o_series: bool,
) -> Model {
    Model {
        id: id.into(),
        name: name.into(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenAI,
        base_url: "https://api.openai.com".into(),
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
        // Conservative placeholders; exact pricing differs per model.
        cost: ModelCost {
            input: 0.0,
            output: 0.0,
            cache_read: 0.0,
            cache_write: 0.0,
        },
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
