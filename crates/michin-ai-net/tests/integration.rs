//! Integration tests for michin-ai providers.
//!
//! These tests hit real LLM APIs and require API keys.
//! They only compile when `--features integration-tests` is passed.
//! They skip silently if no API key is set in the environment.
//!
//! Run with:
//! ```bash
//! OPENAI_API_KEY=sk-... cargo test -p michin-ai --features integration-tests
//! DEEPSEEK_API_KEY=sk-... cargo test -p michin-ai --features integration-tests
//! ```
//!
//! No paid API keys in CI. These are local-only.

#[cfg(feature = "integration-tests")]
mod integration {
    use std::collections::HashMap;

    use michin_ai::LlmProvider;
    use michin_ai::event::EventAccumulator;
    use michin_ai::model::Model;
    use michin_ai::types::{
        Api, ContentBlock, Context, Message, Modality, Provider, StopReason, StreamOptions,
        ThinkingLevel,
    };
    use michin_ai_net::OpenAiCompatProvider;

    /// Helper: create a simple OpenAI model for testing.
    fn test_model_openai() -> Model {
        use michin_ai::model::ModelCompat;
        Model {
            id: "gpt-5.5-instant".into(),
            name: "GPT-5.5 Instant".into(),
            api: Api::OpenAiCompletions,
            provider: Provider::OpenAI,
            base_url: "https://api.openai.com".into(),
            reasoning: false,
            thinking_level_map: HashMap::new(),
            input: vec![Modality::Text],
            context_window: 128_000,
            max_tokens: 16_384,
            compat: ModelCompat::for_openai(),
        }
    }

    /// Helper: create a simple DeepSeek model for testing.
    fn test_model_deepseek() -> Model {
        use michin_ai::model::ModelCompat;
        Model {
            id: "deepseek-v4-flash".into(),
            name: "DeepSeek V4 Flash".into(),
            api: Api::OpenAiCompletions,
            provider: Provider::DeepSeek,
            base_url: "https://api.deepseek.com".into(),
            reasoning: true,
            thinking_level_map: HashMap::from([
                (ThinkingLevel::Off, None),
                (ThinkingLevel::High, Some("high".into())),
            ]),
            input: vec![Modality::Text],
            context_window: 1_000_000,
            max_tokens: 384_000,
            compat: ModelCompat::for_deepseek(),
        }
    }

    /// Helper: send "Hello, respond with just 'ok'." and collect the result.
    async fn simple_ping(provider: &dyn LlmProvider, model: &Model) -> Option<String> {
        let context = Context {
            system: Some(vec![ContentBlock::text(
                "Respond with exactly the word 'ok' and nothing else.",
            )]),
            messages: vec![Message::User {
                content: vec![ContentBlock::text("Hi")],
                timestamp: 0,
            }],
            tools: vec![],
            thinking_level: None,
        };

        let options = StreamOptions {
            max_tokens: Some(10),
            include_usage: false,
            ..Default::default()
        };

        let mut stream = match provider.stream(model, &context, &options).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Stream error: {e}");
                return None;
            }
        };

        let mut acc = EventAccumulator::new();
        use futures::StreamExt;
        while let Some(event) = stream.next().await {
            acc.feed(&event);
        }

        if acc.stop_reason() == Some(StopReason::Stop) {
            let blocks = acc.content_blocks();
            for b in &blocks {
                if let ContentBlock::Text { text } = b {
                    return Some(text.trim().to_lowercase());
                }
            }
        }

        None
    }

    fn has_key(var: &str) -> bool {
        std::env::var(var).ok().filter(|s| !s.is_empty()).is_some()
    }

    #[tokio::test]
    async fn test_openai_simple_stream() {
        if !has_key("OPENAI_API_KEY") {
            eprintln!("Skipping: OPENAI_API_KEY not set");
            return;
        }

        let provider = OpenAiCompatProvider::new();
        let model = test_model_openai();
        let result = simple_ping(&provider, &model).await;

        match result {
            Some(text) => {
                assert!(
                    text.contains("ok"),
                    "Expected response containing 'ok', got: '{text}'"
                );
            }
            None => {
                // Non-fatal — API might be down or key might be invalid
                eprintln!("Warning: OpenAI test returned no valid response");
            }
        }
    }

    #[tokio::test]
    async fn test_deepseek_simple_stream() {
        if !has_key("DEEPSEEK_API_KEY") {
            eprintln!("Skipping: DEEPSEEK_API_KEY not set");
            return;
        }

        let provider = OpenAiCompatProvider::new();
        let model = test_model_deepseek();
        let result = simple_ping(&provider, &model).await;

        match result {
            Some(text) => {
                assert!(
                    text.contains("ok"),
                    "Expected response containing 'ok', got: '{text}'"
                );
            }
            None => {
                eprintln!("Warning: DeepSeek test returned no valid response");
            }
        }
    }

    #[tokio::test]
    async fn test_openai_with_thinking() {
        if !has_key("OPENAI_API_KEY") {
            eprintln!("Skipping: OPENAI_API_KEY not set");
            return;
        }

        // Use a model that supports thinking
        let mut model = test_model_openai();
        model.reasoning = true;
        model.thinking_level_map = HashMap::from([
            (ThinkingLevel::Off, None),
            (ThinkingLevel::Low, Some("low".into())),
            (ThinkingLevel::High, Some("high".into())),
        ]);

        let context = Context {
            system: Some(vec![ContentBlock::text("Answer briefly.")]),
            messages: vec![Message::User {
                content: vec![ContentBlock::text("What is 2+2?")],
                timestamp: 0,
            }],
            tools: vec![],
            thinking_level: Some(ThinkingLevel::Low),
        };

        let provider = OpenAiCompatProvider::new();
        let options = StreamOptions {
            max_tokens: Some(20),
            thinking_level: Some(ThinkingLevel::Low),
            include_usage: false,
            ..Default::default()
        };

        let mut stream = match provider.stream(&model, &context, &options).await {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Thinking stream error: {e}");
                return;
            }
        };

        let mut acc = EventAccumulator::new();
        use futures::StreamExt;
        while let Some(event) = stream.next().await {
            acc.feed(&event);
        }

        let blocks = acc.content_blocks();
        // With thinking, we may get both thinking and text blocks.
        // At minimum, we expect a response.
        assert!(
            !blocks.is_empty(),
            "Expected at least one content block (thinking or text)"
        );
    }
}
