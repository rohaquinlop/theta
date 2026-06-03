// Consolidated tests: provider_registry, model, event, replay, types.
// Merged to reduce test binary count and improve test suite startup time.

mod provider_registry {
    use michin_ai::Api;
    use michin_ai::providers::ProviderRegistry;

    #[test]
    fn test_registry_creation() {
        let reg = ProviderRegistry::new();
        assert!(reg.get(&Api::OpenAiCompletions).is_none());
    }
}

mod model {
    use michin_ai::model::{Model, ModelCompat};
    use michin_ai::{Api, Modality, Provider, ThinkingLevel};
    use std::collections::HashMap;

    fn test_model() -> Model {
        Model {
            id: "test-model".into(),
            name: "Test Model".into(),
            api: Api::OpenAiCompletions,
            provider: Provider::OpenAI,
            base_url: "https://api.openai.com".into(),
            reasoning: true,
            thinking_level_map: HashMap::from([
                (ThinkingLevel::Off, None),
                (ThinkingLevel::High, Some("high".into())),
                (ThinkingLevel::XHigh, Some("max".into())),
            ]),
            input: vec![Modality::Text],
            context_window: 128_000,
            max_tokens: 16_384,
            compat: ModelCompat::for_openai(),
        }
    }

    #[test]
    fn test_thinking_param() {
        let m = test_model();
        assert_eq!(m.thinking_param(ThinkingLevel::Off), None);
        assert_eq!(
            m.thinking_param(ThinkingLevel::High),
            Some("high".to_string())
        );
        assert_eq!(m.thinking_param(ThinkingLevel::Low), None);
    }

    #[test]
    fn test_max_tokens_field_name() {
        let m = test_model();
        assert_eq!(m.max_tokens_field_name(), "max_completion_tokens");
    }

    #[test]
    fn test_requires_reasoning_on_replay() {
        let mut m = test_model();
        assert!(!m.requires_reasoning_on_replay());
        m.compat = ModelCompat::for_deepseek();
        assert!(m.requires_reasoning_on_replay());
    }

    #[test]
    fn test_max_tokens_field_for_non_openai_compat() {
        let mut m = test_model();
        m.compat = ModelCompat::for_deepseek();
        assert_eq!(m.max_tokens_field_name(), "max_tokens");

        m.compat = ModelCompat::for_opencode();
        assert_eq!(m.max_tokens_field_name(), "max_tokens");
    }
}

mod event {
    use michin_ai::event::AssistantMessageEvent;
    use michin_ai::event::EventAccumulator;
    use michin_ai::{ContentBlock, StopReason};

    #[test]
    fn test_accumulator_text_only() {
        let mut acc = EventAccumulator::new();
        acc.feed(&AssistantMessageEvent::Start);
        acc.feed(&AssistantMessageEvent::TextStart);
        acc.feed(&AssistantMessageEvent::text_delta("Hello"));
        acc.feed(&AssistantMessageEvent::text_delta(" world"));
        acc.feed(&AssistantMessageEvent::TextEnd);
        acc.feed(&AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        });

        let blocks = acc.content_blocks();
        assert_eq!(blocks.len(), 1);
        if let ContentBlock::Text { text } = &blocks[0] {
            assert_eq!(text, "Hello world");
        } else {
            panic!("expected Text block");
        }
        assert_eq!(acc.stop_reason(), Some(StopReason::Stop));
    }

    #[test]
    fn test_accumulator_with_thinking() {
        let mut acc = EventAccumulator::new();
        acc.feed(&AssistantMessageEvent::Start);
        acc.feed(&AssistantMessageEvent::ThinkingStart);
        acc.feed(&AssistantMessageEvent::thinking_delta("Let me think..."));
        acc.feed(&AssistantMessageEvent::ThinkingEnd);
        acc.feed(&AssistantMessageEvent::TextStart);
        acc.feed(&AssistantMessageEvent::text_delta("Done."));
        acc.feed(&AssistantMessageEvent::TextEnd);
        acc.feed(&AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        });

        let blocks = acc.content_blocks();
        assert_eq!(blocks.len(), 2);
        assert!(matches!(blocks[0], ContentBlock::Thinking { .. }));
        assert!(matches!(blocks[1], ContentBlock::Text { .. }));
    }

    #[test]
    fn test_accumulator_reset() {
        let mut acc = EventAccumulator::new();
        acc.feed(&AssistantMessageEvent::text_delta("data"));
        acc.reset();
        assert!(acc.content_blocks().is_empty());
        assert!(acc.stop_reason().is_none());
    }

    #[test]
    fn test_tool_call_delta_matches_call_id_prefix() {
        let mut acc = EventAccumulator::new();
        acc.feed(&AssistantMessageEvent::ToolCallStart {
            id: "call_1|item_1".into(),
            name: "read".into(),
        });
        acc.feed(&AssistantMessageEvent::ToolCallDelta {
            id: "call_1".into(),
            arguments: "{\"path\":\"Cargo.toml\"}".into(),
        });
        acc.feed(&AssistantMessageEvent::ToolCallEnd {
            id: "call_1".into(),
        });

        let blocks = acc.content_blocks();
        assert_eq!(blocks.len(), 1);
        assert!(matches!(blocks[0], ContentBlock::ToolCall { .. }));
    }

    #[test]
    fn test_done_tool_use_not_downgraded_by_later_stop() {
        let mut acc = EventAccumulator::new();
        acc.feed(&AssistantMessageEvent::Done {
            stop_reason: StopReason::ToolUse,
            usage: None,
        });
        acc.feed(&AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        });
        assert_eq!(acc.stop_reason(), Some(StopReason::ToolUse));
    }
}

mod replay {
    use michin_ai::replay::{normalize_tool_call_id_for_model, sanitize_messages_for_replay};
    use michin_ai::{
        Api, ContentBlock, Message, Modality, Model, ModelCompat, Provider, StopReason,
    };
    use serde_json::json;

    fn openai_model() -> Model {
        Model {
            id: "gpt-5.5".into(),
            name: "OpenAI".into(),
            api: Api::OpenAiCompletions,
            provider: Provider::OpenAI,
            base_url: "https://api.openai.com".into(),
            reasoning: false,
            thinking_level_map: Default::default(),
            input: vec![Modality::Text],
            context_window: 128_000,
            max_tokens: 16_384,
            compat: ModelCompat::for_openai(),
        }
    }

    #[test]
    fn normalize_pipe_only_tool_call_id_becomes_non_empty() {
        let model = openai_model();
        assert_eq!(normalize_tool_call_id_for_model("|", &model), "tool_call_0");
    }

    #[test]
    fn normalize_pipe_only_tool_call_id_becomes_non_empty_for_deepseek() {
        let mut model = openai_model();
        model.provider = Provider::DeepSeek;
        assert_eq!(normalize_tool_call_id_for_model("|", &model), "tool_call_0");
    }

    #[test]
    fn drops_orphan_tool_result_without_preceding_tool_call() {
        let model = openai_model();
        let messages = vec![
            Message::Assistant {
                content: vec![ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "read".into(),
                    arguments: json!({ "path": "Cargo.toml" }),
                }],
                api: Some(Api::OpenAiCompletions),
                provider: Some(Provider::OpenAI),
                model: Some("gpt-5.5".into()),
                usage: None,
                stop_reason: Some(StopReason::Error),
                error_message: Some("error".into()),
                timestamp: 1,
            },
            Message::ToolResult {
                tool_call_id: "call_1".into(),
                tool_name: "read".into(),
                content: vec![ContentBlock::text("done")],
                details: None,
                is_error: false,
                timestamp: 2,
            },
            Message::User {
                content: vec![ContentBlock::text("continue")],
                timestamp: 3,
            },
        ];

        let (out, stats) = sanitize_messages_for_replay(&messages, &model);
        assert_eq!(stats.dropped_assistant_messages, 1);
        assert!(
            !out.iter().any(|m| matches!(m, Message::ToolResult { .. })),
            "orphan tool result should be dropped when parent assistant tool call is removed"
        );
    }

    #[test]
    fn dedupes_duplicate_tool_results_for_same_pending_call() {
        let model = openai_model();
        let messages = vec![
            Message::Assistant {
                content: vec![ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "read".into(),
                    arguments: json!({ "path": "Cargo.toml" }),
                }],
                api: Some(Api::OpenAiCompletions),
                provider: Some(Provider::OpenAI),
                model: Some("gpt-5.5".into()),
                usage: None,
                stop_reason: Some(StopReason::ToolUse),
                error_message: None,
                timestamp: 1,
            },
            Message::ToolResult {
                tool_call_id: "call_1".into(),
                tool_name: "read".into(),
                content: vec![ContentBlock::text("done-1")],
                details: None,
                is_error: false,
                timestamp: 2,
            },
            Message::ToolResult {
                tool_call_id: "call_1".into(),
                tool_name: "read".into(),
                content: vec![ContentBlock::text("done-2")],
                details: None,
                is_error: false,
                timestamp: 3,
            },
        ];

        let (out, stats) = sanitize_messages_for_replay(&messages, &model);
        let tool_results = out
            .iter()
            .filter(|m| matches!(m, Message::ToolResult { .. }))
            .count();
        assert_eq!(tool_results, 1);
        assert_eq!(stats.deduped_tool_results, 1);
    }
}

mod types {
    use michin_ai::types::approximate_token_count;
    use michin_ai::{ContentBlock, Message};
    use serde_json;

    #[test]
    fn test_approximate_token_count() {
        assert_eq!(approximate_token_count("hello world"), 3);
        assert_eq!(approximate_token_count(""), 0);
        assert_eq!(approximate_token_count("a"), 1);
    }

    #[test]
    fn test_message_token_count() {
        let msg = Message::User {
            content: vec![ContentBlock::text("hello world")],
            timestamp: 0,
        };
        assert_eq!(msg.token_count(), 3);
    }

    #[test]
    fn test_content_block_factories() {
        assert!(matches!(
            ContentBlock::text("hi"),
            ContentBlock::Text { .. }
        ));
        let tc = ContentBlock::tool_call("id1", "read", serde_json::json!({"path": "foo"}));
        assert!(matches!(tc, ContentBlock::ToolCall { .. }));
    }

    #[test]
    fn test_message_tool_result_serializes_to_theta_native_tag() {
        let msg = Message::ToolResult {
            tool_call_id: "c1".into(),
            tool_name: "read".into(),
            content: vec![ContentBlock::text("ok")],
            details: None,
            is_error: false,
            timestamp: 1,
        };
        let v = serde_json::to_value(&msg).expect("serialize");
        assert_eq!(v.get("type").and_then(|x| x.as_str()), Some("tool_result"));
        assert!(v.get("tool_call_id").is_some());
        assert!(v.get("tool_name").is_some());
        assert!(v.get("is_error").is_some());
    }

    #[test]
    fn test_message_tool_result_deserializes_legacy_pi_keys() {
        let v = serde_json::json!({
            "type": "toolResult",
            "toolCallId": "c1",
            "toolName": "read",
            "content": [{"type":"text","text":"ok"}],
            "details": null,
            "isError": false,
            "timestamp": 1
        });
        let msg: Message = serde_json::from_value(v).expect("deserialize");
        assert!(matches!(msg, Message::ToolResult { .. }));
    }
}

mod error_class {
    use michin_ai::{ErrorClass, MichiNError};

    // ── Transient: transport-level errors ──

    #[test]
    fn connection_reset_is_transient() {
        let e = MichiNError::Http("error sending request: connection reset by peer".into());
        assert_eq!(e.class(), ErrorClass::Transient);
    }

    #[test]
    fn broken_pipe_is_transient() {
        let e = MichiNError::Http("broken pipe (os error 32)".into());
        assert_eq!(e.class(), ErrorClass::Transient);
    }

    #[test]
    fn eof_is_transient() {
        let e = MichiNError::Http("unexpected eof during response".into());
        assert_eq!(e.class(), ErrorClass::Transient);
    }

    #[test]
    fn timeout_is_transient() {
        let e = MichiNError::Http("request timeout after 30s".into());
        assert_eq!(e.class(), ErrorClass::Transient);
    }

    #[test]
    fn tls_error_is_transient() {
        let e = MichiNError::Http("tls handshake failed".into());
        assert_eq!(e.class(), ErrorClass::Transient);
    }

    #[test]
    fn sending_request_wrapper_is_transient() {
        // reqwest wraps real errors like "connection reset" under this phrase.
        // The {:#} formatter includes the root cause, but the wrapper alone
        // must also be classified as transient.
        let e = MichiNError::Http("error sending request for url".into());
        assert_eq!(e.class(), ErrorClass::Transient);
    }

    #[test]
    fn dns_error_is_transient() {
        let e = MichiNError::Http("dns resolution failed for api.openai.com".into());
        assert_eq!(e.class(), ErrorClass::Transient);
    }

    #[test]
    fn connection_refused_is_transient() {
        let e = MichiNError::Http("connection refused (os error 61)".into());
        assert_eq!(e.class(), ErrorClass::Transient);
    }

    // ── Permanent: protocol-level errors ──

    #[test]
    fn http_400_is_permanent() {
        let e = MichiNError::Http("400 Bad Request: invalid parameter".into());
        assert_eq!(e.class(), ErrorClass::Permanent);
    }

    #[test]
    fn stream_errors_are_transient() {
        let e = MichiNError::Stream("decode error: invalid utf-8".into());
        assert_eq!(e.class(), ErrorClass::Transient);
    }

    #[test]
    fn stream_ended_early_is_transient() {
        assert_eq!(MichiNError::StreamEndedEarly.class(), ErrorClass::Transient);
    }

    #[test]
    fn api_429_is_transient() {
        let e = MichiNError::ApiError {
            status: 429,
            message: "rate limited".into(),
            retry_after_ms: Some(1000),
        };
        assert_eq!(e.class(), ErrorClass::Transient);
    }

    #[test]
    fn api_500_is_transient() {
        let e = MichiNError::ApiError {
            status: 500,
            message: "internal error".into(),
            retry_after_ms: None,
        };
        assert_eq!(e.class(), ErrorClass::Transient);
    }

    #[test]
    fn api_400_is_permanent() {
        let e = MichiNError::ApiError {
            status: 400,
            message: "bad request".into(),
            retry_after_ms: None,
        };
        assert_eq!(e.class(), ErrorClass::Permanent);
    }

    #[test]
    fn missing_api_key_is_permanent() {
        let e = MichiNError::MissingApiKey {
            provider: michin_ai::Provider::OpenAI,
        };
        assert_eq!(e.class(), ErrorClass::Permanent);
    }

    #[test]
    fn model_not_found_is_permanent() {
        let e = MichiNError::ModelNotFound {
            provider: michin_ai::Provider::OpenAI,
            model_id: "nonexistent".into(),
        };
        assert_eq!(e.class(), ErrorClass::Permanent);
    }

    #[test]
    fn provider_stream_error_is_permanent() {
        let e = MichiNError::ProviderStreamError {
            code: "invalid_request".into(),
            message: "bad parameter".into(),
        };
        assert_eq!(e.class(), ErrorClass::Permanent);
    }

    #[test]
    fn json_error_is_permanent() {
        let e =
            MichiNError::Json(serde_json::from_str::<serde_json::Value>("invalid").unwrap_err());
        assert_eq!(e.class(), ErrorClass::Permanent);
    }
}
