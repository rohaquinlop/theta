use serde_json::json;
use theta_ai::event::AssistantMessageEvent;
use theta_ai::event::EventAccumulator;
use theta_ai::providers::openai_compat::{
    self, apply_thinking_params, build_request_body, convert_message, convert_messages,
    parse_sse_line,
};
use theta_ai::replay::sanitize_messages_for_replay;
use theta_ai::{
    Api, ContentBlock, Context, Message, Modality, Model, ModelCompat, Provider, StopReason,
    ThinkingLevel,
};

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
        cost: Default::default(),
        context_window: 128_000,
        max_tokens: 16_384,
        compat: ModelCompat::for_openai(),
    }
}

fn openai_reasoning_model() -> Model {
    Model {
        id: "gpt-5.5".into(),
        name: "OpenAI".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenAI,
        base_url: "https://api.openai.com".into(),
        reasoning: true,
        thinking_level_map: Default::default(),
        input: vec![Modality::Text],
        cost: Default::default(),
        context_window: 128_000,
        max_tokens: 16_384,
        compat: ModelCompat::for_openai(),
    }
}

fn deepseek_model() -> Model {
    Model {
        id: "deepseek-v4-pro".into(),
        name: "DeepSeek".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::DeepSeek,
        base_url: "https://api.deepseek.com".into(),
        reasoning: true,
        thinking_level_map: Default::default(),
        input: vec![Modality::Text],
        cost: Default::default(),
        context_window: 1_000_000,
        max_tokens: 384_000,
        compat: ModelCompat::for_deepseek(),
    }
}

#[test]
fn test_api_key_env() {
    assert_eq!(
        openai_compat::api_key_env(Provider::OpenAI),
        "OPENAI_API_KEY"
    );
    assert_eq!(
        openai_compat::api_key_env(Provider::DeepSeek),
        "DEEPSEEK_API_KEY"
    );
    assert_eq!(
        openai_compat::api_key_env(Provider::OpenCode),
        "OPENCODE_API_KEY"
    );
}

#[test]
fn test_parse_sse_empty() {
    assert!(parse_sse_line("").is_none());
    assert!(parse_sse_line(": heartbeat").is_none());
    assert!(parse_sse_line("data: [DONE]").is_none());
}

#[test]
fn test_parse_text_delta() {
    let event = parse_sse_line(r#"data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}"#);
    assert!(event.is_some());
    if let Some(AssistantMessageEvent::TextDelta { text }) = event {
        assert_eq!(text, "Hello");
    } else {
        panic!("Expected TextDelta");
    }
}

#[test]
fn test_parse_thinking_delta() {
    let mut parser = openai_compat::OpenAiCompatStreamParser::new();
    let events = parser
        .parse_data(r#"{"choices":[{"delta":{"reasoning_content":"Let me think..."},"index":0}]}"#);
    assert_eq!(events.len(), 2, "expected ThinkingStart + ThinkingDelta");
    assert!(
        matches!(&events[0], AssistantMessageEvent::ThinkingStart),
        "first event should be ThinkingStart"
    );
    if let AssistantMessageEvent::ThinkingDelta { thinking } = &events[1] {
        assert_eq!(thinking, "Let me think...");
    } else {
        panic!("Expected ThinkingDelta as second event");
    }
}

#[test]
fn test_parse_tool_call_start() {
    let event = parse_sse_line(
        r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"read","arguments":""}}]},"index":0}]}"#,
    );
    assert!(event.is_some());
    if let Some(AssistantMessageEvent::ToolCallStart { id, name }) = event {
        assert_eq!(id, "call_123");
        assert_eq!(name, "read");
    } else {
        panic!("Expected ToolCallStart");
    }
}

#[test]
fn test_parse_finish_tool_calls() {
    let event = parse_sse_line(
        r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
    );
    assert!(event.is_some());
    if let Some(AssistantMessageEvent::Done { stop_reason, .. }) = event {
        assert_eq!(stop_reason, StopReason::ToolUse);
    } else {
        panic!("Expected Done with ToolUse");
    }
}

#[test]
fn test_parse_error() {
    let event =
        parse_sse_line(r#"data: {"error":{"code":"rate_limit","message":"Too many requests"}}"#);
    assert!(event.is_some());
    if let Some(AssistantMessageEvent::Error { code, message }) = event {
        assert_eq!(code, "rate_limit");
        assert_eq!(message, "Too many requests");
    } else {
        panic!("Expected Error event");
    }
}

#[test]
fn test_parse_streamed_tool_call_arguments_by_index() {
    let mut parser = openai_compat::OpenAiCompatStreamParser::new();
    let mut accumulator = EventAccumulator::new();

    let chunks = [
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"read","arguments":""}}]},"index":0}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\""}}]},"index":0}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"Cargo.toml\"}"}}]},"index":0}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
    ];

    let events: Vec<AssistantMessageEvent> = chunks
        .iter()
        .flat_map(|chunk| parser.parse_data(chunk))
        .collect();

    assert!(matches!(
        events.first(),
        Some(AssistantMessageEvent::ToolCallStart { id, name })
            if id == "call_123" && name == "read"
    ));
    assert!(matches!(
        events.iter().rev().nth(1),
        Some(AssistantMessageEvent::ToolCallEnd { id }) if id == "call_123"
    ));
    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Done {
            stop_reason: StopReason::ToolUse,
            ..
        })
    ));

    for event in &events {
        accumulator.feed(event);
    }

    let blocks = accumulator.content_blocks();
    assert_eq!(blocks.len(), 1);
    match &blocks[0] {
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
        } => {
            assert_eq!(id, "call_123");
            assert_eq!(name, "read");
            assert_eq!(arguments["path"], "Cargo.toml");
        }
        other => panic!("expected tool call, got {other:?}"),
    }
    assert_eq!(accumulator.stop_reason(), Some(StopReason::ToolUse));
}

#[test]
fn test_parse_tool_call_arguments_before_id_is_retained() {
    let mut parser = openai_compat::OpenAiCompatStreamParser::new();
    let mut accumulator = EventAccumulator::new();

    let chunks = [
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\""}}]},"index":0}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"read","arguments":":\"Cargo.toml\"}"}}]},"index":0}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
    ];

    let events: Vec<AssistantMessageEvent> = chunks
        .iter()
        .flat_map(|chunk| parser.parse_data(chunk))
        .collect();

    for event in &events {
        accumulator.feed(event);
    }

    let blocks = accumulator.content_blocks();
    assert_eq!(blocks.len(), 1);
    match &blocks[0] {
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
        } => {
            assert_eq!(id, "call_abc");
            assert_eq!(name, "read");
            assert_eq!(arguments["path"], "Cargo.toml");
        }
        other => panic!("expected tool call, got {other:?}"),
    }
}

#[test]
fn test_parse_tool_call_id_is_stable_when_id_arrives_late() {
    let mut parser = openai_compat::OpenAiCompatStreamParser::new();

    let chunks = [
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read","arguments":"{\"path\""}}]},"index":0}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_real","function":{"arguments":":\"Cargo.toml\"}"}}]},"index":0}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
    ];

    let events: Vec<AssistantMessageEvent> = chunks
        .iter()
        .flat_map(|chunk| parser.parse_data(chunk))
        .collect();

    let start_ids: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AssistantMessageEvent::ToolCallStart { id, .. } => Some(id.clone()),
            _ => None,
        })
        .collect();
    let end_ids: Vec<String> = events
        .iter()
        .filter_map(|e| match e {
            AssistantMessageEvent::ToolCallEnd { id } => Some(id.clone()),
            _ => None,
        })
        .collect();

    assert_eq!(start_ids, vec!["call_real".to_string()]);
    assert_eq!(end_ids, vec!["call_real".to_string()]);
}

#[test]
fn test_parse_usage_and_done_same_chunk() {
    let mut parser = openai_compat::OpenAiCompatStreamParser::new();
    let chunk = r#"{
        "choices":[{"delta":{"content":""},"finish_reason":"stop","index":0}],
        "usage":{"prompt_tokens":10,"completion_tokens":2}
    }"#;
    let events = parser.parse_data(chunk);
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AssistantMessageEvent::Usage { .. }))
    );
    assert!(events.iter().any(|e| matches!(
        e,
        AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            ..
        }
    )));
}

#[test]
fn test_tool_result_conversion_omits_non_openai_fields() {
    let model = Model {
        id: "test-model".into(),
        name: "Test Model".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenAI,
        base_url: "https://api.openai.com".into(),
        reasoning: false,
        thinking_level_map: Default::default(),
        input: vec![Modality::Text],
        cost: Default::default(),
        context_window: 128_000,
        max_tokens: 16_384,
        compat: ModelCompat::for_openai(),
    };
    let msg = Message::ToolResult {
        tool_call_id: "call_123".into(),
        tool_name: "read".into(),
        content: vec![ContentBlock::Text {
            text: "done".into(),
        }],
        details: None,
        is_error: false,
        timestamp: 0,
    };

    let converted = convert_message(&model, &msg).expect("tool result converts");
    assert_eq!(converted["role"], "tool");
    assert_eq!(converted["tool_call_id"], "call_123");
    assert_eq!(converted["content"], "done");
    assert!(converted.get("is_error").is_none());
}

#[test]
fn test_transform_synthesizes_missing_tool_result() {
    let model = openai_model();
    let messages = vec![
        Message::User {
            content: vec![ContentBlock::text("do thing")],
            timestamp: 1,
        },
        Message::Assistant {
            content: vec![ContentBlock::ToolCall {
                id: "call_1".into(),
                name: "read".into(),
                arguments: json!({"path":"Cargo.toml"}),
            }],
            api: Some(Api::OpenAiCompletions),
            provider: Some(Provider::OpenAI),
            model: Some("gpt-5.5".into()),
            usage: None,
            stop_reason: Some(StopReason::ToolUse),
            error_message: None,
            timestamp: 2,
        },
        Message::User {
            content: vec![ContentBlock::text("next")],
            timestamp: 3,
        },
    ];

    let (transformed, _stats) = sanitize_messages_for_replay(&messages, &model);
    assert!(
        transformed.iter().any(|m| matches!(
            m,
            Message::ToolResult {
                tool_call_id,
                is_error,
                ..
            } if tool_call_id == "call_1" && *is_error
        )),
        "expected synthetic tool result for orphan tool call"
    );
}

#[test]
fn test_transform_drops_aborted_assistant_message() {
    let model = openai_model();
    let messages = vec![
        Message::User {
            content: vec![ContentBlock::text("hi")],
            timestamp: 1,
        },
        Message::Assistant {
            content: vec![ContentBlock::text("partial")],
            api: Some(Api::OpenAiCompletions),
            provider: Some(Provider::OpenAI),
            model: Some("gpt-5.5".into()),
            usage: None,
            stop_reason: Some(StopReason::Aborted),
            error_message: Some("aborted".into()),
            timestamp: 2,
        },
        Message::User {
            content: vec![ContentBlock::text("continue")],
            timestamp: 3,
        },
    ];

    let (transformed, _stats) = sanitize_messages_for_replay(&messages, &model);
    assert_eq!(
        transformed
            .iter()
            .filter(|m| matches!(m, Message::Assistant { .. }))
            .count(),
        0,
        "aborted assistant message should not be replayed"
    );
}

#[test]
fn test_bridge_assistant_inserted_after_tool_before_user() {
    let mut model = deepseek_model();
    model.compat.requires_assistant_after_tool_result = true;
    let ctx = Context {
        system: None,
        messages: vec![
            Message::ToolResult {
                tool_call_id: "call_1".into(),
                tool_name: "read".into(),
                content: vec![ContentBlock::text("ok")],
                details: None,
                is_error: false,
                timestamp: 1,
            },
            Message::User {
                content: vec![ContentBlock::text("next")],
                timestamp: 2,
            },
        ],
        tools: vec![],
        thinking_level: None,
    };
    let msgs = convert_messages(&model, &ctx);
    let arr = msgs.as_array().expect("messages array");
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0]["role"], "tool");
    assert_eq!(arr[1]["role"], "assistant");
    assert_eq!(arr[2]["role"], "user");
}

#[test]
fn test_tools_empty_sent_when_tool_history_present() {
    let model = openai_reasoning_model();
    let ctx = Context {
        system: None,
        messages: vec![
            Message::Assistant {
                content: vec![ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "read".into(),
                    arguments: json!({"path":"Cargo.toml"}),
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
                content: vec![ContentBlock::text("ok")],
                details: None,
                is_error: false,
                timestamp: 2,
            },
        ],
        tools: vec![],
        thinking_level: None,
    };
    let body = build_request_body(&model, &ctx, &theta_ai::StreamOptions::default(), true).unwrap();
    assert!(body.get("tools").is_some());
    assert_eq!(
        body["tools"]
            .as_array()
            .map(|a| a.len())
            .unwrap_or_default(),
        0
    );
}

#[test]
fn test_deepseek_non_off_thinking_falls_back_to_high() {
    let model = deepseek_model();
    let mut body = json!({});
    apply_thinking_params(&mut body, &model, ThinkingLevel::Medium);
    assert!(
        body.get("thinking").is_none(),
        "unmapped level should not send thinking params"
    );

    let mut body_off = json!({});
    apply_thinking_params(&mut body_off, &model, ThinkingLevel::Off);
    assert_eq!(body_off["thinking"]["type"], "disabled");
}

#[test]
fn test_deepseek_request_body_drops_orphan_tool_result_from_aborted_turn() {
    let model = deepseek_model();
    let dirty = vec![
        Message::Assistant {
            content: vec![ContentBlock::ToolCall {
                id: "call_1".into(),
                name: "read".into(),
                arguments: json!({"path":"Cargo.toml"}),
            }],
            api: Some(Api::OpenAiCompletions),
            provider: Some(Provider::DeepSeek),
            model: Some("deepseek-v4-pro".into()),
            usage: None,
            stop_reason: Some(StopReason::Aborted),
            error_message: Some("aborted".into()),
            timestamp: 1,
        },
        Message::ToolResult {
            tool_call_id: "call_1".into(),
            tool_name: "read".into(),
            content: vec![ContentBlock::text("ok")],
            details: None,
            is_error: false,
            timestamp: 2,
        },
        Message::User {
            content: vec![ContentBlock::text("continue")],
            timestamp: 3,
        },
    ];
    let (sanitized, _) = sanitize_messages_for_replay(&dirty, &model);
    let ctx = Context {
        system: None,
        messages: sanitized,
        tools: vec![],
        thinking_level: None,
    };
    let body = build_request_body(&model, &ctx, &theta_ai::StreamOptions::default(), true).unwrap();
    let messages = body["messages"].as_array().expect("messages array");
    assert!(
        !messages
            .iter()
            .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool")),
        "orphan tool messages must not be replayed"
    );
}

#[test]
fn test_parse_legacy_function_call_stream_to_tool_call() {
    let mut parser = openai_compat::OpenAiCompatStreamParser::new();
    let mut accumulator = EventAccumulator::new();
    let chunks = [
        r#"{"choices":[{"delta":{"function_call":{"name":"read","arguments":"{\"path\""}},"index":0}]}"#,
        r#"{"choices":[{"delta":{"function_call":{"arguments":":\"Cargo.toml\"}"}},"index":0}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"function_call","index":0}]}"#,
    ];

    let events: Vec<AssistantMessageEvent> = chunks
        .iter()
        .flat_map(|chunk| parser.parse_data(chunk))
        .collect();

    for event in &events {
        accumulator.feed(event);
    }

    let blocks = accumulator.content_blocks();
    assert_eq!(blocks.len(), 1);
    match &blocks[0] {
        ContentBlock::ToolCall {
            id: _,
            name,
            arguments,
        } => {
            assert_eq!(name, "read");
            assert_eq!(arguments["path"], "Cargo.toml");
        }
        other => panic!("expected tool call, got {other:?}"),
    }
    assert_eq!(accumulator.stop_reason(), Some(StopReason::ToolUse));
}

#[test]
fn test_parse_mixed_function_call_and_tool_calls_yields_single_call() {
    let mut parser = openai_compat::OpenAiCompatStreamParser::new();
    let mut accumulator = EventAccumulator::new();
    let chunks = [
        r#"{"choices":[{"delta":{"function_call":{"name":"read","arguments":"{\"path\""}},"index":0}]}"#,
        r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_real","type":"function","function":{"arguments":":\"Cargo.toml\"}"}}]},"index":0}]}"#,
        r#"{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
    ];

    let events: Vec<AssistantMessageEvent> = chunks
        .iter()
        .flat_map(|chunk| parser.parse_data(chunk))
        .collect();

    for event in &events {
        accumulator.feed(event);
    }
    let blocks = accumulator.content_blocks();
    let tool_calls: Vec<&ContentBlock> = blocks
        .iter()
        .filter(|b| matches!(b, ContentBlock::ToolCall { .. }))
        .collect();
    assert_eq!(tool_calls.len(), 1);
}
