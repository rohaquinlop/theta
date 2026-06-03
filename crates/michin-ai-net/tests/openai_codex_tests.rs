use michin_ai::event::AssistantMessageEvent;
use michin_ai::event::EventAccumulator;
use michin_ai::{Api, ContentBlock, Context, Modality, Model, ModelCompat, Provider, StopReason};
use michin_ai_net::providers::openai_codex::{
    self, build_request_body, codex_url, codex_ws_url, convert_messages, split_tool_call_id,
    stable_tool_call_id,
};

#[test]
fn test_codex_url_resolution() {
    assert_eq!(
        codex_url("https://chatgpt.com/backend-api"),
        "https://chatgpt.com/backend-api/codex/responses"
    );
    assert_eq!(
        codex_url("https://chatgpt.com/backend-api/"),
        "https://chatgpt.com/backend-api/codex/responses"
    );
    assert_eq!(
        codex_url("https://chatgpt.com/backend-api/codex"),
        "https://chatgpt.com/backend-api/codex/responses"
    );
    assert_eq!(
        codex_url("https://chatgpt.com/backend-api/codex/responses"),
        "https://chatgpt.com/backend-api/codex/responses"
    );
}

#[test]
fn test_codex_ws_url() {
    assert_eq!(
        codex_ws_url("https://chatgpt.com/backend-api"),
        "wss://chatgpt.com/backend-api/codex/responses"
    );
}

#[test]
fn test_split_tool_call_id() {
    assert_eq!(split_tool_call_id("a|b"), ("a", "b"));
    assert_eq!(split_tool_call_id("abc"), ("abc", ""));
    assert_eq!(split_tool_call_id("a|b|c"), ("a", "b|c"));
}

#[test]
fn test_stable_tool_call_id_fallbacks() {
    assert_eq!(stable_tool_call_id("", ""), "tool_call_0");
    assert_eq!(stable_tool_call_id("call_1", ""), "call_1|");
}

fn codex_model() -> Model {
    Model {
        id: "gpt-5.5".into(),
        name: "Codex".into(),
        api: Api::OpenAiCodexResponses,
        provider: Provider::OpenAiCodex,
        base_url: "https://chatgpt.com/backend-api".into(),
        reasoning: true,
        thinking_level_map: Default::default(),
        input: vec![Modality::Text],
        context_window: 128_000,
        max_tokens: 16_384,
        compat: ModelCompat::for_openai(),
    }
}

fn codex_ctx() -> Context {
    Context {
        system: None,
        messages: vec![michin_ai::Message::Assistant {
            content: vec![ContentBlock::ToolCall {
                id: "call_abc".into(),
                name: "read".into(),
                arguments: serde_json::json!({"path":"Cargo.toml"}),
            }],
            api: Some(Api::OpenAiCodexResponses),
            provider: Some(Provider::OpenAiCodex),
            model: Some("gpt-5.5".into()),
            usage: None,
            stop_reason: Some(StopReason::ToolUse),
            error_message: None,
            timestamp: 1,
        }],
        tools: vec![],
        thinking_level: None,
    }
}

#[test]
fn test_convert_messages_uses_non_empty_item_id_for_tool_call() {
    let model = codex_model();
    let ctx = codex_ctx();

    let out = convert_messages(&model, &ctx);
    assert_eq!(out[0]["type"], "function_call");
    assert_eq!(out[0]["id"], "item_0");
    assert_eq!(out[0]["call_id"], "call_abc");
}

#[test]
fn test_delta_plus_done_text_no_duplication() {
    let mut parser = openai_codex::CodexEventParser::default();
    let events = [
        parser.parse_event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "m1"}
        })),
        parser.parse_event(&serde_json::json!({
            "type": "response.output_text.delta",
            "item_id": "m1",
            "delta": "hello"
        })),
        parser.parse_event(&serde_json::json!({
            "type": "response.output_text.done",
            "item_id": "m1",
            "text": "hello"
        })),
    ]
    .concat();

    assert!(matches!(events[0], AssistantMessageEvent::TextStart));
    assert!(matches!(events[1], AssistantMessageEvent::TextDelta { .. }));
    assert!(matches!(events[2], AssistantMessageEvent::TextEnd));
    assert_eq!(events.len(), 3);
}

#[test]
fn test_done_text_without_delta_emits_final_text_once() {
    let mut parser = openai_codex::CodexEventParser::default();
    let events = [
        parser.parse_event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "m1"}
        })),
        parser.parse_event(&serde_json::json!({
            "type": "response.output_text.done",
            "item_id": "m1",
            "text": "hello"
        })),
    ]
    .concat();

    assert!(matches!(events[0], AssistantMessageEvent::TextStart));
    assert!(matches!(events[1], AssistantMessageEvent::TextDelta { .. }));
    assert!(matches!(events[2], AssistantMessageEvent::TextEnd));
    assert_eq!(events.len(), 3);
}

#[test]
fn test_output_item_done_message_no_duplicate_text_replay() {
    let mut parser = openai_codex::CodexEventParser::default();
    let events = [
        parser.parse_event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "m1"}
        })),
        parser.parse_event(&serde_json::json!({
            "type": "response.output_item.done",
            "item": {"type": "message", "id": "m1", "content": [
                {"type": "output_text", "text": "hello"}
            ]}
        })),
    ]
    .concat();

    assert_eq!(events.len(), 2);
    assert!(matches!(events[1], AssistantMessageEvent::TextEnd));
}

#[test]
fn test_mixed_multi_item_response_ordering() {
    let mut parser = openai_codex::CodexEventParser::default();
    let events = [
        parser.parse_event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "m1"}
        })),
        parser.parse_event(&serde_json::json!({
            "type": "response.output_text.delta", "item_id": "m1", "delta": "A"
        })),
        parser.parse_event(&serde_json::json!({
            "type": "response.output_item.done",
            "item": {"type": "message", "id": "m1"}
        })),
        parser.parse_event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {"type": "message", "id": "m2"}
        })),
        parser.parse_event(&serde_json::json!({
            "type": "response.output_text.done", "item_id": "m2", "text": "B"
        })),
    ]
    .concat();

    let mut deltas = Vec::new();
    for event in &events {
        if let AssistantMessageEvent::TextDelta { text } = event {
            deltas.push(text.clone());
        }
    }
    assert_eq!(deltas, vec!["A".to_string(), "B".to_string()]);
}

#[test]
fn test_function_call_delta_uses_full_id() {
    let mut parser = openai_codex::CodexEventParser::default();
    let events = [
        parser.parse_event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read"}
        })),
        parser.parse_event(&serde_json::json!({
            "type": "response.function_call_arguments.delta",
            "call_id": "call_1",
            "delta": "{\"path\":\"Cargo.toml\"}"
        })),
        parser.parse_event(&serde_json::json!({
            "type": "response.function_call_arguments.done",
            "call_id": "call_1",
            "id": "fc_1"
        })),
    ]
    .concat();

    assert!(matches!(
        events[0],
        AssistantMessageEvent::ToolCallStart { ref id, .. } if id == "call_1|fc_1"
    ));
    assert!(matches!(
        events[1],
        AssistantMessageEvent::ToolCallDelta { ref id, .. } if id == "call_1|fc_1"
    ));
    assert!(matches!(
        events[2],
        AssistantMessageEvent::ToolCallEnd { ref id } if id == "call_1|fc_1"
    ));
}

#[test]
fn test_response_completed_with_function_call_sets_tool_use() {
    let mut parser = openai_codex::CodexEventParser::default();
    let events = parser.parse_event(&serde_json::json!({
        "type": "response.completed",
        "response": {
            "output": [
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "read"}
            ]
        }
    }));

    assert!(matches!(
        events.last(),
        Some(AssistantMessageEvent::Done {
            stop_reason: StopReason::ToolUse,
            ..
        })
    ));
}

#[test]
fn test_response_completed_emits_final_text_when_no_deltas() {
    let mut parser = openai_codex::CodexEventParser::default();
    let events = parser.parse_event(&serde_json::json!({
        "type": "response.completed",
        "response": {
            "output": [
                {
                    "type": "message",
                    "content": [
                        {"type":"output_text","text":"final answer from completed payload"}
                    ]
                }
            ]
        }
    }));

    assert!(
        events
            .iter()
            .any(|e| matches!(e, AssistantMessageEvent::TextStart))
    );
    assert!(events.iter().any(|e| matches!(
        e,
        AssistantMessageEvent::TextDelta { text } if text.contains("final answer from completed payload")
    )));
    assert!(
        events
            .iter()
            .any(|e| matches!(e, AssistantMessageEvent::TextEnd))
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
fn test_response_completed_function_call_emits_toolcall_events() {
    let mut parser = openai_codex::CodexEventParser::default();
    let events = parser.parse_event(&serde_json::json!({
        "type": "response.completed",
        "response": {
            "output": [
                {
                    "type": "function_call",
                    "id": "fc_9",
                    "call_id": "call_9",
                    "name": "edit",
                    "arguments": "{\"path\":\"a.rs\",\"edits\":[]}"
                }
            ]
        }
    }));

    assert!(events.iter().any(|e| matches!(
        e,
        AssistantMessageEvent::ToolCallStart { id, name } if id == "call_9|fc_9" && name == "edit"
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        AssistantMessageEvent::ToolCallDelta { id, arguments } if id == "call_9|fc_9" && arguments.contains("\"path\"")
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        AssistantMessageEvent::ToolCallEnd { id } if id == "call_9|fc_9"
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        AssistantMessageEvent::Done {
            stop_reason: StopReason::ToolUse,
            ..
        }
    )));
}

#[test]
fn test_output_item_done_function_call_emits_toolcall_events() {
    let mut parser = openai_codex::CodexEventParser::default();
    let events = parser.parse_event(&serde_json::json!({
        "type": "response.output_item.done",
        "item": {
            "type": "function_call",
            "id": "fc_done",
            "call_id": "call_done",
            "name": "edit",
            "arguments": {"path":"a.rs","edits":[]}
        }
    }));

    assert!(events.iter().any(|e| matches!(
        e,
        AssistantMessageEvent::ToolCallStart { id, name } if id == "call_done|fc_done" && name == "edit"
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        AssistantMessageEvent::ToolCallDelta { id, arguments } if id == "call_done|fc_done" && arguments.contains("\"path\"")
    )));
    assert!(events.iter().any(|e| matches!(
        e,
        AssistantMessageEvent::ToolCallEnd { id } if id == "call_done|fc_done"
    )));
}

#[test]
fn test_no_duplicate_tool_arguments_when_added_then_completed_repeat_same_call() {
    let mut parser = openai_codex::CodexEventParser::default();
    let mut acc = EventAccumulator::new();

    let events = [
        parser.parse_event(&serde_json::json!({
            "type": "response.output_item.added",
            "item": {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "read",
                "arguments": "{\"path\":\"Cargo.toml\"}"
            }
        })),
        parser.parse_event(&serde_json::json!({
            "type": "response.completed",
            "response": {
                "output": [{
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "read",
                    "arguments": "{\"path\":\"Cargo.toml\"}"
                }]
            }
        })),
    ]
    .concat();

    for e in &events {
        acc.feed(e);
    }
    let blocks = acc.content_blocks();
    let tc = blocks
        .iter()
        .find_map(|b| match b {
            ContentBlock::ToolCall { arguments, .. } => Some(arguments),
            _ => None,
        })
        .expect("tool call block");
    assert_eq!(tc["path"], "Cargo.toml");
}

#[test]
fn test_parser_marks_done_seen_after_completed_event() {
    let mut parser = openai_codex::CodexEventParser::default();
    assert!(!parser.done_emitted());
    let _ = parser.parse_event(&serde_json::json!({
        "type": "response.completed",
        "response": { "output": [] }
    }));
    assert!(parser.done_emitted());
}

#[test]
fn test_parse_sse_line_accepts_data_without_space() {
    let mut parser = openai_codex::CodexEventParser::default();
    let events = parser.parse_sse_line(
        "data:{\"type\":\"response.output_text.delta\",\"item_id\":\"m1\",\"delta\":\"hello\"}",
    );
    assert!(events.iter().any(|e| matches!(
        e,
        AssistantMessageEvent::TextDelta { text } if text == "hello"
    )));
}

#[test]
fn test_build_request_body_sets_max_output_tokens_when_configured() {
    let model = codex_model();
    let body = build_request_body(
        &model,
        &michin_ai::Context::default(),
        &michin_ai::StreamOptions {
            max_tokens: Some(1234),
            ..Default::default()
        },
    );
    assert_eq!(body["max_output_tokens"], 1234);
}

#[tokio::test]
async fn test_ws_stream_invalid_url_returns_error_not_panic() {
    let body = serde_json::json!({"model":"gpt-5.5","stream":true});
    let result = openai_codex::ws_stream("://bad-url", &body, "", "token", Some(1)).await;
    assert!(result.is_err());
}
