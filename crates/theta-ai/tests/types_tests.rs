use serde_json;
use theta_ai::types::approximate_token_count;
use theta_ai::{ContentBlock, Message};

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
