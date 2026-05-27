use serde_json::json;
use theta_ai::replay::{normalize_tool_call_id_for_model, sanitize_messages_for_replay};
use theta_ai::{Api, ContentBlock, Message, Modality, Model, ModelCompat, Provider, StopReason};

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
