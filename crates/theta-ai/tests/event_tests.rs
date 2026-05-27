use theta_ai::event::AssistantMessageEvent;
use theta_ai::event::EventAccumulator;
use theta_ai::{ContentBlock, StopReason};

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
