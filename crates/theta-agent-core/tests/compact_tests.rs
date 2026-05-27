use theta_agent_core::compact::compact_messages;
use theta_agent_core::{CompactionConfig, CompactionStrategy};
use theta_ai::{ContentBlock, Message};

fn user(text: &str) -> Message {
    Message::User {
        content: vec![ContentBlock::text(text)],
        timestamp: 0,
    }
}

fn assistant(text: &str) -> Message {
    Message::Assistant {
        content: vec![ContentBlock::text(text)],
        api: None,
        provider: None,
        model: None,
        usage: None,
        stop_reason: None,
        error_message: None,
        timestamp: 0,
    }
}

#[test]
fn test_no_compaction_when_under_budget() {
    let msgs = vec![user("hello"), assistant("hi there")];
    let result = compact_messages(
        &msgs,
        0,
        100,
        &CompactionConfig {
            enabled: true,
            reserve_tokens: 0,
            keep_recent_tokens: 0,
            strategy: CompactionStrategy::Textual,
            summary_max_tokens: 512,
        },
    );
    assert_eq!(result.trimmed_count, 0);
    assert_eq!(result.messages.len(), 2);
}

#[test]
fn test_compaction_trims_oldest() {
    let msgs = vec![
        user("a very long message that takes many tokens to represent"),
        assistant("reply 1"),
        user("another very long message with lots of content"),
        assistant("reply 2"),
        user("current question"),
        assistant("current answer"),
    ];
    let result = compact_messages(
        &msgs,
        0,
        20,
        &CompactionConfig {
            enabled: true,
            reserve_tokens: 0,
            keep_recent_tokens: 0,
            strategy: CompactionStrategy::Textual,
            summary_max_tokens: 512,
        },
    );
    assert!(result.trimmed_count > 0);
    assert!(result.messages.len() < msgs.len());
    let has_user = result
        .messages
        .iter()
        .any(|m| matches!(m, Message::User { .. }));
    assert!(has_user);
}

#[test]
fn test_disabled_compaction() {
    let msgs = vec![
        user("message 1"),
        assistant("reply 1"),
        user("message 2"),
        assistant("reply 2"),
    ];
    let result = compact_messages(
        &msgs,
        0,
        2,
        &CompactionConfig {
            enabled: false,
            reserve_tokens: 0,
            keep_recent_tokens: 0,
            strategy: CompactionStrategy::Textual,
            summary_max_tokens: 512,
        },
    );
    assert_eq!(result.trimmed_count, 0);
    assert_eq!(result.messages.len(), 4);
}

#[test]
fn test_reserve_tokens_reduces_available() {
    let msgs = vec![user("short"), assistant("ok")];
    let result = compact_messages(
        &msgs,
        0,
        100,
        &CompactionConfig {
            enabled: true,
            reserve_tokens: 95,
            keep_recent_tokens: 0,
            strategy: CompactionStrategy::Textual,
            summary_max_tokens: 512,
        },
    );
    assert_eq!(result.trimmed_count, 0);
}

#[test]
fn test_system_prompt_accounted() {
    let msgs = vec![
        user("a long introduction message that takes up space"),
        assistant("brief reply"),
        user("latest question"),
    ];
    let result = compact_messages(
        &msgs,
        10,
        20,
        &CompactionConfig {
            enabled: true,
            reserve_tokens: 0,
            keep_recent_tokens: 0,
            strategy: CompactionStrategy::Textual,
            summary_max_tokens: 512,
        },
    );
    assert!(result.trimmed_count > 0);
}

#[test]
fn test_compaction_inserts_summary() {
    let msgs = vec![
        user("old requirement about files"),
        assistant("old answer about files"),
        user("latest question"),
    ];
    let result = compact_messages(
        &msgs,
        0,
        8,
        &CompactionConfig {
            enabled: true,
            reserve_tokens: 0,
            keep_recent_tokens: 0,
            strategy: CompactionStrategy::Textual,
            summary_max_tokens: 512,
        },
    );
    assert!(result.trimmed_count > 0);
    let Message::Assistant { content, .. } = &result.messages[0] else {
        panic!("expected summary assistant message");
    };
    assert!(matches!(
        &content[0],
        ContentBlock::Text { text } if text.contains("Context compacted")
    ));
}
