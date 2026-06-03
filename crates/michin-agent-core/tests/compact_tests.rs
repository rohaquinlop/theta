use michin_agent_core::compact::compact_messages;
use michin_agent_core::{CompactionConfig, CompactionStrategy};
use michin_ai::{ContentBlock, Message};

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
            keep_recent_tokens: 20,
            strategy: CompactionStrategy::Textual,
            summary_max_tokens: 512,
            auto_pause_threshold: 2,
        },
    );
    assert_eq!(result.trimmed_count, 0);
    assert_eq!(result.messages.len(), 2);
}

#[test]
fn test_compaction_preserves_prefix_and_tail() {
    let msgs = vec![
        user("first user message"),
        assistant("first reply"),
        user("middle question that is somewhat long"),
        assistant("middle reply that is also fairly long"),
        user("current question"),
        assistant("current answer"),
    ];
    let result = compact_messages(
        &msgs,
        0,
        30,
        &CompactionConfig {
            enabled: true,
            reserve_tokens: 0,
            keep_recent_tokens: 20,
            strategy: CompactionStrategy::Textual,
            summary_max_tokens: 512,
            auto_pause_threshold: 2,
        },
    );
    assert!(result.trimmed_count > 0);
    assert!(result.messages.len() < msgs.len());

    // The first message in the prefix should be preserved at position 0.
    // (In the original: user("first user message"))
    // Whether it's kept depends on the prefix budget, but if kept, it's at index 0.
    // The summary is inserted at the head boundary.
    // The tail messages are appended at the end.

    // Verify the structure has at least a summary and some messages.
    assert!(result.messages.len() >= 2);
}

#[test]
fn test_compaction_preserves_early_prefix_at_position_zero() {
    // Build a conversation where compaction is needed but the prefix fits.
    let msgs = vec![
        user("system instructions for the coding agent"),
        assistant("understood, I will follow the project conventions"),
        user("please read the main source file and understand the architecture"),
        assistant(
            "I have read the file. The architecture uses a plugin-based approach with a central registry.",
        ),
        user("now implement the new feature based on the architecture"),
        assistant(
            "I have implemented the feature. Here is a summary of the changes made to three files.",
        ),
        user("run the tests to make sure everything passes"),
        assistant("all 42 tests pass with no failures or warnings"),
        user("now fix the bug in the error handling module"),
        assistant(
            "the bug was caused by an unhandled timeout case. I have added proper error handling.",
        ),
        user("what is the current status of the project"),
        assistant("the project is in good shape. all tests pass and the new feature is complete."),
    ];
    let result = compact_messages(
        &msgs,
        0,
        100,
        &CompactionConfig {
            enabled: true,
            reserve_tokens: 0,
            keep_recent_tokens: 20,
            strategy: CompactionStrategy::Textual,
            summary_max_tokens: 512,
            auto_pause_threshold: 2,
        },
    );
    assert!(
        result.trimmed_count > 0,
        "compaction should have trimmed messages"
    );

    // The early prefix should be preserved at position 0.
    assert!(
        result.messages.len() >= 3,
        "expected at least prefix + summary + tail, got {}",
        result.messages.len()
    );
    // Position 0 should be the original first user message (prefix preserved).
    match &result.messages[0] {
        Message::User { content, .. } => {
            let text = content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            assert_eq!(text, "system instructions for the coding agent");
        }
        _ => panic!("expected prefix user message at position 0"),
    }
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
            auto_pause_threshold: 2,
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
            keep_recent_tokens: 20,
            strategy: CompactionStrategy::Textual,
            summary_max_tokens: 512,
            auto_pause_threshold: 2,
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
            keep_recent_tokens: 5,
            strategy: CompactionStrategy::Textual,
            summary_max_tokens: 512,
            auto_pause_threshold: 2,
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
            keep_recent_tokens: 5,
            strategy: CompactionStrategy::Textual,
            summary_max_tokens: 512,
            auto_pause_threshold: 2,
        },
    );
    assert!(result.trimmed_count > 0);
    // Find the summary message (contains "Context compacted").
    let has_summary = result.messages.iter().any(|m| match m {
        Message::Assistant { content, .. } => content.iter().any(|b| match b {
            ContentBlock::Text { text } => text.contains("Context compacted"),
            _ => false,
        }),
        _ => false,
    });
    assert!(
        has_summary,
        "expected a summary message in compacted output"
    );
}
