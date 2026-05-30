use ratatui::style::Style;
use theta_tui::Theme;
use theta_tui::components::chat::{
    Chat, ChatMessage, ChatRole, format_markdown, should_insert_gap,
};

fn rendered_text(lines: &[ratatui::text::Line<'static>]) -> String {
    lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn normalized_rendered_text(lines: &[ratatui::text::Line<'static>]) -> String {
    let raw = rendered_text(lines);
    let mut rows: Vec<&str> = raw.lines().collect();
    while rows.first().is_some_and(|r| r.is_empty()) {
        rows.remove(0);
    }
    while rows.last().is_some_and(|r| r.is_empty()) {
        rows.pop();
    }
    rows.join("\n")
}

#[test]
fn test_markdown_headers() {
    let theme = Theme::default();
    let style = Style::default();
    let lines = format_markdown("# Top\n## Mid\n### Low\ntext", style, &theme, "", 80);
    let rendered: Vec<String> = lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(rendered.iter().any(|l| l.contains("Top")));
    assert!(rendered.iter().any(|l| l.contains("Mid")));
    assert!(rendered.iter().any(|l| l.contains("Low")));
    assert!(rendered.iter().any(|l| l.contains("text")));
}

#[test]
fn test_code_block() {
    let theme = Theme::default();
    let style = Style::default();
    let lines = format_markdown(
        "before\n```rust\nlet x = 1;\n```\nafter",
        style,
        &theme,
        "",
        80,
    );
    assert!(lines.len() >= 3);
}

#[test]
fn test_task_list_markers() {
    let theme = Theme::default();
    let style = Style::default();
    let lines = format_markdown("- [ ] todo\n- [x] done", style, &theme, "", 80);
    let rendered: Vec<String> = lines
        .iter()
        .map(|l| {
            l.spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>()
        })
        .collect();
    assert!(rendered.iter().any(|l| l.contains("☐")));
    assert!(rendered.iter().any(|l| l.contains("☑")));
}

#[test]
fn test_skill_invocation_prefix_rendered() {
    let chat = Chat::new(Theme::default());
    let lines = chat.format_message(
        &ChatMessage {
            role: ChatRole::User,
            text: "/skill:git-commit".into(),
            tool_name: None,
            is_streaming: false,
        },
        80,
    );
    let has_marker = lines.iter().any(|l| {
        let txt = l
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<String>();
        txt.contains("◈ ")
    });
    assert!(has_marker);
}

#[test]
fn test_user_message_uses_bubble_background() {
    let chat = Chat::new(Theme::default());
    let lines = chat.format_message(
        &ChatMessage {
            role: ChatRole::User,
            text: "hello".into(),
            tool_name: None,
            is_streaming: false,
        },
        80,
    );
    let has_bubble_bg = lines
        .iter()
        .flat_map(|l| l.spans.iter())
        .any(|s| s.style.bg == Some(chat.theme.user_bubble));
    assert!(has_bubble_bg);
}

#[test]
fn test_add_message_does_not_force_scroll_to_bottom() {
    let mut chat = Chat::new(Theme::default());
    chat.scroll_top = 5;
    chat.auto_follow_tail = false;
    chat.add_message(ChatMessage {
        role: ChatRole::Assistant,
        text: "new content".into(),
        tool_name: None,
        is_streaming: false,
    });
    assert_eq!(chat.scroll_top, 5);
    assert!(!chat.auto_follow_tail);
}

#[test]
fn test_markdown_golden_nested_lists_and_links() {
    let theme = Theme::default();
    let style = Style::default();
    let md = "# Title\n- parent\n  - child with [link](https://example.com)\n> quote";
    let lines = format_markdown(md, style, &theme, "", 80);
    let got = normalized_rendered_text(&lines);
    let expected = "Title\n• parent\n  • child with link (https://example.com)\nquote";
    assert_eq!(got, expected);
}

#[test]
fn test_markdown_golden_table_wraps_inside_width() {
    let theme = Theme::default();
    let style = Style::default();
    let md = "| Name | Description |\n| --- | --- |\n| A | verylongtoken_without_spaces_that_must_wrap |\n| B | short |";
    let lines = format_markdown(md, style, &theme, "", 34);
    let got = normalized_rendered_text(&lines);
    let expected = "\
| A   | verylongtoken_without_sp |
|     | aces_that_must_wrap      |
| B   | short                    |";
    assert_eq!(got, expected);
}

#[test]
fn compact_tool_completion_updates_started_row() {
    let mut chat = Chat::new(Theme::default());
    chat.add_message(ChatMessage {
        role: ChatRole::Tool,
        text: "running".to_string(),
        tool_name: Some("read".to_string()),
        is_streaming: true,
    });
    chat.complete_tool_compact("read", "done: src/main.rs");
    assert_eq!(chat.messages.len(), 1);
    assert_eq!(chat.messages[0].text, "done: src/main.rs");
    assert!(!chat.messages[0].is_streaming);
}

#[test]
fn inserts_gap_between_role_groups() {
    assert!(should_insert_gap(ChatRole::User, ChatRole::Assistant));
    assert!(!should_insert_gap(ChatRole::Assistant, ChatRole::Thinking));
    assert!(should_insert_gap(ChatRole::Assistant, ChatRole::Tool));
}

#[test]
#[ignore = "perf characterization; run manually"]
fn perf_large_history_render_cache() {
    let mut chat = Chat::new(Theme::default());
    for i in 0..2500 {
        chat.add_message(ChatMessage {
            role: if i % 2 == 0 {
                ChatRole::User
            } else {
                ChatRole::Assistant
            },
            text: format!("message {i} {}", "x".repeat(120)),
            tool_name: None,
            is_streaming: false,
        });
    }
    let start = std::time::Instant::now();
    chat.rebuild_render_cache(120);
    let elapsed = start.elapsed();
    assert!(elapsed.as_secs() < 10);
}

#[test]
fn test_clear_messages() {
    let mut chat = Chat::new(Theme::default());
    chat.add_message(ChatMessage {
        role: ChatRole::User,
        text: "hello".into(),
        tool_name: None,
        is_streaming: false,
    });
    chat.add_message(ChatMessage {
        role: ChatRole::Assistant,
        text: "hi".into(),
        tool_name: None,
        is_streaming: false,
    });
    assert_eq!(chat.messages.len(), 2);

    chat.clear_messages();

    assert!(chat.messages.is_empty());
    assert!(chat.active_tool_message_idx.is_empty());
    assert_eq!(chat.cached_message_count, 0);
    assert!(chat.cache_dirty);
}

/// Reproduction of the skill-message-rendering bug:
/// When ToolStart updates an existing tool message and a Skill message follows,
/// replace_msg_in_cache consumed the inter-message gap line and corrupted
/// the skill message's cached range.
#[test]
fn tool_start_after_skill_message_preserves_skill_in_cache() {
    const W: usize = 80;
    let mut chat = Chat::new(Theme::default());

    // Prime the cache with inner width, then clear to start fresh
    // while keeping cached_inner_width set so append_last_to_cache works.
    chat.rebuild_render_cache(W);
    chat.clear_messages();

    // 1. Assistant message
    chat.add_message(ChatMessage {
        role: ChatRole::Assistant,
        text: "Let me read the skill file.".to_string(),
        tool_name: None,
        is_streaming: false,
    });
    assert_eq!(chat.messages.len(), 1);
    assert!(!chat.cache_dirty, "cache should be up to date after add_message");

    // 2. ToolCallPrepared → upsert_tool_message creates streaming tool message
    let idx = chat.upsert_tool_message("read", "read: (preparing...)", true);
    assert_eq!(chat.messages.len(), 2);
    assert_eq!(chat.cached_msg_ranges.len(), 2);

    // 3. SkillActivated → add_message inserts skill message after tool
    chat.add_message(ChatMessage {
        role: ChatRole::Skill,
        text: "git-commit".to_string(),
        tool_name: None,
        is_streaming: false,
    });
    assert_eq!(chat.messages.len(), 3);
    assert_eq!(chat.cached_msg_ranges.len(), 3);
    assert_eq!(chat.messages[2].role, ChatRole::Skill);

    // 4. ToolStart → upsert_tool_message updates the existing tool message in-place
    //    This is the exact scenario: replace_msg_in_cache on interior message (idx 1)
    //    with a gap (Tool→Skill) between it and the next message (idx 2).
    let updated_idx = chat.upsert_tool_message("read", "read: SKILL.md (done)", false);
    assert_eq!(updated_idx, idx, "should update existing tool message, not create new");
    assert_eq!(chat.messages.len(), 3, "should still have 3 messages");

    // After fix: cache is dirty (fell back to full rebuild path).
    // Rebuild and verify all messages still have correct cache entries.
    chat.rebuild_render_cache(W);
    assert_eq!(
        chat.cached_msg_ranges.len(),
        3,
        "all 3 messages must have cache entries. Got {} ranges",
        chat.cached_msg_ranges.len(),
    );

    // Verify skill message is still there
    assert_eq!(chat.messages[2].role, ChatRole::Skill);
    assert_eq!(chat.messages[2].text, "git-commit");
}
