use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::Rect;
use std::sync::atomic::{AtomicUsize, Ordering};
use theta_tui::Action;
use theta_tui::components::editor::{
    Editor, build_vis_lines, byte_to_vis, file_mention_matches, vis_to_byte,
};
use theta_tui::{Component, Theme};

static ROOT_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn temp_root(name: &str) -> std::path::PathBuf {
    let n = ROOT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!("theta-tui-editor-{name}-{n}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn make_editor(text: &str) -> Editor {
    let root = temp_root("nav");
    let mut ed = Editor::new(Theme::default(), root.clone(), vec![], "send".into());
    ed.focus(true);
    ed.set_text(text);
    ed.cached_width = 80;
    ed.cache_dirty = true;
    ed.rebuild_visual_lines(80);
    ed.clamp_scroll();
    ed
}

// ── Visual line helpers ──

#[test]
fn build_vis_lines_empty() {
    let lines = build_vis_lines("", 80);
    assert_eq!(lines.len(), 1);
    assert!(lines[0].is_empty());
}

#[test]
fn build_vis_lines_single_short_line() {
    let lines = build_vis_lines("hello", 80);
    assert_eq!(lines.len(), 1);
    assert_eq!(lines[0].len(), 5);
}

#[test]
fn build_vis_lines_wraps_at_width() {
    let lines = build_vis_lines("abcdefghij", 5);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0].len(), 5);
    assert_eq!(lines[1].len(), 5);
}

#[test]
fn build_vis_lines_newline_splits() {
    let lines = build_vis_lines("ab\ncd", 80);
    assert_eq!(lines.len(), 2);
    assert_eq!(lines[0].len(), 2);
    assert_eq!(lines[1].len(), 2);
}

#[test]
fn byte_to_vis_roundtrip() {
    let text = "hello world";
    let lines = build_vis_lines(text, 80);
    for (i, _ch) in text.char_indices() {
        let (vl, vc) = byte_to_vis(&lines, text, i);
        let back = vis_to_byte(&lines, text.len(), vl, vc);
        assert_eq!(back, i, "byte {i} -> ({vl},{vc}) -> {back}");
    }
}

#[test]
fn byte_to_vis_wrapped_roundtrip() {
    let text = "abcdefghijklmnop";
    let lines = build_vis_lines(text, 5);
    for (i, _ch) in text.char_indices() {
        let (vl, vc) = byte_to_vis(&lines, text, i);
        let back = vis_to_byte(&lines, text.len(), vl, vc);
        assert_eq!(back, i, "byte {i} -> ({vl},{vc}) -> {back}");
    }
}

#[test]
fn byte_to_vis_handles_empty_lines_after_non_empty_line() {
    let text = "a new message\n\n\n\ntesting";
    let lines = build_vis_lines(text, 80);
    let testing_idx = text.find("testing").unwrap();

    let (vl, vc) = byte_to_vis(&lines, text, testing_idx);
    assert_eq!(vl, 4);
    assert_eq!(vc, 0);

    let (vl2, vc2) = byte_to_vis(&lines, text, 14);
    assert_eq!(vl2, 1);
    assert_eq!(vc2, 0);
}

// ── Cursor navigation ──

#[test]
fn cursor_starts_at_end() {
    let ed = make_editor("hello");
    assert_eq!(ed.cursor, 5);
}

#[test]
fn left_moves_backward() {
    let mut ed = make_editor("abc");
    ed.move_left();
    assert_eq!(ed.cursor, 2);
    ed.move_left();
    assert_eq!(ed.cursor, 1);
    ed.move_left();
    assert_eq!(ed.cursor, 0);
    ed.move_left();
    assert_eq!(ed.cursor, 0);
}

#[test]
fn right_moves_forward() {
    let mut ed = make_editor("abc");
    ed.cursor = 0;
    ed.move_right();
    assert_eq!(ed.cursor, 1);
    ed.move_right();
    assert_eq!(ed.cursor, 2);
    ed.move_right();
    assert_eq!(ed.cursor, 3);
    ed.move_right();
    assert_eq!(ed.cursor, 3);
}

#[test]
fn up_down_on_single_line() {
    let mut ed = make_editor("hello");
    assert!(ed.move_up());
    assert_eq!(ed.cursor, 5);
    ed.move_down();
    assert_eq!(ed.cursor, 5);
}

#[test]
fn up_down_multi_line() {
    let mut ed = make_editor("abcd\nefgh\nijkl");
    ed.cursor = 0;
    ed.after_mutate();
    ed.move_down();
    assert!(
        ed.cursor >= 5 && ed.cursor <= 9,
        "cursor={} should be on efgh",
        ed.cursor
    );
    ed.move_down();
    assert!(
        ed.cursor >= 10 && ed.cursor <= 14,
        "cursor={} should be on ijkl",
        ed.cursor
    );
    assert!(ed.move_down());
    assert_eq!(ed.cursor, 10);
    ed.move_up();
    assert!(
        ed.cursor >= 5 && ed.cursor <= 9,
        "cursor={} should be on efgh",
        ed.cursor
    );
    ed.move_up();
    assert!(ed.cursor <= 4, "cursor={} should be on abcd", ed.cursor);
}

#[test]
fn home_end_multi_line() {
    let mut ed = make_editor("abcd\nefgh");
    ed.move_line_start();
    assert_eq!(
        ed.cursor, 5,
        "home should go to start of current visual line"
    );

    ed.move_up();
    ed.move_line_start();
    assert_eq!(ed.cursor, 0, "home on first visual line");

    ed.move_line_end();
    assert_eq!(ed.cursor, 4, "end on first visual line");
}

#[test]
fn word_left_and_right() {
    let mut ed = make_editor("alpha beta gamma");
    ed.cursor = ed.text.len();
    ed.move_word_left();
    assert_eq!(
        &ed.text[ed.cursor..],
        "gamma",
        "first move_word_left: cursor={}",
        ed.cursor
    );

    ed.move_word_left();
    assert_eq!(
        &ed.text[ed.cursor..],
        "beta gamma",
        "second move_word_left: cursor={}",
        ed.cursor
    );

    ed.move_word_right();
    assert_eq!(
        &ed.text[ed.cursor..],
        " gamma",
        "move_word_right: cursor={}",
        ed.cursor
    );
}

#[test]
fn submit_message_via_enter() {
    let root = temp_root("submit-enter");
    let mut ed = Editor::new(Theme::default(), root, vec![], "send".into());
    ed.focus(true);
    ed.set_text("hello world");
    let action = ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    assert!(matches!(action, Some(Action::SendMessage(ref t)) if t == "hello world"));
    assert!(ed.text().is_empty());
}

#[test]
fn enter_behavior_newline_inserts_newline() {
    let root = temp_root("enter-behavior-newline");
    let mut ed = Editor::new(Theme::default(), root, vec![], "newline".into());
    ed.focus(true);
    ed.set_text("hello");

    let action = ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));

    assert!(action.is_none());
    assert_eq!(ed.text(), "hello\n");
}

#[test]
fn insert_newline_via_shift_enter() {
    let mut ed = make_editor("hello");
    ed.cursor = 3;
    let action = ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::SHIFT,
    )));
    assert!(action.is_none());
    assert_eq!(ed.text, "hel\nlo");
    assert_eq!(ed.cursor, 4);
}

#[test]
fn insert_newline_via_alt_enter() {
    let mut ed = make_editor("hello");
    ed.cursor = 3;
    let action = ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::ALT,
    )));
    assert!(action.is_none());
    assert_eq!(ed.text, "hel\nlo");
    assert_eq!(ed.cursor, 4);
}

#[test]
fn arrow_navigation_after_newline() {
    let mut ed = make_editor("abc");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::ALT,
    )));
    assert_eq!(ed.text, "abc\n");
    assert!(
        ed.cursor > 3,
        "cursor should be on new line, got {}",
        ed.cursor
    );
    ed.move_up();
    assert!(
        ed.cursor <= 3,
        "up should go to first line, got {}",
        ed.cursor
    );
    ed.move_down();
    assert!(
        ed.cursor > 3,
        "down should go to second line, got {}",
        ed.cursor
    );
    ed.insert_char('x');
    assert_eq!(ed.text, "abc\nx");
}

#[test]
fn up_lands_on_empty_line_between_content() {
    let mut ed = make_editor("aaa\n\nbbb");
    ed.cursor = ed.text.len();
    ed.move_up();
    let line_start = ed.cursor;
    assert_eq!(
        &ed.text[line_start..],
        "\nbbb",
        "up from last line should go to empty line, got {:?}",
        &ed.text[line_start..]
    );
    ed.move_up();
    let line_start2 = ed.cursor;
    assert_eq!(
        &ed.text[line_start2..line_start2 + 3],
        "aaa",
        "second up should go to first line"
    );
}

#[test]
fn up_down_through_many_empty_lines() {
    let mut ed = make_editor(&"\n".repeat(20));
    assert_eq!(ed.cursor, 20);
    for i in 1..=20 {
        ed.move_up();
        assert_eq!(
            ed.cursor,
            20 - i,
            "up press {} should land at byte {}",
            i,
            20 - i
        );
    }
    for i in 1..=20 {
        ed.move_down();
        assert_eq!(ed.cursor, i, "down press {} should land at byte {}", i, i);
    }
    assert_eq!(ed.cursor, 20);
}

#[test]
fn submit_via_ctrl_enter() {
    let root = temp_root("submit-ctrl-enter");
    let mut ed = Editor::new(Theme::default(), root, vec![], "send".into());
    ed.focus(true);
    ed.set_text("follow up text");
    let action = ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::CONTROL,
    )));
    assert!(matches!(action, Some(Action::FollowUpMessage(ref t)) if t == "follow up text"));
    assert!(ed.text().is_empty());
}

#[test]
fn tab_inserts_two_spaces() {
    let mut ed = make_editor("hello");
    ed.cursor = 5;
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)));
    assert_eq!(ed.text, "hello  ");
    assert_eq!(ed.cursor, 7);
}

#[test]
fn page_up_down() {
    let mut ed = make_editor(
        "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\nline11\nline12",
    );
    ed.last_inner_area = Some(Rect::new(0, 0, 80, 5));
    ed.cursor = 0;
    ed.move_page_down();
    assert!(ed.cursor > 0, "cursor should have moved down");
    let old = ed.cursor;
    ed.move_page_up();
    assert!(ed.cursor < old, "cursor should have moved back up");
}

#[test]
fn click_position_respects_working_dir_and_text() {
    let mut ed = make_editor("hello\nworld");
    ed.last_inner_area = Some(Rect::new(10, 5, 80, 10));
    let pos = ed.mouse_to_cell(10, 5);
    assert_eq!(pos, Some((0, 0)));
    let pos2 = ed.mouse_to_cell(14, 5);
    assert_eq!(pos2, Some((0, 4)));
    let pos3 = ed.mouse_to_cell(10, 6);
    assert_eq!(pos3, Some((1, 0)));
}

// ── File mention matching ──

#[test]
fn file_mentions_recurse_and_fuzzy_match_relative_paths() {
    let root = temp_root("fuzzy-path");
    std::fs::create_dir_all(root.join("src/components")).unwrap();
    std::fs::write(root.join("src/main.rs"), "").unwrap();
    std::fs::write(root.join("src/components/chat.rs"), "").unwrap();
    std::fs::write(root.join("README.md"), "").unwrap();

    let matches = file_mention_matches(&root, "src chat");

    assert_eq!(matches, vec!["src/components/chat.rs".to_string()]);
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn file_mentions_empty_query_returns_visible_files() {
    let root = temp_root("empty");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/lib.rs"), "").unwrap();
    std::fs::write(root.join("Cargo.toml"), "").unwrap();

    let matches = file_mention_matches(&root, "");

    assert_eq!(
        matches,
        vec!["Cargo.toml".to_string(), "src/lib.rs".to_string()]
    );
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn file_mentions_use_git_exclude_standard_when_available() {
    let root = temp_root("git-aware");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/visible.rs"), "").unwrap();
    std::fs::write(root.join("ignored.log"), "").unwrap();
    std::fs::write(root.join(".gitignore"), "*.log\n").unwrap();
    let _ = std::process::Command::new("git")
        .arg("-C")
        .arg(&root)
        .arg("init")
        .output();

    let matches = file_mention_matches(&root, "");

    assert!(matches.contains(&"src/visible.rs".to_string()));
    assert!(!matches.contains(&"ignored.log".to_string()));
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn file_mentions_skips_hidden_paths_by_default() {
    let root = temp_root("hidden");
    std::fs::create_dir_all(root.join(".git")).unwrap();
    std::fs::write(root.join(".git/config"), "").unwrap();
    std::fs::write(root.join("visible.rs"), "").unwrap();

    let matches = file_mention_matches(&root, "");

    assert_eq!(matches, vec!["visible.rs".to_string()]);
    let _ = std::fs::remove_dir_all(root);
}

// ── Event handling ──

#[test]
fn editor_handles_paste_event() {
    let root = temp_root("paste");
    let mut editor = Editor::new(Theme::default(), root.clone(), vec![], "send".into());
    editor.focus(true);

    editor.handle_event(&Event::Paste("hello\nworld".to_string()));

    assert_eq!(editor.text(), "hello\nworld");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn editor_moves_to_start_with_super_up() {
    let root = temp_root("super-up");
    let mut editor = Editor::new(Theme::default(), root.clone(), vec![], "send".into());
    editor.focus(true);
    editor.set_text("abcdef");

    editor.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SUPER)));
    editor.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('!'),
        KeyModifiers::NONE,
    )));

    assert_eq!(editor.text(), "!abcdef");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn editor_moves_word_left_with_alt_left() {
    let root = temp_root("alt-left");
    let mut editor = Editor::new(Theme::default(), root.clone(), vec![], "send".into());
    editor.focus(true);
    editor.set_text("alpha beta");

    editor.handle_event(&Event::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::ALT)));
    editor.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('!'),
        KeyModifiers::NONE,
    )));

    assert_eq!(editor.text(), "alpha !beta");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn editor_moves_word_right_with_alt_right() {
    let root = temp_root("alt-right");
    let mut editor = Editor::new(Theme::default(), root.clone(), vec![], "send".into());
    editor.focus(true);
    editor.set_text("alpha beta");

    editor.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Home,
        KeyModifiers::NONE,
    )));
    editor.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Right,
        KeyModifiers::ALT,
    )));
    editor.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('!'),
        KeyModifiers::NONE,
    )));

    assert_eq!(editor.text(), "alpha! beta");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn slash_autocomplete_dismisses_after_space() {
    let root = temp_root("slash-dismiss");
    let mut ed = Editor::new(
        Theme::default(),
        root,
        vec!["help".into(), "model".into()],
        "send".into(),
    );
    ed.focus(true);

    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('/'),
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('h'),
        KeyModifiers::NONE,
    )));
    assert!(ed.autocomplete_active());

    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char(' '),
        KeyModifiers::NONE,
    )));
    assert!(!ed.autocomplete_active());
}

#[test]
fn autocomplete_selection_wraps() {
    let root = temp_root("autocomplete-wrap");
    let mut ed = Editor::new(
        Theme::default(),
        root,
        vec!["help".into(), "hello".into(), "model".into()],
        "send".into(),
    );
    ed.focus(true);
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('/'),
        KeyModifiers::NONE,
    )));

    let initial = ed.autocomplete_selected();
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Down,
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Down,
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Down,
        KeyModifiers::NONE,
    )));

    assert_eq!(ed.autocomplete_selected(), initial);
}
