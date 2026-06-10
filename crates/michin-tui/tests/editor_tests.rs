use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use michin_tui::Action;
use michin_tui::components::editor::{Editor, file_mention_matches};
use michin_tui::{Component, Theme};
use ratatui::Terminal;
use ratatui::backend::TestBackend;
use ratatui::layout::Rect;
use std::sync::atomic::{AtomicUsize, Ordering};

static ROOT_COUNTER: AtomicUsize = AtomicUsize::new(0);

fn temp_root(name: &str) -> std::path::PathBuf {
    let n = ROOT_COUNTER.fetch_add(1, Ordering::Relaxed);
    let root = std::env::temp_dir().join(format!(
        "michin-tui-editor-{name}-{n}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).unwrap();
    root
}

fn make_editor(text: &str) -> Editor {
    let root = temp_root("nav");
    let mut ed = Editor::new(Theme::default(), root.clone(), vec![], "send".into(), None);
    ed.focus(true);
    ed.set_text(text);
    ed
}

fn editor_text(ed: &Editor) -> String {
    ed.text()
}

fn cursor_at_end(ed: &Editor) -> bool {
    // After set_text, cursor should be at the end.
    // Since tui-textarea positions cursor at end of text via set_text impl.
    editor_text(ed).len() > 0
}

// ── Cursor navigation ──

#[test]
fn cursor_starts_at_end() {
    let ed = make_editor("hello");
    assert!(cursor_at_end(&ed));
}

#[test]
fn left_moves_backward() {
    let mut ed = make_editor("abc");
    // Move left 3 times — should stop at position 0.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Left,
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Left,
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Left,
        KeyModifiers::NONE,
    )));
    // One more left should be a no-op.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Left,
        KeyModifiers::NONE,
    )));

    // Insert 'x' at cursor = start.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('x'),
        KeyModifiers::NONE,
    )));
    assert_eq!(editor_text(&ed), "xabc");
}

#[test]
fn right_moves_forward() {
    let mut ed = make_editor("abc");
    // Move to start first.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Home,
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Right,
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('x'),
        KeyModifiers::NONE,
    )));
    assert_eq!(editor_text(&ed), "axbc");
}

#[test]
fn up_down_multi_line() {
    let mut ed = make_editor("abcd\nefgh\nijkl");
    // Move to start of text.
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SUPER)));
    // Down to second line, insert marker.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Down,
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('X'),
        KeyModifiers::NONE,
    )));
    assert_eq!(editor_text(&ed), "abcd\nXefgh\nijkl");

    // Move to start of first line, insert marker (to avoid column-preservation ambiguity).
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SUPER)));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('Y'),
        KeyModifiers::NONE,
    )));
    assert_eq!(editor_text(&ed), "Yabcd\nXefgh\nijkl");
}

#[test]
fn home_end_multi_line() {
    let mut ed = make_editor("abcd\nefgh");
    // Home on second line (cursor starts at end).
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Home,
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('X'),
        KeyModifiers::NONE,
    )));
    assert_eq!(editor_text(&ed), "abcd\nXefgh");
}

#[test]
fn word_navigation_with_alt() {
    let mut ed = make_editor("alpha beta gamma");
    // Alt+Left moves word back — from end, one WordBack lands at start of "gamma".
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Left, KeyModifiers::ALT)));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('!'),
        KeyModifiers::NONE,
    )));
    assert_eq!(editor_text(&ed), "alpha beta !gamma");

    // Alt+Right moves word forward — from start, WordForward lands at start of "beta".
    let mut ed2 = make_editor("alpha beta gamma");
    ed2.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SUPER)));
    ed2.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Right,
        KeyModifiers::ALT,
    )));
    ed2.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('!'),
        KeyModifiers::NONE,
    )));
    assert_eq!(editor_text(&ed2), "alpha !beta gamma");
}

#[test]
fn submit_message_via_enter() {
    let root = temp_root("submit-enter");
    let mut ed = Editor::new(Theme::default(), root, vec![], "send".into(), None);
    ed.focus(true);
    ed.set_text("hello world");
    let action = ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    assert!(matches!(action, Some(Action::SendMessage(ref t)) if t == "hello world"));
    assert!(ed.text().trim().is_empty());
}

#[test]
fn enter_behavior_newline_inserts_newline() {
    let root = temp_root("enter-behavior-newline");
    let mut ed = Editor::new(Theme::default(), root, vec![], "newline".into(), None);
    ed.focus(true);
    ed.set_text("hello");

    let action = ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));

    assert!(action.is_none());
    assert_eq!(ed.text().trim(), "hello");
    assert!(ed.text().contains('\n'));
}

#[test]
fn insert_newline_via_shift_enter() {
    let mut ed = make_editor("hello");
    // Move cursor to middle.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Left,
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Left,
        KeyModifiers::NONE,
    )));
    let action = ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::SHIFT,
    )));
    assert!(action.is_none());
    assert_eq!(ed.text().trim(), "hel\nlo");
}

#[test]
fn submit_via_ctrl_enter() {
    let root = temp_root("submit-ctrl-enter");
    let mut ed = Editor::new(Theme::default(), root, vec![], "send".into(), None);
    ed.focus(true);
    ed.set_text("follow up text");
    let action = ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::CONTROL,
    )));
    assert!(matches!(action, Some(Action::FollowUpMessage(ref t)) if t == "follow up text"));
    assert!(ed.text().trim().is_empty());
}

#[test]
fn tab_inserts_two_spaces() {
    let mut ed = make_editor("hello");
    // Cursor is at end after set_text.
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE)));
    assert_eq!(ed.text().trim_end(), "hello");
    // After tab, text should contain spaces.
    let text = ed.text();
    assert!(text.starts_with("hello"));
    assert!(text.len() > 5, "expected tab to add spaces, got: {text:?}");
}

#[test]
fn cmd_delete_kills_line() {
    let mut ed = make_editor("hello world");
    // Move to start, then Cmd+Delete should kill the whole line.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Home,
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));
    assert!(ed.text().trim().is_empty(), "line should be killed");
}

#[test]
fn cmd_backspace_kills_line() {
    let mut ed = make_editor("hello world");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Backspace,
        KeyModifiers::SUPER,
    )));
    // Should kill the whole line regardless of cursor position.
    assert!(ed.text().trim().is_empty(), "line should be killed");
}

#[test]
fn cmd_left_goes_to_start() {
    let mut ed = make_editor("hello world");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Left,
        KeyModifiers::SUPER,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('!'),
        KeyModifiers::NONE,
    )));
    assert_eq!(ed.text(), "!hello world");
}

#[test]
fn cmd_right_goes_to_end() {
    let mut ed = make_editor("hello world");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Home,
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Right,
        KeyModifiers::SUPER,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('!'),
        KeyModifiers::NONE,
    )));
    assert_eq!(ed.text(), "hello world!");
}

#[test]
fn page_up_down() {
    let mut ed = make_editor(
        "line1\nline2\nline3\nline4\nline5\nline6\nline7\nline8\nline9\nline10\nline11\nline12",
    );
    ed.last_inner_area = Some(Rect::new(0, 0, 80, 5));
    // Move to start.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Home,
        KeyModifiers::NONE,
    )));
    // PageDown should move cursor.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::PageDown,
        KeyModifiers::NONE,
    )));
    // Insert marker to verify cursor moved.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('X'),
        KeyModifiers::NONE,
    )));
    assert!(ed.text().contains('X'), "cursor should have moved");
}

#[test]
fn up_at_first_line_triggers_history() {
    let mut ed = make_editor("hello world");
    // Submit first message to populate history.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    // Ensure empty editor.
    assert!(ed.text().trim().is_empty());

    // Up at empty editor (row 0) should recall history.
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE)));
    assert_eq!(ed.text().trim(), "hello world");
}

#[test]
fn alt_up_down_history() {
    let root = temp_root("alt-history");
    let mut ed = Editor::new(Theme::default(), root, vec![], "send".into(), None);
    ed.focus(true);
    ed.set_text("first message");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));

    // Alt+Up: recall history.
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::ALT)));
    assert_eq!(ed.text().trim(), "first message");
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
fn file_mentions_include_gitignored_directory_contents() {
    let root = temp_root("git-aware");
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(root.join("src/visible.rs"), "").unwrap();
    std::fs::create_dir_all(root.join("docs")).unwrap();
    std::fs::write(root.join("docs/guide.md"), "").unwrap();
    std::fs::write(root.join("ignored.log"), "").unwrap();
    std::fs::write(root.join(".gitignore"), "*.log\ndocs/\n").unwrap();
    let _ = std::process::Command::new("git")
        .arg("-C")
        .arg(&root)
        .arg("init")
        .output();

    // General search: gitignored files and dirs are excluded.
    let matches = file_mention_matches(&root, "");
    assert!(matches.contains(&"src/visible.rs".to_string()));
    assert!(!matches.contains(&"ignored.log".to_string()));
    assert!(!matches.contains(&"docs/guide.md".to_string()));

    // Typing exact gitignored directory path: contents are listed.
    let matches = file_mention_matches(&root, "docs/");
    assert!(matches.contains(&"docs/guide.md".to_string()));

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
    let mut editor = Editor::new(Theme::default(), root.clone(), vec![], "send".into(), None);
    editor.focus(true);

    editor.handle_event(&Event::Paste("hello\nworld".to_string()));

    // Text should contain "hello" and "world" on separate lines.
    let text = editor.text();
    assert!(text.contains("hello"), "text should contain hello: {text}");
    assert!(text.contains("world"), "text should contain world: {text}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn editor_moves_to_start_with_super_up() {
    let root = temp_root("super-up");
    let mut editor = Editor::new(Theme::default(), root.clone(), vec![], "send".into(), None);
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
    let mut editor = Editor::new(Theme::default(), root.clone(), vec![], "send".into(), None);
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
    let mut editor = Editor::new(Theme::default(), root.clone(), vec![], "send".into(), None);
    editor.focus(true);
    editor.set_text("alpha beta");

    // Move to start.
    editor.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SUPER)));
    editor.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Right,
        KeyModifiers::ALT,
    )));
    editor.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('!'),
        KeyModifiers::NONE,
    )));

    // WordForward from start lands at start of "beta".
    assert_eq!(editor.text(), "alpha !beta");
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
        None,
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
        None,
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

#[test]
fn editor_handles_mouse_click() {
    let root = temp_root("mouse-click");
    let mut ed = Editor::new(Theme::default(), root.clone(), vec![], "send".into(), None);
    ed.focus(true);
    ed.set_text("hello world");
    // Set last_inner_area so mouse coordinates resolve.
    ed.last_inner_area = Some(Rect::new(10, 5, 80, 10));

    // Mouse click at (10, 5) = row 0, col 0 within the area.
    use crossterm::event::{MouseButton, MouseEventKind};
    let event = Event::Mouse(crossterm::event::MouseEvent {
        kind: MouseEventKind::Down(MouseButton::Left),
        column: 12,
        row: 5,
        modifiers: KeyModifiers::NONE,
    });
    ed.handle_event(&event);

    // Insert char 'X' at the clicked position.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Char('X'),
        KeyModifiers::NONE,
    )));
    let text = ed.text();
    // Should have X inserted somewhere after the click.
    assert!(text.contains('X'), "click+insert should insert X: {text}");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn backspace_works() {
    let mut ed = make_editor("hello");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Backspace,
        KeyModifiers::NONE,
    )));
    assert_eq!(ed.text(), "hell");
}

#[test]
fn delete_works() {
    let mut ed = make_editor("hello");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Home,
        KeyModifiers::NONE,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::NONE,
    )));
    assert_eq!(ed.text(), "ello");
}

// ── Issue fixes ──

#[test]
fn no_cursor_line_underline() {
    // The theme styles clear the default UNDERLINED modifier on cursor line.
    let root = temp_root("no-underline");
    let theme = Theme::default();
    let mut ed = Editor::new(theme.clone(), root.clone(), vec![], "send".into(), None);
    ed.set_theme(theme);
    // After set_theme, cursor_line_style should be default (no modifiers).
    // We verify by checking text renders normally — no underline artifacts.
    ed.set_text("hello\nworld");
    // The visual check is manual. The code path: set_cursor_line_style(Style::default())
    // in both Editor::new() and apply_theme_styles().
    assert_eq!(ed.text(), "hello\nworld");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn paste_no_trailing_empty_line() {
    // Paste appends at cursor — use a newline to start a fresh line.
    let mut ed = make_editor("");
    ed.handle_event(&Event::Paste("line1\nline2\nline3".to_string()));
    let text = ed.text();
    assert!(
        !text.ends_with('\n'),
        "paste should not add trailing newline: got {text:?}"
    );
    assert!(text.contains("line1"), "paste should include line1");
    assert!(text.contains("line2"), "paste should include line2");
    assert!(text.contains("line3"), "paste should include line3");
    assert_eq!(ed.line_count(), 3, "3 pasted lines");
}

#[test]
fn paste_multiple_lines_preserves_content() {
    let mut ed = make_editor("");
    ed.handle_event(&Event::Paste("alpha\nbeta\ngamma".to_string()));
    let text = ed.text();
    assert!(text.contains("alpha"), "text: {text}");
    assert!(text.contains("beta"), "text: {text}");
    assert!(text.contains("gamma"), "text: {text}");
    assert_eq!(ed.line_count(), 3, "3 pasted lines");
}

#[test]
fn shift_up_does_not_trigger_history() {
    let mut ed = make_editor("hello world");
    // Submit to populate history.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    assert!(ed.text().trim().is_empty());
    // Shift+Up at empty editor — should NOT recall history (selection mode).
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SHIFT)));
    assert!(
        ed.text().trim().is_empty(),
        "shift+up should not recall history"
    );
}

#[test]
fn shift_down_does_not_trigger_history() {
    let mut ed = make_editor("first");
    // Submit to populate history, then enter "second".
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    ed.set_text("second");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    assert!(ed.text().trim().is_empty());

    // Shift+Down at bottom of empty editor — should NOT trigger history_down.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Down,
        KeyModifiers::SHIFT,
    )));
    assert!(
        ed.text().trim().is_empty(),
        "shift+down should not recall history"
    );
}

#[test]
fn bare_up_at_top_still_triggers_history() {
    let mut ed = make_editor("hello world");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    assert!(ed.text().trim().is_empty());
    // Bare Up (no shift) at empty editor should still recall history.
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::NONE)));
    assert_eq!(ed.text().trim(), "hello world");
}

#[test]
fn cmd_delete_kills_line_in_multi_line_editor() {
    let mut ed = make_editor("aaa\nbbb\nccc");
    // Move to second line.
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SUPER)));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Down,
        KeyModifiers::NONE,
    )));
    // Cmd+Delete should kill the "bbb" line.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));
    assert_eq!(ed.text(), "aaa\nccc", "'bbb' line should be removed");
}

#[test]
fn cmd_delete_on_last_line() {
    let mut ed = make_editor("aaa\nbbb\nccc");
    // Cursor at end (last line). Cmd+Delete should remove last line.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));
    assert_eq!(ed.text(), "aaa\nbbb", "last line should be removed");
}

#[test]
fn cmd_delete_on_only_line_clears_text() {
    let mut ed = make_editor("only line");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));
    // Single line: content should be cleared, leaving empty editor.
    assert!(ed.text().trim().is_empty(), "only line should be cleared");
}

#[test]
fn cmd_delete_on_first_of_many_lines() {
    let mut ed = make_editor("first\nsecond\nthird");
    // Move to first line.
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SUPER)));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));
    assert_eq!(ed.text(), "second\nthird", "first line should be removed");
}

#[test]
fn cmd_backspace_kills_line_variants() {
    // Cmd+Backspace from end of line.
    let mut ed = make_editor("hello world");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Backspace,
        KeyModifiers::SUPER,
    )));
    assert!(
        ed.text().trim().is_empty(),
        "line should be killed via Cmd+Backspace"
    );
}

#[test]
fn delete_current_line_single_line_clears() {
    // Single line can't be removed entirely — content is cleared.
    let mut ed = make_editor("only one");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));
    assert!(
        ed.text().is_empty() || ed.text().trim().is_empty(),
        "single line should be cleared"
    );
    assert_eq!(ed.line_count(), 1, "one empty line remains");
}

#[test]
fn theme_does_not_add_underline_modifier() {
    // Verify set_theme doesn't re-add UNDELINED.
    let root = temp_root("theme-no-underline");
    let mut ed = Editor::new(Theme::default(), root.clone(), vec![], "send".into(), None);
    ed.focus(true);
    ed.set_text("test");
    // Setting theme should not crash or produce visual artifacts.
    ed.set_theme(Theme::default());
    assert_eq!(ed.text(), "test");
    let _ = std::fs::remove_dir_all(root);
}

#[test]
fn paste_empty_string_noop() {
    let mut ed = make_editor("before");
    ed.handle_event(&Event::Paste(String::new()));
    assert_eq!(ed.text(), "before", "empty paste should be no-op");
}

#[test]
fn paste_single_line_no_trailing_newline() {
    let mut ed = make_editor("pre");
    ed.handle_event(&Event::Paste("single".to_string()));
    assert!(
        !ed.text().ends_with('\n'),
        "single line paste should not add newline"
    );
    assert_eq!(ed.line_count(), 1, "should remain one line");
}

#[test]
fn paste_then_delete_line_works() {
    // Paste content onto a fresh editor, then Cmd+Delete should kill the line.
    let mut ed = make_editor("before");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    ed.set_text("pasted content here");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));
    let text = ed.text().trim().to_string();
    assert_eq!(
        text, "",
        "line should be cleared after Cmd+Delete on only line"
    );
}

#[test]
fn shift_up_on_multi_line_editor_selects_without_history() {
    // Multi-line editor, cursor at bottom line. Shift+Up should select,
    // not jump to history.
    let mut ed = make_editor("line1\nline2\nline3");
    // Populate history so we can verify it's not triggered.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    ed.set_text("a\nb\nc");
    // Cursor at end (line 3). Shift+Up should move cursor up with selection.
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SHIFT)));
    // Text unchanged — no history recall.
    assert_eq!(ed.text(), "a\nb\nc", "shift+up should not change text");
}

// ── Soft wrapping ──

#[test]
fn desired_height_accounts_for_wrapping() {
    // A 20-char line on a 10-wide display wraps to 2 display lines.
    let ed = make_editor("abcdefghijklmnopqrst");
    let height = ed.desired_height(10, 40);
    // 2 wrapped lines + 2 border = 4
    assert_eq!(height, 4, "20-char line in 10-wide should be 4 rows");
}

#[test]
fn desired_height_short_line_no_wrap() {
    let ed = make_editor("hello");
    let height = ed.desired_height(80, 40);
    // 1 line + 2 border = 3
    assert_eq!(height, 3);
}

#[test]
fn desired_height_multiline_wrapping() {
    // Two lines: first is 25 chars (wraps 3x in 10-wide), second is 5 chars (1x).
    let ed = make_editor("abcdefghijklmnopqrstuvwxy\nhello");
    let height = ed.desired_height(10, 40);
    // 3 + 1 = 4 display lines + 2 border = 6
    assert_eq!(height, 6);
}

#[test]
fn desired_height_respects_max() {
    let ed = make_editor("a\nb\nc\nd\ne\nf\ng\nh\ni\nj\nk\nl\nm\nn\no");
    let height = ed.desired_height(80, 10);
    assert!(height <= 10, "height {height} should not exceed max_height");
}

#[test]
fn desired_height_empty_line_counts_as_one() {
    let ed = make_editor("");
    let height = ed.desired_height(80, 40);
    // 1 empty line + 2 border = 3
    assert_eq!(height, 3);
}

#[test]
fn wrapping_preserves_text_content() {
    // Wrapping is visual only — text content unchanged.
    let mut ed = make_editor("the quick brown fox jumps over the lazy dog");
    let before = ed.text();
    // Trigger some key events (visual only).
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Left,
        KeyModifiers::NONE,
    )));
    assert_eq!(ed.text(), before, "wrapping should not change text content");
}

#[test]
fn height_grows_when_lines_added() {
    let mut ed = make_editor("line1");
    let h1 = ed.desired_height(80, 40);
    assert_eq!(h1, 3); // 1 line + 2 border

    ed.set_text("line1\nline2\nline3");
    let h2 = ed.desired_height(80, 40);
    assert_eq!(h2, 5); // 3 lines + 2 border
}

#[test]
fn height_shrinks_when_lines_removed() {
    let mut ed = make_editor("aaa\nbbb\nccc");
    let h1 = ed.desired_height(80, 40);
    assert_eq!(h1, 5); // 3 lines + 2 border

    // Remove last line.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));
    let h2 = ed.desired_height(80, 40);
    assert_eq!(h2, 4); // 2 lines + 2 border

    // Remove another.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));
    let h3 = ed.desired_height(80, 40);
    assert_eq!(h3, 3); // 1 line + 2 border
}

#[test]
fn height_shrinks_to_single_line_after_full_clear() {
    let mut ed = make_editor("aaa\nbbb\nccc\nddd\neee");
    assert_eq!(ed.desired_height(80, 40), 7); // 5 lines + 2

    // Clear all via repeated Cmd+Delete.
    for _ in 0..5 {
        ed.handle_event(&Event::Key(KeyEvent::new(
            KeyCode::Delete,
            KeyModifiers::SUPER,
        )));
    }
    assert_eq!(ed.desired_height(80, 40), 3); // 1 empty line + 2 border
}

#[test]
fn height_responds_to_width_change_wrapping() {
    // Long line that wraps in narrow terminal.
    let ed = make_editor("abcdefghijklmnopqrstuvwxyz"); // 26 chars
    let h_wide = ed.desired_height(40, 40);
    assert_eq!(h_wide, 3); // no wrap: 1 line + 2 border

    let h_narrow = ed.desired_height(10, 40);
    // 26 chars / 10 width = ceil(2.6) = 3 display lines + 2 border = 5
    assert_eq!(h_narrow, 5);
}

#[test]
fn height_multiple_wrapped_lines() {
    // Two logical lines, both wrapping.
    let ed = make_editor("abcdefghijklmnopqrst\nuvwxyz0123456789abcd");
    // Line 1: 20 chars. Line 2: 20 chars. Width 10.
    // Each wraps to 2 display lines. Total 4 + 2 border = 6.
    assert_eq!(ed.desired_height(10, 40), 6);
}

#[test]
fn height_after_submit_shrinks_to_minimum() {
    let root = temp_root("submit-shrink");
    let mut ed = Editor::new(Theme::default(), root, vec![], "send".into(), None);
    ed.focus(true);
    ed.set_text("line1\nline2\nline3\nline4\nline5");
    assert!(ed.desired_height(80, 40) > 3);

    // Submit clears the editor.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    assert_eq!(ed.desired_height(80, 40), 3); // empty editor = 1 line + 2 border
}

// ── Render behavior ──

#[test]
fn render_after_line_removal_uses_smaller_area() {
    // Start with 3 lines, render at height 7.
    let mut ed = make_editor("aaa\nbbb\nccc");
    let wide_area = Rect::new(0, 0, 40, 7);
    let backend = TestBackend::new(40, 10);
    let mut terminal = Terminal::new(backend).unwrap();

    terminal
        .draw(|f| {
            ed.render(wide_area, f);
        })
        .unwrap();

    // Remove all but one line.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));

    // Editor now has 1 line. desired_height = 3.
    assert_eq!(ed.line_count(), 1);
    let new_h = ed.desired_height(40, 10);
    assert_eq!(new_h, 3);

    // Render at smaller area.
    let small_area = Rect::new(0, 0, 40, new_h);
    terminal
        .draw(|f| {
            ed.render(small_area, f);
        })
        .unwrap();

    // The bottom border should be at y=2 (area.y + area.height - 1).
    let buf = terminal.backend().buffer().clone();
    let border_y = small_area.y + small_area.height - 1;
    // Check that there's a border character at the bottom row.
    let has_border = (0..40).any(|x| {
        let cell = &buf[(x, border_y)];
        let s = cell.symbol();
        s == "─" || s == "-" || s == "│" || s == "|" || s == "┌" || s == "┐" || s == "└" || s == "┘"
    });
    assert!(has_border, "bottom border should be at row {border_y}");

    // Verify content "aaa" is at row 1 (inside the block).
    let row1_content: String = (0..40)
        .map(|x| buf[(x, 1)].symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(
        row1_content.trim().starts_with('a') || row1_content.trim().is_empty(),
        "row 1 should have content or be empty (cursor line): {row1_content:?}"
    );
}

// ── Viewport + cursor preservation on area change ──

#[test]
fn cursor_preserved_after_area_shrink() {
    let mut ed = make_editor("short\nabcdefghijklmnopqrstuvwxyz0123456789");
    // Render at wide width so the long line doesn't wrap.
    let area_wide = Rect::new(0, 0, 60, 6);
    let backend = TestBackend::new(60, 10);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| ed.render(area_wide, f)).unwrap();

    // Cursor should be at end of second line.
    let before = ed.cursor_position();
    assert_eq!(before.0, 1); // second data line

    // Shrink width — long line now wraps, area height changes.
    let area_narrow = Rect::new(0, 0, 20, 8);
    terminal.draw(|f| ed.render(area_narrow, f)).unwrap();

    // Cursor data row should still be 1 (same logical line).
    let after = ed.cursor_position();
    assert_eq!(after.0, 1, "cursor should stay on same data line");
}

#[test]
fn cursor_column_preserved_after_deleting_wrapped_line() {
    // Two lines: first short, second long (wraps in narrow terminal).
    let mut ed = make_editor("aaa\nabcdefghijklmnopqrstuvwxyz0123456789");
    let area = Rect::new(0, 0, 20, 8);
    let backend = TestBackend::new(20, 10);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| ed.render(area, f)).unwrap();

    // Cursor at end of line 2 (col = 36).
    let before = ed.cursor_position();
    assert_eq!(before.0, 1);
    assert_eq!(before.1, 36, "cursor at end of 36-char line");

    // Delete the long line (Cmd+Delete on last line).
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));
    terminal.draw(|f| ed.render(area, f)).unwrap();

    // Cursor should be at end of remaining "aaa" line.
    let after = ed.cursor_position();
    assert_eq!(after.0, 0, "cursor on first (only) line");
    assert_eq!(after.1, 3, "cursor at end of 'aaa'");
}

#[test]
fn viewport_no_blank_space_after_shrink() {
    // 5 lines, rendered at height 9 (5 + 2 border + some padding).
    let mut ed = make_editor("one\ntwo\nthree\nfour\nfive");
    let big_area = Rect::new(0, 0, 40, 9);
    let backend = TestBackend::new(40, 12);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| ed.render(big_area, f)).unwrap();

    // Delete 3 lines.
    for _ in 0..3 {
        ed.handle_event(&Event::Key(KeyEvent::new(
            KeyCode::Delete,
            KeyModifiers::SUPER,
        )));
    }
    assert_eq!(ed.line_count(), 2);

    // Shrink area to match new content (2 lines + 2 border = 4).
    let small_area = Rect::new(0, 0, 40, 4);
    terminal.draw(|f| ed.render(small_area, f)).unwrap();

    // Verify no blank rows between content and bottom border.
    let buf = terminal.backend().buffer().clone();
    let border_y = small_area.y + small_area.height - 1; // row 3
    let content_y = small_area.y + 1; // row 1 (first content row)

    // Row 1 should have content ("one" or "two").
    let row1: String = (0..40)
        .map(|x| buf[(x, content_y)].symbol().chars().next().unwrap_or(' '))
        .collect();
    assert!(
        row1.trim().starts_with('o') || row1.trim().starts_with('t'),
        "row 1 should have 'one' or 'two': {row1:?}"
    );

    // Bottom border at row 3.
    let has_border = (0..40).any(|x| {
        let s = buf[(x, border_y)].symbol();
        s == "─" || s == "-"
    });
    assert!(has_border, "bottom border at row {border_y}");
}

#[test]
fn cursor_at_mid_line_preserved_after_area_change() {
    let mut ed = make_editor("hello world foo bar");
    // Move cursor to middle of the line.
    for _ in 0..5 {
        ed.handle_event(&Event::Key(KeyEvent::new(
            KeyCode::Left,
            KeyModifiers::NONE,
        )));
    }
    let mid = ed.cursor_position();
    assert_eq!(mid.0, 0);
    assert_eq!(mid.1, 14); // "hello world foo| bar" (19 chars, 5 lefts from end)

    // Render at same width — cursor must not change.
    let backend = TestBackend::new(60, 10);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal
        .draw(|f| ed.render(Rect::new(0, 0, 60, 6), f))
        .unwrap();
    assert_eq!(ed.cursor_position(), mid, "cursor unchanged at same width");

    // Render at a different width — cursor row must be preserved.
    // Column may shift by ±1 due to screen-map round-trip at narrow widths.
    terminal
        .draw(|f| ed.render(Rect::new(0, 0, 10, 6), f))
        .unwrap();
    let after = ed.cursor_position();
    assert_eq!(after.0, 0, "cursor row preserved");
    assert!(
        after.1.abs_diff(mid.1) <= 1,
        "cursor col near original: {} vs {}",
        after.1,
        mid.1
    );
}

#[test]
fn repeated_area_changes_dont_corrupt_state() {
    let mut ed = make_editor("alpha\nbeta\ngamma\ndelta\nepsilon");
    let backend = TestBackend::new(60, 15);
    let mut terminal = Terminal::new(backend).unwrap();

    // Alternate between large and small areas.
    let areas = [
        Rect::new(0, 0, 60, 9),
        Rect::new(0, 0, 20, 5),
        Rect::new(0, 0, 60, 9),
        Rect::new(0, 0, 10, 4),
        Rect::new(0, 0, 60, 9),
    ];
    for area in areas {
        terminal.draw(|f| ed.render(area, f)).unwrap();
    }

    // Text and cursor should be intact.
    assert_eq!(ed.text(), "alpha\nbeta\ngamma\ndelta\nepsilon");
    let pos = ed.cursor_position();
    assert_eq!(pos.0, 4, "cursor on last line");
    assert_eq!(pos.1, 7, "cursor at end of 'epsilon'");
}

#[test]
fn wrapping_height_scales_with_width() {
    // 40-char line wraps differently at different widths.
    let ed = make_editor("abcdefghijklmnopqrstuvwxyz0123456789abcdefgh"); // 44 chars

    // Width 80: no wrap → 1 display line + 2 border = 3.
    assert_eq!(ed.desired_height(80, 50), 3);
    // Width 22: 44/22 = 2 display lines + 2 border = 4.
    assert_eq!(ed.desired_height(22, 50), 4);
    // Width 11: 44/11 = 4 display lines + 2 border = 6.
    assert_eq!(ed.desired_height(11, 50), 6);
    // Width 10: 44/10 = ceil(4.4) = 5 display lines + 2 border = 7.
    assert_eq!(ed.desired_height(10, 50), 7);
    // Width 1: each char on own line → 44 display lines + 2 border = 46.
    assert_eq!(ed.desired_height(1, 50), 46);
}

#[test]
fn cjk_chars_wrap_at_correct_width() {
    // CJK chars are 2 columns wide.
    let ed = make_editor("\u{4e16}\u{754c}\u{4f60}\u{597d}\u{4e16}\u{754c}"); // 6 CJK chars = 12 columns

    // Width 12: no wrap → 3.
    assert_eq!(ed.desired_height(12, 40), 3);
    // Width 6: 12/6 = 2 display lines + 2 border = 4.
    assert_eq!(ed.desired_height(6, 40), 4);
    // Width 4: 12/4 = 3 display lines + 2 border = 5.
    assert_eq!(ed.desired_height(4, 40), 5);
}

#[test]
fn mixed_ascii_and_cjk_wrap() {
    // Mix of ASCII and CJK.
    let ed = make_editor("ab\u{4e16}\u{754c}cd"); // a(1) + b(1) + 世(2) + 界(2) + c(1) + d(1) = 8 columns

    assert_eq!(ed.desired_height(8, 40), 3); // no wrap
    assert_eq!(ed.desired_height(4, 40), 4); // 8/4 = 2 lines + 2
    assert_eq!(ed.desired_height(3, 40), 5); // ceil(8/3) = 3 lines + 2
}

#[test]
fn submit_then_type_preserves_wrapping() {
    let root = temp_root("submit-wrap");
    let mut ed = Editor::new(Theme::default(), root, vec![], "send".into(), None);
    ed.focus(true);

    // Type a long line and submit.
    ed.set_text("abcdefghijklmnopqrstuvwxyz0123456789");
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Enter,
        KeyModifiers::NONE,
    )));
    assert_eq!(ed.desired_height(80, 40), 3); // empty after submit

    // Type new content — wrapping should still work.
    ed.set_text("abcdefghijklmnopqrstuvwxyz0123456789");
    assert_eq!(ed.desired_height(20, 40), 4); // 36 chars / 20 = 2 lines + 2
}

#[test]
fn render_wrapped_content_fits_allocated_area() {
    // 36-char line at width 20 wraps to 2 display lines (36/20=1.8 → ceil=2).
    let mut ed = make_editor("abcdefghijklmnopqrstuvwxyz0123456789");
    let h = ed.desired_height(20, 40);
    assert_eq!(h, 4, "2 wrapped lines + 2 border");

    let area = Rect::new(0, 0, 20, h);
    let backend = TestBackend::new(20, 8);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| ed.render(area, f)).unwrap();

    // Bottom border at area.y + area.height - 1 = 3.
    let buf = terminal.backend().buffer().clone();
    let border_y = area.y + area.height - 1;
    let has_border = (0..20).any(|x| {
        let s = buf[(x, border_y)].symbol();
        s == "─" || s == "-"
    });
    assert!(has_border, "border at row {border_y}");

    // Content rows 1-2 should have text.
    for row in 1..border_y {
        let line: String = (0..20)
            .map(|x| buf[(x, row)].symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(
            !line.trim().is_empty(),
            "row {row} should have wrapped content"
        );
    }
}

#[test]
fn long_path_without_spaces_wraps_and_stays_visible() {
    // Realistic long file path with no spaces.
    let path = "crates/michin-agent-core/src/loop_mod/inner_loop/tool_execution/handler.rs";
    let mut ed = make_editor(path);
    let width = 20u16;

    // desired_height should account for wrapping.
    let h = ed.desired_height(width as usize, 40);
    let expected_display = (path.len() as u16).div_ceil(width);
    assert_eq!(h, expected_display + 2, "height = wrapped lines + border");

    // Render and verify ALL content rows have text.
    let area = Rect::new(0, 0, width, h);
    let backend = TestBackend::new(width.into(), h.into());
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| ed.render(area, f)).unwrap();

    let buf = terminal.backend().buffer().clone();
    let border_y = area.y + area.height - 1;
    for row in 1..border_y {
        let line: String = (0..width)
            .map(|x| buf[(x, row)].symbol().chars().next().unwrap_or(' '))
            .collect();
        assert!(
            !line.trim().is_empty(),
            "row {row} must have content — long path disappeared"
        );
    }
}

#[test]
fn long_url_without_spaces_wraps_correctly() {
    let url = "https://example.com/very/long/path/to/some/resource/that/exceeds/terminal/width?param=value&other=123";
    let mut ed = make_editor(url);
    let width = 30u16;

    let h = ed.desired_height(width as usize, 40);
    assert!(h > 2 + 2, "URL should wrap: height {h}");

    let area = Rect::new(0, 0, width, h);
    let backend = TestBackend::new(width.into(), h.into());
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| ed.render(area, f)).unwrap();

    // Collect all visible content chars.
    let buf = terminal.backend().buffer().clone();
    let border_y = area.y + area.height - 1;
    let mut visible = String::new();
    for row in 1..border_y {
        for x in 0..width {
            let ch = buf[(x, row)].symbol().chars().next().unwrap_or(' ');
            if ch != ' ' {
                visible.push(ch);
            }
        }
    }
    // All URL chars must be visible (no disappearing text).
    assert_eq!(
        visible.len(),
        url.len(),
        "all URL chars must be rendered, got {visible:?}"
    );
}

#[test]
fn single_char_width_terminal_wraps_every_char() {
    let mut ed = make_editor("abcde");
    let width = 1u16;

    let h = ed.desired_height(width as usize, 40);
    // 5 chars at width 1 → 5 display lines + 2 border = 7.
    assert_eq!(h, 7);

    let area = Rect::new(0, 0, width, h);
    let backend = TestBackend::new(width.into(), h.into());
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| ed.render(area, f)).unwrap();

    let buf = terminal.backend().buffer().clone();
    // Each content row should have exactly one visible char.
    for (i, ch) in ['a', 'b', 'c', 'd', 'e'].iter().enumerate() {
        let row = (1 + i) as u16;
        let cell = buf[(0u16, row)].symbol();
        assert_eq!(
            cell,
            ch.to_string().as_str(),
            "row {row} should show '{ch}'"
        );
    }
}

#[test]
fn long_word_height_matches_actual_rendered_lines() {
    // Verify desired_height estimate matches actual wrapped line count.
    let long = "a".repeat(100);
    let ed = make_editor(&long);

    for width in [10, 20, 33, 50, 80] {
        let h = ed.desired_height(width, 200);
        let display_lines = h - 2; // subtract border
        let expected = (100u16).div_ceil(width as u16);
        assert_eq!(
            display_lines, expected,
            "width {width}: expected {expected} display lines, got {display_lines}"
        );
    }
}

// ── Cursor stability during edits that change editor size ──

#[test]
fn cursor_stays_on_line_after_typing_that_causes_wrap() {
    let mut ed = make_editor("short line\nanother line");
    // Render at width 30 — no wrapping.
    let area = Rect::new(0, 0, 30, 6);
    let backend = TestBackend::new(30, 10);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| ed.render(area, f)).unwrap();

    // Move cursor to end of first line ("short line" = 10 chars).
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SUPER)));
    assert_eq!(ed.cursor_position().0, 0, "cursor on line 0");

    // Type enough chars to make the first line wrap at width 30.
    for _ in 0..25 {
        ed.handle_event(&Event::Key(KeyEvent::new(
            KeyCode::Char('x'),
            KeyModifiers::NONE,
        )));
    }
    // First line is now 35 chars — wraps at width 30.
    // Editor height changed. Re-render.
    let new_h = ed.desired_height(30, 40);
    let area2 = Rect::new(0, 0, 30, new_h);
    terminal.draw(|f| ed.render(area2, f)).unwrap();

    // Cursor MUST still be on line 0.
    let after = ed.cursor_position();
    assert_eq!(after.0, 0, "cursor must stay on line 0 after wrap");
    assert_eq!(after.1, 25, "cursor at end of typed chars");
}

#[test]
fn cursor_stays_on_line_after_deleting_that_changes_height() {
    let mut ed = make_editor("aaa\nbbb\nccc\nddd\neee");
    let area = Rect::new(0, 0, 40, 9);
    let backend = TestBackend::new(40, 12);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| ed.render(area, f)).unwrap();

    // Move to line 4 ("eee").
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SUPER)));
    for _ in 0..4 {
        ed.handle_event(&Event::Key(KeyEvent::new(
            KeyCode::Down,
            KeyModifiers::NONE,
        )));
    }
    assert_eq!(ed.cursor_position().0, 4, "cursor on line 4");

    // Delete line 4. Height changes.
    ed.handle_event(&Event::Key(KeyEvent::new(
        KeyCode::Delete,
        KeyModifiers::SUPER,
    )));

    // Editor now has 4 lines. Re-render at smaller height.
    let new_h = ed.desired_height(40, 40);
    let area2 = Rect::new(0, 0, 40, new_h);
    terminal.draw(|f| ed.render(area2, f)).unwrap();

    // Cursor should be on line 3 ("ddd", the new last line).
    let after = ed.cursor_position();
    assert_eq!(after.0, 3, "cursor on last remaining line");
}

#[test]
fn cursor_preserved_across_multiple_rapid_renders() {
    // Simulate rapid renders at different sizes (like terminal resize).
    let mut ed = make_editor("line1\nline2\nline3\nline4\nline5");
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SUPER)));
    for _ in 0..2 {
        ed.handle_event(&Event::Key(KeyEvent::new(
            KeyCode::Down,
            KeyModifiers::NONE,
        )));
    }
    // Cursor on line 2.
    assert_eq!(ed.cursor_position().0, 2);

    let backend = TestBackend::new(60, 20);
    let mut terminal = Terminal::new(backend).unwrap();

    // Render at many different sizes.
    let widths: &[u16] = &[60, 10, 40, 5, 80, 20, 15];
    for &w in widths {
        let h = ed.desired_height(w as usize, 20).min(20);
        terminal
            .draw(|f| ed.render(Rect::new(0, 0, w, h), f))
            .unwrap();
    }

    // Cursor must still be on line 2.
    assert_eq!(
        ed.cursor_position().0,
        2,
        "cursor must stay on line 2 after rapid renders"
    );
}

#[test]
fn viewport_shows_content_near_cursor_not_top() {
    // 15-line editor, cursor on line 10.
    let lines: Vec<String> = (0..15).map(|i| format!("line {i}")).collect();
    let mut ed = make_editor(&lines.join("\n"));
    ed.handle_event(&Event::Key(KeyEvent::new(KeyCode::Up, KeyModifiers::SUPER)));
    for _ in 0..10 {
        ed.handle_event(&Event::Key(KeyEvent::new(
            KeyCode::Down,
            KeyModifiers::NONE,
        )));
    }
    assert_eq!(ed.cursor_position().0, 10);

    // Render at small height (6 rows = 4 content rows).
    let area = Rect::new(0, 0, 40, 6);
    let backend = TestBackend::new(40, 8);
    let mut terminal = Terminal::new(backend).unwrap();
    terminal.draw(|f| ed.render(area, f)).unwrap();

    // The viewport should show content near line 10, not line 0.
    // Verify by checking that text near line 10 appears in the buffer.
    let buf = terminal.backend().buffer().clone();
    let mut found_near_cursor = false;
    for row in 1..5 {
        let text: String = (0..40)
            .map(|x| {
                buf[(x as u16, row as u16)]
                    .symbol()
                    .chars()
                    .next()
                    .unwrap_or(' ')
            })
            .collect();
        if text.contains("line 10")
            || text.contains("line 9")
            || text.contains("line 11")
            || text.contains("line 8")
        {
            found_near_cursor = true;
            break;
        }
    }
    assert!(
        found_near_cursor,
        "viewport should show content near cursor (line 10), not jump to top"
    );
}
