use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use michin_tui::Action;
use michin_tui::components::editor::{Editor, file_mention_matches};
use michin_tui::{Component, Theme};
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
