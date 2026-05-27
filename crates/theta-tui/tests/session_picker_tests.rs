use theta_tui::Theme;
use theta_tui::components::session_picker::{SessionInfo, SessionPicker, session_row_label};

fn mk(id: &str, title: &str, created_at: u64, messages: usize) -> SessionInfo {
    SessionInfo {
        id: id.to_string(),
        title: title.to_string(),
        model: None,
        branch: None,
        token_count: 0,
        created_at,
        message_count: messages,
    }
}

#[test]
fn cycle_sort_mode_reorders_and_preserves_selection() {
    let sessions = vec![
        mk("a", "zeta", 3000, 2),
        mk("b", "alpha", 1000, 10),
        mk("c", "beta", 2000, 5),
    ];
    let mut picker = SessionPicker::new(sessions, Theme::default());
    assert_eq!(picker.selected_session().map(|s| s.id.as_str()), Some("a"));

    picker.select_down();
    let selected = picker.selected_session().map(|s| s.id.clone());
    picker.cycle_sort_mode();
    assert_eq!(picker.sort_mode_label(), "oldest");
    assert_eq!(picker.selected_session().map(|s| s.id.clone()), selected);

    picker.cycle_sort_mode();
    assert_eq!(picker.sort_mode_label(), "title");
    assert_eq!(picker.sessions[0].title, "alpha");

    picker.cycle_sort_mode();
    assert_eq!(picker.sort_mode_label(), "messages");
    assert_eq!(picker.sessions[0].message_count, 10);
}

#[test]
fn session_row_label_aligns_both_separators() {
    let session = SessionInfo {
        id: "s1".to_string(),
        title: "conversation".to_string(),
        model: Some("gpt-5.5".to_string()),
        branch: Some("feature/ui".to_string()),
        token_count: 3200,
        created_at: 1_000_000_000_000,
        message_count: 18,
    };
    let max_w = 21usize;
    let max_when = 8usize;

    let short = "conversation".to_string();
    let row_short = session_row_label(&session, &short, "18h ago", max_w, max_when);
    let row_justnow = session_row_label(&session, &short, "just now", max_w, max_when);
    let long = "quite long title here".to_string();
    let row_long = session_row_label(&session, &long, "5m ago", max_w, max_when);

    let seps: Vec<(usize, usize)> = [&row_short, &row_justnow, &row_long]
        .iter()
        .map(|r| {
            let first = r.find('│').unwrap();
            let second = r
                .char_indices()
                .filter(|(_, c)| *c == '│')
                .nth(1)
                .map(|(i, _)| i)
                .unwrap();
            (first, second)
        })
        .collect();

    assert_eq!(seps[0].0, seps[1].0, "first │ should align across rows");
    assert_eq!(seps[0].0, seps[2].0, "first │ should align across rows");
    assert_eq!(seps[0].1, seps[1].1, "second │ should align across rows");
    assert_eq!(seps[0].1, seps[2].1, "second │ should align across rows");
}

#[test]
fn truncation_handles_multi_byte_chars_safely() {
    let title = "áéíóú — accented chars";
    let title_chars: Vec<char> = title.chars().collect();
    assert!(title_chars.len() > 5);
    let truncated: String = title_chars[..5].iter().collect();
    assert_eq!(truncated.chars().count(), 5);
    assert_eq!(truncated, "áéíóú");
}
