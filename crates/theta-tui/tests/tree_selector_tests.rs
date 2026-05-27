use theta_tui::components::session_picker::SessionInfo;
use theta_tui::components::tree_selector::tree_row_label;

#[test]
fn row_label_prioritizes_branch_and_model() {
    let session = SessionInfo {
        id: "s1".to_string(),
        title: "long session title".to_string(),
        model: Some("gpt-5.5".to_string()),
        branch: Some("feature/readability".to_string()),
        token_count: 0,
        created_at: 0,
        message_count: 12,
    };
    let row = tree_row_label(&session);
    assert!(row.starts_with("feature/readability | gpt-5.5"));
    assert!(row.contains("12 msgs"));
}
