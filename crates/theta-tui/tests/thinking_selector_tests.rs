use theta_tui::Theme;
use theta_tui::components::thinking_selector::{ThinkingLevelEntry, ThinkingSelector};

fn test_levels() -> Vec<ThinkingLevelEntry> {
    vec![
        ThinkingLevelEntry {
            id: "off".into(),
            label: "Disabled".into(),
        },
        ThinkingLevelEntry {
            id: "high".into(),
            label: "High".into(),
        },
        ThinkingLevelEntry {
            id: "xhigh".into(),
            label: "X-High (Max)".into(),
        },
    ]
}

#[test]
fn select_moves_within_bounds() {
    let mut selector = ThinkingSelector::new(Theme::default());
    selector.show(test_levels(), None);
    assert_eq!(selector.selected_level(), Some("off"));

    selector.select_down();
    assert_eq!(selector.selected_level(), Some("high"));

    selector.select_down();
    assert_eq!(selector.selected_level(), Some("xhigh"));

    selector.select_down();
    assert_eq!(selector.selected_level(), Some("xhigh"));
}

#[test]
fn select_up_stays_in_bounds() {
    let mut selector = ThinkingSelector::new(Theme::default());
    selector.show(test_levels(), None);
    selector.selected = 2;
    selector.list_state.select(Some(2));

    selector.select_up();
    assert_eq!(selector.selected_level(), Some("high"));

    selector.select_up();
    assert_eq!(selector.selected_level(), Some("off"));

    selector.select_up();
    assert_eq!(selector.selected_level(), Some("off"));
}

#[test]
fn show_selects_current_level() {
    let mut selector = ThinkingSelector::new(Theme::default());
    selector.show(test_levels(), Some("high"));
    assert_eq!(selector.selected_level(), Some("high"));

    selector.show(test_levels(), Some("xhigh"));
    assert_eq!(selector.selected_level(), Some("xhigh"));

    selector.show(test_levels(), Some("low"));
    assert_eq!(selector.selected_level(), Some("off"));
}
