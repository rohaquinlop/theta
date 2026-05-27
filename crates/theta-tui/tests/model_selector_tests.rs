use theta_tui::Theme;
use theta_tui::components::model_selector::{ModelEntry, ModelSelector, format_model_row};

#[test]
fn model_row_places_identity_first() {
    let row = format_model_row(&ModelEntry {
        id: "gpt-5.5".to_string(),
        name: "GPT 5.5".to_string(),
        provider: "openai".to_string(),
        context_window: 128_000,
    });
    assert!(row.starts_with("gpt-5.5"));
    assert!(row.contains("GPT 5.5"));
    assert!(row.contains("openai"));
}

#[test]
fn filter_matches_provider() {
    let mut selector = ModelSelector::new(
        vec![
            ModelEntry {
                id: "gpt-5.5".to_string(),
                name: "GPT 5.5".to_string(),
                provider: "openai".to_string(),
                context_window: 128_000,
            },
            ModelEntry {
                id: "deepseek-v4-flash-free".to_string(),
                name: "DeepSeek V4 Flash Free".to_string(),
                provider: "opencode".to_string(),
                context_window: 200_000,
            },
        ],
        Theme::default(),
    );
    selector.show();
    for c in "opencode".chars() {
        selector.push_query(c);
    }

    let selected = selector.selected_model().expect("expected selected model");
    assert_eq!(selected.provider, "opencode");
    assert_eq!(selector.filtered.len(), 1);
}
