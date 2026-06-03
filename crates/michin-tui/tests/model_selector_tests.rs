use michin_tui::Theme;
use michin_tui::components::model_selector::{ModelEntry, ModelSelector, format_model_row};

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
        vec![],
        Theme::default(),
    );
    selector.show();
    for c in "opencode".chars() {
        selector.push_query(c);
    }

    let selected = selector.selected_model().expect("expected selected model");
    assert_eq!(selected.provider, "opencode");
    // Only one model should remain after filtering.
    // Down should stay at 0 since there's only one result.
    selector.select_down();
    let selected = selector.selected_model().expect("expected selected model");
    assert_eq!(selected.provider, "opencode");
}

#[test]
fn favorites_appear_first_in_selection_order() {
    let mut selector = ModelSelector::new(
        vec![
            ModelEntry {
                id: "gpt-5.5".to_string(),
                name: "GPT 5.5".to_string(),
                provider: "openai".to_string(),
                context_window: 128_000,
            },
            ModelEntry {
                id: "o4".to_string(),
                name: "O4".to_string(),
                provider: "openai".to_string(),
                context_window: 200_000,
            },
        ],
        vec!["o4".to_string()],
        Theme::default(),
    );
    selector.show();
    // First selected model should be the favorited one (o4).
    let first = selector.selected_model().expect("expected selected model");
    assert_eq!(first.id, "o4");
    assert!(selector.selected_is_favorite());
}

#[test]
fn select_down_moves_through_favorites_then_others() {
    let mut selector = ModelSelector::new(
        vec![
            ModelEntry {
                id: "gpt-5.5".to_string(),
                name: "GPT 5.5".to_string(),
                provider: "openai".to_string(),
                context_window: 128_000,
            },
            ModelEntry {
                id: "o4".to_string(),
                name: "O4".to_string(),
                provider: "openai".to_string(),
                context_window: 200_000,
            },
            ModelEntry {
                id: "deepseek-v4-pro".to_string(),
                name: "DeepSeek V4 Pro".to_string(),
                provider: "deepseek".to_string(),
                context_window: 1_000_000,
            },
        ],
        vec!["o4".to_string()],
        Theme::default(),
    );
    selector.show();
    // o4 (favorite) is first.
    assert_eq!(selector.selected_model().unwrap().id, "o4");
    // Next should be gpt-5.5 (first non-favorite).
    selector.select_down();
    assert_eq!(selector.selected_model().unwrap().id, "gpt-5.5");
    assert!(!selector.selected_is_favorite());
    // Next should be deepseek-v4-pro.
    selector.select_down();
    assert_eq!(selector.selected_model().unwrap().id, "deepseek-v4-pro");
}

#[test]
fn filtering_shows_matching_favorites_first() {
    let mut selector = ModelSelector::new(
        vec![
            ModelEntry {
                id: "gpt-5.5".to_string(),
                name: "GPT 5.5".to_string(),
                provider: "openai".to_string(),
                context_window: 128_000,
            },
            ModelEntry {
                id: "o4".to_string(),
                name: "O4".to_string(),
                provider: "openai".to_string(),
                context_window: 200_000,
            },
            ModelEntry {
                id: "deepseek-v4-pro".to_string(),
                name: "DeepSeek V4 Pro".to_string(),
                provider: "deepseek".to_string(),
                context_window: 1_000_000,
            },
        ],
        vec!["o4".to_string()],
        Theme::default(),
    );
    selector.show();
    // Filter by "openai" — both gpt-5.5 and o4 match, o4 is favorite so comes first.
    for c in "openai".chars() {
        selector.push_query(c);
    }
    assert_eq!(selector.selected_model().unwrap().id, "o4");
    assert!(selector.selected_is_favorite());
    selector.select_down();
    assert_eq!(selector.selected_model().unwrap().id, "gpt-5.5");
    assert!(!selector.selected_is_favorite());
}

#[test]
fn set_favorites_updates_display_order() {
    let mut selector = ModelSelector::new(
        vec![
            ModelEntry {
                id: "gpt-5.5".to_string(),
                name: "GPT 5.5".to_string(),
                provider: "openai".to_string(),
                context_window: 128_000,
            },
            ModelEntry {
                id: "o4".to_string(),
                name: "O4".to_string(),
                provider: "openai".to_string(),
                context_window: 200_000,
            },
        ],
        vec![],
        Theme::default(),
    );
    selector.show();
    // Initially no favorites — gpt-5.5 is first.
    assert_eq!(selector.selected_model().unwrap().id, "gpt-5.5");
    assert!(!selector.selected_is_favorite());
    // Add o4 as favorite via set_favorites.
    selector.set_favorites(vec!["o4".to_string()]);
    // o4 should now be first.
    assert_eq!(selector.selected_model().unwrap().id, "o4");
    assert!(selector.selected_is_favorite());
}

#[test]
fn favorited_model_not_duplicated_in_all_section() {
    let mut selector = ModelSelector::new(
        vec![
            ModelEntry {
                id: "gpt-5.5".to_string(),
                name: "GPT 5.5".to_string(),
                provider: "openai".to_string(),
                context_window: 128_000,
            },
            ModelEntry {
                id: "o4".to_string(),
                name: "O4".to_string(),
                provider: "openai".to_string(),
                context_window: 200_000,
            },
            ModelEntry {
                id: "deepseek-v4-pro".to_string(),
                name: "DeepSeek V4 Pro".to_string(),
                provider: "deepseek".to_string(),
                context_window: 1_000_000,
            },
        ],
        vec![],
        Theme::default(),
    );
    selector.show();
    // Initially 3 models, no favorites.
    selector.set_favorites(vec!["o4".to_string()]);
    // Walk all entries — o4 should appear exactly once.
    let mut seen = std::collections::HashSet::new();
    for &idx in selector.display_order() {
        let id = &selector.all_models()[idx].id;
        assert!(seen.insert(id.clone()), "duplicate entry for {id}");
    }
    assert_eq!(seen.len(), 3, "expected 3 unique models in display_order");
    // o4 must not be in the 'other' part.
    let fav_count = selector.favorite_count();
    for &idx in &selector.display_order()[fav_count..] {
        assert_ne!(
            selector.all_models()[idx].id,
            "o4",
            "favorited model leaked into All section"
        );
    }
}
