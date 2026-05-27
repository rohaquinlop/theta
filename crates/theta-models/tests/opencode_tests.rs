use theta_models::opencode;

#[test]
fn test_all_models_valid() {
    for m in opencode::models() {
        assert!(!m.id.is_empty());
        assert_eq!(m.provider, theta_ai::Provider::OpenCode);
        assert!(m.context_window > 0);
    }
}

#[test]
fn test_free_models_are_excluded() {
    for id in opencode::FREE_MODEL_IDS {
        assert!(
            opencode::is_free(id),
            "free model {id} should be recognized"
        );
    }
    assert!(!opencode::is_free("gpt-5.5"));
}

#[test]
fn test_paid_models_have_cost() {
    let cost = opencode::known_cost("gpt-5.5");
    assert!(cost.input > 0.0);
    assert!(cost.output > 0.0);
}
