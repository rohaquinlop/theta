use michin_models::opencode;

#[test]
fn test_all_models_valid() {
    for m in opencode::models() {
        assert!(!m.id.is_empty());
        assert_eq!(m.provider, michin_ai::Provider::OpenCode);
        assert!(m.context_window > 0);
    }
}

#[test]
fn test_reasoning_models() {
    // Build a few models through the same path as fetch_models
    // and verify reasoning is set correctly.
    let reasoning_ids = [
        "gpt-5.5",
        "gpt-5.4-nano",
        "claude-sonnet-4",
        "claude-haiku-4-5",
        "gemini-3-flash",
        "deepseek-v4-flash-free",
        "qwen3.6-plus",
    ];
    for id in reasoning_ids {
        assert!(
            opencode::supports_reasoning(id),
            "{id} should support reasoning"
        );
    }
    let non_reasoning_ids = [
        "minimax-m2.7",
        "minimax-m2.5-free",
        "mimo-v2.5-free",
        "glm-5.1",
        "kimi-k2.6",
        "grok-build-0.1",
        "big-pickle",
        "nemotron-3-super-free",
    ];
    for id in non_reasoning_ids {
        assert!(
            !opencode::supports_reasoning(id),
            "{id} should NOT support reasoning"
        );
    }
}
