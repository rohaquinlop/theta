use std::collections::HashMap;
use theta_agent_core::{AgentState, RunReport};
use theta_ai::model::{Model, ModelCompat};
use theta_ai::{Api, Modality, ModelCost, Provider};

fn test_model() -> Model {
    Model {
        id: "test-model".into(),
        name: "Test".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenAI,
        base_url: "https://example.invalid".into(),
        reasoning: false,
        thinking_level_map: HashMap::new(),
        input: vec![Modality::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 16_384,
        compat: ModelCompat::for_openai(),
    }
}

#[test]
fn run_event_redacts_sensitive_fields() {
    let model = test_model();
    let mut state = AgentState::new(model, vec![]);
    state.current_run_id = Some("run-1".to_string());
    state.current_run_report = Some(RunReport {
        run_id: "run-1".to_string(),
        started_at_ms: 1,
        finished_at_ms: None,
        outcome: None,
        events: vec![],
    });
    state.push_run_event(
        "test",
        [
            ("access_token".to_string(), "sk-live-secret".to_string()),
            ("authorization".to_string(), "Bearer abc".to_string()),
            ("normal".to_string(), "ok".to_string()),
        ],
    );
    let report = state.current_run_report.expect("report exists");
    let fields = &report.events[0].fields;
    assert_eq!(
        fields.get("access_token").map(String::as_str),
        Some("[REDACTED]")
    );
    assert_eq!(
        fields.get("authorization").map(String::as_str),
        Some("[REDACTED]")
    );
    assert_eq!(fields.get("normal").map(String::as_str), Some("ok"));
}
