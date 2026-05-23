//! Scenario matrix regression tests for production hardening behavior.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;

use theta_agent_core::*;
use theta_ai::event::AssistantMessageEvent;
use theta_ai::model::{Model, ModelCatalog, ModelCompat};
use theta_ai::providers::ProviderRegistry;
use theta_ai::types::{
    Api, ContentBlock, Context, Message, Modality, ModelCost, Provider as ProviderKind,
    SimpleStreamOptions, StopReason, StreamOptions,
};
use theta_ai::{LlmProvider, ThetaError};

type EventStream<'a> = Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send + 'a>>;

struct MockProvider {
    responses: std::sync::Mutex<Vec<Result<Vec<AssistantMessageEvent>, ThetaError>>>,
}

impl MockProvider {
    fn new(responses: Vec<Result<Vec<AssistantMessageEvent>, ThetaError>>) -> Self {
        Self {
            responses: std::sync::Mutex::new(responses),
        }
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    async fn stream<'a>(
        &'a self,
        _model: &Model,
        _context: &Context,
        _options: &StreamOptions,
    ) -> Result<EventStream<'a>, ThetaError> {
        let response = {
            let mut guard = self.responses.lock().expect("responses lock");
            if guard.is_empty() {
                Ok(vec![AssistantMessageEvent::Done {
                    stop_reason: StopReason::Stop,
                    usage: None,
                }])
            } else {
                guard.remove(0)
            }
        }?;
        Ok(Box::pin(futures::stream::iter(response)))
    }

    async fn stream_simple<'a>(
        &'a self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> Result<EventStream<'a>, ThetaError> {
        let so = StreamOptions {
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            ..Default::default()
        };
        self.stream(model, context, &so).await
    }
}

struct MatrixTool {
    name: String,
    sleep_ms: u64,
}

#[async_trait]
impl AgentTool for MatrixTool {
    fn name(&self) -> &str {
        &self.name
    }
    fn description(&self) -> &str {
        "matrix tool"
    }
    fn label(&self) -> &str {
        &self.name
    }
    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "command": { "type": "string" }
            }
        })
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        _args: serde_json::Value,
        _signal: Option<tokio_util::sync::CancellationToken>,
        _on_update: Option<ToolUpdateSender>,
    ) -> Result<ToolResult, AgentError> {
        if self.sleep_ms > 0 {
            tokio::time::sleep(Duration::from_millis(self.sleep_ms)).await;
        }
        Ok(ToolResult {
            tool_call_id: tool_call_id.to_string(),
            tool_name: self.name.clone(),
            content: vec![ContentBlock::text("ok")],
            details: None,
            is_error: false,
        })
    }
}

fn test_model() -> Model {
    Model {
        id: "test-model".into(),
        name: "Test Model".into(),
        api: Api::OpenAiCompletions,
        provider: ProviderKind::OpenAI,
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

struct Catalog {
    model: Model,
}
impl ModelCatalog for Catalog {
    fn find(&self, _provider: ProviderKind, _model_id: &str) -> Option<&Model> {
        Some(&self.model)
    }
    fn list(&self) -> Vec<&Model> {
        vec![&self.model]
    }
    fn list_by_provider(&self, _provider: ProviderKind) -> Vec<&Model> {
        vec![&self.model]
    }
}

fn make_registry(provider: MockProvider) -> Arc<ProviderRegistry> {
    let mut reg = ProviderRegistry::new();
    reg.register(Api::OpenAiCompletions, Box::new(provider));
    Arc::new(reg)
}

#[tokio::test]
async fn scenario_no_tool_promise_retries_then_executes() {
    let model = test_model();
    let provider = MockProvider::new(vec![
        Ok(vec![
            AssistantMessageEvent::text_delta("On it. I'll implement now."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ]),
        Ok(vec![
            AssistantMessageEvent::ToolCallStart {
                id: "call_1".into(),
                name: "mock".into(),
            },
            AssistantMessageEvent::tool_call_delta("call_1", r#"{"command":"echo ok"}"#),
            AssistantMessageEvent::ToolCallEnd {
                id: "call_1".into(),
            },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ]),
        Ok(vec![
            AssistantMessageEvent::text_delta("done"),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ]),
    ]);

    let agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    agent
        .add_tool(Arc::new(MatrixTool {
            name: "mock".into(),
            sleep_ms: 0,
        }))
        .await;
    agent
        .prompt(vec![ContentBlock::text("implement this change")])
        .await
        .expect("prompt should succeed");
    let state = agent.state().await;
    assert!(
        state
            .messages
            .iter()
            .any(|m| matches!(m, Message::ToolResult { .. }))
    );
}

#[tokio::test]
async fn scenario_explicit_blocker_stops_without_tool_exec() {
    let model = test_model();
    let provider = MockProvider::new(vec![
        Ok(vec![
            AssistantMessageEvent::text_delta("I need more detail before implementing."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ]),
        Ok(vec![
            AssistantMessageEvent::text_delta("I still need more detail before implementing."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ]),
    ]);
    let agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    agent
        .prompt(vec![ContentBlock::text("implement this")])
        .await
        .expect("prompt should finish with blocker");
    let state = agent.state().await;
    assert_eq!(
        state.last_turn_end_reason,
        Some(TurnEndReason::BlockedMissingInfo)
    );
    assert!(
        !state
            .messages
            .iter()
            .any(|m| matches!(m, Message::ToolResult { .. }))
    );
}

#[tokio::test]
async fn scenario_provider_timeout_or_transient_failure_retries() {
    let model = test_model();
    let provider = MockProvider::new(vec![
        Err(ThetaError::ApiError {
            status: 503,
            message: "temporary".into(),
            retry_after_ms: None,
        }),
        Ok(vec![
            AssistantMessageEvent::text_delta("recovered"),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ]),
    ]);

    let mut agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    let mut cfg = AgentLoopConfig::default();
    cfg.retry.max_retries = 1;
    agent.set_config(cfg);
    agent
        .prompt(vec![ContentBlock::text("explain")])
        .await
        .expect("should recover after retry");
    let state = agent.state().await;
    assert_eq!(state.last_turn_end_reason, Some(TurnEndReason::Completed));
}

#[tokio::test]
async fn scenario_tool_timeout_emits_tool_error() {
    let model = test_model();
    let provider = MockProvider::new(vec![
        Ok(vec![
            AssistantMessageEvent::ToolCallStart {
                id: "slow_1".into(),
                name: "slow".into(),
            },
            AssistantMessageEvent::tool_call_delta("slow_1", r#"{"command":"sleep"}"#),
            AssistantMessageEvent::ToolCallEnd {
                id: "slow_1".into(),
            },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ]),
        Ok(vec![
            AssistantMessageEvent::text_delta("post"),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ]),
    ]);
    let mut agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    agent
        .add_tool(Arc::new(MatrixTool {
            name: "slow".into(),
            sleep_ms: 100,
        }))
        .await;
    let mut cfg = AgentLoopConfig::default();
    cfg.tool_watchdog.hard_timeout_ms = 10;
    agent.set_config(cfg);

    agent
        .prompt(vec![ContentBlock::text("implement something")])
        .await
        .expect("turn should continue with tool error result");
    let state = agent.state().await;
    assert!(state.messages.iter().any(|m| {
        matches!(
            m,
            Message::ToolResult {
                tool_name,
                is_error: true,
                ..
            } if tool_name == "slow"
        )
    }));
}

#[tokio::test]
async fn scenario_repeated_tool_signature_stops_loop() {
    let model = test_model();
    let mut responses = Vec::new();
    for i in 0..5 {
        let id = format!("c{i}");
        responses.push(Ok(vec![
            AssistantMessageEvent::ToolCallStart {
                id: id.clone(),
                name: "mock".into(),
            },
            AssistantMessageEvent::tool_call_delta(&id, r#"{"command":"same"}"#),
            AssistantMessageEvent::ToolCallEnd { id },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ]));
    }
    let provider = MockProvider::new(responses);
    let mut agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    agent
        .add_tool(Arc::new(MatrixTool {
            name: "mock".into(),
            sleep_ms: 0,
        }))
        .await;
    let mut cfg = AgentLoopConfig::default();
    cfg.max_same_tool_call_repeats = Some(2);
    agent.set_config(cfg);
    agent
        .prompt(vec![ContentBlock::text("implement this")])
        .await
        .expect("loop should stop gracefully");
    let state = agent.state().await;
    assert!(
        state
            .messages
            .iter()
            .filter(|m| matches!(m, Message::ToolResult { .. }))
            .count()
            <= 3
    );
}

#[tokio::test]
async fn scenario_analyze_only_mutation_attempt_is_blocked_without_mutation_request() {
    let model = test_model();
    let provider = MockProvider::new(vec![Ok(vec![
        AssistantMessageEvent::ToolCallStart {
            id: "c1".into(),
            name: "write".into(),
        },
        AssistantMessageEvent::tool_call_delta("c1", r#"{"path":"x","content":"y"}"#),
        AssistantMessageEvent::ToolCallEnd { id: "c1".into() },
        AssistantMessageEvent::Done {
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
    ])]);
    let agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    agent
        .add_tool(Arc::new(MatrixTool {
            name: "write".into(),
            sleep_ms: 0,
        }))
        .await;
    agent
        .prompt(vec![ContentBlock::text("review and analyze this code")])
        .await
        .expect("turn should complete with safety rejection");
    let state = agent.state().await;
    assert_eq!(
        state.last_turn_end_reason,
        Some(TurnEndReason::SafetyRejected)
    );
}

#[tokio::test]
async fn scenario_commit_tool_call_is_blocked_without_explicit_user_request() {
    let model = test_model();
    let provider = MockProvider::new(vec![Ok(vec![
        AssistantMessageEvent::ToolCallStart {
            id: "c_commit".into(),
            name: "bash".into(),
        },
        AssistantMessageEvent::tool_call_delta("c_commit", r#"{"command":"git commit -m test"}"#),
        AssistantMessageEvent::ToolCallEnd {
            id: "c_commit".into(),
        },
        AssistantMessageEvent::Done {
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
    ])]);
    let agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    agent
        .add_tool(Arc::new(MatrixTool {
            name: "bash".into(),
            sleep_ms: 0,
        }))
        .await;
    agent
        .prompt(vec![ContentBlock::text(
            "review the changes and run validations",
        )])
        .await
        .expect("turn should complete with safety rejection");
    let state = agent.state().await;
    assert_eq!(
        state.last_turn_end_reason,
        Some(TurnEndReason::SafetyRejected)
    );
}

#[tokio::test]
async fn scenario_commit_tool_call_is_allowed_when_user_explicitly_requests_commit() {
    let model = test_model();
    let provider = MockProvider::new(vec![Ok(vec![
        AssistantMessageEvent::ToolCallStart {
            id: "c_commit".into(),
            name: "bash".into(),
        },
        AssistantMessageEvent::tool_call_delta("c_commit", r#"{"command":"git commit -m test"}"#),
        AssistantMessageEvent::ToolCallEnd {
            id: "c_commit".into(),
        },
        AssistantMessageEvent::Done {
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
    ])]);
    let agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    agent
        .add_tool(Arc::new(MatrixTool {
            name: "bash".into(),
            sleep_ms: 0,
        }))
        .await;
    agent
        .prompt(vec![ContentBlock::text("run tests and commit the changes")])
        .await
        .expect("turn should execute commit tool call");
    let state = agent.state().await;
    assert!(
        state
            .messages
            .iter()
            .any(|m| matches!(m, Message::ToolResult { tool_name, .. } if tool_name == "bash"))
    );
}

#[tokio::test]
async fn scenario_file_mutation_is_blocked_without_explicit_user_request() {
    let model = test_model();
    let provider = MockProvider::new(vec![Ok(vec![
        AssistantMessageEvent::ToolCallStart {
            id: "c_write".into(),
            name: "write".into(),
        },
        AssistantMessageEvent::tool_call_delta(
            "c_write",
            r#"{"path":"tmp.txt","content":"hello"}"#,
        ),
        AssistantMessageEvent::ToolCallEnd {
            id: "c_write".into(),
        },
        AssistantMessageEvent::Done {
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
    ])]);
    let agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    agent
        .add_tool(Arc::new(MatrixTool {
            name: "write".into(),
            sleep_ms: 0,
        }))
        .await;
    agent
        .prompt(vec![ContentBlock::text(
            "review and inspect the current changes",
        )])
        .await
        .expect("turn should complete with safety rejection");
    let state = agent.state().await;
    assert_eq!(
        state.last_turn_end_reason,
        Some(TurnEndReason::SafetyRejected)
    );
}

#[tokio::test]
async fn scenario_dependency_mutation_is_blocked_without_explicit_user_request() {
    let model = test_model();
    let provider = MockProvider::new(vec![Ok(vec![
        AssistantMessageEvent::ToolCallStart {
            id: "c_dep".into(),
            name: "bash".into(),
        },
        AssistantMessageEvent::tool_call_delta("c_dep", r#"{"command":"npm install"}"#),
        AssistantMessageEvent::ToolCallEnd { id: "c_dep".into() },
        AssistantMessageEvent::Done {
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
    ])]);
    let agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    agent
        .add_tool(Arc::new(MatrixTool {
            name: "bash".into(),
            sleep_ms: 0,
        }))
        .await;
    agent
        .prompt(vec![ContentBlock::text("review and inspect current setup")])
        .await
        .expect("turn should complete with safety rejection");
    let state = agent.state().await;
    assert_eq!(
        state.last_turn_end_reason,
        Some(TurnEndReason::SafetyRejected)
    );
}

#[tokio::test]
async fn scenario_dependency_mutation_is_allowed_with_explicit_request() {
    let model = test_model();
    let provider = MockProvider::new(vec![Ok(vec![
        AssistantMessageEvent::ToolCallStart {
            id: "c_dep".into(),
            name: "bash".into(),
        },
        AssistantMessageEvent::tool_call_delta("c_dep", r#"{"command":"npm install"}"#),
        AssistantMessageEvent::ToolCallEnd { id: "c_dep".into() },
        AssistantMessageEvent::Done {
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
    ])]);
    let agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    agent
        .add_tool(Arc::new(MatrixTool {
            name: "bash".into(),
            sleep_ms: 0,
        }))
        .await;
    agent
        .prompt(vec![ContentBlock::text(
            "install dependencies and run validations",
        )])
        .await
        .expect("turn should execute dependency mutation tool call");
    let state = agent.state().await;
    assert!(
        state
            .messages
            .iter()
            .any(|m| matches!(m, Message::ToolResult { tool_name, .. } if tool_name == "bash"))
    );
}

#[tokio::test]
async fn scenario_vcs_mutation_is_blocked_without_explicit_user_request() {
    let model = test_model();
    let provider = MockProvider::new(vec![Ok(vec![
        AssistantMessageEvent::ToolCallStart {
            id: "c_vcs".into(),
            name: "bash".into(),
        },
        AssistantMessageEvent::tool_call_delta("c_vcs", r#"{"command":"git branch feature/test"}"#),
        AssistantMessageEvent::ToolCallEnd { id: "c_vcs".into() },
        AssistantMessageEvent::Done {
            stop_reason: StopReason::ToolUse,
            usage: None,
        },
    ])]);
    let agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    agent
        .add_tool(Arc::new(MatrixTool {
            name: "bash".into(),
            sleep_ms: 0,
        }))
        .await;
    agent
        .prompt(vec![ContentBlock::text("inspect repository status only")])
        .await
        .expect("turn should complete with safety rejection");
    let state = agent.state().await;
    assert_eq!(
        state.last_turn_end_reason,
        Some(TurnEndReason::SafetyRejected)
    );
}

#[tokio::test]
#[ignore = "soak test; run manually for long-run stability characterization"]
async fn scenario_long_run_soak_stability() {
    let model = test_model();
    let mut responses = Vec::new();
    for i in 0..100 {
        let id = format!("soak_{i}");
        responses.push(Ok(vec![
            AssistantMessageEvent::ToolCallStart {
                id: id.clone(),
                name: "mock".into(),
            },
            AssistantMessageEvent::tool_call_delta(&id, r#"{"command":"echo ok"}"#),
            AssistantMessageEvent::ToolCallEnd { id },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ]));
    }
    responses.push(Ok(vec![
        AssistantMessageEvent::text_delta("soak-complete"),
        AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        },
    ]));

    let provider = MockProvider::new(responses);
    let agent = Agent::new(
        model.clone(),
        make_registry(provider),
        Arc::new(Catalog { model }),
    );
    agent
        .add_tool(Arc::new(MatrixTool {
            name: "mock".into(),
            sleep_ms: 0,
        }))
        .await;
    agent
        .prompt(vec![ContentBlock::text("implement this thoroughly")])
        .await
        .expect("soak run should finish");
    let state = agent.state().await;
    assert_eq!(state.last_turn_end_reason, Some(TurnEndReason::Completed));
}
