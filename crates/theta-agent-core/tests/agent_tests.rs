//! Tests for theta-agent-core.
//!
//! Uses a mock LLM provider to test the agent loop without hitting real APIs.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

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

// ── Mock Provider ─────────────────────────────────────────────

/// A mock LLM provider that returns pre-configured event sequences.
struct MockProvider {
    /// Events to emit per call. Each stream() call pops the first entry.
    events: std::sync::Mutex<Vec<Vec<AssistantMessageEvent>>>,
    /// Track call count.
    call_count: std::sync::atomic::AtomicU32,
    /// Optional: block until released (for testing concurrency).
    block_until_released: std::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

impl MockProvider {
    fn new(responses: Vec<Vec<AssistantMessageEvent>>) -> Self {
        Self {
            events: std::sync::Mutex::new(responses),
            call_count: std::sync::atomic::AtomicU32::new(0),
            block_until_released: std::sync::Mutex::new(None),
        }
    }

    /// Make the first stream() call block until the returned Sender is triggered.
    fn set_blocking(&self) -> tokio::sync::oneshot::Sender<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        *self.block_until_released.lock().unwrap() = Some(rx);
        tx
    }
}

type EventStream<'a> = Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send + 'a>>;

#[async_trait]
impl LlmProvider for MockProvider {
    async fn stream<'a>(
        &'a self,
        _model: &Model,
        _context: &Context,
        _options: &StreamOptions,
    ) -> Result<EventStream<'a>, ThetaError> {
        // If blocking is set, wait until released.
        let rx = {
            let mut guard = self.block_until_released.lock().unwrap();
            guard.take()
        };
        if let Some(rx) = rx {
            let _ = rx.await;
        }

        self.call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let response = {
            let mut events = self.events.lock().unwrap();
            if events.is_empty() {
                vec![AssistantMessageEvent::Done {
                    stop_reason: StopReason::Stop,
                    usage: None,
                }]
            } else {
                events.remove(0)
            }
        };

        Ok(Box::pin(futures::stream::iter(response)))
    }

    async fn stream_simple<'a>(
        &'a self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> Result<EventStream<'a>, ThetaError> {
        let stream_opts = StreamOptions {
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            ..Default::default()
        };
        self.stream(model, context, &stream_opts).await
    }
}

// ── Mock Tool ─────────────────────────────────────────────────

struct MockTool {
    name: String,
    description: String,
    label: String,
    mode: ToolExecutionMode,
}

impl MockTool {
    fn new(name: &str, mode: ToolExecutionMode) -> Self {
        Self {
            name: name.to_string(),
            description: format!("Tool: {name}"),
            label: name.to_string(),
            mode,
        }
    }
}

#[async_trait]
impl AgentTool for MockTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn label(&self) -> &str {
        &self.label
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "input": { "type": "string" }
            }
        })
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        self.mode
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        _args: serde_json::Value,
        _signal: Option<tokio_util::sync::CancellationToken>,
        _on_update: Option<ToolUpdateSender>,
    ) -> Result<ToolResult, AgentError> {
        Ok(ToolResult {
            tool_call_id: tool_call_id.to_string(),
            tool_name: self.name.clone(),
            content: vec![ContentBlock::text(format!("result from {tool_call_id}"))],
            details: None,
            is_error: false,
        })
    }
}

// ── Helpers ───────────────────────────────────────────────────

fn test_model() -> Model {
    Model {
        id: "test-model".into(),
        name: "Test Model".into(),
        api: Api::OpenAiCompletions,
        provider: ProviderKind::OpenAI,
        base_url: "https://test.api".into(),
        reasoning: false,
        thinking_level_map: HashMap::new(),
        input: vec![Modality::Text],
        cost: ModelCost::default(),
        context_window: 128_000,
        max_tokens: 16_384,
        compat: ModelCompat::for_openai(),
    }
}

struct TestModelCatalog {
    model: Model,
}

impl ModelCatalog for TestModelCatalog {
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

// ── Tests ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_agent_text_response() {
    let model = test_model();
    let mock = MockProvider::new(vec![vec![
        AssistantMessageEvent::text_delta("Hello"),
        AssistantMessageEvent::text_delta(" world"),
        AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        },
    ]]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Agent::new(model, registry, catalog);
    let mut rx = agent.subscribe();

    let handle = tokio::spawn(async move {
        agent.prompt(vec![ContentBlock::text("Hi")]).await.unwrap();
    });

    // Collect events.
    let mut texts = Vec::new();
    loop {
        match rx.recv().await.unwrap() {
            AgentEvent::TextDelta { text } => texts.push(text),
            AgentEvent::AgentEnd { .. } => break,
            _ => {}
        }
    }

    handle.await.unwrap();

    assert_eq!(texts.join(""), "Hello world");
}

#[tokio::test]
async fn test_agent_tool_loop() {
    let model = test_model();

    // Response 1: ask for a tool call.
    // Response 2: text after tool result.
    let mock = MockProvider::new(vec![
        vec![
            AssistantMessageEvent::ToolCallStart {
                id: "call_1".into(),
                name: "mock".into(),
            },
            AssistantMessageEvent::tool_call_delta("call_1", r#"{"input":"test"}"#),
            AssistantMessageEvent::ToolCallEnd {
                id: "call_1".into(),
            },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::text_delta("Tool result received"),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
    ]);

    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Agent::new(model, registry, catalog);
    agent
        .add_tool(Arc::new(MockTool::new("mock", ToolExecutionMode::Parallel)))
        .await;

    let mut rx = agent.subscribe();

    let handle = tokio::spawn(async move {
        agent
            .prompt(vec![ContentBlock::text("Do something")])
            .await
            .unwrap();
    });

    let mut tool_starts = 0;
    let mut tool_ends = 0;
    let mut texts = Vec::new();

    loop {
        match rx.recv().await.unwrap() {
            AgentEvent::TextDelta { text } => texts.push(text),
            AgentEvent::ToolExecutionStart { .. } => tool_starts += 1,
            AgentEvent::ToolExecutionEnd { .. } => tool_ends += 1,
            AgentEvent::AgentEnd { .. } => break,
            _ => {}
        }
    }

    handle.await.unwrap();

    assert_eq!(tool_starts, 1, "should have 1 tool execution start");
    assert_eq!(tool_ends, 1, "should have 1 tool execution end");
    assert_eq!(texts.join(""), "Tool result received");
}

#[tokio::test]
async fn test_agent_retries_empty_assistant_turn() {
    let model = test_model();
    let mock = MockProvider::new(vec![
        vec![AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        }],
        vec![
            AssistantMessageEvent::text_delta("Recovered response"),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
    ]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Agent::new(model, registry, catalog);
    agent
        .prompt(vec![ContentBlock::text("Explain changes")])
        .await
        .unwrap();

    let state = agent.state().await;
    let last_assistant = state
        .messages
        .iter()
        .rev()
        .find_map(|msg| match msg {
            Message::Assistant { content, .. } => Some(content),
            _ => None,
        })
        .expect("assistant message should exist");

    let text = last_assistant
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");

    assert!(text.contains("Recovered response"));
}

#[tokio::test]
async fn test_agent_retries_when_execution_promised_without_tool_calls() {
    let model = test_model();
    let mock = MockProvider::new(vec![
        vec![
            AssistantMessageEvent::text_delta("On it. I'll implement now."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::ToolCallStart {
                id: "call_1".into(),
                name: "mock".into(),
            },
            AssistantMessageEvent::tool_call_delta("call_1", r#"{"input":"go"}"#),
            AssistantMessageEvent::ToolCallEnd {
                id: "call_1".into(),
            },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::text_delta("Implemented."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
    ]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let mut config = AgentLoopConfig::default();
    config.max_tool_rounds = Some(5);

    let mut agent = Agent::new(model, registry, catalog);
    agent.set_config(config);
    agent
        .add_tool(Arc::new(MockTool::new(
            "mock",
            ToolExecutionMode::Sequential,
        )))
        .await;

    agent
        .prompt(vec![ContentBlock::text("implement it")])
        .await
        .unwrap();

    let state = agent.state().await;
    let has_tool_result = state
        .messages
        .iter()
        .any(|msg| matches!(msg, Message::ToolResult { .. }));
    assert!(
        has_tool_result,
        "execution retry should drive an actual tool call"
    );

    let last_assistant_text = state
        .messages
        .iter()
        .rev()
        .find_map(|msg| match msg {
            Message::Assistant { content, .. } => Some(
                content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            _ => None,
        })
        .unwrap_or_default();
    assert!(last_assistant_text.contains("Implemented."));
}

#[tokio::test]
async fn test_agent_allows_long_progressive_tool_runs_without_round_cap() {
    let model = test_model();
    let mut responses: Vec<Vec<AssistantMessageEvent>> = Vec::new();
    for i in 0..25 {
        let call_id = format!("call_progress_{i}");
        let args = format!(r#"{{"input":"step-{i}"}}"#);
        responses.push(vec![
            AssistantMessageEvent::ToolCallStart {
                id: call_id.clone(),
                name: "mock".into(),
            },
            AssistantMessageEvent::tool_call_delta(&call_id, &args),
            AssistantMessageEvent::ToolCallEnd { id: call_id },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ]);
    }
    responses.push(vec![
        AssistantMessageEvent::text_delta("Completed after long tool run."),
        AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        },
    ]);

    let mock = MockProvider::new(responses);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Agent::new(model, registry, catalog);
    agent
        .add_tool(Arc::new(MockTool::new(
            "mock",
            ToolExecutionMode::Sequential,
        )))
        .await;
    agent
        .prompt(vec![ContentBlock::text("implement this large change")])
        .await
        .unwrap();

    let state = agent.state().await;
    let tool_results = state
        .messages
        .iter()
        .filter(|msg| {
            matches!(
                msg,
                Message::ToolResult { tool_name, .. } if tool_name == "mock"
            )
        })
        .count();
    assert_eq!(
        tool_results, 25,
        "progressive, non-identical tool calls should not be capped by default"
    );

    let last_assistant_text = state
        .messages
        .iter()
        .rev()
        .find_map(|msg| match msg {
            Message::Assistant { content, .. } => Some(
                content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n"),
            ),
            _ => None,
        })
        .unwrap_or_default();
    assert!(
        last_assistant_text.contains("Completed after long tool run."),
        "turn should reach final assistant completion after many tool rounds"
    );
}

#[tokio::test]
async fn test_agent_stops_repeated_identical_tool_call_loop() {
    let model = test_model();
    let mut responses: Vec<Vec<AssistantMessageEvent>> = Vec::new();
    for i in 0..7 {
        let call_id = format!("call_repeat_{i}");
        responses.push(vec![
            AssistantMessageEvent::ToolCallStart {
                id: call_id.clone(),
                name: "mock".into(),
            },
            AssistantMessageEvent::tool_call_delta(&call_id, r#"{"input":"same"}"#),
            AssistantMessageEvent::ToolCallEnd { id: call_id },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ]);
    }

    let mock = MockProvider::new(responses);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Arc::new(Agent::new(model, registry, catalog));
    agent
        .add_tool(Arc::new(MockTool::new(
            "mock",
            ToolExecutionMode::Sequential,
        )))
        .await;

    let mut rx = agent.subscribe();
    let agent_clone = agent.clone();
    let handle = tokio::spawn(async move {
        agent_clone
            .prompt(vec![ContentBlock::text("implement and keep trying")])
            .await
            .unwrap();
    });

    let mut saw_repeat_guard_error = false;
    loop {
        match rx.recv().await.unwrap() {
            AgentEvent::Error { message } => {
                if message.contains("repeated identical tool call loop") {
                    saw_repeat_guard_error = true;
                }
            }
            AgentEvent::AgentEnd { .. } => break,
            _ => {}
        }
    }

    handle.await.unwrap();

    let state = agent.state().await;
    let tool_results = state
        .messages
        .iter()
        .filter(|msg| {
            matches!(
                msg,
                Message::ToolResult { tool_name, .. } if tool_name == "mock"
            )
        })
        .count();
    assert_eq!(
        tool_results, 6,
        "repeat guard should stop before executing the 7th identical call"
    );
    assert!(
        saw_repeat_guard_error,
        "agent should emit a diagnostic error when repeat guard triggers"
    );
}

#[tokio::test]
async fn test_action_turn_retries_without_promise_when_no_tools_and_no_blocker() {
    let model = test_model();
    let mock = MockProvider::new(vec![
        vec![
            AssistantMessageEvent::text_delta("Plan: edit config and run tests."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::ToolCallStart {
                id: "call_2".into(),
                name: "mock".into(),
            },
            AssistantMessageEvent::tool_call_delta("call_2", r#"{"input":"run"}"#),
            AssistantMessageEvent::ToolCallEnd {
                id: "call_2".into(),
            },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::text_delta("Done."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
    ]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Agent::new(model, registry, catalog);
    agent
        .add_tool(Arc::new(MockTool::new(
            "mock",
            ToolExecutionMode::Sequential,
        )))
        .await;
    agent
        .prompt(vec![ContentBlock::text("implement this change")])
        .await
        .unwrap();

    let state = agent.state().await;
    assert!(
        state
            .messages
            .iter()
            .any(|msg| matches!(msg, Message::ToolResult { .. })),
        "action turn should retry and execute tool calls"
    );
}

#[tokio::test]
async fn test_action_turn_with_explicit_blocker_does_not_retry() {
    let model = test_model();
    let mock = MockProvider::new(vec![
        vec![
            AssistantMessageEvent::text_delta(
                "What should I implement exactly? Please provide target file.",
            ),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::text_delta("This second response should not be reached."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
    ]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Agent::new(model, registry, catalog);
    agent
        .prompt(vec![ContentBlock::text("implement it")])
        .await
        .unwrap();

    let state = agent.state().await;
    let has_retry_injected_user_message = state.messages.iter().any(|msg| match msg {
        Message::User { content, .. } => content.iter().any(|b| {
            matches!(b, ContentBlock::Text { text } if text.contains("This is an action request. Execute now by calling required tools first."))
        }),
        _ => false,
    });
    assert!(
        !has_retry_injected_user_message,
        "explicit blocker should end turn without injecting corrective retry"
    );
}

#[tokio::test]
async fn test_inspection_turn_retries_and_executes_read_only_tool() {
    let model = test_model();
    let mock = MockProvider::new(vec![
        vec![
            AssistantMessageEvent::text_delta("I can inspect if you want."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::ToolCallStart {
                id: "call_i1".into(),
                name: "read".into(),
            },
            AssistantMessageEvent::tool_call_delta("call_i1", r#"{"input":"inspect"}"#),
            AssistantMessageEvent::ToolCallEnd {
                id: "call_i1".into(),
            },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::text_delta("Inspected."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
    ]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Agent::new(model, registry, catalog);
    agent
        .add_tool(Arc::new(MockTool::new(
            "read",
            ToolExecutionMode::Sequential,
        )))
        .await;
    agent
        .prompt(vec![ContentBlock::text(
            "explain what changed and inspect files",
        )])
        .await
        .unwrap();

    let state = agent.state().await;
    assert!(
        state
            .messages
            .iter()
            .any(|msg| matches!(msg, Message::ToolResult { .. })),
        "inspection turn should retry and execute at least one tool"
    );
}

#[tokio::test]
async fn test_commit_turn_retries_and_executes_git_bash_tool() {
    let model = test_model();
    let mock = MockProvider::new(vec![
        vec![
            AssistantMessageEvent::text_delta("I can run git status if you want."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::ToolCallStart {
                id: "call_g1".into(),
                name: "bash".into(),
            },
            AssistantMessageEvent::tool_call_delta(
                "call_g1",
                r#"{"command":"git status --short"}"#,
            ),
            AssistantMessageEvent::ToolCallEnd {
                id: "call_g1".into(),
            },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::text_delta("Git status checked."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
    ]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Agent::new(model, registry, catalog);
    agent
        .add_tool(Arc::new(MockTool::new(
            "bash",
            ToolExecutionMode::Sequential,
        )))
        .await;
    agent
        .prompt(vec![ContentBlock::text("check git status and commit plan")])
        .await
        .unwrap();

    let state = agent.state().await;
    assert!(
        state.messages.iter().any(|msg| {
            matches!(
                msg,
                Message::ToolResult { tool_name, .. } if tool_name == "bash"
            )
        }),
        "commit-op turn should retry and run git command via bash tool"
    );
}

#[tokio::test]
async fn test_agent_abort() {
    let model = test_model();

    let mock = MockProvider::new(vec![vec![
        AssistantMessageEvent::text_delta("This will be aborted..."),
        AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        },
    ]]);
    // Make the mock block until we signal it.
    let release_tx = mock.set_blocking();

    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Arc::new(Agent::new(model, registry, catalog));
    let agent_clone = agent.clone();

    let handle = tokio::spawn(async move {
        let result = agent_clone
            .prompt(vec![ContentBlock::text("Say something long")])
            .await;
        result
    });

    // Wait for prompt to start, then abort.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    agent.abort().unwrap();

    // Release the mock so it can respond (though it should be aborted by then).
    let _ = release_tx.send(());

    let result = handle.await.unwrap();
    assert!(result.is_err());
    match result {
        Err(AgentError::Aborted) => {}
        other => panic!("expected Aborted, got {other:?}"),
    }
}

#[tokio::test]
async fn test_agent_already_running() {
    let model = test_model();
    let mock = MockProvider::new(vec![vec![AssistantMessageEvent::text_delta("ok")]]);
    let release_tx = mock.set_blocking();

    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Arc::new(Agent::new(model, registry, catalog));
    let agent_clone = agent.clone();

    // Start a prompt in background (will block).
    let handle = tokio::spawn(async move {
        let _ = agent_clone.prompt(vec![ContentBlock::text("hi")]).await;
    });

    // Give it time to start and acquire the active_run lock.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Try to start another prompt while first is running.
    let result = agent.prompt(vec![ContentBlock::text("another")]).await;
    assert!(matches!(result, Err(AgentError::AlreadyRunning)));

    // Release and clean up.
    agent.abort().unwrap();
    let _ = release_tx.send(());
    let _ = handle.await;
}

#[tokio::test]
async fn test_agent_follow_up() {
    let model = test_model();

    // Two text responses.
    let mock = MockProvider::new(vec![
        vec![
            AssistantMessageEvent::text_delta("First response"),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::text_delta("Second response"),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
    ]);

    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Arc::new(Agent::new(model, registry, catalog));

    // Queue a follow-up before starting.
    agent.follow_up(vec![ContentBlock::text("follow up question")]);

    let mut rx = agent.subscribe();

    let handle = tokio::spawn(async move {
        agent
            .prompt(vec![ContentBlock::text("first question")])
            .await
            .unwrap();
    });

    let mut texts = Vec::new();
    let mut turns = 0;
    loop {
        match rx.recv().await.unwrap() {
            AgentEvent::TurnStart { .. } => turns += 1,
            AgentEvent::TextDelta { text } => texts.push(text),
            AgentEvent::AgentEnd { .. } => break,
            _ => {}
        }
    }

    handle.await.unwrap();

    assert!(texts.join("").contains("First response"));
    assert!(texts.join("").contains("Second response"));
    assert_eq!(turns, 2, "should have 2 turns due to follow-up");
}

#[tokio::test]
async fn test_agent_event_subscription() {
    let model = test_model();
    let mock = MockProvider::new(vec![vec![
        AssistantMessageEvent::text_delta("Hello"),
        AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        },
    ]]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Agent::new(model, registry, catalog);
    let mut rx = agent.subscribe();

    let handle = tokio::spawn(async move {
        agent.prompt(vec![ContentBlock::text("hi")]).await.unwrap();
    });

    let mut has_start = false;
    let mut has_turn_start = false;
    let mut has_message_start = false;
    let mut has_text = false;
    let mut has_message_end = false;
    let mut has_turn_end = false;
    let mut has_agent_end = false;

    while let Ok(event) = rx.recv().await {
        match event {
            AgentEvent::AgentStart => has_start = true,
            AgentEvent::TurnStart { .. } => has_turn_start = true,
            AgentEvent::MessageStart => has_message_start = true,
            AgentEvent::TextDelta { .. } => has_text = true,
            AgentEvent::MessageEnd { .. } => has_message_end = true,
            AgentEvent::TurnEnd { .. } => has_turn_end = true,
            AgentEvent::AgentEnd { .. } => {
                has_agent_end = true;
                break;
            }
            _ => {}
        }
    }

    handle.await.unwrap();

    assert!(has_start);
    assert!(has_turn_start);
    assert!(has_message_start);
    assert!(has_text);
    assert!(has_message_end);
    assert!(has_turn_end);
    assert!(has_agent_end);
}

#[tokio::test]
async fn test_agent_state_transcript() {
    let model = test_model();
    let mock = MockProvider::new(vec![vec![
        AssistantMessageEvent::text_delta("response"),
        AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        },
    ]]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Agent::new(model, registry, catalog);

    agent
        .prompt(vec![ContentBlock::text("hello")])
        .await
        .unwrap();

    let state = agent.state().await;
    assert_eq!(
        state.messages.len(),
        2,
        "should have user + assistant messages"
    );

    // First message should be user.
    assert!(matches!(state.messages[0], Message::User { .. }));

    // Second should be assistant.
    assert!(matches!(state.messages[1], Message::Assistant { .. }));
    if let Message::Assistant { content, .. } = &state.messages[1] {
        assert_eq!(content.len(), 1);
        if let ContentBlock::Text { text } = &content[0] {
            assert_eq!(text, "response");
        } else {
            panic!("expected Text block");
        }
    }
}

#[tokio::test]
async fn test_agent_steer() {
    let model = test_model();

    // Two responses: first is interrupted by steer, second responds to steering.
    let mock = MockProvider::new(vec![
        // First stream: interrupted
        vec![
            AssistantMessageEvent::text_delta("This will be interrupted..."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
        // Second stream: after steering
        vec![
            AssistantMessageEvent::text_delta("Steered response"),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
    ]);
    // Block so steer() can interrupt before first response completes.
    let release_tx = mock.set_blocking();

    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        model: model.clone(),
    });

    let agent = Arc::new(Agent::new(model, registry, catalog));
    let agent_clone = agent.clone();

    let mut rx = agent.subscribe();

    let handle = tokio::spawn(async move {
        agent_clone
            .prompt(vec![ContentBlock::text("initial question")])
            .await
    });

    // Wait for prompt to start streaming, then steer.
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    agent.steer(vec![ContentBlock::text("steering override")]);

    // Release the mock so it can respond to both calls.
    let _ = release_tx.send(());

    // Collect events.
    let mut texts = Vec::new();
    loop {
        match rx.recv().await.unwrap() {
            AgentEvent::TextDelta { text } => texts.push(text),
            AgentEvent::AgentEnd { aborted } => {
                // With steering, the agent should complete (not abort).
                assert!(!aborted, "agent should complete after steering");
                break;
            }
            _ => {}
        }
    }

    let result = handle.await.unwrap();
    assert!(result.is_ok(), "steered prompt should succeed");

    // Should contain the steered response, not the interrupted one.
    let full = texts.join("");
    assert!(
        full.contains("Steered response"),
        "should contain steered response, got: {full}"
    );
}
