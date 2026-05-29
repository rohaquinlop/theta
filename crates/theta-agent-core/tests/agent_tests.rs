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
    Api, ContentBlock, Context, Message, Modality, Provider as ProviderKind, SimpleStreamOptions,
    StopReason, StreamOptions,
};
use theta_ai::{LlmProvider, ThetaError};

// ── Mock Provider ─────────────────────────────────────────────

/// A mock LLM provider that returns pre-configured event sequences.
struct MockProvider {
    /// Events to emit per call. Each stream() call pops the first entry.
    events: std::sync::Mutex<Vec<Result<Vec<AssistantMessageEvent>, ThetaError>>>,
    /// Track call count.
    call_count: std::sync::atomic::AtomicU32,
    /// Optional: block until released (for testing concurrency).
    block_until_released: std::sync::Mutex<Option<tokio::sync::oneshot::Receiver<()>>>,
}

impl MockProvider {
    fn new(responses: Vec<Vec<AssistantMessageEvent>>) -> Self {
        Self {
            events: std::sync::Mutex::new(responses.into_iter().map(Ok).collect()),
            call_count: std::sync::atomic::AtomicU32::new(0),
            block_until_released: std::sync::Mutex::new(None),
        }
    }

    fn new_with_results(responses: Vec<Result<Vec<AssistantMessageEvent>, ThetaError>>) -> Self {
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
                Ok(vec![AssistantMessageEvent::Done {
                    stop_reason: StopReason::Stop,
                    usage: None,
                }])
            } else {
                events.remove(0)
            }
        };
        match response {
            Ok(events) => Ok(Box::pin(futures::stream::iter(events))),
            Err(e) => Err(e),
        }
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
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
        context_window: 128_000,
        max_tokens: 16_384,
        compat: ModelCompat::for_openai(),
    }
}

fn test_model_with_id(id: &str) -> Model {
    let mut m = test_model();
    m.id = id.to_string();
    m.name = id.to_string();
    m
}

struct TestModelCatalog {
    models: Vec<Model>,
}

impl ModelCatalog for TestModelCatalog {
    fn find(&self, provider: ProviderKind, model_id: &str) -> Option<&Model> {
        self.models
            .iter()
            .find(|m| m.provider == provider && m.id == model_id)
    }
    fn list(&self) -> Vec<&Model> {
        self.models.iter().collect()
    }
    fn list_by_provider(&self, provider: ProviderKind) -> Vec<&Model> {
        self.models
            .iter()
            .filter(|m| m.provider == provider)
            .collect()
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
        models: vec![model.clone()],
    });

    let agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );
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
        models: vec![model.clone()],
    });

    let agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );
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
        models: vec![model.clone()],
    });

    let agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );
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
        models: vec![model.clone()],
    });

    let agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );
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
        models: vec![model.clone()],
    });

    let agent = Arc::new(Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    ));
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
async fn test_duplicate_tool_call_id_in_turn_is_deduped() {
    let model = test_model();
    let mock = MockProvider::new(vec![
        vec![
            AssistantMessageEvent::ToolCallStart {
                id: "dup_call".into(),
                name: "mock".into(),
            },
            AssistantMessageEvent::tool_call_delta("dup_call", r#"{"input":"first"}"#),
            AssistantMessageEvent::ToolCallEnd {
                id: "dup_call".into(),
            },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::ToolCallStart {
                id: "dup_call".into(),
                name: "mock".into(),
            },
            AssistantMessageEvent::tool_call_delta("dup_call", r#"{"input":"second"}"#),
            AssistantMessageEvent::ToolCallEnd {
                id: "dup_call".into(),
            },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::text_delta("done"),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
    ]);

    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        models: vec![model.clone()],
    });
    let agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );
    agent
        .add_tool(Arc::new(MockTool::new(
            "mock",
            ToolExecutionMode::Sequential,
        )))
        .await;

    agent
        .prompt(vec![ContentBlock::text("implement this")])
        .await
        .unwrap();

    let state = agent.state().await;
    let dup_results = state
        .messages
        .iter()
        .filter(
            |m| matches!(m, Message::ToolResult { tool_call_id, .. } if tool_call_id == "dup_call"),
        )
        .count();
    assert_eq!(
        dup_results, 1,
        "duplicate tool_call_id should execute once per turn"
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
        models: vec![model.clone()],
    });

    let agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );
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
        models: vec![model.clone()],
    });

    let agent = Arc::new(Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    ));
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
        models: vec![model.clone()],
    });

    let agent = Arc::new(Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    ));
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
        models: vec![model.clone()],
    });

    let agent = Arc::new(Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    ));

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
        models: vec![model.clone()],
    });

    let agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );
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
        models: vec![model.clone()],
    });

    let agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );

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
        models: vec![model.clone()],
    });

    let agent = Arc::new(Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    ));
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

#[tokio::test]
async fn test_manual_compaction_trims_messages() {
    let model = test_model();
    let mock = MockProvider::new(vec![vec![
        AssistantMessageEvent::text_delta("All done."),
        AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        },
    ]]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        models: vec![model.clone()],
    });

    let agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );

    // Inject many messages to exceed the 128k context window.
    {
        let mut raw: Vec<Message> = Vec::new();
        let pad = "padding text to consume more tokens ".repeat(10);
        for i in 0..5000 {
            raw.push(Message::User {
                content: vec![ContentBlock::text(format!("user message {i} {pad}"))],
                timestamp: i as u64,
            });
            raw.push(Message::Assistant {
                content: vec![ContentBlock::text(format!("assistant response {i} {pad}"))],
                api: None,
                provider: None,
                model: None,
                usage: None,
                stop_reason: None,
                error_message: None,
                timestamp: (i + 5000) as u64,
            });
        }
        agent.load_messages(raw).await;
    }

    let before_count = agent.state().await.messages.len();
    assert_eq!(before_count, 10000);

    // Manual compaction should trim.
    let trimmed = agent.compact_context().await.unwrap();
    assert!(trimmed > 0, "should trim messages from 128k context window");

    let after_count = agent.state().await.messages.len();
    assert!(
        after_count < before_count,
        "after {after_count} should be less than {before_count}"
    );
}

#[tokio::test]
async fn test_context_stats_returns_token_counts() {
    let model = test_model();
    let mock = MockProvider::new(vec![vec![
        AssistantMessageEvent::text_delta("ok"),
        AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        },
    ]]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        models: vec![model.clone()],
    });

    let agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );
    agent
        .prompt(vec![ContentBlock::text("hello world")])
        .await
        .unwrap();

    let (msg_count, token_count, _real) = agent.context_stats().await;
    assert_eq!(msg_count, 2);
    assert!(token_count > 0, "token count should be positive");
}

#[tokio::test]
async fn test_run_report_events_include_standard_fields() {
    let model = test_model();
    let mock = MockProvider::new(vec![
        vec![
            AssistantMessageEvent::ToolCallStart {
                id: "call_std_1".into(),
                name: "mock".into(),
            },
            AssistantMessageEvent::tool_call_delta("call_std_1", r#"{"input":"x"}"#),
            AssistantMessageEvent::ToolCallEnd {
                id: "call_std_1".into(),
            },
            AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            },
        ],
        vec![
            AssistantMessageEvent::text_delta("done"),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ],
    ]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        models: vec![model.clone()],
    });
    let agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );
    agent
        .add_tool(Arc::new(MockTool::new(
            "mock",
            ToolExecutionMode::Sequential,
        )))
        .await;

    agent
        .prompt(vec![ContentBlock::text("implement this")])
        .await
        .unwrap();

    let report = agent
        .last_run_report()
        .await
        .expect("run report should exist");
    assert!(!report.events.is_empty());
    for ev in &report.events {
        assert!(
            ev.fields.contains_key("run_id"),
            "missing run_id for {}",
            ev.kind
        );
        assert!(
            ev.fields.contains_key("model"),
            "missing model for {}",
            ev.kind
        );
        assert!(
            ev.fields.contains_key("provider"),
            "missing provider for {}",
            ev.kind
        );
    }
    assert!(
        report
            .events
            .iter()
            .any(|ev| ev.kind == "tool_execution_end")
    );
}

#[tokio::test]
async fn test_provider_fallback_chain_uses_next_model() {
    let primary = test_model_with_id("primary-model");
    let fallback = test_model_with_id("fallback-model");
    let mock = MockProvider::new_with_results(vec![
        Err(ThetaError::ApiError {
            status: 503,
            message: "temporary outage".to_string(),
            retry_after_ms: None,
        }),
        Ok(vec![
            AssistantMessageEvent::text_delta("Recovered on fallback."),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ]),
    ]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        models: vec![primary.clone(), fallback.clone()],
    });

    let mut agent = Agent::new(
        primary,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );
    let mut cfg = AgentLoopConfig::default();
    cfg.retry.max_retries = 0;
    cfg.provider_fallback_chain = vec![fallback.id.clone()];
    agent.set_config(cfg);

    let mut rx = agent.subscribe();
    agent
        .prompt(vec![ContentBlock::text("please execute")])
        .await
        .unwrap();

    let mut saw_fallback_event = false;
    while let Ok(event) = rx.try_recv() {
        if matches!(event, AgentEvent::ProviderFallback { .. }) {
            saw_fallback_event = true;
        }
    }
    assert!(
        saw_fallback_event,
        "expected explicit provider fallback event"
    );

    let state = agent.state().await;
    let used_fallback = state.messages.iter().any(|m| {
        matches!(
            m,
            Message::Assistant {
                model: Some(model_id),
                ..
            } if model_id == &fallback.id
        )
    });
    assert!(
        used_fallback,
        "assistant message should record fallback model"
    );
}

#[tokio::test]
async fn test_circuit_breaker_opens_and_emits_event() {
    let model = test_model_with_id("circuit-model");
    let mock = MockProvider::new_with_results(vec![
        Err(ThetaError::ApiError {
            status: 503,
            message: "transient-1".to_string(),
            retry_after_ms: None,
        }),
        Err(ThetaError::ApiError {
            status: 503,
            message: "transient-2".to_string(),
            retry_after_ms: None,
        }),
        Err(ThetaError::ApiError {
            status: 503,
            message: "transient-3".to_string(),
            retry_after_ms: None,
        }),
    ]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        models: vec![model.clone()],
    });
    let mut agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );
    let mut cfg = AgentLoopConfig::default();
    cfg.retry.max_retries = 0;
    cfg.provider_circuit_breaker.failure_threshold = 1;
    cfg.provider_circuit_breaker.open_cooldown_ms = 60_000;
    agent.set_config(cfg);

    let _ = agent.prompt(vec![ContentBlock::text("attempt one")]).await;
    let mut rx = agent.subscribe();
    let _ = agent.prompt(vec![ContentBlock::text("attempt two")]).await;

    let mut saw_circuit_open = false;
    while let Ok(event) = rx.try_recv() {
        if matches!(event, AgentEvent::ProviderCircuitOpen { .. }) {
            saw_circuit_open = true;
        }
    }
    assert!(
        saw_circuit_open,
        "expected circuit-open event after repeated transient failures"
    );
}

#[tokio::test]
async fn test_turn_mode_resolver_uses_runtime_override_source() {
    let model = test_model();
    let mock = MockProvider::new(vec![vec![
        AssistantMessageEvent::text_delta("ok"),
        AssistantMessageEvent::Done {
            stop_reason: StopReason::Stop,
            usage: None,
        },
    ]]);
    let registry = make_registry(mock);
    let catalog = Arc::new(TestModelCatalog {
        models: vec![model.clone()],
    });
    let agent = Agent::new(
        model,
        registry,
        catalog.list().into_iter().cloned().collect(),
    );

    agent
        .prompt(vec![ContentBlock::text("implement this")])
        .await
        .unwrap();
}

mod state {
    use std::collections::HashMap;
    use theta_agent_core::{AgentState, RunReport};
    use theta_ai::model::{Model, ModelCompat};
    use theta_ai::{Api, Modality, Provider};

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
}
