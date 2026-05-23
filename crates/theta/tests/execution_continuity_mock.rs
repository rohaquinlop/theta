use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use tempfile::TempDir;
use theta::session::SessionManager;
use theta::system_prompt::build_system_prompt;
use theta_agent_core::{
    Agent, AgentError, AgentTool, ToolExecutionMode, ToolResult, ToolUpdateSender,
};
use theta_ai::event::AssistantMessageEvent;
use theta_ai::model::{Model, ModelCatalog, ModelCompat};
use theta_ai::providers::ProviderRegistry;
use theta_ai::types::{
    Api, ContentBlock, Context, Message, Modality, ModelCost, Provider as ProviderKind,
    SimpleStreamOptions, StopReason, StreamOptions,
};
use theta_ai::{LlmProvider, ThetaError};

struct PromptSensitiveMockProvider {
    call_count: std::sync::atomic::AtomicU32,
}

struct ReplayValidationProvider;

#[async_trait]
impl LlmProvider for ReplayValidationProvider {
    async fn stream<'a>(
        &'a self,
        _model: &Model,
        context: &Context,
        _options: &StreamOptions,
    ) -> Result<EventStream<'a>, ThetaError> {
        let mut pending = std::collections::HashSet::new();
        for msg in &context.messages {
            match msg {
                Message::Assistant { content, .. } => {
                    for b in content {
                        if let ContentBlock::ToolCall { id, .. } = b {
                            pending.insert(id.clone());
                        }
                    }
                }
                Message::ToolResult { tool_call_id, .. } => {
                    pending.remove(tool_call_id);
                }
                Message::User { .. } => {
                    if !pending.is_empty() {
                        return Ok(Box::pin(futures::stream::iter(vec![
                            AssistantMessageEvent::Error {
                                code: "invalid_replay".into(),
                                message: "orphan tool call replayed".into(),
                            },
                        ])));
                    }
                }
                _ => {}
            }
        }

        Ok(Box::pin(futures::stream::iter(vec![
            AssistantMessageEvent::text_delta("follow-up ok"),
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            },
        ])))
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

type EventStream<'a> = Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send + 'a>>;

#[async_trait]
impl LlmProvider for PromptSensitiveMockProvider {
    async fn stream<'a>(
        &'a self,
        _model: &Model,
        context: &Context,
        _options: &StreamOptions,
    ) -> Result<EventStream<'a>, ThetaError> {
        let call = self
            .call_count
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let events = if call == 0 {
            let system = context
                .system
                .as_ref()
                .map(|blocks| {
                    blocks
                        .iter()
                        .filter_map(|b| match b {
                            ContentBlock::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                })
                .unwrap_or_default();

            let has_execute_guardrail = system.contains("do the work in this turn");
            let has_function_calling_guardrail =
                system.contains("Invoke tools using function-calling");

            if has_execute_guardrail && has_function_calling_guardrail && !context.tools.is_empty()
            {
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
                ]
            } else {
                vec![
                    AssistantMessageEvent::text_delta("I will do it next."),
                    AssistantMessageEvent::Done {
                        stop_reason: StopReason::Stop,
                        usage: None,
                    },
                ]
            }
        } else {
            vec![
                AssistantMessageEvent::text_delta("Implemented via tool."),
                AssistantMessageEvent::Done {
                    stop_reason: StopReason::Stop,
                    usage: None,
                },
            ]
        };

        Ok(Box::pin(futures::stream::iter(events)))
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

struct MockTool;

#[async_trait]
impl AgentTool for MockTool {
    fn name(&self) -> &str {
        "mock"
    }

    fn description(&self) -> &str {
        "Mock tool"
    }

    fn label(&self) -> &str {
        "mock"
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
        ToolExecutionMode::Parallel
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
            tool_name: "mock".into(),
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

#[tokio::test]
async fn system_prompt_guardrails_drive_tool_execution_with_mock_provider() {
    let model = test_model();
    let provider = PromptSensitiveMockProvider {
        call_count: std::sync::atomic::AtomicU32::new(0),
    };
    let mut registry = ProviderRegistry::new();
    registry.register(Api::OpenAiCompletions, Box::new(provider));

    let agent = Agent::new(
        model.clone(),
        Arc::new(registry),
        Arc::new(TestModelCatalog { model }),
    );
    agent.add_tool(Arc::new(MockTool)).await;

    let wd = std::env::current_dir().expect("cwd");
    let system = build_system_prompt(&wd, "test-model", Some("medium"), None).await;
    agent.set_system_prompt(system).await;

    agent
        .prompt(vec![ContentBlock::text("implement it")])
        .await
        .expect("prompt should succeed");

    let state = agent.state().await;

    let tool_result_count = state
        .messages
        .iter()
        .filter(|m| matches!(m, Message::ToolResult { .. }))
        .count();
    assert_eq!(tool_result_count, 1, "expected one tool result message");

    let assistant_text = state
        .messages
        .iter()
        .filter_map(|m| match m {
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
        .collect::<Vec<_>>()
        .join("\n");

    assert!(
        assistant_text.contains("Implemented via tool."),
        "expected assistant to continue after tool call"
    );
}

#[tokio::test]
async fn persisted_session_with_orphan_toolcall_is_sanitized_on_resume() {
    let tmp = TempDir::new().expect("tmp");
    let mgr = SessionManager::with_dir(tmp.path().join("sessions"));
    let mut session = mgr.create(Some("test-model")).await.expect("create");

    // Persist broken history: assistant tool call without matching tool result.
    let broken = Message::Assistant {
        content: vec![ContentBlock::ToolCall {
            id: "call_orphan".into(),
            name: "mock".into(),
            arguments: serde_json::json!({"input":"x"}),
        }],
        api: Some(Api::OpenAiCompletions),
        provider: Some(ProviderKind::OpenAI),
        model: Some("test-model".into()),
        usage: None,
        stop_reason: Some(StopReason::ToolUse),
        error_message: None,
        timestamp: 1,
    };
    mgr.append_entry(&mut session, &broken)
        .await
        .expect("append");
    mgr.append_entry(
        &mut session,
        &Message::User {
            content: vec![ContentBlock::text("second ask")],
            timestamp: 2,
        },
    )
    .await
    .expect("append2");

    let reopened = mgr
        .open(std::path::Path::new(
            session.file_path.file_name().expect("file"),
        ))
        .await
        .expect("open");

    let model = test_model();
    let mut registry = ProviderRegistry::new();
    registry.register(Api::OpenAiCompletions, Box::new(ReplayValidationProvider));
    let agent = Agent::new(
        model.clone(),
        Arc::new(registry),
        Arc::new(TestModelCatalog { model }),
    );
    agent.load_messages(reopened.messages).await;

    agent
        .prompt(vec![ContentBlock::text("third ask")])
        .await
        .expect("resume prompt should succeed");
}
