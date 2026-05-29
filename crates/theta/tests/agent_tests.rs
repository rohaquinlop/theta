// Consolidated tests: edit_tool, prompts, tools, rpc, mentions, execution_continuity_mock.
// Merged to reduce test binary count and improve test suite startup time.

mod edit_tool {
    use theta::tools::edit::make_diff_preview;

    #[test]
    fn diff_preview_uses_git_like_headers() {
        let before = "a\nb\nc\n";
        let after = "a\nB\nc\n";
        let diff = make_diff_preview("src/lib.rs", before, after);
        assert!(diff.contains("--- a/src/lib.rs"));
        assert!(diff.contains("+++ b/src/lib.rs"));
        assert!(diff.contains("@@"));
    }
}

mod prompts {
    use std::collections::HashMap;
    use theta::prompts::PromptTemplate;

    #[test]
    fn test_template_resolve() {
        let tpl = PromptTemplate {
            name: "test".into(),
            body: "Hello {{name}}, your project is {{project}}.".into(),
        };
        let mut vars = HashMap::new();
        vars.insert("name".into(), "Alice".into());
        vars.insert("project".into(), "Theta".into());

        let resolved = tpl.resolve(&vars);
        assert_eq!(resolved, "Hello Alice, your project is Theta.");
    }

    #[test]
    fn test_template_unresolved() {
        let tpl = PromptTemplate {
            name: "test".into(),
            body: "Hello {{name}}.".into(),
        };
        let vars = HashMap::new();
        let resolved = tpl.resolve(&vars);
        assert_eq!(resolved, "Hello {{name}}.");
    }
}

mod tools {
    use std::path::{Path, PathBuf};
    use theta::tools::{ToolContext, resolve_path, shorten_path};

    #[test]
    fn resolve_path_keeps_absolute_path() {
        let ctx = ToolContext::new(PathBuf::from("/tmp/theta-workdir"));
        let resolved = resolve_path(&ctx, "/Users/rhafid/.theta");
        assert_eq!(resolved, PathBuf::from("/Users/rhafid/.theta"));
    }

    #[test]
    fn resolve_path_resolves_relative_from_workdir() {
        let ctx = ToolContext::new(PathBuf::from("/tmp/theta-workdir"));
        let resolved = resolve_path(&ctx, "src/main.rs");
        assert_eq!(resolved, PathBuf::from("/tmp/theta-workdir/src/main.rs"));
    }

    #[test]
    fn shorten_path_replaces_home_with_tilde() {
        let home = dirs::home_dir().unwrap();
        let path = home.join("projects/theta");
        let result = shorten_path(&path);
        assert!(result.starts_with("~/"));
        assert!(result.ends_with("projects/theta"));
    }

    #[test]
    fn shorten_path_leaves_non_home_path_unchanged() {
        let result = shorten_path(Path::new("/tmp/theta-workdir"));
        assert_eq!(result, "/tmp/theta-workdir");
    }
}

mod rpc {
    use theta::config::{AuthConfig, ProviderToken, ThetaConfig};
    use theta::rpc::resolve_auth_for_model;
    use theta_models::BuiltInCatalog;

    #[tokio::test]
    async fn resolve_auth_for_model_falls_back_to_authenticated_provider() {
        let catalog = BuiltInCatalog::new();
        let mut cfg = ThetaConfig::default();
        cfg.auth = AuthConfig {
            tokens: vec![ProviderToken {
                provider: "openai-codex".into(),
                token: "codex-token".into(),
                expires_at: None,
                obtained_at: 1,
            }],
            oauth_tokens: vec![],
        };

        let (model, key) = resolve_auth_for_model(&cfg, &catalog, "gpt-5.5")
            .await
            .expect("fallback should resolve");
        assert_eq!(model.provider, theta_ai::Provider::OpenAiCodex);
        assert_eq!(key, "codex-token");
    }

    #[tokio::test]
    async fn resolve_auth_for_model_returns_explicit_error_when_no_auth() {
        let catalog = BuiltInCatalog::new();
        let cfg = ThetaConfig::default();
        let err = resolve_auth_for_model(&cfg, &catalog, "gpt-5.5")
            .await
            .expect_err("expected missing auth error");
        assert!(
            err.to_string().contains("no auth token for 'openai'"),
            "unexpected error: {err}"
        );
    }
}

mod mentions {
    use tempfile::TempDir;
    use theta::mentions::expand_file_mentions;
    use theta::mentions::extract_mentions;
    use theta_ai::ContentBlock;

    #[test]
    fn extracts_plain_and_quoted_mentions() {
        assert_eq!(
            extract_mentions(r#"read @src/main.rs and @"docs/user guide.md""#),
            vec!["docs/user guide.md".to_string(), "src/main.rs".to_string()]
        );
    }

    #[test]
    fn ignores_email_addresses() {
        assert!(extract_mentions("a@b.com").is_empty());
    }

    #[tokio::test]
    async fn expands_existing_file_mentions() {
        let tmp = TempDir::new().unwrap();
        std::fs::create_dir_all(tmp.path().join("src")).unwrap();
        std::fs::write(tmp.path().join("src/main.rs"), "fn main() {}\n").unwrap();

        let blocks = expand_file_mentions(tmp.path(), "read @src/main.rs").await;
        let ContentBlock::Text { text } = &blocks[0] else {
            panic!("expected text block");
        };
        assert!(text.contains("# Referenced files"));
        assert!(text.contains("## @src/main.rs"));
        assert!(text.contains("fn main() {}"));
    }
}

mod execution_continuity {
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
    use theta_ai::model::{Model, ModelCompat};
    use theta_ai::providers::ProviderRegistry;
    use theta_ai::types::{
        Api, ContentBlock, Context, Message, Modality, Provider as ProviderKind,
        SimpleStreamOptions, StopReason, StreamOptions,
    };
    use theta_ai::{LlmProvider, ThetaError};

    type EventStream<'a> = Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send + 'a>>;

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

        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
    }

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

                let has_execute_guardrail = system.contains("the full");
                let has_function_calling_guardrail = system.contains("function-calling");

                if has_execute_guardrail
                    && has_function_calling_guardrail
                    && !context.tools.is_empty()
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

        fn as_any(&self) -> &dyn std::any::Any {
            self
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
            context_window: 128_000,
            max_tokens: 16_384,
            compat: ModelCompat::for_openai(),
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

        let agent = Agent::new(model.clone(), Arc::new(registry), vec![model.clone()]);
        agent.add_tool(Arc::new(MockTool)).await;

        let wd = std::env::current_dir().expect("cwd");
        let system = build_system_prompt(&wd, "test-model", Some("medium"), Some(250_000)).await;
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
        let agent = Agent::new(model.clone(), Arc::new(registry), vec![model.clone()]);
        agent.load_messages(reopened.messages).await;

        agent
            .prompt(vec![ContentBlock::text("third ask")])
            .await
            .expect("resume prompt should succeed");
    }
}
