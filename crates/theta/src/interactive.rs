//! Interactive TUI mode — connects the agent to the terminal UI.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use theta_agent_core::agent::Agent;
use theta_agent_core::events::AgentEvent;
use theta_ai::providers::default_registry;
use theta_ai::{Model, ModelCatalog, Provider};
use theta_models::BuiltInCatalog;
use theta_models::opencode;
use theta_tui::App;
use theta_tui::app::{HistoryEntry, TuiAction, TuiEvent};
use theta_tui::components::CommandEntry;
use theta_tui::components::{ModelEntry, SessionInfo, known_providers};
use theta_tui::theme::Theme;
use tokio::sync::{RwLock, mpsc};

use crate::config::ThetaConfig;
use crate::session::SessionManager;
use crate::system_prompt::build_system_prompt_with_skills;
use crate::tools::ToolContext;
use crate::tools::builtin_tools;

/// Shared agent handle — None until auth is resolved.
type AgentCell = Arc<RwLock<Option<Arc<Agent>>>>;

/// Run the TUI interactive mode.
pub async fn run_tui(
    config: &ThetaConfig,
    working_dir: &Path,
    model_id: &str,
    thinking: &str,
    initial_prompt: Option<&str>,
) -> anyhow::Result<()> {
    // Build runtime model snapshot (hydrate OpenCode model list if available).
    let catalog = BuiltInCatalog::new();
    let runtime_models_cell: Arc<RwLock<Vec<Model>>> =
        Arc::new(RwLock::new(resolve_runtime_models(&catalog).await));
    let runtime_models = runtime_models_cell.read().await.clone();

    let model = find_model_by_id(&runtime_models, model_id)
        .ok_or_else(|| anyhow::anyhow!("model not found: {model_id}"))?
        .clone();

    // Resolve auth. If the default model's provider has no token,
    // try other providers that DO have auth (e.g., user logged in
    // via Codex but default model is from OpenAI provider).
    let provider_str = provider_to_string(model.provider);
    let mut auth_config = config.auth.clone();
    let model_entries = available_model_entries(&runtime_models, &mut auth_config).await;
    let api_key = auth_config.get_api_key(&provider_str).await;

    // Fallback: if no auth for the default model's provider, check
    // other providers and find a matching model.
    let (model, model_id, api_key) = if api_key.is_none() {
        let alt_providers = [
            ("openai-codex", Provider::OpenAiCodex),
            ("openai", Provider::OpenAI),
            ("deepseek", Provider::DeepSeek),
            ("opencode", Provider::OpenCode),
        ];
        let mut found: Option<(theta_ai::Model, String, String)> = None;
        for (prov_str, prov) in &alt_providers {
            if prov_str == &provider_str {
                continue; // already checked
            }
            if let Some(key) = auth_config.get_api_key(prov_str).await
                && let Some(m) = runtime_models.iter().find(|m| {
                    m.provider == *prov && (m.id == model_id || m.id.starts_with(model_id))
                })
            {
                found = Some((m.clone(), m.id.clone(), key));
                break;
            }
        }
        if let Some((m, mid, key)) = found {
            (m, mid, Some(key))
        } else {
            (model, model_id.to_string(), None)
        }
    } else {
        (model, model_id.to_string(), api_key)
    };
    let has_auth = api_key.is_some();

    // Create channels between TUI and agent bridge.
    let (event_tx_raw, mut event_rx_raw) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let (message_tx, mut message_rx) = mpsc::unbounded_channel::<String>();
    let (action_tx, mut action_rx) = mpsc::unbounded_channel();

    // Lazy session: created on first message, None until then.
    // No session file is written for login-only or no-message runs.
    let session_id_cell: Arc<RwLock<Option<String>>> = Arc::new(RwLock::new(None));

    // Shared agent cell — populated immediately if auth is available,
    // populated by the action handler after login otherwise.
    let agent_cell: AgentCell = Arc::new(RwLock::new(None));

    // Build the theme.
    let theme = match config.theme.as_deref() {
        Some("monokai") => Theme::monokai(),
        _ => Theme::default(),
    };

    // Shared notification channel: the ScriptHooks after_tool_call callback
    // signals this whenever tool execution may have changed extension state.
    // The TUI poller wakes on this instead of polling on a timer.
    let status_notify = Arc::new(tokio::sync::Notify::new());

    // ------------------------------------------------------------------
    // Always spawn the action handler first (handles login + agent init).
    // ------------------------------------------------------------------
    let action_agent_cell = agent_cell.clone();
    let action_event_tx = event_tx_raw.clone();
    let action_session_id_cell = session_id_cell.clone();
    let action_working_dir = working_dir.to_path_buf();
    let action_model_id = model_id.clone();
    let action_model = model.clone();
    let action_thinking = thinking.to_string();
    let action_catalog = BuiltInCatalog::new();
    let action_runtime_models_cell = runtime_models_cell.clone();
    let action_config = config.clone();
    let action_status_notify = status_notify.clone();
    tokio::spawn(async move {
        while let Some(action) = action_rx.recv().await {
            handle_tui_action(
                action,
                &action_agent_cell,
                &action_session_id_cell,
                &action_event_tx,
                &action_working_dir,
                &action_model_id,
                &action_thinking,
                &action_model,
                &action_catalog,
                &action_runtime_models_cell,
                &action_config,
                &action_status_notify,
            )
            .await;
        }
    });

    // ------------------------------------------------------------------
    // If we have auth, create the agent now and spawn event bridge.
    // ------------------------------------------------------------------
    if let Some(ref key) = api_key {
        let agent = create_agent(
            &model,
            key,
            config,
            working_dir,
            &model_id,
            thinking,
            &status_notify,
        )
        .await?;
        let agent = Arc::new(agent);
        *agent_cell.write().await = Some(agent.clone());
        let persisted = crate::settings::load_settings().await;
        spawn_event_bridge(
            agent.clone(),
            event_tx_raw.clone(),
            persisted.tool_progress_hz.max(1),
        );

        // Send initial valid thinking levels for the model.
        let levels = compute_valid_thinking_levels(&model);
        let tl = thinking_level_to_string(parse_thinking_level(thinking));
        let _ = event_tx_raw.send(TuiEvent::ThinkingLevels {
            levels,
            current: tl,
        });

        // Poll extension status rows — wait on notify from hook evaluations.
        // Reads the current agent from agent_cell so it works across agent
        // replacements (e.g. after login).
        let ext_agent_cell = agent_cell.clone();
        let ext_event_tx = event_tx_raw.clone();
        let ext_notify = status_notify.clone();
        tokio::spawn(async move {
            loop {
                ext_notify.notified().await;
                let Some(agent) = ext_agent_cell.read().await.clone() else {
                    continue;
                };
                let rows = agent.hooks().tui_status_rows();
                let lines = agent.hooks().tui_status_lines();
                if !rows.is_empty() || !lines.is_empty() {
                    let payload = to_extension_payload(rows, lines);
                    let _ = ext_event_tx.send(TuiEvent::ExtensionStatus(payload));
                }
            }
        });

        // Persist the model + thinking for the next session.
        let mut s = crate::settings::load_settings().await;
        s.last_model = Some(model_id.clone());
        s.last_thinking = Some(thinking.to_string());
        crate::settings::save_settings(&s).await.ok();
    }

    let skills = crate::skills::discover_skills(working_dir).await;

    // ------------------------------------------------------------------
    // Spawn message handler — waits for agent, creates session lazily.
    // ------------------------------------------------------------------
    let msg_agent_cell = agent_cell.clone();
    let msg_event_tx = event_tx_raw.clone();
    let msg_working_dir = working_dir.to_path_buf();
    let msg_session_id_cell = session_id_cell.clone();
    let msg_skills = skills.clone();
    tokio::spawn(async move {
        // Wait for agent to be available (block until login completes).
        let agent = wait_for_agent(&msg_agent_cell).await;
        let session_mgr = SessionManager::new(&msg_working_dir);
        while let Some(message) = message_rx.recv().await {
            // Reload agent in case it was replaced (model switch, etc.).
            let agent = msg_agent_cell.read().await.clone().unwrap_or(agent.clone());

            // Lazy session creation on first real message — no session
            // file is left behind for login-only or no-message runs.
            if msg_session_id_cell.read().await.is_none() {
                // Read current model from agent — not from closure-captured
                // model_id which may be stale after a model switch.
                let current_model_id = agent.state().await.model.id.clone();
                match session_mgr.create(Some(&current_model_id)).await {
                    Ok(session) => {
                        let id = session
                            .meta
                            .as_ref()
                            .map(|m| m.id.clone())
                            .unwrap_or_default();
                        let _ = msg_event_tx.send(TuiEvent::SessionCreated {
                            id: id.clone(),
                            model: current_model_id,
                        });
                        *msg_session_id_cell.write().await = Some(id);
                    }
                    Err(e) => {
                        let _ = msg_event_tx
                            .send(TuiEvent::Error(format!("Failed to create session: {e}")));
                        continue;
                    }
                }
            }
            let Some(sid) = msg_session_id_cell.read().await.clone() else {
                continue;
            };
            let run_id = format!("run-{}", rand::random::<u64>());
            let turn_id = format!("turn-{}", rand::random::<u64>());
            if let Err(e) = session_mgr
                .mark_run_in_progress(&sid, &run_id, &turn_id)
                .await
            {
                tracing::warn!("failed to mark session run in progress: {e}");
            }

            let expanded_message = expand_skill_message(&message, &msg_skills);

            let blocks =
                crate::mentions::expand_file_mentions(&msg_working_dir, &expanded_message).await;
            if let Err(e) = agent.prompt(blocks).await {
                tracing::error!("agent prompt failed: {e}");
                let _ = msg_event_tx.send(TuiEvent::Error(format_error_chain(&e)));
                let _ = session_mgr
                    .mark_run_completed(&sid, Some("ProviderFailure"))
                    .await;
                continue;
            }
            let state = agent.state().await;
            match session_mgr.open_by_id(&sid).await {
                Ok(mut session) => {
                    if let Err(e) = session_mgr
                        .append_missing_entries(&mut session, &state.messages)
                        .await
                    {
                        tracing::error!("failed to persist session entries: {e}");
                        let _ = msg_event_tx
                            .send(TuiEvent::Error(format!("Failed to persist session: {e}")));
                    } else {
                        let _ = session_mgr
                            .mark_run_completed(
                                &sid,
                                state
                                    .last_turn_end_reason
                                    .map(|r| format!("{r:?}"))
                                    .as_deref(),
                            )
                            .await;
                    }
                }
                Err(e) => {
                    tracing::error!("failed to open session {sid} for persistence: {e}");
                    let _ = msg_event_tx.send(TuiEvent::Error(format!(
                        "Failed to open active session for persistence: {e}"
                    )));
                }
            }
        }
    });

    // ------------------------------------------------------------------
    // Build available commands + skills for the / picker.
    let mut commands = vec![
        CommandEntry {
            name: "help".into(),
            description: "Show available commands".into(),
        },
        CommandEntry {
            name: "model".into(),
            description: "Pick model from available authenticated models".into(),
        },
        CommandEntry {
            name: "thinking".into(),
            description: "Set thinking level (off/minimal/low/medium/high/xhigh)".into(),
        },
        CommandEntry {
            name: "effort".into(),
            description: "Alias for /thinking".into(),
        },
        CommandEntry {
            name: "clear".into(),
            description: "Clear the chat display".into(),
        },
        CommandEntry {
            name: "session".into(),
            description: "Show session info (tokens, context window, compaction)".into(),
        },
        CommandEntry {
            name: "compact".into(),
            description: "Manually compact context to fit in context window".into(),
        },
        CommandEntry {
            name: "fork".into(),
            description: "Fork the current session".into(),
        },
        CommandEntry {
            name: "new".into(),
            description: "Start a new unsaved session".into(),
        },
        CommandEntry {
            name: "sessions".into(),
            description: "List recent sessions to resume".into(),
        },
        CommandEntry {
            name: "resume".into(),
            description: "Alias for /sessions".into(),
        },
        CommandEntry {
            name: "tree".into(),
            description: "Open session tree selector".into(),
        },
        CommandEntry {
            name: "login".into(),
            description: "Log in to a provider".into(),
        },
        CommandEntry {
            name: "skills".into(),
            description: "List available skills".into(),
        },
        CommandEntry {
            name: "exit".into(),
            description: "Exit Theta".into(),
        },
        CommandEntry {
            name: "cancel".into(),
            description: "Cancel current agent execution".into(),
        },
    ];

    // Skills as /skill:<name> commands.
    for skill in &skills {
        commands.push(CommandEntry {
            name: format!("skill:{}", skill.name),
            description: skill.description.clone(),
        });
    }

    // Build and run the TUI.
    // ------------------------------------------------------------------
    let persisted = crate::settings::load_settings().await;
    let mut app = App::new(
        theme.clone(),
        &model.id,
        "", // session created lazily on first message
        thinking,
        theta_tui::app::SettingsPayload {
            steering_mode: persisted.steering_mode,
            follow_up_mode: persisted.follow_up_mode,
            transport_preference: persisted.transport_preference,
            show_thinking: persisted.show_thinking,
            tool_progress_hz: persisted.tool_progress_hz,
            enter_behavior: persisted.enter_behavior,
        },
        model_entries,
        commands,
        working_dir.to_path_buf(),
        event_rx,
        message_tx,
        action_tx,
        Some(crate::window_title(working_dir)),
    );

    // If auth is missing, start the login flow immediately.
    if !has_auth {
        let providers = known_providers(
            config.auth.has_token("openai"),
            config.auth.has_token("openai-codex"),
            config.auth.has_token("deepseek"),
            config.auth.has_token("opencode"),
        );
        app.start_login_flow(providers);
    }

    // Send initial prompt if provided (and show it in chat as user message).
    if let Some(prompt) = initial_prompt {
        app.send_initial_message(prompt.to_string());
    }

    // Forward all events directly — no coalescing. Coalescing
    // progress here causes a "BUM!" effect where progress accumulates
    // silently and flushes only on the next non-progress event.
    tokio::spawn(async move {
        while let Some(event) = event_rx_raw.recv().await {
            let _ = event_tx.send(event);
        }
    });

    app.run().await?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a fully configured agent.
async fn create_agent(
    model: &Model,
    api_key: &str,
    config: &ThetaConfig,
    working_dir: &Path,
    model_id: &str,
    thinking: &str,
    status_notify: &Arc<tokio::sync::Notify>,
) -> anyhow::Result<Agent> {
    let catalog = BuiltInCatalog::new();
    let available_models: Vec<theta_ai::Model> = catalog.list().into_iter().cloned().collect();
    let registry = default_registry();
    registry.set_api_key(model.provider, api_key);

    let tool_ctx = ToolContext::new(working_dir.to_path_buf());
    let mut agent = Agent::new(model.clone(), Arc::new(registry), available_models);
    agent.set_config(crate::config::to_agent_config(config));
    for tool in builtin_tools(tool_ctx) {
        agent.add_tool(tool).await;
    }

    let system_blocks = build_system_prompt_with_skills(
        working_dir,
        model_id,
        Some(thinking),
        &config.startup_skills,
    )
    .await;
    agent.set_system_prompt(system_blocks).await;

    // Load script hooks from ~/.theta/extensions/*.rhai and ./.theta/extensions/*.rhai.
    if let Some(hooks) =
        crate::scripts::load_script_hooks(working_dir, Arc::clone(status_notify)).await
    {
        agent.set_hooks(hooks);
    }

    // Apply thinking level from settings.
    let tl = parse_thinking_level(thinking);
    agent.set_thinking_level(tl).await;

    Ok(agent)
}

/// Parse a thinking level string into a ThinkingLevel enum.
fn parse_thinking_level(level: &str) -> theta_ai::ThinkingLevel {
    match level.to_lowercase().as_str() {
        "off" => theta_ai::ThinkingLevel::Off,
        "minimal" => theta_ai::ThinkingLevel::Minimal,
        "low" => theta_ai::ThinkingLevel::Low,
        "medium" => theta_ai::ThinkingLevel::Medium,
        "high" => theta_ai::ThinkingLevel::High,
        "xhigh" => theta_ai::ThinkingLevel::XHigh,
        _ => theta_ai::ThinkingLevel::Off,
    }
}

/// Compute the list of valid thinking level strings for a model.
fn compute_valid_thinking_levels(model: &theta_ai::Model) -> Vec<String> {
    let all_levels = [
        ("off", theta_ai::ThinkingLevel::Off),
        ("minimal", theta_ai::ThinkingLevel::Minimal),
        ("low", theta_ai::ThinkingLevel::Low),
        ("medium", theta_ai::ThinkingLevel::Medium),
        ("high", theta_ai::ThinkingLevel::High),
        ("xhigh", theta_ai::ThinkingLevel::XHigh),
    ];
    all_levels
        .into_iter()
        .filter(|(_, level)| {
            // Off is always valid; other levels need a mapping.
            *level == theta_ai::ThinkingLevel::Off || model.thinking_param(*level).is_some()
        })
        .map(|(name, _)| name.to_string())
        .collect()
}

/// Convert a ThinkingLevel to its string representation.
fn thinking_level_to_string(level: theta_ai::ThinkingLevel) -> String {
    match level {
        theta_ai::ThinkingLevel::Off => "off".to_string(),
        theta_ai::ThinkingLevel::Minimal => "minimal".to_string(),
        theta_ai::ThinkingLevel::Low => "low".to_string(),
        theta_ai::ThinkingLevel::Medium => "medium".to_string(),
        theta_ai::ThinkingLevel::High => "high".to_string(),
        theta_ai::ThinkingLevel::XHigh => "xhigh".to_string(),
    }
}

/// Spawn the event bridge — subscribes to agent events, forwards to TUI.
fn spawn_event_bridge(agent: Arc<Agent>, event_tx: mpsc::UnboundedSender<TuiEvent>, _hz: u64) {
    tokio::spawn(async move {
        let reserve_tokens = agent.config().compaction.reserve_tokens;
        let context_window = agent.state().await.model.context_window;
        let mut events = agent.subscribe();
        let mut tool_names: HashMap<String, String> = HashMap::new();
        let mut tool_args: HashMap<String, String> = HashMap::new(); // id -> raw args JSON
        let mut saw_assistant_text_delta = false;
        let mut saw_thinking_delta = false;
        let mut latest_turn_end_reason = "completed".to_string();

        loop {
            let received = events.recv().await;
            match received {
                Ok(AgentEvent::MessageStart) => {
                    saw_assistant_text_delta = false;
                    saw_thinking_delta = false;
                    tool_args.clear();
                }
                Ok(AgentEvent::TextDelta { text }) => {
                    saw_assistant_text_delta = true;
                    let _ = event_tx.send(TuiEvent::TextDelta(text));
                }
                Ok(AgentEvent::ThinkingDelta { thinking }) => {
                    saw_thinking_delta = true;
                    let _ = event_tx.send(TuiEvent::ThinkingDelta(thinking));
                }
                Ok(AgentEvent::ThinkingStart) => {
                    let _ = event_tx.send(TuiEvent::ThinkingStart);
                }
                Ok(AgentEvent::ThinkingEnd) => {
                    let _ = event_tx.send(TuiEvent::ThinkingEnd);
                }
                Ok(AgentEvent::ToolCallStart { id, name }) => {
                    // Forward LLM-side tool call preparation so the TUI can show
                    // tools appearing during the response stream (before execution).
                    let _ = event_tx.send(TuiEvent::ToolCallPrepared { name, id });
                }
                Ok(AgentEvent::ToolCallDelta { id, arguments }) => {
                    // Accumulate streamed arguments for this tool call.
                    let entry = tool_args.entry(id).or_default();
                    entry.push_str(&arguments);
                }
                Ok(AgentEvent::ToolExecutionStart {
                    tool_call_id: id,
                    tool_name: name,
                }) => {
                    let args = tool_args.remove(&id);
                    tool_names.insert(id.clone(), name.clone());

                    // Detect skill loading: read tool targeting SKILL.md
                    if name == "read"
                        && let Some(ref raw_args) = args
                        && let Ok(json) = serde_json::from_str::<serde_json::Value>(raw_args)
                        && let Some(path) = json.get("path").and_then(|v| v.as_str())
                    {
                        let lower_path = path.to_lowercase();
                        if lower_path.ends_with("skill.md") || lower_path.contains("/skill.md") {
                            // Extract skill name from path
                            // Path like: skills/web-research/SKILL.md -> web-research
                            // Path like: .theta/skills/caveman/SKILL.md -> caveman
                            let skill_name = std::path::Path::new(path)
                                .parent()
                                .and_then(|p| p.file_name())
                                .and_then(|n| n.to_str())
                                .unwrap_or("")
                                .to_string();
                            if !skill_name.is_empty() {
                                let _ =
                                    event_tx.send(TuiEvent::SkillActivated { name: skill_name });
                            }
                        }
                    }

                    let _ = event_tx.send(TuiEvent::ToolStart { name, id, args });
                }
                Ok(AgentEvent::ToolExecutionProgress {
                    tool_call_id: id,
                    output,
                }) => {
                    // Forward progress directly. The TUI discards it during render,
                    // so this is just a pass-through.
                    let name = tool_names.get(&id).cloned().unwrap_or_else(|| id.clone());
                    let _ = event_tx.send(TuiEvent::ToolProgress {
                        name,
                        message: output,
                    });
                }
                Ok(AgentEvent::ToolExecutionEnd { result }) => {
                    let summary = format_tool_summary(&result, 2200);
                    tool_names.remove(&result.tool_call_id);
                    tool_args.remove(&result.tool_call_id); // cleanup stale args
                    let _ = event_tx.send(TuiEvent::ToolEnd {
                        id: result.tool_call_id,
                        name: result.tool_name,
                        is_error: result.is_error,
                        summary,
                    });
                }
                Ok(AgentEvent::MessageEnd { message }) => {
                    // Forward token usage to TUI status bar.
                    if let theta_ai::Message::Assistant { content, usage, .. } = &message {
                        if let Some(u) = usage {
                            let avail = context_window.saturating_sub(reserve_tokens);
                            let pct = if avail > 0 {
                                (u.input_tokens as f64 / avail as f64 * 100.0) as u32
                            } else {
                                0
                            };
                            let _ = event_tx.send(TuiEvent::ContextTokens {
                                tokens: u.input_tokens,
                                pct,
                            });
                        }
                        if !saw_assistant_text_delta {
                            let final_text = content
                                .iter()
                                .filter_map(|b| match b {
                                    theta_ai::ContentBlock::Text { text } => Some(text.as_str()),
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n");
                            if !final_text.is_empty() {
                                let _ = event_tx.send(TuiEvent::TextDelta(final_text));
                            }
                        }
                        if !saw_thinking_delta {
                            let final_thinking = content
                                .iter()
                                .filter_map(|b| match b {
                                    theta_ai::ContentBlock::Thinking { thinking, .. } => {
                                        Some(thinking.as_str())
                                    }
                                    _ => None,
                                })
                                .collect::<Vec<_>>()
                                .join("\n");
                            if !final_thinking.is_empty() {
                                let _ = event_tx.send(TuiEvent::ThinkingDelta(final_thinking));
                            }
                        }
                    }
                    // Forward MessageEnd so the TUI knows LLM streaming is
                    // complete and tool execution is about to begin.
                    let _ = event_tx.send(TuiEvent::MessageEnd);
                    saw_assistant_text_delta = false;
                    saw_thinking_delta = false;
                }
                Ok(AgentEvent::TurnStart { .. }) => {
                    let _ = event_tx.send(TuiEvent::TurnStart);
                }
                Ok(AgentEvent::TurnEnd { .. }) => {
                    let _ = event_tx.send(TuiEvent::TurnEnd {
                        stop_reason: latest_turn_end_reason.clone(),
                    });
                }
                Ok(AgentEvent::TurnDecision {
                    reason, details, ..
                }) => {
                    let _ = event_tx.send(TuiEvent::TurnDecision {
                        reason: format!("{reason:?}"),
                        details,
                    });
                }
                Ok(AgentEvent::TurnTerminated {
                    reason, details, ..
                }) => {
                    latest_turn_end_reason = format!("{reason:?}");
                    if !matches!(reason, theta_agent_core::types::TurnEndReason::Completed) {
                        let detail = details.trim();
                        let message = if detail.is_empty() {
                            format!("Turn ended: {reason:?}")
                        } else {
                            format!("Turn ended: {reason:?}\n{detail}")
                        };
                        let _ = event_tx.send(TuiEvent::Info(message));
                    }
                }
                Ok(AgentEvent::SafetyDecision {
                    decision,
                    tool_name,
                    details,
                    ..
                }) => {
                    if matches!(
                        decision,
                        theta_agent_core::types::SafetyDecisionKind::Rejected
                    ) {
                        let _ = event_tx.send(TuiEvent::Info(format!(
                            "Safety policy rejected {tool_name}: {details}"
                        )));
                    }
                }
                Ok(AgentEvent::AgentEnd { aborted }) => {
                    let _ = event_tx.send(TuiEvent::AgentEnd { aborted });
                }
                Ok(AgentEvent::ContextCompacted {
                    trimmed_count,
                    tokens_before,
                    tokens_after,
                }) => {
                    let _ = event_tx.send(TuiEvent::ContextCompacted {
                        trimmed_count,
                        tokens_before,
                        tokens_after,
                    });
                }
                Ok(AgentEvent::Retrying { attempt, delay_ms }) => {
                    let _ = event_tx.send(TuiEvent::Retrying { attempt, delay_ms });
                }
                Ok(AgentEvent::ReplaySanitized {
                    dropped_assistant_messages,
                    synthesized_tool_results,
                    normalized_tool_call_ids,
                    deduped_tool_results,
                }) => {
                    let _ = event_tx.send(TuiEvent::Info(format!(
                        "replay sanitized: dropped_assistant={dropped_assistant_messages}, synthesized_tool_results={synthesized_tool_results}, normalized_tool_call_ids={normalized_tool_call_ids}, deduped_tool_results={deduped_tool_results}"
                    )));
                }
                Ok(AgentEvent::Error { message }) => {
                    let _ = event_tx.send(TuiEvent::Error(message));
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    // Log but don't forward to TUI — just continue.
                    // The next recv() will return the most recent event.
                    tracing::warn!("event bridge lagged by {n} events; continuing");
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                _ => {}
            }
        }
    });
}
async fn wait_for_agent(cell: &AgentCell) -> Arc<Agent> {
    loop {
        if let Some(agent) = cell.read().await.clone() {
            return agent;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}

/// Handle a single TUI action (switched model, /login, /fork, etc.).
#[allow(clippy::too_many_arguments)]
async fn handle_tui_action(
    action: TuiAction,
    agent_cell: &AgentCell,
    session_id_cell: &Arc<RwLock<Option<String>>>,
    event_tx: &mpsc::UnboundedSender<TuiEvent>,
    working_dir: &Path,
    model_id: &str,
    thinking: &str,
    model: &Model,
    catalog: &BuiltInCatalog,
    runtime_models_cell: &Arc<RwLock<Vec<Model>>>,
    config: &ThetaConfig,
    status_notify: &Arc<tokio::sync::Notify>,
) {
    match action {
        TuiAction::StartCodexOAuth => {
            tracing::info!("Starting Codex OAuth login flow");
            let _ = event_tx.send(TuiEvent::Info(
                "Sign in to ChatGPT in your browser...".into(),
            ));
            match crate::oauth::codex::login_codex().await {
                Ok(creds) => {
                    // Save OAuth credentials.
                    match crate::config::load_auth(None).await {
                        Ok(mut auth) => {
                            auth.set_oauth_token(
                                "openai-codex",
                                &creds.access_token,
                                &creds.refresh_token,
                                creds.expires_at,
                            );
                            if let Err(e) = crate::config::save_auth(&auth, None).await {
                                let _ = event_tx
                                    .send(TuiEvent::Error(format!("Failed to save token: {e}")));
                                return;
                            }
                            if let Some(agent) = agent_cell.read().await.clone() {
                                agent
                                    .set_api_key(Provider::OpenAiCodex, creds.access_token.clone());
                            }
                            // If initial login, find Codex model variant and create agent.
                            if agent_cell.read().await.is_none() {
                                let codex_catalog = BuiltInCatalog::new();
                                let codex_model = codex_catalog
                                    .list()
                                    .into_iter()
                                    .find(|cm| {
                                        cm.id == model_id && cm.provider == Provider::OpenAiCodex
                                    })
                                    .cloned()
                                    .unwrap_or_else(|| model.clone());

                                match create_agent(
                                    &codex_model,
                                    &creds.access_token,
                                    config,
                                    working_dir,
                                    &codex_model.id,
                                    thinking,
                                    status_notify,
                                )
                                .await
                                {
                                    Ok(agent) => {
                                        let agent = Arc::new(agent);
                                        let hz =
                                            crate::settings::load_settings().await.tool_progress_hz;
                                        spawn_event_bridge(agent.clone(), event_tx.clone(), hz);
                                        *agent_cell.write().await = Some(agent);
                                        let _ = event_tx.send(TuiEvent::Info(
                                            "Connected to ChatGPT Plus. Ready.".into(),
                                        ));
                                        // Persist model + thinking.
                                        let mut s = crate::settings::load_settings().await;
                                        s.last_model = Some(codex_model.id.clone());
                                        s.last_thinking = Some(thinking.to_string());
                                        crate::settings::save_settings(&s).await.ok();
                                    }
                                    Err(e) => {
                                        let _ = event_tx.send(TuiEvent::Error(format!(
                                            "Failed to start agent: {e}"
                                        )));
                                    }
                                }
                            }

                            refresh_runtime_models(catalog, runtime_models_cell).await;
                            let runtime_models = runtime_models_cell.read().await.clone();
                            let refreshed_models =
                                available_model_entries(&runtime_models, &mut auth).await;
                            let _ = event_tx.send(TuiEvent::UpdateModels(refreshed_models));
                        }
                        Err(e) => {
                            let _ =
                                event_tx.send(TuiEvent::Error(format!("Failed to load auth: {e}")));
                        }
                    }
                }
                Err(e) => {
                    tracing::error!("Codex OAuth failed: {e}");
                    let _ = event_tx.send(TuiEvent::Error(format!("OAuth login failed: {e}")));
                }
            }
        }
        TuiAction::LoginResult { provider, token } => {
            // Save token.
            match crate::config::load_auth(None).await {
                Ok(mut auth) => {
                    auth.set_token(&provider, &token, None);
                    if let Err(e) = crate::config::save_auth(&auth, None).await {
                        let _ =
                            event_tx.send(TuiEvent::Error(format!("Failed to save token: {e}")));
                        return;
                    }
                    if let Some(agent) = agent_cell.read().await.clone()
                        && let Some(provider_kind) = provider_from_string(&provider)
                    {
                        agent.set_api_key(provider_kind, token.clone());
                    }
                    // If this was the initial login (no agent yet), create the agent now.
                    if agent_cell.read().await.is_none() {
                        match create_agent(
                            model,
                            &token,
                            config,
                            working_dir,
                            model_id,
                            thinking,
                            status_notify,
                        )
                        .await
                        {
                            Ok(agent) => {
                                let agent = Arc::new(agent);
                                let hz = crate::settings::load_settings().await.tool_progress_hz;
                                spawn_event_bridge(agent.clone(), event_tx.clone(), hz);
                                *agent_cell.write().await = Some(agent);
                                let _ = event_tx.send(TuiEvent::Info(format!(
                                    "Connected to {provider}. Ready."
                                )));
                                // Persist model + thinking.
                                let mut s = crate::settings::load_settings().await;
                                s.last_model = Some(model_id.to_string());
                                s.last_thinking = Some(thinking.to_string());
                                crate::settings::save_settings(&s).await.ok();
                            }
                            Err(e) => {
                                let _ = event_tx
                                    .send(TuiEvent::Error(format!("Failed to start agent: {e}")));
                            }
                        }
                    }

                    refresh_runtime_models(catalog, runtime_models_cell).await;
                    let runtime_models = runtime_models_cell.read().await.clone();
                    let refreshed_models =
                        available_model_entries(&runtime_models, &mut auth).await;
                    let _ = event_tx.send(TuiEvent::UpdateModels(refreshed_models));
                }
                Err(e) => {
                    let _ = event_tx.send(TuiEvent::Error(format!("Failed to load auth: {e}")));
                }
            }
        }
        TuiAction::SwitchModel {
            model_id,
            provider,
            request_id,
        } => {
            let acknowledge = |event_tx: &mpsc::UnboundedSender<TuiEvent>| {
                let _ = event_tx.send(TuiEvent::ActionAck { request_id });
            };

            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                acknowledge(event_tx);
                return;
            };
            let runtime_models = runtime_models_cell.read().await.clone();
            let model = provider
                .as_deref()
                .and_then(|p| find_model_by_provider_and_id(&runtime_models, p, &model_id))
                .or_else(|| find_model_by_id(&runtime_models, &model_id));

            if let Some(m) = model {
                let provider = provider_to_string(m.provider);
                let key = match crate::config::load_auth(None).await {
                    Ok(mut auth) => match auth.get_api_key(&provider).await {
                        Some(key) => key,
                        None => {
                            let _ = event_tx.send(TuiEvent::Error(format!(
                                "Model {model_id} is unavailable: missing auth for {provider}"
                            )));
                            acknowledge(event_tx);
                            return;
                        }
                    },
                    Err(e) => {
                        let _ = event_tx.send(TuiEvent::Error(format!("Failed to load auth: {e}")));
                        acknowledge(event_tx);
                        return;
                    }
                };

                agent.set_api_key(m.provider, key);
                let levels = compute_valid_thinking_levels(&m);
                agent.set_model(m).await;
                // Read thinking level before rebuilding prompt so runtime
                // context shows the correct level and startup skills are re-applied.
                let state = agent.state().await;
                let current_thinking = thinking_level_to_string(state.thinking_level);
                drop(state);
                let blocks = build_system_prompt_with_skills(
                    working_dir,
                    &model_id,
                    Some(&current_thinking),
                    &config.startup_skills,
                )
                .await;
                agent.set_system_prompt(blocks).await;
                let _ = event_tx.send(TuiEvent::Info(format!(
                    "Switched to {model_id} ({provider})"
                )));
                let _ = event_tx.send(TuiEvent::ModelSwitched {
                    model: model_id.to_string(),
                });
                let _ = event_tx.send(TuiEvent::ThinkingLevels {
                    levels,
                    current: current_thinking,
                });
                // Persist model preference (merge with existing settings).
                let mut s = crate::settings::load_settings().await;
                s.last_model = Some(model_id.to_string());
                crate::settings::save_settings(&s).await.ok();
                acknowledge(event_tx);
            } else {
                let _ = event_tx.send(TuiEvent::Error(format!("Model not found: {model_id}")));
                acknowledge(event_tx);
            }
        }
        TuiAction::SetThinking { level, request_id } => {
            let acknowledge = |event_tx: &mpsc::UnboundedSender<TuiEvent>| {
                let _ = event_tx.send(TuiEvent::ActionAck { request_id });
            };

            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                acknowledge(event_tx);
                return;
            };
            let normalized = level.to_lowercase();
            let tl = match normalized.as_str() {
                "off" => theta_ai::ThinkingLevel::Off,
                "minimal" => theta_ai::ThinkingLevel::Minimal,
                "low" => theta_ai::ThinkingLevel::Low,
                "medium" => theta_ai::ThinkingLevel::Medium,
                "high" => theta_ai::ThinkingLevel::High,
                "xhigh" => theta_ai::ThinkingLevel::XHigh,
                _ => {
                    let _ = event_tx.send(TuiEvent::Error(format!(
                        "Invalid thinking level: {level}. Use off/minimal/low/medium/high/xhigh"
                    )));
                    acknowledge(event_tx);
                    return;
                }
            };
            agent.set_thinking_level(tl).await;
            // Persist thinking preference (merge with existing settings).
            let mut s = crate::settings::load_settings().await;
            s.last_thinking = Some(normalized.clone());
            crate::settings::save_settings(&s).await.ok();
            let _ = event_tx.send(TuiEvent::ThinkingSet { level: normalized });
            acknowledge(event_tx);
        }
        TuiAction::ShowThinkingSelector => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };
            let state = agent.state().await;
            let levels = compute_valid_thinking_levels(&state.model);
            let current_str = thinking_level_to_string(state.thinking_level);
            let _ = event_tx.send(TuiEvent::ThinkingLevels {
                levels,
                current: current_str,
            });
        }
        TuiAction::ForkSession => {
            if agent_cell.read().await.is_none() {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            }
            let Some(ref sid) = *session_id_cell.read().await else {
                let _ = event_tx.send(TuiEvent::Error("No active session".into()));
                return;
            };
            let session_mgr = SessionManager::new(working_dir);
            match session_mgr.open_by_id(sid).await {
                Ok(session) => match session_mgr.fork(&session, Some(model_id)).await {
                    Ok(forked) => {
                        let new_id = forked
                            .meta
                            .as_ref()
                            .map(|m| m.id.clone())
                            .unwrap_or_default();
                        let _ = event_tx.send(TuiEvent::SessionCreated {
                            id: new_id.clone(),
                            model: model_id.to_string(),
                        });
                        *session_id_cell.write().await = Some(new_id);
                    }
                    Err(e) => {
                        let _ = event_tx.send(TuiEvent::Error(format!("Fork failed: {e}")));
                    }
                },
                Err(e) => {
                    let _ = event_tx.send(TuiEvent::Error(format!("Cannot open session: {e}")));
                }
            }
        }
        TuiAction::ShowSessions => {
            let session_mgr = SessionManager::new(working_dir);
            if let Ok(sessions) = session_mgr.list().await {
                let infos: Vec<SessionInfo> = sessions
                    .into_iter()
                    .map(|m| SessionInfo {
                        id: m.id,
                        title: m.title.unwrap_or_else(|| "(untitled)".into()),
                        model: m.model,
                        branch: m.branch,
                        token_count: m.token_count,
                        created_at: m.created_at,
                        message_count: m.message_count,
                    })
                    .collect();
                let _ = event_tx.send(TuiEvent::SessionPicker(infos));
            }
        }
        TuiAction::ShowTree(filter) => {
            let session_mgr = SessionManager::new(working_dir);
            if let Ok(sessions) = session_mgr.list().await {
                let mut infos: Vec<SessionInfo> = Vec::new();
                for m in sessions {
                    let pass = match filter.as_str() {
                        "default" => m.message_count > 0,
                        "labeled-only" => m.title.as_deref().is_some_and(|t| !t.trim().is_empty()),
                        "all" => true,
                        "no-tools" | "user-only" => {
                            if let Ok(s) = session_mgr.open_by_id(&m.id).await {
                                let has_tool = s
                                    .messages
                                    .iter()
                                    .any(|msg| matches!(msg, theta_ai::Message::ToolResult { .. }));
                                let has_assistant = s
                                    .messages
                                    .iter()
                                    .any(|msg| matches!(msg, theta_ai::Message::Assistant { .. }));
                                if filter == "no-tools" {
                                    !has_tool
                                } else {
                                    !has_assistant && !has_tool
                                }
                            } else {
                                false
                            }
                        }
                        _ => true,
                    };
                    if pass {
                        infos.push(SessionInfo {
                            id: m.id,
                            title: m.title.unwrap_or_else(|| "(untitled)".into()),
                            model: m.model,
                            branch: m.branch,
                            token_count: m.token_count,
                            created_at: m.created_at,
                            message_count: m.message_count,
                        });
                    }
                }
                let _ = event_tx.send(TuiEvent::TreeSessions {
                    sessions: infos,
                    filter,
                });
            }
        }
        TuiAction::ResumeSession(id) => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };
            let session_mgr = SessionManager::new(working_dir);
            match session_mgr.open_by_id(&id).await {
                Ok(session) => {
                    let messages = session.messages.clone();
                    let recap = session_recap(&session);
                    agent.load_messages(messages.clone()).await;
                    let state = agent.state().await;
                    let mid = state
                        .last_model_id()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| model_id.to_string());
                    let current_thinking = thinking_level_to_string(state.thinking_level);
                    drop(state);
                    let blocks = build_system_prompt_with_skills(
                        working_dir,
                        &mid,
                        Some(&current_thinking),
                        &config.startup_skills,
                    )
                    .await;
                    agent.set_system_prompt(blocks).await;
                    *session_id_cell.write().await = Some(id.clone());
                    let _ = event_tx.send(TuiEvent::SessionCreated {
                        id: id.clone(),
                        model: mid.clone(),
                    });

                    // Emit context token stats from the loaded session.
                    let (_, _, last_input) = agent.context_stats().await;
                    if let Some(tokens) = last_input {
                        let state = agent.state().await;
                        let avail = state
                            .model
                            .context_window
                            .saturating_sub(agent.config().compaction.reserve_tokens);
                        let pct = if avail > 0 {
                            (tokens as f64 / avail as f64 * 100.0) as u32
                        } else {
                            0
                        };
                        let _ = event_tx.send(TuiEvent::ContextTokens { tokens, pct });
                    }

                    // Send history to display in chat.
                    let history: Vec<HistoryEntry> = messages
                        .into_iter()
                        .flat_map(|msg| message_to_history_entries(&msg))
                        .collect();
                    let _ = event_tx.send(TuiEvent::Info(recap));
                    if !history.is_empty() {
                        let _ = event_tx.send(TuiEvent::LoadHistory(history));
                    }
                }
                Err(e) => {
                    let _ =
                        event_tx.send(TuiEvent::Error(format!("Failed to resume session: {e}")));
                }
            }
        }
        TuiAction::NewSession => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };

            // Clear in-memory transcript — no session file until first message.
            agent.load_messages(Vec::new()).await;
            // Read current model and thinking level from agent state so
            // runtime context shows the correct values and startup skills are
            // re-applied.
            let state = agent.state().await;
            let current_model_id = state.model.id.clone();
            let current_thinking = thinking_level_to_string(state.thinking_level);
            drop(state);
            let blocks = build_system_prompt_with_skills(
                working_dir,
                &current_model_id,
                Some(&current_thinking),
                &config.startup_skills,
            )
            .await;
            agent.set_system_prompt(blocks).await;

            // Clear the chat display.
            let _ = event_tx.send(TuiEvent::ClearChat);

            *session_id_cell.write().await = None;
            let _ = event_tx.send(TuiEvent::SessionCreated {
                id: "".to_string(),
                model: current_model_id,
            });
            let _ = event_tx.send(TuiEvent::Info(
                "Started new unsaved session (saved on first message).".into(),
            ));
        }
        TuiAction::Steer(text) => {
            let Some(agent) = agent_cell.read().await.clone() else {
                return;
            };
            agent.steer(vec![theta_ai::ContentBlock::Text { text }]);
            let (steer, follow_up) = agent.queue_lengths();
            let _ = event_tx.send(TuiEvent::QueueStatus { steer, follow_up });
        }
        TuiAction::ShowSessionInfo => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };
            let state = agent.state().await;
            let (msg_count, approx_tokens, real_input_tokens) = agent.context_stats().await;
            let _ = event_tx.send(TuiEvent::SessionInfo {
                message_count: msg_count,
                approx_tokens,
                real_input_tokens,
                context_window: state.model.context_window,
                compaction_enabled: agent.config().compaction.enabled,
                reserve_tokens: agent.config().compaction.reserve_tokens,
                keep_recent_tokens: agent.config().compaction.keep_recent_tokens,
                model_id: state.model.id.clone(),
                provider: provider_to_string(state.model.provider),
            });
        }
        TuiAction::CompactContext => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };
            let _ = event_tx.send(TuiEvent::SetAgentState("compacting".into()));
            match agent.compact_context().await {
                Ok(trimmed) => {
                    let _ = event_tx.send(TuiEvent::SetAgentState("idle".into()));
                    // Update the ctx% in the status bar.
                    let (_, approx_tokens, _) = agent.context_stats().await;
                    let state = agent.state().await;
                    let avail = state
                        .model
                        .context_window
                        .saturating_sub(agent.config().compaction.reserve_tokens);
                    let pct = if avail > 0 {
                        (approx_tokens as f64 / avail as f64 * 100.0) as u32
                    } else {
                        0
                    };
                    let _ = event_tx.send(TuiEvent::ContextTokens {
                        tokens: approx_tokens,
                        pct,
                    });
                    if trimmed > 0 {
                        let _ = event_tx.send(TuiEvent::Info(format!(
                            "Compacted {trimmed} old messages from context."
                        )));
                    } else {
                        let _ = event_tx.send(TuiEvent::Info(
                            "No older messages to compact — context already minimal.".into(),
                        ));
                    }
                }
                Err(e) => {
                    let _ = event_tx.send(TuiEvent::SetAgentState("idle".into()));
                    let _ = event_tx.send(TuiEvent::Error(format!("Compaction failed: {e}")));
                }
            }
        }
        TuiAction::ShowRunTimeline => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };
            let Some(report) = agent.last_run_report().await else {
                let _ = event_tx.send(TuiEvent::Info(
                    "No completed run report is available yet.".into(),
                ));
                return;
            };
            let mut lines = vec![
                format!("Run timeline: {}", report.run_id),
                format!("Started: {}", report.started_at_ms),
                format!(
                    "Finished: {}",
                    report
                        .finished_at_ms
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "n/a".to_string())
                ),
                format!(
                    "Outcome: {}",
                    report
                        .outcome
                        .map(|r| format!("{r:?}"))
                        .unwrap_or_else(|| "n/a".to_string())
                ),
                "Events:".to_string(),
            ];
            for ev in report.events.iter().take(50) {
                let mut fields = ev
                    .fields
                    .iter()
                    .filter(|(k, _)| !matches!(k.as_str(), "run_id" | "model" | "provider"))
                    .take(3)
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>();
                fields.sort();
                let suffix = if fields.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", fields.join(", "))
                };
                lines.push(format!("  - {} {}{}", ev.ts_ms, ev.kind, suffix));
            }
            let _ = event_tx.send(TuiEvent::Info(lines.join("\n")));
        }
        TuiAction::FollowUp(text) => {
            let Some(agent) = agent_cell.read().await.clone() else {
                return;
            };
            agent.follow_up(vec![theta_ai::ContentBlock::Text { text }]);
            let (steer, follow_up) = agent.queue_lengths();
            let _ = event_tx.send(TuiEvent::QueueStatus { steer, follow_up });
        }
        TuiAction::AbortAgent => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Info("No agent to cancel.".into()));
                return;
            };
            match agent.abort() {
                Ok(()) => {
                    tracing::info!("agent cancelled by user");
                }
                Err(theta_agent_core::error::AgentError::NotRunning) => {
                    let _ = event_tx.send(TuiEvent::Error(
                        "No active agent execution to cancel.".into(),
                    ));
                }
                Err(e) => {
                    tracing::warn!("failed to cancel agent: {e}");
                    let _ = event_tx.send(TuiEvent::Error(format!("Failed to cancel: {e}")));
                }
            }
        }
    }
}

async fn available_model_entries(
    models: &[Model],
    auth: &mut crate::config::AuthConfig,
) -> Vec<ModelEntry> {
    let mut provider_has_auth: HashMap<String, bool> = HashMap::new();
    for provider in ["openai", "openai-codex", "deepseek", "opencode"] {
        provider_has_auth.insert(
            provider.to_string(),
            auth.get_api_key(provider).await.is_some(),
        );
    }

    let mut entries: Vec<ModelEntry> = models
        .iter()
        .filter(|m| {
            provider_has_auth
                .get(&provider_to_string(m.provider))
                .copied()
                .unwrap_or(false)
        })
        .map(|m| ModelEntry {
            id: m.id.clone(),
            name: m.name.clone(),
            provider: provider_to_string(m.provider),
            context_window: m.context_window,
        })
        .collect();

    entries.sort_by(|a, b| {
        a.provider
            .cmp(&b.provider)
            .then_with(|| a.id.cmp(&b.id))
            .then_with(|| a.name.cmp(&b.name))
    });
    entries
}

fn find_model_by_id(catalog: &[theta_ai::Model], id: &str) -> Option<theta_ai::Model> {
    catalog.iter().find(|m| m.id == id).cloned().or_else(|| {
        if id == "opencode" {
            catalog
                .iter()
                .find(|m| m.provider == Provider::OpenCode)
                .cloned()
        } else {
            None
        }
    })
}

fn find_model_by_provider_and_id(
    catalog: &[theta_ai::Model],
    provider: &str,
    id: &str,
) -> Option<theta_ai::Model> {
    catalog
        .iter()
        .find(|m| provider_to_string(m.provider) == provider && m.id == id)
        .cloned()
}

async fn refresh_runtime_models(
    catalog: &BuiltInCatalog,
    runtime_models_cell: &Arc<RwLock<Vec<Model>>>,
) {
    let refreshed = resolve_runtime_models(catalog).await;
    *runtime_models_cell.write().await = refreshed;
}

async fn resolve_runtime_models(catalog: &BuiltInCatalog) -> Vec<Model> {
    let mut models: Vec<Model> = catalog.list().into_iter().cloned().collect();
    let fetched = opencode::fetch_models().await;
    if !fetched.is_empty() {
        models.retain(|m| m.provider != Provider::OpenCode);
        models.extend(fetched);
    }
    models
}

/// Build an ExtensionStatusPayload from Rhai row callbacks + legacy status lines.
/// Legacy status lines are mapped to row[0].left.
fn to_extension_payload(
    rows: Vec<theta_agent_core::types::ExtensionStatusRow>,
    lines: Vec<(String, String)>,
) -> theta_tui::app::ExtensionStatusPayload {
    // Count rows from tui.row() callbacks that have actual content.
    let extension_row_count = rows.iter().filter(|r| !r.is_empty()).count();

    let mut all_rows: Vec<theta_tui::components::status::StatusRow> = rows
        .into_iter()
        .map(|r| theta_tui::components::status::StatusRow {
            left: r.left,
            center: r.center,
            right: r.right,
        })
        .collect();

    // Merge legacy status lines into row[0].left
    if !lines.is_empty() {
        if all_rows.is_empty() {
            all_rows.push(theta_tui::components::status::StatusRow::default());
        }
        let mut merged = all_rows[0].left.clone();
        for (key, text) in &lines {
            if text.starts_with('[') && text.contains(':') {
                merged.push(text.clone());
            } else {
                merged.push(format!("[{key}:{text}]"));
            }
        }
        all_rows[0].left = merged;
    }

    theta_tui::app::ExtensionStatusPayload {
        rows: all_rows,
        extension_row_count,
    }
}

fn provider_to_string(provider: Provider) -> String {
    match provider {
        Provider::OpenAI => "openai".into(),
        Provider::OpenAiCodex => "openai-codex".into(),
        Provider::DeepSeek => "deepseek".into(),
        Provider::OpenCode => "opencode".into(),
        Provider::OpenCodeGo => "opencode-go".into(),
    }
}

fn provider_from_string(provider: &str) -> Option<Provider> {
    match provider {
        "openai" => Some(Provider::OpenAI),
        "openai-codex" => Some(Provider::OpenAiCodex),
        "deepseek" => Some(Provider::DeepSeek),
        "opencode" => Some(Provider::OpenCode),
        "opencode-go" => Some(Provider::OpenCodeGo),
        _ => None,
    }
}

fn format_tool_summary(result: &theta_agent_core::types::ToolResult, max_chars: usize) -> String {
    let details = result.details.as_ref();
    let summary = match result.tool_name.as_str() {
        "read" => {
            if let Some(d) = details {
                let path = d
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(unknown)");
                let total_lines = d.get("total_lines").and_then(|v| v.as_u64()).unwrap_or(0);
                let offset = d.get("offset").and_then(|v| v.as_u64()).unwrap_or(1);
                let lines_read = d.get("lines_read").and_then(|v| v.as_u64()).unwrap_or(0);
                format!(
                    "read {path}\nlines {offset}-{end} of {total_lines}",
                    end = offset.saturating_add(lines_read.saturating_sub(1))
                )
            } else {
                "read done".to_string()
            }
        }
        "edit" => {
            if let Some(d) = details {
                let path = d
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(unknown)");
                let changes = d.get("changes").and_then(|v| v.as_u64()).unwrap_or(0);
                let diff = d.get("diff").and_then(|v| v.as_str()).unwrap_or("");
                if diff.is_empty() {
                    format!("edit {path}\n{changes} change(s)")
                } else {
                    format!("edit {path}\n{changes} change(s)\n{diff}")
                }
            } else {
                "edit done".to_string()
            }
        }
        "bash" => {
            if let Some(d) = details {
                let exit = d
                    .get("exit_code")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "null".to_string());
                let timed_out = d
                    .get("timed_out")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                if timed_out {
                    "bash timeout".to_string()
                } else {
                    format!("bash done (exit={exit})")
                }
            } else {
                "bash done".to_string()
            }
        }
        "grep" => {
            if let Some(d) = details {
                let pattern = d
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(unknown)");
                let count = d.get("match_count").and_then(|v| v.as_u64()).unwrap_or(0);
                format!("grep /{pattern}/\n{count} match(es)")
            } else {
                "grep done".to_string()
            }
        }
        "ls" => {
            if let Some(d) = details {
                let path = d
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(unknown)");
                let count = d.get("entry_count").and_then(|v| v.as_u64()).unwrap_or(0);
                format!(
                    "ls {path}\n{count} entr{suffix}",
                    suffix = if count == 1 { "y" } else { "ies" }
                )
            } else {
                "ls done".to_string()
            }
        }
        "find" => {
            if let Some(d) = details {
                let pattern = d
                    .get("pattern")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(unknown)");
                let count = d.get("match_count").and_then(|v| v.as_u64()).unwrap_or(0);
                format!("find {pattern}\n{count} match(es)")
            } else {
                "find done".to_string()
            }
        }
        "write" => {
            if let Some(d) = details {
                let path = d
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(unknown)");
                let bytes = d.get("bytes_written").and_then(|v| v.as_u64()).unwrap_or(0);
                format!("write {path}\n{bytes} bytes")
            } else {
                "write done".to_string()
            }
        }
        _ => content_blocks_to_text(&result.content, max_chars),
    };
    truncate_chars(&summary, max_chars)
}

fn content_blocks_to_text(content: &[theta_ai::ContentBlock], max_chars: usize) -> String {
    let text = content
        .iter()
        .filter_map(|block| match block {
            theta_ai::ContentBlock::Text { text } => Some(text.as_str()),
            theta_ai::ContentBlock::Image { .. } => Some("[image]"),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    truncate_chars(&text, max_chars)
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn format_error_chain(error: &dyn std::error::Error) -> String {
    let mut out = error.to_string();
    let mut source = error.source();
    while let Some(err) = source {
        out.push_str("\ncaused by: ");
        out.push_str(&err.to_string());
        source = err.source();
    }
    out
}

fn expand_skill_message(message: &str, skills: &[crate::skills::Skill]) -> String {
    let Some(skill_cmd) = message.strip_prefix("/skill:") else {
        return message.to_string();
    };

    let space_index = skill_cmd.find(' ');
    let skill_name = match space_index {
        Some(idx) => &skill_cmd[..idx],
        None => skill_cmd,
    };
    let args = match space_index {
        Some(idx) => skill_cmd[idx + 1..].trim(),
        None => "",
    };

    let Some(skill) = skills.iter().find(|s| s.name == skill_name) else {
        return message.to_string();
    };

    let base_dir = skill.location.parent().unwrap_or(skill.location.as_path());
    let skill_block = format!(
        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
        skill.name,
        skill.location.display(),
        base_dir.display(),
        skill.body.trim()
    );

    if args.is_empty() {
        format!(
            "{skill_block}\n\nExecute this skill now for the current request. Do not only acknowledge loading the skill."
        )
    } else {
        format!("{skill_block}\n\n{args}")
    }
}

fn session_recap(session: &crate::session::Session) -> String {
    let meta = session.meta.as_ref();
    let title = meta
        .and_then(|m| m.title.as_deref())
        .unwrap_or("(untitled)");
    let model = meta.and_then(|m| m.model.as_deref()).unwrap_or("unknown");
    let branch = meta.and_then(|m| m.branch.as_deref()).unwrap_or("-");
    let messages = session.messages.len();
    let tokens = meta.map(|m| m.token_count).unwrap_or_else(|| {
        session
            .messages
            .iter()
            .map(theta_ai::Message::token_count)
            .sum()
    });
    let last_user = session
        .messages
        .iter()
        .rev()
        .find_map(|msg| match msg {
            theta_ai::Message::User { content, .. } => Some(content_blocks_to_text(content, 160)),
            _ => None,
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "(none)".into());

    format!(
        "Resumed: {title}\nModel: {model}\nBranch: {branch}\nMessages: {messages}, approx tokens: {tokens}\nLast user message: {last_user}"
    )
}

/// Convert a session message to a history entry for display.
fn message_to_history_entries(msg: &theta_ai::Message) -> Vec<HistoryEntry> {
    match msg {
        theta_ai::Message::User { content, .. } => {
            let text = content
                .iter()
                .filter_map(|b| match b {
                    theta_ai::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            vec![HistoryEntry {
                role: "user".into(),
                text,
            }]
        }
        theta_ai::Message::Assistant { content, .. } => {
            let text = content
                .iter()
                .filter_map(|b| match b {
                    theta_ai::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            let thinking = content
                .iter()
                .filter_map(|b| match b {
                    theta_ai::ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            let mut out = Vec::new();
            if !text.is_empty() {
                out.push(HistoryEntry {
                    role: "assistant".into(),
                    text,
                });
            }
            if !thinking.is_empty() {
                out.push(HistoryEntry {
                    role: "thinking".into(),
                    text: thinking,
                });
            }
            out
        }
        theta_ai::Message::ToolResult { tool_name, .. } => vec![HistoryEntry {
            role: "tool".into(),
            text: format!("[{tool_name}] done"),
        }],
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;

    use super::{expand_skill_message, format_tool_summary};
    use theta_agent_core::types::ToolResult;

    #[test]
    fn read_summary_is_compact() {
        let result = ToolResult {
            tool_call_id: "id".into(),
            tool_name: "read".into(),
            content: vec![],
            details: Some(serde_json::json!({
                "path": "/tmp/a.rs",
                "total_lines": 100,
                "offset": 11,
                "lines_read": 20
            })),
            is_error: false,
        };
        let s = format_tool_summary(&result, 200);
        assert!(s.contains("read /tmp/a.rs"));
        assert!(s.contains("lines 11-30 of 100"));
        assert!(!s.contains("fn "));
    }

    #[test]
    fn edit_summary_includes_diff() {
        let result = ToolResult {
            tool_call_id: "id".into(),
            tool_name: "edit".into(),
            content: vec![],
            details: Some(serde_json::json!({
                "path": "/tmp/a.rs",
                "changes": 1,
                "diff": "@@ -1 +1 @@\n-a\n+b"
            })),
            is_error: false,
        };
        let s = format_tool_summary(&result, 200);
        assert!(s.contains("edit /tmp/a.rs"));
        assert!(s.contains("1 change(s)"));
        assert!(s.contains("@@ -1 +1 @@"));
    }

    #[test]
    fn skill_command_without_args_executes_now() {
        let skill = crate::skills::Skill {
            name: "git-commit".into(),
            description: "Commit workflow".into(),
            location: PathBuf::from("/tmp/skills/git-commit/SKILL.md"),
            body: "Do commit workflow".into(),
            extra: HashMap::new(),
        };
        let s = expand_skill_message("/skill:git-commit", &[skill]);
        assert!(s.contains("<skill name=\"git-commit\""));
        assert!(s.contains("Execute this skill now"));
        assert!(s.contains("Do not only acknowledge loading the skill"));
    }

    #[test]
    fn skill_command_with_args_preserves_args_only() {
        let skill = crate::skills::Skill {
            name: "git-commit".into(),
            description: "Commit workflow".into(),
            location: PathBuf::from("/tmp/skills/git-commit/SKILL.md"),
            body: "Do commit workflow".into(),
            extra: HashMap::new(),
        };
        let s = expand_skill_message("/skill:git-commit commit all staged", &[skill]);
        assert!(s.contains("commit all staged"));
        assert!(!s.contains("Do not only acknowledge loading the skill"));
    }
}
