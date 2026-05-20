//! Interactive TUI mode — connects the agent to the terminal UI.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use theta_agent_core::agent::Agent;
use theta_agent_core::events::AgentEvent;
use theta_ai::providers::default_registry;
use theta_ai::{Model, ModelCatalog, Provider};
use theta_models::BuiltInCatalog;
use theta_tui::App;
use theta_tui::app::{HistoryEntry, TuiAction, TuiEvent};
use theta_tui::components::CommandEntry;
use theta_tui::components::{ModelEntry, SessionInfo, known_providers};
use theta_tui::theme::Theme;
use tokio::sync::{RwLock, mpsc};

use crate::config::ThetaConfig;
use crate::session::SessionManager;
use crate::system_prompt::build_system_prompt;
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
    // Resolve model.
    let catalog = BuiltInCatalog::new();

    let model = find_model_by_id(&catalog, model_id)
        .ok_or_else(|| anyhow::anyhow!("model not found: {model_id}"))?
        .clone();

    // Resolve auth. If the default model's provider has no token,
    // try other providers that DO have auth (e.g., user logged in
    // via Codex but default model is from OpenAI provider).
    let provider_str = provider_to_string(model.provider);
    let mut auth_config = config.auth.clone();
    let model_entries = available_model_entries(&catalog, &mut auth_config).await;
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
                && let Some(m) = catalog.list().into_iter().find(|m| {
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

    // ------------------------------------------------------------------
    // Always spawn the action handler first (handles login + agent init).
    // ------------------------------------------------------------------
    let action_agent_cell = agent_cell.clone();
    let action_event_tx = event_tx.clone();
    let action_session_id_cell = session_id_cell.clone();
    let action_working_dir = working_dir.to_path_buf();
    let action_model_id = model_id.clone();
    let action_model = model.clone();
    let action_thinking = thinking.to_string();
    let action_catalog = BuiltInCatalog::new();
    let action_config = config.clone();
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
                &action_config,
            )
            .await;
        }
    });

    // ------------------------------------------------------------------
    // If we have auth, create the agent now and spawn event bridge.
    // ------------------------------------------------------------------
    if let Some(ref key) = api_key {
        let agent = create_agent(&model, key, config, working_dir, &model_id, thinking).await?;
        let agent = Arc::new(agent);
        *agent_cell.write().await = Some(agent.clone());
        spawn_event_bridge(agent, event_tx.clone());

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
    let msg_event_tx = event_tx.clone();
    let msg_working_dir = working_dir.to_path_buf();
    let msg_session_id_cell = session_id_cell.clone();
    let msg_model_id = model_id.to_string();
    let msg_skills = skills.clone();
    tokio::spawn(async move {
        // Wait for agent to be available (block until login completes).
        let agent = wait_for_agent(&msg_agent_cell).await;
        let session_mgr = SessionManager::new(&msg_working_dir);
        while let Some(message) = message_rx.recv().await {
            // Reload agent in case it was replaced (model switch, etc.).
            let agent = msg_agent_cell.read().await.clone().unwrap_or(agent.clone());

            let expanded_message = if let Some(skill_cmd) = message.strip_prefix("/skill:") {
                let space_index = skill_cmd.find(' ');
                let skill_name = match space_index {
                    Some(idx) => &skill_cmd[..idx],
                    None => skill_cmd,
                };
                let args = match space_index {
                    Some(idx) => skill_cmd[idx + 1..].trim(),
                    None => "",
                };

                if let Some(skill) = msg_skills.iter().find(|s| s.name == skill_name) {
                    let base_dir = skill.location.parent().unwrap_or(skill.location.as_path());
                    let skill_block = format!(
                        "<skill name=\"{}\" location=\"{}\">\nReferences are relative to {}.\n\n{}\n</skill>",
                        skill.name,
                        skill.location.display(),
                        base_dir.display(),
                        skill.body.trim()
                    );
                    if args.is_empty() {
                        skill_block
                    } else {
                        format!("{skill_block}\n\n{args}")
                    }
                } else {
                    message.clone()
                }
            } else {
                message.clone()
            };

            let blocks =
                crate::mentions::expand_file_mentions(&msg_working_dir, &expanded_message).await;
            if let Err(e) = agent.prompt(blocks).await {
                tracing::error!("agent prompt failed: {e}");
                let _ = msg_event_tx.send(TuiEvent::Error(format!("{e}")));
                continue;
            }

            // Lazy session creation on first real message — no session
            // file is left behind for login-only or no-message runs.
            if msg_session_id_cell.read().await.is_none() {
                match session_mgr.create(Some(&msg_model_id)).await {
                    Ok(session) => {
                        let id = session
                            .meta
                            .as_ref()
                            .map(|m| m.id.clone())
                            .unwrap_or_default();
                        let _ = msg_event_tx.send(TuiEvent::SessionCreated {
                            id: id.clone(),
                            model: msg_model_id.clone(),
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

            // Persist any state messages not yet in session storage.
            let Some(ref sid) = *msg_session_id_cell.read().await else {
                continue;
            };
            let state = agent.state().await;
            match session_mgr.open_by_id(sid).await {
                Ok(mut session) => {
                    if let Err(e) = session_mgr
                        .append_missing_entries(&mut session, &state.messages)
                        .await
                    {
                        tracing::error!("failed to persist session entries: {e}");
                        let _ = msg_event_tx
                            .send(TuiEvent::Error(format!("Failed to persist session: {e}")));
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
            description: "Set thinking level (off/low/medium/high)".into(),
        },
        CommandEntry {
            name: "clear".into(),
            description: "Clear the chat display".into(),
        },
        CommandEntry {
            name: "session".into(),
            description: "Show current session info".into(),
        },
        CommandEntry {
            name: "fork".into(),
            description: "Fork the current session".into(),
        },
        CommandEntry {
            name: "sessions".into(),
            description: "List recent sessions to resume".into(),
        },
        CommandEntry {
            name: "tree".into(),
            description: "Open session tree selector".into(),
        },
        CommandEntry {
            name: "settings".into(),
            description: "Open settings selector".into(),
        },
        CommandEntry {
            name: "login".into(),
            description: "Log in to a provider".into(),
        },
        CommandEntry {
            name: "skills".into(),
            description: "List available skills".into(),
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
        },
        model_entries,
        commands,
        working_dir.to_path_buf(),
        event_rx,
        message_tx,
        action_tx,
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
) -> anyhow::Result<Agent> {
    let catalog = BuiltInCatalog::new();
    let mut registry = default_registry();
    registry.set_api_key(model.provider, api_key);

    let tool_ctx = ToolContext::new(working_dir.to_path_buf());
    let mut agent = Agent::new(model.clone(), Arc::new(registry), Arc::new(catalog));
    agent.set_config(crate::config::to_agent_config(config));
    for tool in builtin_tools(tool_ctx) {
        agent.add_tool(tool).await;
    }

    let system_blocks = build_system_prompt(working_dir, model_id, Some(thinking)).await;
    agent.set_system_prompt(system_blocks).await;

    Ok(agent)
}

/// Spawn the event bridge — subscribes to agent events, forwards to TUI.
fn spawn_event_bridge(agent: Arc<Agent>, event_tx: mpsc::UnboundedSender<TuiEvent>) {
    tokio::spawn(async move {
        let mut events = agent.subscribe();
        let mut tool_names: HashMap<String, String> = HashMap::new();
        let mut saw_assistant_text_delta = false;
        loop {
            match events.recv().await {
                Ok(AgentEvent::MessageStart) => {
                    saw_assistant_text_delta = false;
                }
                Ok(AgentEvent::TextDelta { text }) => {
                    saw_assistant_text_delta = true;
                    let _ = event_tx.send(TuiEvent::TextDelta(text));
                }
                Ok(AgentEvent::ThinkingDelta { thinking }) => {
                    let _ = event_tx.send(TuiEvent::ThinkingDelta(thinking));
                }
                Ok(AgentEvent::ToolCallStart { .. }) => {}
                Ok(AgentEvent::ToolExecutionStart {
                    tool_call_id: id,
                    tool_name: name,
                }) => {
                    tool_names.insert(id.clone(), name.clone());
                    let _ = event_tx.send(TuiEvent::ToolStart { name, id });
                }
                Ok(AgentEvent::ToolExecutionProgress {
                    tool_call_id: id,
                    output,
                }) => {
                    let _ = event_tx.send(TuiEvent::ToolProgress {
                        name: tool_names.get(&id).cloned().unwrap_or(id),
                        message: output,
                    });
                }
                Ok(AgentEvent::ToolExecutionEnd { result }) => {
                    let summary = format_tool_summary(&result, 2200);
                    tool_names.remove(&result.tool_call_id);
                    let _ = event_tx.send(TuiEvent::ToolEnd {
                        id: result.tool_call_id,
                        name: result.tool_name,
                        is_error: result.is_error,
                        summary,
                    });
                }
                Ok(AgentEvent::MessageEnd { message }) => {
                    if !saw_assistant_text_delta
                        && let theta_ai::Message::Assistant { content, .. } = message
                    {
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
                    saw_assistant_text_delta = false;
                }
                Ok(AgentEvent::TurnStart { .. }) => {
                    let _ = event_tx.send(TuiEvent::TurnStart);
                }
                Ok(AgentEvent::TurnEnd { .. }) => {
                    let _ = event_tx.send(TuiEvent::TurnEnd {
                        stop_reason: "stop".into(),
                    });
                }
                Ok(AgentEvent::AgentEnd { .. }) => {
                    let _ = event_tx.send(TuiEvent::AgentEnd);
                }
                Ok(AgentEvent::ContextCompacted { trimmed_count, .. }) => {
                    let _ = event_tx.send(TuiEvent::ContextCompacted { trimmed_count });
                }
                Ok(AgentEvent::Retrying { attempt, delay_ms }) => {
                    let _ = event_tx.send(TuiEvent::Retrying { attempt, delay_ms });
                }
                Ok(AgentEvent::Error { message }) => {
                    let _ = event_tx.send(TuiEvent::Error(message));
                }
                Err(tokio::sync::broadcast::error::RecvError::Lagged(n)) => {
                    let _ = event_tx.send(TuiEvent::Error(format!("lagged by {n} events")));
                }
                Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                _ => {}
            }
        }
    });
}

/// Block until an agent is available in the cell.
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
    config: &ThetaConfig,
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
                                )
                                .await
                                {
                                    Ok(agent) => {
                                        let agent = Arc::new(agent);
                                        spawn_event_bridge(agent.clone(), event_tx.clone());
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

                            let refreshed_models =
                                available_model_entries(catalog, &mut auth).await;
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
                    // If this was the initial login (no agent yet), create the agent now.
                    if agent_cell.read().await.is_none() {
                        match create_agent(model, &token, config, working_dir, model_id, thinking)
                            .await
                        {
                            Ok(agent) => {
                                let agent = Arc::new(agent);
                                spawn_event_bridge(agent.clone(), event_tx.clone());
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

                    let refreshed_models = available_model_entries(catalog, &mut auth).await;
                    let _ = event_tx.send(TuiEvent::UpdateModels(refreshed_models));
                }
                Err(e) => {
                    let _ = event_tx.send(TuiEvent::Error(format!("Failed to load auth: {e}")));
                }
            }
        }
        TuiAction::SwitchModel { model_id, provider } => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };
            let model = provider
                .as_deref()
                .and_then(|p| find_model_by_provider_and_id(catalog, p, &model_id))
                .or_else(|| find_model_by_id(catalog, &model_id));

            if let Some(m) = model {
                let provider = provider_to_string(m.provider);
                match crate::config::load_auth(None).await {
                    Ok(mut auth) => {
                        if auth.get_api_key(&provider).await.is_none() {
                            let _ = event_tx.send(TuiEvent::Error(format!(
                                "Model {model_id} is unavailable: missing auth for {provider}"
                            )));
                            return;
                        }
                    }
                    Err(e) => {
                        let _ = event_tx.send(TuiEvent::Error(format!("Failed to load auth: {e}")));
                        return;
                    }
                }

                agent.set_model(m).await;
                let blocks = build_system_prompt(working_dir, &model_id, None).await;
                agent.set_system_prompt(blocks).await;
                let _ = event_tx.send(TuiEvent::Info(format!(
                    "Switched to {model_id} ({provider})"
                )));
                // Persist model preference (merge with existing settings).
                let mut s = crate::settings::load_settings().await;
                s.last_model = Some(model_id.to_string());
                crate::settings::save_settings(&s).await.ok();
            } else {
                let _ = event_tx.send(TuiEvent::Error(format!("Model not found: {model_id}")));
            }
        }
        TuiAction::SetThinking(level) => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };
            let tl = match level.to_lowercase().as_str() {
                "off" => theta_ai::ThinkingLevel::Off,
                "low" => theta_ai::ThinkingLevel::Low,
                "medium" => theta_ai::ThinkingLevel::Medium,
                "high" => theta_ai::ThinkingLevel::High,
                _ => {
                    let _ = event_tx.send(TuiEvent::Error(format!(
                        "Invalid thinking level: {level}. Use off/low/medium/high"
                    )));
                    return;
                }
            };
            agent.set_thinking_level(tl).await;
            // Persist thinking preference (merge with existing settings).
            let mut s = crate::settings::load_settings().await;
            s.last_thinking = Some(level.to_string());
            crate::settings::save_settings(&s).await.ok();
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
                    let mid = agent
                        .state()
                        .await
                        .last_model_id()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| model_id.to_string());
                    let blocks = build_system_prompt(working_dir, &mid, None).await;
                    agent.set_system_prompt(blocks).await;
                    *session_id_cell.write().await = Some(id.clone());
                    let _ = event_tx.send(TuiEvent::SessionCreated {
                        id: id.clone(),
                        model: mid.clone(),
                    });

                    // Send history to display in chat.
                    let history: Vec<HistoryEntry> = messages
                        .into_iter()
                        .filter_map(|msg| message_to_history(&msg))
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

            // Lazy session behavior: clear in-memory transcript, but do not
            // create a session file until first real message.
            agent.load_messages(Vec::new()).await;
            let blocks = build_system_prompt(working_dir, model_id, None).await;
            agent.set_system_prompt(blocks).await;

            *session_id_cell.write().await = None;
            let _ = event_tx.send(TuiEvent::SessionCreated {
                id: "".to_string(),
                model: model_id.to_string(),
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
        TuiAction::FollowUp(text) => {
            let Some(agent) = agent_cell.read().await.clone() else {
                return;
            };
            agent.follow_up(vec![theta_ai::ContentBlock::Text { text }]);
            let (steer, follow_up) = agent.queue_lengths();
            let _ = event_tx.send(TuiEvent::QueueStatus { steer, follow_up });
        }
        TuiAction::SaveSettings(payload) => {
            let mut s = crate::settings::load_settings().await;
            s.steering_mode = payload.steering_mode;
            s.follow_up_mode = payload.follow_up_mode;
            s.transport_preference = payload.transport_preference;
            s.show_thinking = payload.show_thinking;
            let _ = crate::settings::save_settings(&s).await;
            let _ = event_tx.send(TuiEvent::Info("Settings saved".into()));
        }
        TuiAction::ShowSettings => {}
    }
}

async fn available_model_entries(
    catalog: &BuiltInCatalog,
    auth: &mut crate::config::AuthConfig,
) -> Vec<ModelEntry> {
    let mut provider_has_auth: HashMap<String, bool> = HashMap::new();
    for provider in ["openai", "openai-codex", "deepseek", "opencode"] {
        provider_has_auth.insert(
            provider.to_string(),
            auth.get_api_key(provider).await.is_some(),
        );
    }

    let mut entries: Vec<ModelEntry> = catalog
        .list()
        .into_iter()
        .map(|m| {
            let provider = provider_to_string(m.provider);
            let auth_suffix = if provider_has_auth.get(&provider).copied().unwrap_or(false) {
                ""
            } else {
                " [auth required]"
            };
            ModelEntry {
                id: m.id.clone(),
                name: format!("{}{auth_suffix}", m.name),
                provider,
                context_window: m.context_window,
            }
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

fn find_model_by_id(catalog: &BuiltInCatalog, id: &str) -> Option<theta_ai::Model> {
    catalog.list().into_iter().find(|m| m.id == id).cloned()
}

fn find_model_by_provider_and_id(
    catalog: &BuiltInCatalog,
    provider: &str,
    id: &str,
) -> Option<theta_ai::Model> {
    catalog
        .list()
        .into_iter()
        .find(|m| provider_to_string(m.provider) == provider && m.id == id)
        .cloned()
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
fn message_to_history(msg: &theta_ai::Message) -> Option<HistoryEntry> {
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
            Some(HistoryEntry {
                role: "user".into(),
                text,
            })
        }
        theta_ai::Message::Assistant { content, .. } => {
            let text = content
                .iter()
                .filter_map(|b| match b {
                    theta_ai::ContentBlock::Text { text } => Some(text.as_str()),
                    theta_ai::ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
                    theta_ai::ContentBlock::ToolCall { name, .. } => Some(name.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() {
                None
            } else {
                Some(HistoryEntry {
                    role: "assistant".into(),
                    text,
                })
            }
        }
        theta_ai::Message::ToolResult { tool_name, .. } => Some(HistoryEntry {
            role: "tool".into(),
            text: format!("[{tool_name}] done"),
        }),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::format_tool_summary;
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
}
