//! Interactive TUI mode — connects the agent to the terminal UI.

use std::path::Path;
use std::sync::Arc;

use theta_agent_core::agent::Agent;
use theta_agent_core::events::AgentEvent;
use theta_ai::providers::default_registry;
use theta_ai::{ContentBlock, Model, ModelCatalog, Provider};
use theta_models::BuiltInCatalog;
use theta_tui::App;
use theta_tui::app::{HistoryEntry, TuiAction, TuiEvent};
use theta_tui::components::command_picker::CommandEntry;
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

    // Build model entries for the model selector.
    let model_entries: Vec<ModelEntry> = catalog
        .list()
        .into_iter()
        .map(|m| ModelEntry {
            id: m.id.clone(),
            name: m.name.clone(),
            provider: format!("{:?}", m.provider).to_lowercase(),
            context_window: m.context_window,
        })
        .collect();

    let model = find_model_by_id(&catalog, model_id)
        .ok_or_else(|| anyhow::anyhow!("model not found: {model_id}"))?
        .clone();

    // Resolve auth. If the default model's provider has no token,
    // try other providers that DO have auth (e.g., user logged in
    // via Codex but default model is from OpenAI provider).
    let provider_str = provider_to_string(model.provider);
    let mut auth_config = config.auth.clone();
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
                    m.provider == *prov
                        && (m.id == model_id
                            || m.id.starts_with(model_id))
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
    let (message_tx, mut message_rx) = mpsc::unbounded_channel();
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

    // ------------------------------------------------------------------
    // Spawn message handler — waits for agent, creates session lazily.
    // ------------------------------------------------------------------
    let msg_agent_cell = agent_cell.clone();
    let msg_event_tx = event_tx.clone();
    let msg_working_dir = working_dir.to_path_buf();
    let msg_session_id_cell = session_id_cell.clone();
    let msg_model_id = model_id.to_string();
    tokio::spawn(async move {
        let mut saved_count: usize = 0;
        // Wait for agent to be available (block until login completes).
        let agent = wait_for_agent(&msg_agent_cell).await;
        let session_mgr = SessionManager::new(&msg_working_dir);
        while let Some(message) = message_rx.recv().await {
            // Reload agent in case it was replaced (model switch, etc.).
            let agent = msg_agent_cell.read().await.clone().unwrap_or(agent.clone());
            let blocks = vec![ContentBlock::text(&message)];
            if let Err(e) = agent.prompt(blocks).await {
                tracing::error!("agent prompt failed: {e}");
                let _ = msg_event_tx.send(TuiEvent::Error(format!("{e}")));
                break;
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

            // Save only new messages since last save.
            let Some(ref sid) = *msg_session_id_cell.read().await else {
                continue;
            };
            let state = agent.state().await;
            if let Ok(mut session) = session_mgr.open_by_id(sid).await {
                for msg in &state.messages[saved_count..] {
                    session_mgr.append_entry(&mut session, msg).await.ok();
                }
                saved_count = state.messages.len();
            }
        }
    });

    // ------------------------------------------------------------------
    // Show session picker only if prior sessions exist (no initial prompt).
    // ------------------------------------------------------------------
    let session_mgr = SessionManager::new(working_dir);
    if initial_prompt.is_none()
        && let Ok(sessions) = session_mgr.list().await
        && !sessions.is_empty()
    {
        let infos: Vec<SessionInfo> = sessions
            .into_iter()
            .map(|m| SessionInfo {
                id: m.id,
                title: m.title.unwrap_or_else(|| "(untitled)".into()),
                model: m.model,
                created_at: m.created_at,
                message_count: m.message_count,
            })
            .collect();
        let _ = event_tx.send(TuiEvent::SessionPicker(infos));
    }

    // ------------------------------------------------------------------
    // Build available commands + skills for the / picker.
    let skills = crate::skills::discover_skills(working_dir).await;
    let mut commands = vec![
        CommandEntry {
            name: "help".into(),
            description: "Show available commands".into(),
        },
        CommandEntry {
            name: "model".into(),
            description: "Switch model (e.g., /model gpt-5.5)".into(),
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
            name: "login".into(),
            description: "Log in to a provider".into(),
        },
    ];

    // Skills.
    for skill in &skills {
        commands.push(CommandEntry {
            name: skill.name.clone(),
            description: skill.description.clone(),
        });
    }

    // Build and run the TUI.
    // ------------------------------------------------------------------
    let msg_tx_for_prompt = message_tx.clone();

    let mut app = App::new(
        theme.clone(),
        &model.id,
        "", // session created lazily on first message
        thinking,
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

    // Send initial prompt if provided.
    if let Some(prompt) = initial_prompt {
        let _ = msg_tx_for_prompt.send(prompt.to_string());
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
        loop {
            match events.recv().await {
                Ok(AgentEvent::TextDelta { text }) => {
                    let _ = event_tx.send(TuiEvent::TextDelta(text));
                }
                Ok(AgentEvent::ThinkingDelta { thinking }) => {
                    let _ = event_tx.send(TuiEvent::ThinkingDelta(thinking));
                }
                Ok(AgentEvent::ToolCallStart { id, name }) => {
                    let _ = event_tx.send(TuiEvent::ToolStart { name, id });
                }
                Ok(AgentEvent::ToolExecutionProgress {
                    tool_call_id: _,
                    output: _,
                }) => {}
                Ok(AgentEvent::ToolExecutionEnd { result }) => {
                    let output = format_tool_result(&result);
                    let _ = event_tx.send(TuiEvent::ToolEnd {
                        name: result.tool_name,
                        output,
                    });
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
            let _ =
                event_tx.send(TuiEvent::Info("Sign in to ChatGPT in your browser...".into()));
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
                                        cm.id == model_id
                                            && cm.provider == Provider::OpenAiCodex
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
                }
                Err(e) => {
                    let _ = event_tx.send(TuiEvent::Error(format!("Failed to load auth: {e}")));
                }
            }
        }
        TuiAction::SwitchModel(new_model) => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };
            if let Some(m) = find_model_by_id(catalog, &new_model) {
                agent.set_model(m).await;
                let blocks = build_system_prompt(working_dir, &new_model, None).await;
                agent.set_system_prompt(blocks).await;
                let _ = event_tx.send(TuiEvent::Info(format!("Switched to {new_model}")));
                // Persist model preference (merge with existing settings).
                let mut s = crate::settings::load_settings().await;
                s.last_model = Some(new_model.to_string());
                crate::settings::save_settings(&s).await.ok();
            } else {
                let _ = event_tx.send(TuiEvent::Error(format!("Model not found: {new_model}")));
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
                        created_at: m.created_at,
                        message_count: m.message_count,
                    })
                    .collect();
                let _ = event_tx.send(TuiEvent::SessionPicker(infos));
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
            let session_mgr = SessionManager::new(working_dir);
            if let Ok(session) = session_mgr.create(Some(model_id)).await {
                let sid = session
                    .meta
                    .as_ref()
                    .map(|m| m.id.clone())
                    .unwrap_or_default();
                *session_id_cell.write().await = Some(sid.clone());
                let _ = event_tx.send(TuiEvent::SessionCreated {
                    id: sid,
                    model: model_id.to_string(),
                });
            }
        }
    }
}

fn find_model_by_id(catalog: &BuiltInCatalog, id: &str) -> Option<theta_ai::Model> {
    catalog.list().into_iter().find(|m| m.id == id).cloned()
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
                    theta_ai::ContentBlock::Thinking { thinking, .. } => {
                        Some(thinking.as_str())
                    }
                    theta_ai::ContentBlock::ToolCall { name, .. } => {
                        Some(name.as_str())
                    }
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
        theta_ai::Message::ToolResult { tool_name, content, .. } => {
            let text = content
                .iter()
                .filter_map(|b| match b {
                    theta_ai::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            Some(HistoryEntry {
                role: "tool".into(),
                text: format!("[{tool_name}] {text}"),
            })
        }
        _ => None,
    }
}

fn format_tool_result(result: &theta_agent_core::ToolResult) -> String {
    // Format content blocks into a readable summary.
    let summary: String = result
        .content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => text.clone(),
            ContentBlock::Image { .. } => "[image]".into(),
            ContentBlock::ToolCall { name, .. } => format!("[tool_call: {name}]"),
            ContentBlock::Thinking { thinking, .. } => thinking.clone(),
            ContentBlock::ToolResult { tool_name, .. } => {
                format!("[tool_result: {tool_name}]",)
            }
        })
        .collect::<Vec<_>>()
        .join("\n");

    if result.is_error {
        format!("Error: {summary}")
    } else {
        summary
    }
}
