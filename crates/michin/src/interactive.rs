//! Interactive TUI mode — connects the agent to the terminal UI.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use michin_agent_core::agent::Agent;
use michin_agent_core::events::AgentEvent;
use michin_ai::{Model, ModelCatalog, Provider};
use michin_ai_net::default_registry;
use michin_models::BuiltInCatalog;
use michin_models::opencode;
use michin_models::xiaomi;
use michin_tui::App;
use michin_tui::app::{HistoryEntry, TuiAction, TuiEvent};
use michin_tui::components::{CommandEntry, MimoClusterEntry};
use michin_tui::components::{ModelEntry, SessionInfo, known_providers};
use michin_tui::theme::Theme;
use tokio::sync::{RwLock, mpsc};

use crate::config::MichiNConfig;
use crate::session::SessionManager;
use crate::settings::LastSession;
use crate::skills;
use crate::system_prompt::{
    SystemPromptConfig, build_active_overlays, build_system_prompt_with_skills,
    discover_nested_agents, find_context_file,
};
use crate::tools::ToolContext;
use crate::tools::builtin_tools;

/// Shared agent handle — None until auth is resolved.
type AgentCell = Arc<RwLock<Option<Arc<Agent>>>>;

pub async fn run_tui(
    config: &MichiNConfig,
    working_dir: &Path,
    model_id: &str,
    thinking: &str,
    initial_prompt: Option<&str>,
    caveman_mode: Option<String>,
) -> anyhow::Result<()> {
    let catalog = BuiltInCatalog::new();
    let runtime_models_cell: Arc<RwLock<Vec<Model>>> = Arc::new(RwLock::new(
        resolve_runtime_models(&catalog, None, None).await,
    ));
    let runtime_models = runtime_models_cell.read().await.clone();

    let model = find_model_by_id(&runtime_models, model_id)
        .ok_or_else(|| anyhow::anyhow!("model not found: {model_id}"))?
        .clone();

    let provider_str = provider_to_string(model.provider);
    let mut auth_config = config.auth.clone();
    let model_entries = available_model_entries(&runtime_models, &mut auth_config).await;
    let api_key = auth_config.get_api_key(&provider_str).await;

    // If no auth for the default model's provider, find a fallback with auth.
    let (model, model_id, api_key) = if api_key.is_none() {
        let alt_providers = [
            ("openai-codex", Provider::OpenAiCodex),
            ("openai", Provider::OpenAI),
            ("deepseek", Provider::DeepSeek),
            ("opencode", Provider::OpenCode),
            ("xiaomi", Provider::XiaomiMiMo),
        ];
        let mut found: Option<(michin_ai::Model, String, String)> = None;
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

    let (event_tx_raw, mut event_rx_raw) = mpsc::unbounded_channel();
    let (event_tx, event_rx) = mpsc::unbounded_channel();
    let (message_tx, mut message_rx) = mpsc::unbounded_channel::<String>();
    let (action_tx, mut action_rx) = mpsc::unbounded_channel();

    // Session created lazily on first message.
    let session_id_cell: Arc<RwLock<Option<String>>> = Arc::new(RwLock::new(None));

    // Agent populated immediately if auth available, deferred until after login otherwise.
    let agent_cell: AgentCell = Arc::new(RwLock::new(None));

    let user_themes = michin_tui::theme::load_user_themes();
    let initial_theme_name = config.theme.as_deref().unwrap_or("default");
    let theme = Theme::named_with_users(initial_theme_name, &user_themes);

    // ScriptHooks notify after each tool execution to wake up the TUI poller.
    let status_notify = Arc::new(tokio::sync::Notify::new());

    // ── Action handler (login + agent init) ──
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

    // ── Event bridge — if we have auth, create agent now ──
    if let Some(ref key) = api_key {
        let agent = create_agent(&AgentCreateCtx {
            model: &model,
            api_key: key,
            config,
            working_dir,
            model_id: &model_id,
            thinking,
            status_notify: &status_notify,
            caveman_mode: caveman_mode.as_deref(),
        })
        .await?;
        let agent = Arc::new(agent);

        *agent_cell.write().await = Some(agent.clone());
        let persisted = crate::settings::load_settings().await;
        spawn_event_bridge(
            agent.clone(),
            event_tx_raw.clone(),
            persisted.tool_progress_hz.max(1),
        );

        // Apply caveman mode if starting with it active.
        // Do this AFTER spawn_event_bridge so the event reaches the TUI.
        if let Some(ref level) = caveman_mode {
            agent.set_caveman_mode(Some(level.clone())).await;
            let _ = event_tx_raw.send(TuiEvent::CavemanModeToggled {
                level: Some(level.clone()),
            });
        }

        // Send initial valid thinking levels for the model.
        let levels = compute_valid_thinking_levels(&model);
        let parsed = parse_thinking_level(thinking);
        let tl = normalize_thinking_level(&model, parsed, &levels);
        let _ = event_tx_raw.send(TuiEvent::ThinkingLevels {
            levels,
            current: tl,
            show_selector: false,
        });

        // Send startup banner.
        {
            let working_dir = working_dir.to_path_buf();
            let tx = event_tx_raw.clone();
            tokio::spawn(async move {
                if let Some(banner) = build_startup_banner(&working_dir).await {
                    let _ = tx.send(TuiEvent::Info(banner));
                }
            });
        }

        // Send caveman mode state for status bar badge on first paint.
        if let Some(ref level) = caveman_mode {
            let _ = event_tx_raw.send(TuiEvent::CavemanModeToggled {
                level: Some(level.clone()),
            });
        }

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
        s.set_model_thinking(&provider_to_string(model.provider), &model_id, thinking);
        crate::settings::save_settings(&s).await.ok();
    }

    let skills = crate::skills::discover_skills(working_dir).await;

    // ── Spawn message handler (waits for agent, creates session lazily) ──
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
            let prompt_result = agent.prompt(blocks).await;
            let aborted = matches!(&prompt_result, Err(e) if matches!(e, michin_agent_core::error::AgentError::Aborted));
            if let Err(e) = &prompt_result {
                tracing::error!("agent prompt failed: {e}");
                let _ = msg_event_tx.send(TuiEvent::Error(format_error_chain(e)));
            }

            // Always persist accumulated messages — even on abort or error.
            // append_missing_entries is idempotent (fingerprint dedup), and
            // the agent loop already skips persisting incomplete assistant
            // messages from aborted streams.
            let end_reason = if aborted {
                Some("AbortedByUser")
            } else if prompt_result.is_err() {
                Some("ProviderFailure")
            } else {
                None
            };
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
                    } else if let Some(reason) = end_reason {
                        let _ = session_mgr.mark_run_completed(&sid, Some(reason)).await;
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

    // ── Build available commands + skills for / picker ──
    let mut commands = vec![
        CommandEntry {
            name: "help".into(),
            description: "Show available commands".into(),
        },
        CommandEntry {
            name: "keys".into(),
            description: "Show keyboard shortcuts".into(),
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
            name: "plan".into(),
            description: "Toggle plan mode (explore/plan without modifying source code)".into(),
        },
        CommandEntry {
            name: "caveman".into(),
            description: "Toggle caveman compact mode (off/lite/full/ultra/wenyan-*)".into(),
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
            name: "themes".into(),
            description: "Open theme picker with live preview".into(),
        },
        CommandEntry {
            name: "status".into(),
            description: "Show live runtime status snapshot".into(),
        },
        CommandEntry {
            name: "timeline".into(),
            description: "Show compact timeline from latest run report".into(),
        },
        CommandEntry {
            name: "diag".into(),
            description: "Toggle diagnostic event stream (on/off)".into(),
        },
        CommandEntry {
            name: "tools-rate".into(),
            description: "Set tool progress update rate (1-60 Hz)".into(),
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
            name: "settings".into(),
            description: "Open settings panel (steering, thinking, diffs, context cap, etc.)"
                .into(),
        },
        CommandEntry {
            name: "exit".into(),
            description: "Exit MichiN".into(),
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
    // ── Build and run TUI ──
    let persisted = crate::settings::load_settings().await;
    let mut app = App::new(
        theme.clone(),
        user_themes,
        initial_theme_name,
        &model.id,
        "", // session created lazily on first message
        thinking,
        michin_tui::app::SettingsPayload {
            steering_mode: persisted.steering_mode,
            follow_up_mode: persisted.follow_up_mode,
            transport_preference: persisted.transport_preference,
            show_thinking: persisted.show_thinking,
            show_tool_diffs: persisted.show_tool_diffs,
            tool_progress_hz: persisted.tool_progress_hz,
            enter_behavior: persisted.enter_behavior,
            max_context_window: persisted.max_context_window,
            auto_escalate: persisted.auto_escalate,
        },
        model_entries,
        persisted.favorite_models.clone(),
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
            config.auth.has_token("xiaomi"),
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

// ── Helpers ──

/// Parameters for creating a fully configured agent.
struct AgentCreateCtx<'a> {
    model: &'a Model,
    api_key: &'a str,
    config: &'a MichiNConfig,
    working_dir: &'a Path,
    model_id: &'a str,
    thinking: &'a str,
    status_notify: &'a Arc<tokio::sync::Notify>,
    caveman_mode: Option<&'a str>,
}

async fn create_agent(ctx: &AgentCreateCtx<'_>) -> anyhow::Result<Agent> {
    let catalog = BuiltInCatalog::new();
    let available_models: Vec<michin_ai::Model> = catalog.list().into_iter().cloned().collect();
    let registry = default_registry();
    registry.set_api_key(ctx.model.provider, ctx.api_key);

    let tool_ctx = ToolContext::new(ctx.working_dir.to_path_buf());
    let mut agent = Agent::new(ctx.model.clone(), Arc::new(registry), available_models);
    let mut loop_config = crate::config::to_agent_config(ctx.config);
    let settings = crate::settings::load_settings().await;
    loop_config.max_context_window = settings.max_context_window;
    agent.set_config(loop_config);

    // Restore saved MiMo cluster URL so token-plan keys route to the
    // user's preferred region instead of defaulting to SGP.
    if ctx.model.provider == michin_ai::types::Provider::XiaomiMiMo
        && let Some(ref cluster_url) = settings.mimo_cluster_url
    {
        agent.set_mimo_base_url(cluster_url);
    }

    for tool in builtin_tools(tool_ctx) {
        agent.add_tool(tool).await;
    }

    let (system_blocks, resource_blocks) = build_system_prompt_with_skills(
        ctx.working_dir,
        &SystemPromptConfig {
            model_id: ctx.model_id,
            thinking_level: Some(ctx.thinking),
            max_context_window: settings.max_context_window,
        },
    )
    .await;
    agent.set_system_prompt(system_blocks).await;
    if !resource_blocks.is_empty() {
        agent.set_resource_context(resource_blocks).await;
    }

    // Set initial volatile overlays: RESPONSE_CONTRACT always active.
    let overlays = build_active_overlays(false, ctx.caveman_mode);
    agent.set_volatile_overlays(overlays).await;

    // Wire auto-escalation if enabled in settings and current model is DeepSeek.
    if settings.auto_escalate && ctx.model.provider == michin_ai::Provider::DeepSeek {
        let esc_id = settings
            .escalation_model
            .as_deref()
            .unwrap_or("deepseek-v4-pro");
        let available_models: Vec<michin_ai::Model> =
            BuiltInCatalog::new().list().into_iter().cloned().collect();
        if let Some(esc_model) = available_models.iter().find(|m| m.id == esc_id)
            && esc_model.provider == michin_ai::Provider::DeepSeek
        {
            agent.set_escalation_model(Some(esc_model.clone())).await;
        }
    }

    // Load script hooks from ~/.michin/extensions/*.rhai and ./.michin/extensions/*.rhai.
    if let Some(hooks) =
        crate::scripts::load_script_hooks(ctx.working_dir, Arc::clone(ctx.status_notify)).await
    {
        agent.set_hooks(hooks);
    }

    // Load custom tools from ~/.michin/tools/*.rhai and ./.michin/tools/*.rhai.
    for tool in crate::scripts::load_custom_tools(ctx.working_dir).await {
        agent.add_tool(tool).await;
    }

    // Apply thinking level from settings.
    let tl = parse_thinking_level(ctx.thinking);
    agent.set_thinking_level(tl).await;

    Ok(agent)
}

/// Parse a thinking level string into a ThinkingLevel enum.
fn parse_thinking_level(level: &str) -> michin_ai::ThinkingLevel {
    match level.to_lowercase().as_str() {
        "off" => michin_ai::ThinkingLevel::Off,
        "enabled" => michin_ai::ThinkingLevel::Minimal,
        "minimal" => michin_ai::ThinkingLevel::Minimal,
        "low" => michin_ai::ThinkingLevel::Low,
        "medium" => michin_ai::ThinkingLevel::Medium,
        "high" => michin_ai::ThinkingLevel::High,
        "xhigh" | "max" => michin_ai::ThinkingLevel::XHigh,
        _ => michin_ai::ThinkingLevel::Off,
    }
}

/// Compute the list of valid thinking level strings for a model.
fn compute_valid_thinking_levels(model: &michin_ai::Model) -> Vec<String> {
    let all_levels: &[(&str, michin_ai::ThinkingLevel)] = &[
        ("off", michin_ai::ThinkingLevel::Off),
        ("minimal", michin_ai::ThinkingLevel::Minimal),
        ("low", michin_ai::ThinkingLevel::Low),
        ("medium", michin_ai::ThinkingLevel::Medium),
        ("high", michin_ai::ThinkingLevel::High),
        ("xhigh", michin_ai::ThinkingLevel::XHigh),
    ];
    // Deduplicate by the actual param value sent to the provider.
    // For binary-thinking providers (like MiMo), all non-Off levels
    // share the same param, so only one non-Off entry appears.
    let mut seen_params: std::collections::HashSet<Option<String>> =
        std::collections::HashSet::new();
    all_levels
        .iter()
        .filter_map(|(_name, level)| {
            if *level == michin_ai::ThinkingLevel::Off {
                // Always include Off.
                seen_params.insert(None);
                return Some("off".to_string());
            }
            let param = model.thinking_param(*level);
            param.filter(|p| seen_params.insert(Some(p.clone())))
        })
        .collect()
}

/// Convert a ThinkingLevel to its string representation.
fn thinking_level_to_string(level: michin_ai::ThinkingLevel) -> String {
    match level {
        michin_ai::ThinkingLevel::Off => "off".to_string(),
        michin_ai::ThinkingLevel::Minimal => "minimal".to_string(),
        michin_ai::ThinkingLevel::Low => "low".to_string(),
        michin_ai::ThinkingLevel::Medium => "medium".to_string(),
        michin_ai::ThinkingLevel::High => "high".to_string(),
        michin_ai::ThinkingLevel::XHigh => "xhigh".to_string(),
        michin_ai::ThinkingLevel::Max => "max".to_string(),
    }
}

/// Normalize a ThinkingLevel to match the provider's param in the valid levels list.
/// For binary-thinking providers (MiMo), maps generic levels like "minimal" to the
/// provider-native ID (e.g. "enabled"). Falls back to the generic string if no match.
fn normalize_thinking_level(
    model: &michin_ai::Model,
    level: michin_ai::ThinkingLevel,
    valid_levels: &[String],
) -> String {
    let generic = thinking_level_to_string(level);
    if valid_levels.contains(&generic) {
        return generic;
    }
    // Try to find a matching provider param in the valid levels.
    // Deduplicate by param value for binary-thinking providers (MiMo).
    let param = model.thinking_param(level);
    if let Some(p) = param
        && valid_levels.contains(&p)
    {
        return p;
    }
    generic
}

/// Spawn the event bridge — subscribes to agent events, forwards to TUI.
fn spawn_event_bridge(agent: Arc<Agent>, event_tx: mpsc::UnboundedSender<TuiEvent>, _hz: u64) {
    tokio::spawn(async move {
        let reserve_tokens = agent.config().compaction.reserve_tokens;
        let model_ctx = agent.state().await.model.context_window;
        let context_window = agent.config().effective_context_window(model_ctx);
        let mut events = agent.subscribe();
        let mut tool_names: HashMap<String, String> = HashMap::new();
        let mut saw_assistant_text_delta = false;
        let mut saw_thinking_delta = false;
        let mut latest_turn_end_reason = "completed".to_string();

        loop {
            let received = events.recv().await;
            match received {
                Ok(AgentEvent::MessageStart) => {
                    saw_assistant_text_delta = false;
                    saw_thinking_delta = false;
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
                Ok(AgentEvent::ToolExecutionStart {
                    tool_call_id: id,
                    tool_name: name,
                    arguments,
                }) => {
                    let args = arguments.and_then(|v| serde_json::to_string(&v).ok());
                    tool_names.insert(id.clone(), name.clone());

                    // Detect skill loading: read tool targeting SKILL.md
                    if name == "read"
                        && let Some(ref raw_args) = args
                        && let Ok(json) = serde_json::from_str::<serde_json::Value>(raw_args)
                        && let Some(path) = json.get("path").and_then(|v| v.as_str())
                    {
                        let lower_path = path.to_lowercase();
                        if lower_path.ends_with("skill.md") || lower_path.contains("/skill.md") {
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
                    let _ = event_tx.send(TuiEvent::ToolEnd {
                        id: result.tool_call_id,
                        name: result.tool_name,
                        is_error: result.is_error,
                        summary,
                        details: result.details.clone(),
                    });
                }
                Ok(AgentEvent::MessageEnd { message }) => {
                    // Forward token usage to TUI status bar.
                    if let michin_ai::Message::Assistant { content, usage, .. } = &message {
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
                                    michin_ai::ContentBlock::Text { text } => Some(text.as_str()),
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
                                    michin_ai::ContentBlock::Thinking { thinking, .. } => {
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
                    if !matches!(reason, michin_agent_core::types::TurnEndReason::Completed) {
                        let detail = details.trim();
                        // Provider failures: log the full error chain via tracing,
                        // but keep the chat message short — the user can't act on
                        // HTTP connection traces.
                        if matches!(
                            reason,
                            michin_agent_core::types::TurnEndReason::ProviderFailure
                        ) {
                            tracing::warn!(
                                ?reason,
                                detail = %detail,
                                "turn terminated with provider failure"
                            );
                            // Include the error detail so the user can diagnose.
                            // Clip to a reasonable length; the full body is in tracing logs.
                            let short = truncate_chars(detail, 500);
                            let _ =
                                event_tx.send(TuiEvent::Info(format!("Provider error: {short}",)));
                        } else {
                            let message = if detail.is_empty() {
                                format!("Turn ended: {reason:?}")
                            } else {
                                format!("Turn ended: {reason:?}\n{detail}")
                            };
                            let _ = event_tx.send(TuiEvent::Info(message));
                        }
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
                        michin_agent_core::types::SafetyDecisionKind::Rejected
                    ) {
                        let _ = event_tx.send(TuiEvent::Info(format!(
                            "Safety policy rejected {tool_name}: {details}"
                        )));
                    }
                }
                Ok(AgentEvent::AgentEnd { aborted }) => {
                    let _ = event_tx.send(TuiEvent::AgentEnd { aborted });
                }
                Ok(AgentEvent::PlanModeToggled { enabled }) => {
                    let _ = event_tx.send(TuiEvent::PlanModeToggled { enabled });
                }
                Ok(AgentEvent::CavemanModeToggled { level }) => {
                    let _ = event_tx.send(TuiEvent::CavemanModeToggled { level });
                }
                Ok(AgentEvent::ModelEscalated {
                    from: _,
                    to,
                    is_escalation,
                }) => {
                    let _ = event_tx.send(TuiEvent::ModelEscalated {
                        from: String::new(),
                        to,
                        is_escalation,
                    });
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
                Ok(AgentEvent::CompactionPaused {
                    context_window,
                    reserve_tokens,
                }) => {
                    let _ = event_tx.send(TuiEvent::CompactionPaused {
                        context_window,
                        reserve_tokens,
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
                    tracing::debug!(
                        dropped_assistant_messages,
                        synthesized_tool_results,
                        normalized_tool_call_ids,
                        deduped_tool_results,
                        "replay sanitized"
                    );
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
                Ok(AgentEvent::AgentStart) => {
                    // Agent lifecycle — informational, no TUI event needed.
                }
                Ok(AgentEvent::ToolCallDelta { .. }) => {
                    // Streaming tool-call args — handled by LLM-side processing.
                }
                Ok(AgentEvent::ToolCallEnd { .. }) => {
                    // Streaming tool-call completion — handled by LLM-side processing.
                }
                Ok(AgentEvent::ProviderCircuitOpen { key, retry_in_ms }) => {
                    tracing::debug!(
                        provider_key = %key,
                        retry_in_ms = retry_in_ms,
                        "provider circuit breaker open"
                    );
                }
                Ok(AgentEvent::ProviderFallback {
                    from_model,
                    to_model,
                    reason,
                }) => {
                    tracing::debug!(
                        from_model = %from_model,
                        to_model = %to_model,
                        reason = %reason,
                        "provider fallback triggered"
                    );
                }
                Ok(AgentEvent::CacheShapeReport {
                    bust_reason,
                    per_tool_tokens: _,
                }) => {
                    // Only emitted on cache miss — hits are silent.
                    tracing::debug!(
                        bust_reason = %bust_reason,
                        "prefix cache miss"
                    );
                }
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
    config: &MichiNConfig,
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

                                match create_agent(&AgentCreateCtx {
                                    model: &codex_model,
                                    api_key: &creds.access_token,
                                    config,
                                    working_dir,
                                    model_id: &codex_model.id,
                                    thinking,
                                    status_notify,
                                    caveman_mode: None, // caveman starts off on fresh login
                                })
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

                                        // Show startup banner.
                                        if let Some(banner) =
                                            build_startup_banner(working_dir).await
                                        {
                                            let _ = event_tx.send(TuiEvent::Info(banner));
                                        }

                                        // Persist model + thinking.
                                        let mut s = crate::settings::load_settings().await;
                                        s.set_model_thinking(
                                            &provider_to_string(codex_model.provider),
                                            &codex_model.id,
                                            thinking,
                                        );
                                        crate::settings::save_settings(&s).await.ok();
                                    }
                                    Err(e) => {
                                        let _ = event_tx.send(TuiEvent::Error(format!(
                                            "Failed to start agent: {e}"
                                        )));
                                    }
                                }
                            }

                            refresh_runtime_models(
                                catalog,
                                runtime_models_cell,
                                auth.get_api_key("opencode").await.as_deref(),
                                auth.get_api_key("xiaomi").await.as_deref(),
                            )
                            .await;
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
                        match create_agent(&AgentCreateCtx {
                            model,
                            api_key: &token,
                            config,
                            working_dir,
                            model_id,
                            thinking,
                            status_notify,
                            caveman_mode: None, // caveman starts off on fresh login
                        })
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

                                // Show startup banner.
                                if let Some(banner) = build_startup_banner(working_dir).await {
                                    let _ = event_tx.send(TuiEvent::Info(banner));
                                }

                                // Persist model + thinking.
                                let mut s = crate::settings::load_settings().await;
                                s.set_model_thinking(
                                    &provider_to_string(model.provider),
                                    model_id,
                                    thinking,
                                );
                                crate::settings::save_settings(&s).await.ok();
                            }
                            Err(e) => {
                                let _ = event_tx
                                    .send(TuiEvent::Error(format!("Failed to start agent: {e}")));
                            }
                        }
                    }

                    refresh_runtime_models(
                        catalog,
                        runtime_models_cell,
                        auth.get_api_key("opencode").await.as_deref(),
                        auth.get_api_key("xiaomi").await.as_deref(),
                    )
                    .await;
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
                let (saved_thinking, _had_saved) = {
                    let s = crate::settings::load_settings().await;
                    match s.thinking_for_model(&provider, &model_id) {
                        Some(t) => (t.to_string(), true),
                        None => {
                            let state = agent.state().await;
                            let tl = thinking_level_to_string(state.thinking_level);
                            drop(state);
                            (tl, false)
                        }
                    }
                };
                let is_valid = levels.contains(&saved_thinking);
                let normalized_saved = if is_valid {
                    Some(saved_thinking.clone())
                } else {
                    let tl = parse_thinking_level(&saved_thinking);
                    let param = m.thinking_param(tl);
                    param.and_then(|p| levels.iter().find(|l| **l == p).cloned())
                };
                let show_thinking_selector = normalized_saved.is_none();
                let current_thinking = match normalized_saved {
                    Some(t) => t,
                    None => "off".to_string(),
                };

                agent.set_model(m).await;

                if !show_thinking_selector {
                    let tl = parse_thinking_level(&current_thinking);
                    agent.set_thinking_level(tl).await;
                } else {
                    agent
                        .set_thinking_level(michin_ai::ThinkingLevel::Off)
                        .await;
                }
                let max_ctx = agent.config().max_context_window;
                let settings = crate::settings::load_settings().await;
                let (blocks, resource_blocks) = build_system_prompt_with_skills(
                    working_dir,
                    &SystemPromptConfig {
                        model_id: &model_id,
                        thinking_level: Some(&current_thinking),
                        max_context_window: max_ctx,
                    },
                )
                .await;
                agent.set_system_prompt(blocks).await;
                if !resource_blocks.is_empty() {
                    agent.set_resource_context(resource_blocks).await;
                }
                let plan_mode = agent.plan_mode().await;
                let overlays = build_active_overlays(plan_mode, settings.caveman_mode.as_deref());
                agent.set_volatile_overlays(overlays).await;
                let _ = event_tx.send(TuiEvent::Info(format!(
                    "Switched to {model_id} ({provider})"
                )));
                let _ = event_tx.send(TuiEvent::ModelSwitched {
                    model: model_id.to_string(),
                    provider: provider.clone(),
                });
                let _ = event_tx.send(TuiEvent::ThinkingLevels {
                    levels,
                    current: current_thinking.clone(),
                    show_selector: show_thinking_selector,
                });
                // Persist model + thinking preference (merge with existing settings).
                let mut s = crate::settings::load_settings().await;
                if agent.plan_mode().await {
                    // In plan mode: update plan_session only, keep last_session
                    // untouched so toggle-off can restore the pre-plan model.
                    s.model_thinking_map
                        .entry(provider.clone())
                        .or_default()
                        .insert(model_id.to_string(), current_thinking.clone());
                    s.plan_session = Some(LastSession {
                        provider,
                        model: model_id.to_string(),
                        thinking: Some(current_thinking.clone()),
                    });
                } else {
                    s.set_model_thinking(&provider, &model_id, &current_thinking);
                }
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
                "off" => michin_ai::ThinkingLevel::Off,
                "enabled" => michin_ai::ThinkingLevel::Minimal,
                "minimal" => michin_ai::ThinkingLevel::Minimal,
                "low" => michin_ai::ThinkingLevel::Low,
                "medium" => michin_ai::ThinkingLevel::Medium,
                "high" => michin_ai::ThinkingLevel::High,
                "xhigh" | "max" => michin_ai::ThinkingLevel::XHigh,
                _ => {
                    let _ = event_tx.send(TuiEvent::Error(format!(
                        "Invalid thinking level: {level}. Use off/enabled/minimal/low/medium/high/xhigh"
                    )));
                    acknowledge(event_tx);
                    return;
                }
            };
            agent.set_thinking_level(tl).await;
            // Persist thinking preference to the per-model map.
            // Store the canonical enum string so parse_thinking_level
            // can roundtrip correctly on restart.
            let canonical = thinking_level_to_string(tl);
            let mut s = crate::settings::load_settings().await;
            {
                let state = agent.state().await;
                s.set_model_thinking(
                    &provider_to_string(state.model.provider),
                    &state.model.id,
                    &canonical,
                );
            }
            crate::settings::save_settings(&s).await.ok();
            let _ = event_tx.send(TuiEvent::ThinkingSet { level: canonical });
            acknowledge(event_tx);
        }
        TuiAction::ShowThinkingSelector => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };
            let state = agent.state().await;
            let levels = compute_valid_thinking_levels(&state.model);
            // Normalize the current level to match the provider param (e.g.
            // "minimal" → "enabled" for binary-thinking providers).
            let current_str = normalize_thinking_level(&state.model, state.thinking_level, &levels);
            let _ = event_tx.send(TuiEvent::ThinkingLevels {
                levels,
                current: current_str,
                show_selector: true,
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
                        last_active_at: m.last_active_at,
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
                    let recap = session_recap(&session);
                    agent.load_messages(messages.clone()).await;
                    let state = agent.state().await;
                    let mid = state
                        .last_model_id()
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| model_id.to_string());
                    let current_thinking = thinking_level_to_string(state.thinking_level);
                    drop(state);
                    let max_ctx = agent.config().max_context_window;
                    let settings = crate::settings::load_settings().await;
                    let (blocks, resource_blocks) = build_system_prompt_with_skills(
                        working_dir,
                        &SystemPromptConfig {
                            model_id: &mid,
                            thinking_level: Some(&current_thinking),
                            max_context_window: max_ctx,
                        },
                    )
                    .await;
                    agent.set_system_prompt(blocks).await;
                    if !resource_blocks.is_empty() {
                        agent.set_resource_context(resource_blocks).await;
                    }
                    let plan_mode = agent.plan_mode().await;
                    let overlays =
                        build_active_overlays(plan_mode, settings.caveman_mode.as_deref());
                    agent.set_volatile_overlays(overlays).await;
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
                    // Always send LoadHistory so old messages are cleared.
                    let _ = event_tx.send(TuiEvent::LoadHistory(history));
                    let _ = event_tx.send(TuiEvent::Info(recap));
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
            // runtime context shows the correct values.
            let state = agent.state().await;
            let current_model_id = state.model.id.clone();
            let current_thinking = thinking_level_to_string(state.thinking_level);
            drop(state);
            let max_ctx = agent.config().max_context_window;
            let settings = crate::settings::load_settings().await;
            let (blocks, resource_blocks) = build_system_prompt_with_skills(
                working_dir,
                &SystemPromptConfig {
                    model_id: &current_model_id,
                    thinking_level: Some(&current_thinking),
                    max_context_window: max_ctx,
                },
            )
            .await;
            agent.set_system_prompt(blocks).await;
            if !resource_blocks.is_empty() {
                agent.set_resource_context(resource_blocks).await;
            }
            let plan_mode = agent.plan_mode().await;
            let overlays = build_active_overlays(plan_mode, settings.caveman_mode.as_deref());
            agent.set_volatile_overlays(overlays).await;

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

            // Show startup banner on new session.
            if let Some(banner) = build_startup_banner(working_dir).await {
                let _ = event_tx.send(TuiEvent::Info(banner));
            }
        }
        TuiAction::Steer(text) => {
            let Some(agent) = agent_cell.read().await.clone() else {
                return;
            };
            agent.steer(vec![michin_ai::ContentBlock::Text { text }]);
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
                context_window: agent
                    .config()
                    .effective_context_window(state.model.context_window),
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
                    let avail = agent
                        .config()
                        .effective_context_window(state.model.context_window)
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
            agent.follow_up(vec![michin_ai::ContentBlock::Text { text }]);
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
                Err(michin_agent_core::error::AgentError::NotRunning) => {
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
        TuiAction::ToggleFavoriteModel { model_id } => {
            let mut s = crate::settings::load_settings().await;
            let is_fav = s.toggle_favorite_model(&model_id);
            crate::settings::save_settings(&s).await.ok();
            let _ = event_tx.send(TuiEvent::ModelFavoritesUpdated {
                favorites: s.favorite_models.clone(),
                toggled_model: model_id,
                is_favorite: is_fav,
            });
        }
        TuiAction::ShowMimoClusters => {
            // Measure latencies and send results back to TUI.
            // Always show the modal — the user may want to switch clusters.
            // Use the stored API key to filter clusters (token-plan keys only
            // work on their assigned region).
            let api_key = if let Some(agent) = agent_cell.read().await.clone() {
                agent.provider_key(Provider::XiaomiMiMo).unwrap_or_default()
            } else {
                String::new()
            };
            let clusters = crate::mimo_cluster::measure_cluster_latencies(&api_key).await;
            let entries: Vec<MimoClusterEntry> = clusters
                .into_iter()
                .map(|c| MimoClusterEntry {
                    label: c.label,
                    url: c.url,
                    latency_ms: c.latency_ms,
                })
                .collect();
            let s = crate::settings::load_settings().await;
            let _ = event_tx.send(TuiEvent::MimoClusterResults {
                clusters: entries,
                current_url: s.mimo_cluster_url,
            });
        }
        TuiAction::SelectMimoCluster { url } => {
            // Store in settings and set on the provider.
            let mut s = crate::settings::load_settings().await;
            s.mimo_cluster_url = Some(url.clone());
            crate::settings::save_settings(&s).await.ok();

            // Set on the provider for immediate effect.
            if let Some(agent) = agent_cell.read().await.clone() {
                agent.set_mimo_base_url(&url);
            }

            let _ = event_tx.send(TuiEvent::Info(format!("MiMo cluster set to {url}")));
        }
        TuiAction::SetTheme { name } => {
            let mut cfg = match crate::config::load_config(None).await {
                Ok(c) => c,
                Err(e) => {
                    let _ = event_tx.send(TuiEvent::Error(format!("Failed to load config: {e}")));
                    return;
                }
            };
            cfg.theme = Some(name.clone());
            if let Err(e) = crate::config::save_config(&cfg, None).await {
                let _ = event_tx.send(TuiEvent::Error(format!("Failed to save theme: {e}")));
                return;
            }
            let _ = event_tx.send(TuiEvent::Info(format!(
                "Theme '{name}' saved to config.toml"
            )));
        }
        TuiAction::UpdateSettings {
            steering_mode,
            follow_up_mode,
            transport_preference,
            show_thinking,
            show_tool_diffs,
            tool_progress_hz,
            enter_behavior,
            max_context_window,
            auto_escalate,
        } => {
            let mut settings = crate::settings::load_settings().await;
            settings.steering_mode = steering_mode.clone();
            settings.follow_up_mode = follow_up_mode.clone();
            settings.transport_preference = transport_preference;
            settings.show_thinking = show_thinking;
            settings.show_tool_diffs = show_tool_diffs;
            settings.tool_progress_hz = tool_progress_hz;
            settings.enter_behavior = enter_behavior;
            settings.max_context_window = max_context_window;
            settings.auto_escalate = auto_escalate;
            if let Err(e) = crate::settings::save_settings(&settings).await {
                let _ = event_tx.send(TuiEvent::Error(format!("Failed to save settings: {e}")));
                return;
            }
            let _ = event_tx.send(TuiEvent::SettingsApplied {
                steering_mode: steering_mode.clone(),
                follow_up_mode: follow_up_mode.clone(),
                show_thinking,
                show_tool_diffs,
                tool_progress_hz,
            });
            let _ = event_tx.send(TuiEvent::Info("Settings saved to settings.json".into()));
        }
        TuiAction::TogglePlanMode => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };

            let enable = !agent.plan_mode().await;
            let mut settings = crate::settings::load_settings().await;

            if enable {
                // Save current model+thinking as last_session so toggle-off restores them.
                let (current_provider, current_model_id, current_thinking) = {
                    let state = agent.state().await;
                    (
                        provider_to_string(state.model.provider),
                        state.model.id.clone(),
                        thinking_level_to_string(state.thinking_level),
                    )
                };
                settings.set_last_session(
                    &current_provider,
                    &current_model_id,
                    Some(&current_thinking),
                );

                // Switch to the plan model if one is already saved.
                let plan_sess = settings.plan_session.clone();
                if let Some(ref sess) = plan_sess {
                    let runtime_models = runtime_models_cell.read().await.clone();
                    if let Some(m) =
                        find_model_by_provider_and_id(&runtime_models, &sess.provider, &sess.model)
                    {
                        let provider_str = provider_to_string(m.provider);
                        let mut auth = crate::config::load_auth(None).await.unwrap_or_default();
                        if let Some(key) = auth.get_api_key(&provider_str).await {
                            let levels = compute_valid_thinking_levels(&m);
                            let saved_thinking =
                                sess.thinking.clone().unwrap_or_else(|| "off".to_string());
                            let is_valid = levels.contains(&saved_thinking);
                            let normalized = if is_valid {
                                Some(saved_thinking.clone())
                            } else {
                                let tl = parse_thinking_level(&saved_thinking);
                                let param = m.thinking_param(tl);
                                param.and_then(|p| levels.iter().find(|l| **l == p).cloned())
                            };
                            let current_thinking = normalized.as_deref().unwrap_or("off");
                            let show_selector = normalized.is_none();

                            agent.set_api_key(m.provider, key);
                            agent.set_model(m).await;
                            if !show_selector {
                                agent
                                    .set_thinking_level(parse_thinking_level(current_thinking))
                                    .await;
                            } else {
                                agent
                                    .set_thinking_level(michin_ai::ThinkingLevel::Off)
                                    .await;
                            }
                            let max_ctx = agent.config().max_context_window;
                            let (blocks, resource_blocks) = build_system_prompt_with_skills(
                                working_dir,
                                &SystemPromptConfig {
                                    model_id: &sess.model,
                                    thinking_level: Some(current_thinking),
                                    max_context_window: max_ctx,
                                },
                            )
                            .await;
                            agent.set_system_prompt(blocks).await;
                            if !resource_blocks.is_empty() {
                                agent.set_resource_context(resource_blocks).await;
                            }
                            let overlays =
                                build_active_overlays(true, settings.caveman_mode.as_deref());
                            agent.set_volatile_overlays(overlays).await;
                            let _ = event_tx.send(TuiEvent::ModelSwitched {
                                model: sess.model.clone(),
                                provider: sess.provider.clone(),
                            });
                            let _ = event_tx.send(TuiEvent::ThinkingLevels {
                                levels,
                                current: current_thinking.to_string(),
                                show_selector,
                            });
                        }
                    }
                } else {
                    // First time: snapshot current as plan.
                    settings.plan_session = Some(LastSession {
                        provider: current_provider,
                        model: current_model_id,
                        thinking: Some(current_thinking),
                    });
                }
            } else {
                // Restore the regular last_session model.
                if let Some(ref sess) = settings.last_session {
                    let runtime_models = runtime_models_cell.read().await.clone();
                    if let Some(m) =
                        find_model_by_provider_and_id(&runtime_models, &sess.provider, &sess.model)
                    {
                        let provider_str = provider_to_string(m.provider);
                        let mut auth = crate::config::load_auth(None).await.unwrap_or_default();
                        if let Some(key) = auth.get_api_key(&provider_str).await {
                            let levels = compute_valid_thinking_levels(&m);
                            let saved_thinking =
                                sess.thinking.clone().unwrap_or_else(|| "off".to_string());
                            let is_valid = levels.contains(&saved_thinking);
                            let normalized = if is_valid {
                                Some(saved_thinking.clone())
                            } else {
                                let tl = parse_thinking_level(&saved_thinking);
                                let param = m.thinking_param(tl);
                                param.and_then(|p| levels.iter().find(|l| **l == p).cloned())
                            };
                            let current_thinking = normalized.as_deref().unwrap_or("off");
                            let show_selector = normalized.is_none();

                            agent.set_api_key(m.provider, key);
                            agent.set_model(m).await;
                            if !show_selector {
                                agent
                                    .set_thinking_level(parse_thinking_level(current_thinking))
                                    .await;
                            } else {
                                agent
                                    .set_thinking_level(michin_ai::ThinkingLevel::Off)
                                    .await;
                            }
                            let max_ctx = agent.config().max_context_window;
                            let (blocks, resource_blocks) = build_system_prompt_with_skills(
                                working_dir,
                                &SystemPromptConfig {
                                    model_id: &sess.model,
                                    thinking_level: Some(current_thinking),
                                    max_context_window: max_ctx,
                                },
                            )
                            .await;
                            agent.set_system_prompt(blocks).await;
                            if !resource_blocks.is_empty() {
                                agent.set_resource_context(resource_blocks).await;
                            }
                            let overlays =
                                build_active_overlays(false, settings.caveman_mode.as_deref());
                            agent.set_volatile_overlays(overlays).await;
                            let _ = event_tx.send(TuiEvent::ModelSwitched {
                                model: sess.model.clone(),
                                provider: sess.provider.clone(),
                            });
                            let _ = event_tx.send(TuiEvent::ThinkingLevels {
                                levels,
                                current: current_thinking.to_string(),
                                show_selector,
                            });
                        }
                    }
                }
            }

            agent.set_plan_mode(enable).await;
            crate::settings::save_settings(&settings).await.ok();

            // For first-time toggle (no model switch happened), still rebuild
            // system prompt and swap overlays.
            let state = agent.state().await;
            let model_id = state.model.id.clone();
            let thinking = thinking_level_to_string(state.thinking_level);
            let provider_str = provider_to_string(state.model.provider);
            drop(state);
            let max_ctx = agent.config().max_context_window;
            let settings = crate::settings::load_settings().await;
            let (blocks, resource_blocks) = build_system_prompt_with_skills(
                working_dir,
                &SystemPromptConfig {
                    model_id: &model_id,
                    thinking_level: Some(&thinking),
                    max_context_window: max_ctx,
                },
            )
            .await;
            agent.set_system_prompt(blocks).await;
            if !resource_blocks.is_empty() {
                agent.set_resource_context(resource_blocks).await;
            }
            let plan_mode = agent.plan_mode().await;
            let overlays = build_active_overlays(plan_mode, settings.caveman_mode.as_deref());
            agent.set_volatile_overlays(overlays).await;

            let _ = event_tx.send(TuiEvent::Info(format!(
                "Plan mode {} (model: {model_id}, provider: {provider_str})",
                if enable { "on" } else { "off" }
            )));
        }
        TuiAction::ToggleCavemanMode { level } => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };

            let normalized = normalize_caveman_level(level.as_deref());
            agent.set_caveman_mode(normalized.clone()).await;

            // Persist
            let mut settings = crate::settings::load_settings().await;
            settings.caveman_mode = normalized.clone();
            crate::settings::save_settings(&settings).await.ok();

            // Swap volatile overlays instead of rebuilding system prompt.
            let plan_mode = agent.plan_mode().await;
            let overlays = build_active_overlays(plan_mode, normalized.as_deref());
            agent.set_volatile_overlays(overlays).await;

            let msg = match normalized {
                Some(ref l) => format!("Caveman mode on ({l})"),
                None => "Caveman mode off.".into(),
            };
            let _ = event_tx.send(TuiEvent::Info(msg));
            let _ = event_tx.send(TuiEvent::CavemanModeToggled { level: normalized });
        }
        TuiAction::ToggleAutoEscalate => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };

            {
                let state = agent.state().await;
                if state.model.provider != Provider::DeepSeek {
                    let _ = event_tx.send(TuiEvent::Info(
                        "Auto-escalation only supported for DeepSeek models.".into(),
                    ));
                    return;
                }
            }

            let mut settings = crate::settings::load_settings().await;
            let new_state = !settings.auto_escalate;
            settings.auto_escalate = new_state;

            if new_state {
                let esc_id = settings
                    .escalation_model
                    .as_deref()
                    .unwrap_or("deepseek-v4-pro");
                let runtime_models = runtime_models_cell.read().await.clone();
                if let Some(esc_model) = runtime_models.iter().find(|m| m.id == esc_id) {
                    agent.set_escalation_model(Some(esc_model.clone())).await;
                } else if settings.escalation_model.is_none() {
                    settings.auto_escalate = false;
                    crate::settings::save_settings(&settings).await.ok();
                    let provider = {
                        let state = agent.state().await;
                        provider_to_string(state.model.provider)
                    };
                    let _ = event_tx.send(TuiEvent::ShowEscalationSelector { provider });
                    return;
                } else {
                    let _ = event_tx.send(TuiEvent::Error(format!(
                        "Escalation model '{esc_id}' not found."
                    )));
                    settings.auto_escalate = false;
                }
            } else {
                agent.set_escalation_model(None).await;
            }

            crate::settings::save_settings(&settings).await.ok();
            let _ = event_tx.send(TuiEvent::Info(if new_state {
                "Auto-escalation on.".into()
            } else {
                "Auto-escalation off.".into()
            }));
        }
        TuiAction::SetEscalationModel { model_id } => {
            let Some(agent) = agent_cell.read().await.clone() else {
                let _ = event_tx.send(TuiEvent::Error("Agent not ready".into()));
                return;
            };

            let runtime_models = runtime_models_cell.read().await.clone();
            if let Some(esc_model) = runtime_models.iter().find(|m| m.id == model_id) {
                if esc_model.provider != Provider::DeepSeek {
                    let _ = event_tx.send(TuiEvent::Error(format!(
                        "Escalation only supported for DeepSeek models (got {}).",
                        provider_to_string(esc_model.provider)
                    )));
                    return;
                }

                let mut settings = crate::settings::load_settings().await;
                let was_off = !settings.auto_escalate;
                settings.escalation_model = Some(model_id.clone());
                if was_off {
                    settings.auto_escalate = true;
                }
                crate::settings::save_settings(&settings).await.ok();

                agent.set_escalation_model(Some(esc_model.clone())).await;
                let msg = if was_off {
                    format!("Escalation model set to {model_id}. Auto-escalation enabled.")
                } else {
                    format!("Escalation model set to {model_id}.")
                };
                let _ = event_tx.send(TuiEvent::Info(msg));
            } else {
                let _ = event_tx.send(TuiEvent::Error(format!("Model '{model_id}' not found.")));
            }
        }
        TuiAction::ShowEscalationSelector => {
            let Some(agent) = agent_cell.read().await.clone() else {
                return;
            };
            let provider = {
                let state = agent.state().await;
                provider_to_string(state.model.provider)
            };
            let _ = event_tx.send(TuiEvent::ShowEscalationSelector { provider });
        }
    }
}

/// Normalize a caveman level string. Returns None for "off" or invalid values.
fn normalize_caveman_level(input: Option<&str>) -> Option<String> {
    match input
        .map(|s| s.to_lowercase().trim().to_string())
        .as_deref()
    {
        None | Some("off") => None,
        Some("lite") => Some("lite".into()),
        Some("full") => Some("full".into()),
        Some("ultra") => Some("ultra".into()),
        Some("wenyan-lite") => Some("wenyan-lite".into()),
        Some("wenyan-full") => Some("wenyan-full".into()),
        Some("wenyan-ultra") => Some("wenyan-ultra".into()),
        Some(other) => {
            tracing::warn!("unknown caveman level: {other}, ignoring");
            None
        }
    }
}

async fn available_model_entries(
    models: &[Model],
    auth: &mut crate::config::AuthConfig,
) -> Vec<ModelEntry> {
    let mut provider_has_auth: HashMap<String, bool> = HashMap::new();
    for provider in ["openai", "openai-codex", "deepseek", "opencode", "xiaomi"] {
        provider_has_auth.insert(
            provider.to_string(),
            auth.get_api_key(provider).await.is_some(),
        );
    }

    let settings = crate::settings::load_settings().await;
    let disabled: std::collections::HashSet<&str> = settings
        .disabled_models
        .iter()
        .map(|s| s.as_str())
        .collect();

    let mut entries: Vec<ModelEntry> = models
        .iter()
        .filter(|m| {
            if disabled.contains(m.id.as_str()) {
                return false;
            }
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

fn find_model_by_id(catalog: &[michin_ai::Model], id: &str) -> Option<michin_ai::Model> {
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
    catalog: &[michin_ai::Model],
    provider: &str,
    id: &str,
) -> Option<michin_ai::Model> {
    catalog
        .iter()
        .find(|m| provider_to_string(m.provider) == provider && m.id == id)
        .cloned()
}

async fn refresh_runtime_models(
    catalog: &BuiltInCatalog,
    runtime_models_cell: &Arc<RwLock<Vec<Model>>>,
    opencode_api_key: Option<&str>,
    mimo_api_key: Option<&str>,
) {
    let refreshed = resolve_runtime_models(catalog, opencode_api_key, mimo_api_key).await;
    *runtime_models_cell.write().await = refreshed;
}

async fn resolve_runtime_models(
    catalog: &BuiltInCatalog,
    opencode_api_key: Option<&str>,
    mimo_api_key: Option<&str>,
) -> Vec<Model> {
    let mut models: Vec<Model> = catalog.list().into_iter().cloned().collect();
    let fetched = opencode::fetch_models(opencode_api_key).await;
    if !fetched.is_empty() {
        models.retain(|m| m.provider != Provider::OpenCode);
        models.extend(fetched);
    }
    let fetched = xiaomi::fetch_models(mimo_api_key).await;
    if !fetched.is_empty() {
        models.retain(|m| m.provider != Provider::XiaomiMiMo);
        models.extend(fetched);
    }
    models
}

/// Build an ExtensionStatusPayload from Rhai row callbacks + legacy status lines.
/// Legacy status lines are mapped to row[0].left.
fn to_extension_payload(
    rows: Vec<michin_agent_core::types::ExtensionStatusRow>,
    lines: Vec<(String, String)>,
) -> michin_tui::app::ExtensionStatusPayload {
    // Count rows from tui.row() callbacks that have actual content.
    let extension_row_count = rows.iter().filter(|r| !r.is_empty()).count();

    let mut all_rows: Vec<michin_tui::components::status::StatusRow> = rows
        .into_iter()
        .map(|r| michin_tui::components::status::StatusRow {
            left: r.left,
            center: r.center,
            right: r.right,
        })
        .collect();

    // Merge legacy status lines into row[0].left
    if !lines.is_empty() {
        if all_rows.is_empty() {
            all_rows.push(michin_tui::components::status::StatusRow::default());
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

    michin_tui::app::ExtensionStatusPayload {
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
        Provider::XiaomiMiMo => "xiaomi".into(),
    }
}

fn provider_from_string(provider: &str) -> Option<Provider> {
    match provider {
        "openai" => Some(Provider::OpenAI),
        "openai-codex" => Some(Provider::OpenAiCodex),
        "deepseek" => Some(Provider::DeepSeek),
        "opencode" => Some(Provider::OpenCode),
        "opencode-go" => Some(Provider::OpenCodeGo),
        "xiaomi" => Some(Provider::XiaomiMiMo),
        _ => None,
    }
}

pub fn format_tool_summary(
    result: &michin_agent_core::types::ToolResult,
    max_chars: usize,
) -> String {
    let details = result.details.as_ref();
    let summary = match result.tool_name.as_str() {
        "read" => {
            if let Some(d) = details {
                let path = d
                    .get("path")
                    .and_then(|v| v.as_str())
                    .unwrap_or("(unknown)");
                let total_lines = d.get("total_lines").and_then(|v| v.as_u64()).unwrap_or(0);
                if total_lines == 0 {
                    format!("read {path}")
                } else {
                    let offset = d.get("offset").and_then(|v| v.as_u64()).unwrap_or(1);
                    let lines_read = d.get("lines_read").and_then(|v| v.as_u64()).unwrap_or(0);
                    format!(
                        "read {path}:{offset}-{end}",
                        end = offset.saturating_add(lines_read.saturating_sub(1))
                    )
                }
            } else {
                "read".to_string()
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
                    format!("edit {path}  {changes} change(s)")
                } else {
                    let (added, removed) = count_diff_lines(diff);
                    format!("edit {path}  [+{added}/-{removed}]")
                }
            } else {
                "edit".to_string()
            }
        }
        "bash" => {
            if let Some(d) = details {
                let exit = d
                    .get("exit_code")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "null".to_string());
                let base = if let Some(cmd) = d.get("command").and_then(|v| v.as_str()) {
                    format!("bash: {cmd}")
                } else {
                    "bash".to_string()
                };
                if result.is_error {
                    format!("{base} (exit={exit})")
                } else {
                    base
                }
            } else {
                "bash".to_string()
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
                "write".to_string()
            }
        }
        _ => summarize_custom_tool(result, max_chars),
    };
    truncate_chars(&summary, max_chars)
}

/// Summarize output from a custom tool (not read/edit/bash/write).
/// If the output is JSON, extract the most informative fields.
/// Otherwise truncate the raw text.
fn summarize_custom_tool(
    result: &michin_agent_core::types::ToolResult,
    max_chars: usize,
) -> String {
    let text = content_blocks_to_text(&result.content, usize::MAX);
    // Try to parse as JSON and extract key fields for a compact summary.
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
        if value.is_object() {
            return summarize_json_tool_result(&value, &result.tool_name, max_chars);
        } else if value.is_array() {
            // Array of objects (e.g. search results) — show count + first title.
            if let Some(arr) = value.as_array() {
                let count = arr.len();
                let first = arr.first().and_then(|v| {
                    v.get("title")
                        .or_else(|| v.get("url"))
                        .and_then(|t| t.as_str())
                        .map(|s| truncate_chars(s, 80))
                });
                return match first {
                    Some(f) => format!("{} ({count} results): {f}", result.tool_name),
                    None => format!("{}: {count} results", result.tool_name),
                };
            }
        }
    }
    // Not JSON or unparseable — truncate raw text.
    truncate_chars(&text, max_chars)
}

/// Summarize a JSON object tool result by extracting the most useful keys.
fn summarize_json_tool_result(
    value: &serde_json::Value,
    tool_name: &str,
    max_chars: usize,
) -> String {
    let obj = value.as_object().unwrap();
    // Priority fields: title, url, status, workdir (for fetch tools).
    let mut parts: Vec<String> = Vec::new();
    if let Some(t) = obj.get("title").and_then(|v| v.as_str())
        && !t.is_empty()
    {
        parts.push(format!("\"{}\"", truncate_chars(t, 60)));
    }
    if let Some(u) = obj.get("url").and_then(|v| v.as_str()) {
        parts.push(truncate_chars(u, 50));
    }
    if let Some(s) = obj.get("status") {
        parts.push(format!("status={}", s));
    }
    if let Some(err) = obj.get("error") {
        parts.push(format!("error={}", err));
    }
    if parts.is_empty() {
        // Fallback: use first non-trivial value.
        for (k, v) in obj.iter().take(3) {
            let s = v.to_string();
            if !s.is_empty() && s != "null" {
                parts.push(format!("{}={}", k, truncate_chars(&s, 40)));
            }
        }
    }
    let summary = format!("{}: {}", tool_name, parts.join(", "));
    truncate_chars(&summary, max_chars)
}

fn content_blocks_to_text(content: &[michin_ai::ContentBlock], max_chars: usize) -> String {
    let text = content
        .iter()
        .filter_map(|block| match block {
            michin_ai::ContentBlock::Text { text } => Some(text.as_str()),
            michin_ai::ContentBlock::Image { .. } => Some("[image]"),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n");
    truncate_chars(&text, max_chars)
}

/// Count added (+) and removed (-) lines in a unified diff.
/// Excludes the header lines (---, +++, @@).
fn count_diff_lines(diff: &str) -> (usize, usize) {
    let mut added = 0usize;
    let mut removed = 0usize;
    for line in diff.lines() {
        if line.starts_with('+') && !line.starts_with("+++") {
            added += 1;
        } else if line.starts_with('-') && !line.starts_with("---") {
            removed += 1;
        }
    }
    (added, removed)
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

pub fn expand_skill_message(message: &str, skills: &[crate::skills::Skill]) -> String {
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
            .map(michin_ai::Message::token_count)
            .sum()
    });
    let last_user = session
        .messages
        .iter()
        .rev()
        .find_map(|msg| match msg {
            michin_ai::Message::User { content, .. } => Some(content_blocks_to_text(content, 160)),
            _ => None,
        })
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "(none)".into());

    format!(
        "Resumed: {title}\nModel: {model}\nBranch: {branch}\nMessages: {messages}, approx tokens: {tokens}\nLast user message: {last_user}"
    )
}

/// Convert a session message to a history entry for display.
fn message_to_history_entries(msg: &michin_ai::Message) -> Vec<HistoryEntry> {
    match msg {
        michin_ai::Message::User { content, .. } => {
            let text = content
                .iter()
                .filter_map(|b| match b {
                    michin_ai::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            vec![HistoryEntry {
                role: "user".into(),
                text,
            }]
        }
        michin_ai::Message::Assistant { content, .. } => {
            let text = content
                .iter()
                .filter_map(|b| match b {
                    michin_ai::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            let thinking = content
                .iter()
                .filter_map(|b| match b {
                    michin_ai::ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
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
        michin_ai::Message::ToolResult {
            tool_name,
            details,
            is_error,
            ..
        } => {
            let status = if *is_error { " (failed)" } else { "" };
            let summary = match tool_name.as_str() {
                "read" => {
                    if let Some(d) = details {
                        let path = d
                            .get("path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("(unknown)");
                        let total_lines =
                            d.get("total_lines").and_then(|v| v.as_u64()).unwrap_or(0);
                        if total_lines == 0 {
                            format!("read {path}{status}")
                        } else {
                            let offset = d.get("offset").and_then(|v| v.as_u64()).unwrap_or(1);
                            let lines_read =
                                d.get("lines_read").and_then(|v| v.as_u64()).unwrap_or(0);
                            let end = offset.saturating_add(lines_read.saturating_sub(1));
                            format!("read {path}:{offset}-{end}{status}")
                        }
                    } else {
                        format!("read{status}")
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
                            format!("edit {path}  {changes} change(s){status}")
                        } else {
                            let (added, removed) = count_diff_lines(diff);
                            format!("edit {path}  [+{added}/-{removed}]{status}")
                        }
                    } else {
                        format!("edit{status}")
                    }
                }
                "bash" => {
                    if let Some(d) = details {
                        let exit = d
                            .get("exit_code")
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| "null".to_string());
                        let base = if let Some(cmd) = d.get("command").and_then(|v| v.as_str()) {
                            format!("bash: {cmd}")
                        } else {
                            "bash".to_string()
                        };
                        if *is_error {
                            format!("{base} (exit={exit})")
                        } else {
                            base
                        }
                    } else {
                        format!("bash{status}")
                    }
                }
                "write" => {
                    if let Some(d) = details {
                        let path = d
                            .get("path")
                            .and_then(|v| v.as_str())
                            .unwrap_or("(unknown)");
                        let bytes = d.get("bytes_written").and_then(|v| v.as_u64()).unwrap_or(0);
                        format!("write {path}\n{bytes} bytes{status}")
                    } else {
                        format!("write{status}")
                    }
                }
                _ => format!("{tool_name}{status}"),
            };
            vec![HistoryEntry {
                role: "tool".into(),
                text: summary,
            }]
        }
        _ => Vec::new(),
    }
}

/// Build the startup banner shown in chat at session start.
pub async fn build_startup_banner(working_dir: &Path) -> Option<String> {
    let art = "┌─────────────────┐\n│  /\\_/\\          │\n| ( o.o )         │\n│  > ^ <          │\n│  michin@dev:~$  │\n└─────────────────┘";

    let mut banner = String::with_capacity(2048);

    banner.push_str(art);
    banner.push_str("\n\n");

    // ── [Context] ──
    let context_files = build_context_files_list(working_dir).await;
    if !context_files.is_empty() {
        banner.push_str("[Context]\n");
        for f in &context_files {
            banner.push_str(&format!("  {f}\n"));
        }
        banner.push('\n');
    }

    // ── [Skills] ──
    let discovered = skills::discover_skills(working_dir).await;
    if !discovered.is_empty() {
        let names: Vec<&str> = discovered.iter().map(|s| s.name.as_str()).collect();
        banner.push_str("[Skills]\n");
        banner.push_str(&format!("  {}\n", names.join(", ")));
    }

    if banner.lines().all(|l| l.trim().is_empty()) {
        None
    } else {
        Some(banner)
    }
}

/// Build a compact list of context files discovered for the working directory.
async fn build_context_files_list(working_dir: &Path) -> Vec<String> {
    let mut files = Vec::new();

    // AGENTS.md and any nested AGENTS.md.
    if let Some(root) = find_context_file(working_dir, "AGENTS.md").await {
        files.push(root.display().to_string());

        let project_root = root.parent().unwrap_or(working_dir);
        let nested = discover_nested_agents(project_root).await;
        for (rel_path, _) in &nested {
            files.push(format!("  {rel_path}/AGENTS.md"));
        }
    }

    // .michin/context.md
    let ctx = working_dir.join(".michin").join("context.md");
    if ctx.exists() {
        files.push(ctx.display().to_string());
    }

    files
}
