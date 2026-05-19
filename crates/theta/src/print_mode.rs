//! Print mode: non-interactive agent loop that streams events to stdout.

use std::path::Path;
use std::sync::Arc;

use theta_agent_core::agent::Agent;
use theta_agent_core::events::AgentEvent;
use theta_ai::providers::default_registry;
use theta_ai::{ContentBlock, ModelCatalog};
use theta_models::BuiltInCatalog;
use tokio::sync::broadcast;

use crate::config::ThetaConfig;
use crate::session::SessionManager;
use crate::system_prompt::build_system_prompt;
use crate::tools::ToolContext;
use crate::tools::builtin_tools;

/// Run a prompt session in print mode.
pub async fn run_prompt_print_mode(
    config: &ThetaConfig,
    working_dir: &Path,
    model_id: &str,
    prompt: &str,
    session_id: &str,
) -> anyhow::Result<()> {
    let session_mgr = SessionManager::new(working_dir);
    let mut session = session_mgr.open_by_id(session_id).await?;

    // Resolve model + auth with provider fallback.
    let catalog = BuiltInCatalog::new();
    let (model, api_key) = resolve_auth(config, &catalog, model_id).await?;

    // Provider registry.
    let mut registry = default_registry();
    registry.set_api_key(model.provider, &api_key);

    // Register tools.
    let tool_ctx = ToolContext::new(working_dir.to_path_buf());
    let mut agent = Agent::new(model.clone(), Arc::new(registry), Arc::new(catalog));
    agent.set_config(crate::config::to_agent_config(config));
    for tool in builtin_tools(tool_ctx) {
        agent.add_tool(tool).await;
    }

    // Build and set the system prompt for prompt mode.
    let system_blocks = build_system_prompt(working_dir, model_id, Some("medium")).await;
    agent.set_system_prompt(system_blocks).await;

    let agent = Arc::new(agent);

    // Subscribe to events.
    let mut events = agent.subscribe();

    // Spawn the agent loop.
    let prompt_owned = prompt.to_string();
    let agent_for_spawn = agent.clone();
    let agent_handle = tokio::spawn(async move {
        agent_for_spawn
            .prompt(vec![ContentBlock::Text {
                text: prompt_owned.clone(),
            }])
            .await
    });

    // Consume events.
    let mut aborted = false;
    loop {
        match events.recv().await {
            Ok(event) => match event {
                AgentEvent::TextDelta { text } => {
                    print!("{text}");
                }
                AgentEvent::ThinkingDelta { thinking } => {
                    eprintln!("[thinking] {thinking}");
                }
                AgentEvent::ToolCallStart { name, .. } => {
                    eprintln!("[tool:{name}] calling...");
                }
                AgentEvent::ToolExecutionStart { tool_name, .. } => {
                    eprintln!("[tool:{tool_name}] running...");
                }
                AgentEvent::ToolExecutionEnd { result } => {
                    let status = if result.is_error { "error" } else { "done" };
                    eprintln!("[tool:{}] {status}", result.tool_name);
                }
                AgentEvent::Error { message } => {
                    eprintln!("[error] {message}");
                }
                AgentEvent::AgentEnd { aborted: a } => {
                    aborted = a;
                    break;
                }
                _ => {}
            },
            Err(broadcast::error::RecvError::Lagged(n)) => {
                eprintln!("[warning] event lag: {n} messages skipped");
            }
            Err(broadcast::error::RecvError::Closed) => {
                break;
            }
        }
    }

    // Wait for agent to finish.
    let result = agent_handle.await?;

    if aborted {
        eprintln!("\n[aborted]");
    }

    result.map_err(|e| anyhow::anyhow!("agent error: {e}"))?;

    // Save conversation to session file.
    let state = agent.state().await;
    for msg in &state.messages {
        session_mgr.append_entry(&mut session, msg).await?;
    }

    Ok(())
}

fn provider_to_string(provider: theta_ai::Provider) -> String {
    match provider {
        theta_ai::Provider::OpenAI => "openai".into(),
        theta_ai::Provider::OpenAiCodex => "openai-codex".into(),
        theta_ai::Provider::DeepSeek => "deepseek".into(),
        theta_ai::Provider::OpenCode | theta_ai::Provider::OpenCodeGo => "opencode".into(),
    }
}

/// Resolve auth for a model with provider fallback.
/// If the model's provider has no token, try other providers.
async fn resolve_auth(
    config: &ThetaConfig,
    catalog: &BuiltInCatalog,
    model_id: &str,
) -> anyhow::Result<(theta_ai::Model, String)> {
    let model = find_model_by_id(catalog, model_id)
        .ok_or_else(|| anyhow::anyhow!("model not found: {model_id}"))?
        .clone();

    let provider_str = provider_to_string(model.provider);
    let mut auth_config = config.auth.clone();
    let api_key = auth_config.get_api_key(&provider_str).await;

    // If no auth for this provider, try others.
    if let Some(ref key) = api_key {
        return Ok((model, key.clone()));
    }

    let alt_providers = [
        ("openai-codex", theta_ai::Provider::OpenAiCodex),
        ("openai", theta_ai::Provider::OpenAI),
        ("deepseek", theta_ai::Provider::DeepSeek),
        ("opencode", theta_ai::Provider::OpenCode),
    ];
    for (prov_str, prov) in &alt_providers {
        if prov_str == &provider_str {
            continue;
        }
        if let Some(key) = auth_config.get_api_key(prov_str).await
            && let Some(m) = catalog.list().into_iter().find(|m| {
                m.provider == *prov
                    && (m.id == model_id || m.id.starts_with(model_id))
            })
        {
            return Ok((m.clone(), key));
        }
    }

    anyhow::bail!("{}", crate::config::auth_error_message(&provider_str));
}

/// Continue the latest session in print mode.
pub async fn run_continue_print_mode(
    config: &ThetaConfig,
    working_dir: &Path,
    model_id: &str,
    follow_up: Option<&str>,
) -> anyhow::Result<()> {
    let session_mgr = SessionManager::new(working_dir);
    let mut session = session_mgr.resume().await?;
    let _sid = session
        .meta
        .as_ref()
        .map(|m| m.id.clone())
        .unwrap_or_default();

    // Detect model from session's last assistant message, or use provided one.
    let effective_model = session
        .messages
        .iter()
        .rev()
        .find_map(|m| match m {
            theta_ai::Message::Assistant { model, .. } => model.clone(),
            _ => None,
        })
        .unwrap_or_else(|| model_id.to_string());

    let catalog = BuiltInCatalog::new();
    let (model, api_key) = resolve_auth(config, &catalog, &effective_model).await?;

    // Provider registry.
    let mut registry = default_registry();
    registry.set_api_key(model.provider, &api_key);

    // Register tools.
    let tool_ctx = ToolContext::new(working_dir.to_path_buf());
    let mut agent = Agent::new(model.clone(), Arc::new(registry), Arc::new(catalog));
    agent.set_config(crate::config::to_agent_config(config));
    for tool in builtin_tools(tool_ctx) {
        agent.add_tool(tool).await;
    }

    // Build and set system prompt.
    let system_blocks = build_system_prompt(working_dir, &effective_model, Some("medium")).await;
    agent.set_system_prompt(system_blocks).await;

    // Load past messages into agent state.
    agent.load_messages(session.messages.clone()).await;

    let agent = Arc::new(agent);
    let mut events = agent.subscribe();

    // Spawn agent: either prompt with follow-up or continue_.
    let agent_for_spawn = agent.clone();
    let agent_handle = if let Some(text) = follow_up {
        let text = text.to_string();
        tokio::spawn(async move {
            agent_for_spawn
                .prompt(vec![ContentBlock::Text { text }])
                .await
        })
    } else {
        tokio::spawn(async move { agent_for_spawn.continue_().await })
    };

    // Consume events.
    let mut aborted = false;
    loop {
        match events.recv().await {
            Ok(event) => match event {
                AgentEvent::TextDelta { text } => {
                    print!("{text}");
                }
                AgentEvent::ThinkingDelta { thinking } => {
                    eprintln!("[thinking] {thinking}");
                }
                AgentEvent::ToolCallStart { name, .. } => {
                    eprintln!("[tool:{name}] calling...");
                }
                AgentEvent::ToolExecutionEnd { result } => {
                    let status = if result.is_error { "error" } else { "done" };
                    eprintln!("[tool:{}] {status}", result.tool_name);
                }
                AgentEvent::Error { message } => {
                    eprintln!("[error] {message}");
                }
                AgentEvent::AgentEnd { aborted: a } => {
                    aborted = a;
                    break;
                }
                _ => {}
            },
            Err(broadcast::error::RecvError::Lagged(n)) => {
                eprintln!("[warning] event lag: {n} messages skipped");
            }
            Err(broadcast::error::RecvError::Closed) => {
                break;
            }
        }
    }

    let result = agent_handle.await?;
    if aborted {
        eprintln!("\n[aborted]");
    }
    result.map_err(|e| anyhow::anyhow!("agent error: {e}"))?;

    // Save new messages to session.
    let state = agent.state().await;
    let saved_count = session.messages.len();
    for msg in &state.messages[saved_count..] {
        session_mgr.append_entry(&mut session, msg).await?;
    }

    Ok(())
}

/// Resume a specific session in print mode.
pub async fn run_resume_print_mode(
    config: &ThetaConfig,
    working_dir: &Path,
    session_id: &str,
    follow_up: Option<&str>,
) -> anyhow::Result<()> {
    let session_mgr = SessionManager::new(working_dir);

    let sessions = session_mgr.list().await?;
    let meta = sessions
        .iter()
        .find(|m| m.id == session_id)
        .ok_or_else(|| anyhow::anyhow!("session not found: {session_id}"))?;

    let mut session = session_mgr.open_by_id(session_id).await?;

    // Detect model from session, or use config default.
    let effective_model = meta
        .model
        .clone()
        .or_else(|| {
            session.messages.iter().rev().find_map(|m| match m {
                theta_ai::Message::Assistant { model, .. } => model.clone(),
                _ => None,
            })
        })
        .unwrap_or_else(|| {
            config
                .model
                .default
                .clone()
                .unwrap_or_else(|| "gpt-5.5".into())
        });

    let catalog = BuiltInCatalog::new();
    let (model, api_key) = resolve_auth(config, &catalog, &effective_model).await?;

    let mut registry = default_registry();
    registry.set_api_key(model.provider, &api_key);

    let tool_ctx = ToolContext::new(working_dir.to_path_buf());
    let mut agent = Agent::new(model.clone(), Arc::new(registry), Arc::new(catalog));
    agent.set_config(crate::config::to_agent_config(config));
    for tool in builtin_tools(tool_ctx) {
        agent.add_tool(tool).await;
    }

    let system_blocks = build_system_prompt(working_dir, &effective_model, Some("medium")).await;
    agent.set_system_prompt(system_blocks).await;

    agent.load_messages(session.messages.clone()).await;

    let agent = Arc::new(agent);
    let mut events = agent.subscribe();

    let agent_for_spawn = agent.clone();
    let agent_handle = if let Some(text) = follow_up {
        let text = text.to_string();
        tokio::spawn(async move {
            agent_for_spawn
                .prompt(vec![ContentBlock::Text { text }])
                .await
        })
    } else {
        tokio::spawn(async move { agent_for_spawn.continue_().await })
    };

    let mut aborted = false;
    loop {
        match events.recv().await {
            Ok(event) => match event {
                AgentEvent::TextDelta { text } => {
                    print!("{text}");
                }
                AgentEvent::ThinkingDelta { thinking } => {
                    eprintln!("[thinking] {thinking}");
                }
                AgentEvent::ToolCallStart { name, .. } => {
                    eprintln!("[tool:{name}] calling...");
                }
                AgentEvent::ToolExecutionEnd { result } => {
                    let status = if result.is_error { "error" } else { "done" };
                    eprintln!("[tool:{}] {status}", result.tool_name);
                }
                AgentEvent::Error { message } => {
                    eprintln!("[error] {message}");
                }
                AgentEvent::AgentEnd { aborted: a } => {
                    aborted = a;
                    break;
                }
                _ => {}
            },
            Err(broadcast::error::RecvError::Lagged(n)) => {
                eprintln!("[warning] event lag: {n} messages skipped");
            }
            Err(broadcast::error::RecvError::Closed) => {
                break;
            }
        }
    }

    let result = agent_handle.await?;
    if aborted {
        eprintln!("\n[aborted]");
    }
    result.map_err(|e| anyhow::anyhow!("agent error: {e}"))?;

    let state = agent.state().await;
    let saved_count = session.messages.len();
    for msg in &state.messages[saved_count..] {
        session_mgr.append_entry(&mut session, msg).await?;
    }

    Ok(())
}

/// Find a model by ID across all providers in the catalog.
fn find_model_by_id(catalog: &BuiltInCatalog, model_id: &str) -> Option<theta_ai::Model> {
    catalog
        .list()
        .into_iter()
        .find(|m| m.id == model_id)
        .cloned()
}
