//! Print mode: non-interactive agent loop that streams events to stdout.

use std::path::Path;
use std::sync::Arc;

use michin_agent_core::agent::Agent;
use michin_agent_core::events::AgentEvent;
use michin_ai::ModelCatalog;
use michin_ai_net::default_registry;
use michin_models::BuiltInCatalog;
use tokio::sync::broadcast;

use crate::config::MichiNConfig;
use crate::session::SessionManager;
use crate::system_prompt::{
    SystemPromptConfig, build_active_overlays, build_resource_context, build_system_prompt,
};
use crate::tools::ToolContext;
use crate::tools::builtin_tools;

/// Run a prompt session in print mode.
pub async fn run_prompt_print_mode(
    config: &MichiNConfig,
    working_dir: &Path,
    model_id: &str,
    prompt: &str,
    session_id: &str,
) -> anyhow::Result<()> {
    let session_mgr = SessionManager::new(working_dir);
    let mut session = session_mgr.open_by_id(session_id).await?;

    // Resolve model + auth with provider fallback.
    let catalog = BuiltInCatalog::new();
    let available_models: Vec<michin_ai::Model> = catalog.list().into_iter().cloned().collect();
    let (model, api_key) = resolve_auth(config, &catalog, model_id).await?;

    // Provider registry.
    let registry = default_registry();
    registry.set_api_key(model.provider, &api_key);

    // Register tools.
    let tool_ctx = ToolContext::new(working_dir.to_path_buf());
    let mut agent = Agent::new(model.clone(), Arc::new(registry), available_models);
    agent.set_config(crate::config::to_agent_config(config));
    for tool in builtin_tools(tool_ctx, None) {
        agent.add_tool(tool).await;
    }

    // Load custom tools from ~/.michin/tools/*.rhai and ./.michin/tools/*.rhai.
    for tool in crate::scripts::load_custom_tools(working_dir).await {
        agent.add_tool(tool).await;
    }

    // Build and set the system prompt for prompt mode.
    let system_blocks = build_system_prompt(
        working_dir,
        &SystemPromptConfig {
            model_id,
            thinking_level: Some("medium"),
            max_context_window: Some(250_000),
        },
    )
    .await;
    agent.set_system_prompt(system_blocks).await;
    let resource_blocks = build_resource_context(working_dir).await;
    if !resource_blocks.is_empty() {
        agent.set_resource_context(resource_blocks).await;
    }
    agent
        .set_volatile_overlays(build_active_overlays(false, None))
        .await;

    let agent = Arc::new(agent);

    // Subscribe to events.
    let mut events = agent.subscribe();

    // Spawn the agent loop.
    let prompt_owned = prompt.to_string();
    let agent_for_spawn = agent.clone();
    let mention_working_dir = working_dir.to_path_buf();
    let agent_handle = tokio::spawn(async move {
        agent_for_spawn
            .prompt(
                crate::mentions::expand_file_mentions(&mention_working_dir, &prompt_owned).await,
            )
            .await
    });

    // Consume events.
    let mut aborted = false;
    let mut tool_errors = 0u32;
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
                    if result.is_error {
                        tool_errors += 1;
                    }
                    eprintln!("[tool:{}] {status}", result.tool_name);
                    let summary = format_tool_log_summary(&result);
                    if !summary.is_empty() {
                        eprintln!("{summary}");
                    }
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

    if tool_errors > 0 {
        anyhow::bail!("{tool_errors} tool error(s)");
    }

    Ok(())
}

fn provider_to_string(provider: michin_ai::Provider) -> String {
    match provider {
        michin_ai::Provider::OpenAI => "openai".into(),
        michin_ai::Provider::OpenAiCodex => "openai-codex".into(),
        michin_ai::Provider::DeepSeek => "deepseek".into(),
        michin_ai::Provider::OpenCode | michin_ai::Provider::OpenCodeGo => "opencode".into(),
        michin_ai::Provider::XiaomiMiMo => "xiaomi".into(),
    }
}

/// Resolve auth for a model with provider fallback.
/// If the model's provider has no token, try other providers.
async fn resolve_auth(
    config: &MichiNConfig,
    catalog: &BuiltInCatalog,
    model_id: &str,
) -> anyhow::Result<(michin_ai::Model, String)> {
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
        ("openai-codex", michin_ai::Provider::OpenAiCodex),
        ("openai", michin_ai::Provider::OpenAI),
        ("deepseek", michin_ai::Provider::DeepSeek),
        ("opencode", michin_ai::Provider::OpenCode),
        ("xiaomi", michin_ai::Provider::XiaomiMiMo),
    ];
    for (prov_str, prov) in &alt_providers {
        if prov_str == &provider_str {
            continue;
        }
        if let Some(key) = auth_config.get_api_key(prov_str).await
            && let Some(m) = catalog
                .list()
                .into_iter()
                .find(|m| m.provider == *prov && (m.id == model_id || m.id.starts_with(model_id)))
        {
            return Ok((m.clone(), key));
        }
    }

    anyhow::bail!("{}", crate::config::auth_error_message(&provider_str));
}

/// Continue the latest session in print mode.
pub async fn run_continue_print_mode(
    config: &MichiNConfig,
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
            michin_ai::Message::Assistant { model, .. } => model.clone(),
            _ => None,
        })
        .unwrap_or_else(|| model_id.to_string());

    let catalog = BuiltInCatalog::new();
    let available_models: Vec<michin_ai::Model> = catalog.list().into_iter().cloned().collect();
    let (model, api_key) = resolve_auth(config, &catalog, &effective_model).await?;

    // Provider registry.
    let registry = default_registry();
    registry.set_api_key(model.provider, &api_key);

    // Register tools.
    let tool_ctx = ToolContext::new(working_dir.to_path_buf());
    let mut agent = Agent::new(model.clone(), Arc::new(registry), available_models);
    agent.set_config(crate::config::to_agent_config(config));
    for tool in builtin_tools(tool_ctx, None) {
        agent.add_tool(tool).await;
    }
    for tool in crate::scripts::load_custom_tools(working_dir).await {
        agent.add_tool(tool).await;
    }

    // Build and set system prompt.
    let system_blocks = build_system_prompt(
        working_dir,
        &SystemPromptConfig {
            model_id: &effective_model,
            thinking_level: Some("medium"),
            max_context_window: Some(250_000),
        },
    )
    .await;
    agent.set_system_prompt(system_blocks).await;
    let resource_blocks = build_resource_context(working_dir).await;
    if !resource_blocks.is_empty() {
        agent.set_resource_context(resource_blocks).await;
    }
    agent
        .set_volatile_overlays(build_active_overlays(false, None))
        .await;

    // Load past messages into agent state.
    agent.load_messages(session.messages.clone()).await;

    let agent = Arc::new(agent);
    let mut events = agent.subscribe();

    // Spawn agent: either prompt with follow-up or continue_.
    let agent_for_spawn = agent.clone();
    let agent_handle = if let Some(text) = follow_up {
        let text = text.to_string();
        let mention_working_dir = working_dir.to_path_buf();
        tokio::spawn(async move {
            agent_for_spawn
                .prompt(crate::mentions::expand_file_mentions(&mention_working_dir, &text).await)
                .await
        })
    } else {
        tokio::spawn(async move { agent_for_spawn.continue_().await })
    };

    // Consume events.
    let mut aborted = false;
    let mut tool_errors = 0u32;
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
                    if result.is_error {
                        tool_errors += 1;
                    }
                    eprintln!("[tool:{}] {status}", result.tool_name);
                    let summary = format_tool_log_summary(&result);
                    if !summary.is_empty() {
                        eprintln!("{summary}");
                    }
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

    if tool_errors > 0 {
        anyhow::bail!("{tool_errors} tool error(s)");
    }

    Ok(())
}

/// Resume a specific session in print mode.
pub async fn run_resume_print_mode(
    config: &MichiNConfig,
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
                michin_ai::Message::Assistant { model, .. } => model.clone(),
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
    let available_models: Vec<michin_ai::Model> = catalog.list().into_iter().cloned().collect();
    let (model, api_key) = resolve_auth(config, &catalog, &effective_model).await?;

    let registry = default_registry();
    registry.set_api_key(model.provider, &api_key);

    let tool_ctx = ToolContext::new(working_dir.to_path_buf());
    let mut agent = Agent::new(model.clone(), Arc::new(registry), available_models);
    agent.set_config(crate::config::to_agent_config(config));
    for tool in builtin_tools(tool_ctx, None) {
        agent.add_tool(tool).await;
    }
    for tool in crate::scripts::load_custom_tools(working_dir).await {
        agent.add_tool(tool).await;
    }

    let system_blocks = build_system_prompt(
        working_dir,
        &SystemPromptConfig {
            model_id: &effective_model,
            thinking_level: Some("medium"),
            max_context_window: Some(250_000),
        },
    )
    .await;
    agent.set_system_prompt(system_blocks).await;
    let resource_blocks = build_resource_context(working_dir).await;
    if !resource_blocks.is_empty() {
        agent.set_resource_context(resource_blocks).await;
    }
    agent
        .set_volatile_overlays(build_active_overlays(false, None))
        .await;

    agent.load_messages(session.messages.clone()).await;

    let agent = Arc::new(agent);
    let mut events = agent.subscribe();

    let agent_for_spawn = agent.clone();
    let agent_handle = if let Some(text) = follow_up {
        let text = text.to_string();
        let mention_working_dir = working_dir.to_path_buf();
        tokio::spawn(async move {
            agent_for_spawn
                .prompt(crate::mentions::expand_file_mentions(&mention_working_dir, &text).await)
                .await
        })
    } else {
        tokio::spawn(async move { agent_for_spawn.continue_().await })
    };

    let mut aborted = false;
    let mut tool_errors = 0u32;
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
                    if result.is_error {
                        tool_errors += 1;
                    }
                    eprintln!("[tool:{}] {status}", result.tool_name);
                    let summary = format_tool_log_summary(&result);
                    if !summary.is_empty() {
                        eprintln!("{summary}");
                    }
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

    if tool_errors > 0 {
        anyhow::bail!("{tool_errors} tool error(s)");
    }

    Ok(())
}

fn format_tool_log_summary(result: &michin_agent_core::types::ToolResult) -> String {
    let Some(details) = result.details.as_ref() else {
        return String::new();
    };

    match result.tool_name.as_str() {
        "read" => {
            let path = details
                .get("path")
                .and_then(|v| v.as_str())
                .unwrap_or("(unknown)");
            let offset = details.get("offset").and_then(|v| v.as_u64()).unwrap_or(1);
            let lines_read = details
                .get("lines_read")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            format!(
                "  read {path}:{offset}-{end}",
                end = offset.saturating_add(lines_read.saturating_sub(1))
            )
        }
        "edit" => {
            let changes = details.get("changes").and_then(|v| v.as_u64()).unwrap_or(0);
            let diff = details.get("diff").and_then(|v| v.as_str()).unwrap_or("");
            if diff.is_empty() {
                format!("  {changes} change(s)")
            } else {
                format!("  {changes} change(s)\n{diff}")
            }
        }
        "bash" => {
            let exit = details
                .get("exit_code")
                .map(|v| v.to_string())
                .unwrap_or_else(|| "null".to_string());
            format!("  exit={exit}")
        }
        "write" => {
            let bytes = details
                .get("bytes_written")
                .and_then(|v| v.as_u64())
                .unwrap_or(0);
            format!("  {bytes} bytes")
        }
        _ => String::new(),
    }
}

/// Find a model by ID across all providers in the catalog.
fn find_model_by_id(catalog: &BuiltInCatalog, model_id: &str) -> Option<michin_ai::Model> {
    catalog
        .list()
        .into_iter()
        .find(|m| m.id == model_id)
        .cloned()
}
