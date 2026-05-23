//! JSON-RPC-ish stdin/stdout mode for editor integrations.

use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use theta_agent_core::agent::Agent;
use theta_agent_core::events::AgentEvent;
use theta_ai::ModelCatalog;
use theta_ai::providers::default_registry;
use theta_models::BuiltInCatalog;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::broadcast;

use crate::config::ThetaConfig;
use crate::session::SessionManager;
use crate::system_prompt::build_system_prompt;
use crate::tools::{ToolContext, builtin_tools};

#[derive(Debug, Deserialize)]
struct RpcRequest {
    id: serde_json::Value,
    method: String,
    #[serde(default)]
    params: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct RpcResponse {
    id: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    result: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

pub async fn run_rpc(config: &ThetaConfig, working_dir: &Path) -> anyhow::Result<()> {
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        let response = match serde_json::from_str::<RpcRequest>(&line) {
            Ok(request) => handle_request(request, config, working_dir).await,
            Err(e) => RpcResponse {
                id: serde_json::Value::Null,
                result: None,
                error: Some(format!("invalid request: {e}")),
            },
        };

        println!("{}", serde_json::to_string(&response)?);
        std::io::stdout().flush().ok();
    }

    Ok(())
}

async fn handle_request(
    request: RpcRequest,
    config: &ThetaConfig,
    working_dir: &Path,
) -> RpcResponse {
    let result = match request.method.as_str() {
        "ping" => Ok(serde_json::json!({"ok": true})),
        "sessions" => list_sessions(working_dir).await,
        "prompt" => prompt(request.params, config, working_dir).await,
        method => Err(anyhow::anyhow!("unknown method: {method}")),
    };

    match result {
        Ok(value) => RpcResponse {
            id: request.id,
            result: Some(value),
            error: None,
        },
        Err(e) => RpcResponse {
            id: request.id,
            result: None,
            error: Some(e.to_string()),
        },
    }
}

async fn list_sessions(working_dir: &Path) -> anyhow::Result<serde_json::Value> {
    let sessions = SessionManager::new(working_dir).list().await?;
    Ok(serde_json::to_value(sessions)?)
}

async fn prompt(
    params: serde_json::Value,
    config: &ThetaConfig,
    working_dir: &Path,
) -> anyhow::Result<serde_json::Value> {
    let text = params
        .get("text")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("missing params.text"))?;
    let model_id = params
        .get("model")
        .and_then(|v| v.as_str())
        .or(config.model.default.as_deref())
        .unwrap_or("gpt-5.5");
    let thinking = params
        .get("thinking")
        .and_then(|v| v.as_str())
        .or(config.thinking.default.as_deref())
        .unwrap_or("medium");

    let catalog = BuiltInCatalog::new();
    let (model, key) = resolve_auth_for_model(config, &catalog, model_id).await?;

    let registry = default_registry();
    registry.set_api_key(model.provider, key);

    let mut agent = Agent::new(model.clone(), Arc::new(registry), Arc::new(catalog));
    agent.set_config(crate::config::to_agent_config(config));
    for tool in builtin_tools(ToolContext::new(working_dir.to_path_buf())) {
        agent.add_tool(tool).await;
    }
    let system_blocks =
        build_system_prompt(working_dir, model_id, Some(thinking), Some(text)).await;
    agent.set_system_prompt(system_blocks).await;

    let agent = Arc::new(agent);
    let mut events = agent.subscribe();
    let agent_for_spawn = agent.clone();
    let text = text.to_string();
    let mention_working_dir = working_dir.to_path_buf();
    let handle = tokio::spawn(async move {
        agent_for_spawn
            .prompt(crate::mentions::expand_file_mentions(&mention_working_dir, &text).await)
            .await
    });

    let mut output = String::new();
    let mut tool_errors = 0u32;
    loop {
        match events.recv().await {
            Ok(AgentEvent::TextDelta { text }) => output.push_str(&text),
            Ok(AgentEvent::ToolExecutionEnd { result }) if result.is_error => tool_errors += 1,
            Ok(AgentEvent::AgentEnd { .. }) => break,
            Ok(_) => {}
            Err(broadcast::error::RecvError::Lagged(_)) => {}
            Err(broadcast::error::RecvError::Closed) => break,
        }
    }

    handle
        .await?
        .map_err(|e| anyhow::anyhow!("agent error: {e}"))?;
    let run_report = agent
        .last_run_report()
        .await
        .map(serde_json::to_value)
        .transpose()?;
    Ok(serde_json::json!({
        "text": output,
        "tool_errors": tool_errors,
        "run_report": run_report,
    }))
}

fn provider_to_string(provider: theta_ai::Provider) -> &'static str {
    match provider {
        theta_ai::Provider::OpenAI => "openai",
        theta_ai::Provider::OpenAiCodex => "openai-codex",
        theta_ai::Provider::DeepSeek => "deepseek",
        theta_ai::Provider::OpenCode | theta_ai::Provider::OpenCodeGo => "opencode",
    }
}

async fn resolve_auth_for_model(
    config: &ThetaConfig,
    catalog: &BuiltInCatalog,
    model_id: &str,
) -> anyhow::Result<(theta_ai::Model, String)> {
    let model = catalog
        .list()
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| anyhow::anyhow!("model not found: {model_id}"))?
        .clone();
    let provider = provider_to_string(model.provider);
    let mut auth = config.auth.clone();
    if let Some(key) = auth.get_api_key(provider).await {
        return Ok((model, key));
    }

    let alt_providers = [
        ("openai-codex", theta_ai::Provider::OpenAiCodex),
        ("openai", theta_ai::Provider::OpenAI),
        ("deepseek", theta_ai::Provider::DeepSeek),
        ("opencode", theta_ai::Provider::OpenCode),
    ];
    for (prov_str, prov) in &alt_providers {
        if *prov_str == provider {
            continue;
        }
        if let Some(key) = auth.get_api_key(prov_str).await
            && let Some(m) = catalog
                .list()
                .into_iter()
                .find(|m| m.provider == *prov && (m.id == model_id || m.id.starts_with(model_id)))
        {
            return Ok((m.clone(), key));
        }
    }

    Err(anyhow::anyhow!(
        "{}",
        crate::config::auth_error_message(provider)
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AuthConfig, ProviderToken, ThetaConfig};

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
