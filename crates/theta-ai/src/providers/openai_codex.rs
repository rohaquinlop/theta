//! OpenAI Codex provider — ChatGPT Plus subscription authentication.
//!
//! ChatGPT Plus subscribers get access to Codex, which exposes models
//! through `https://chatgpt.com/backend-api`. Authentication uses the
//! ChatGPT session token (JWT) instead of an API key.
//!
//! ## Setup
//!
//! 1. Extract your ChatGPT session token from your browser:
//!    - Open chatgpt.com → DevTools → Application → Cookies
//!    - Find the `__Secure-next-auth.session-token` cookie
//!    - Or use the `/login` flow in theta to authenticate via OAuth
//!
//! 2. Set environment variable:
//!    ```bash
//!    export OPENAI_CODEX_TOKEN="<your-session-token>"
//!    ```
//!
//! 3. Use `--model codex-gpt-5.5` or set codex as default provider.

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde_json::Value;
use tracing;

use crate::error::ThetaError;
use crate::event::AssistantMessageEvent;
use crate::model::Model;
use crate::provider::{EventStream, Provider};
use crate::providers::openai_compat::{apply_thinking_params, convert_messages, parse_sse_line};
use crate::types::{ContentBlock, Context, SimpleStreamOptions, StopReason, StreamOptions};

/// The OpenAI Codex provider — uses ChatGPT Plus session tokens.
pub struct OpenAiCodexProvider {
    client: Client,
}

impl OpenAiCodexProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

impl Default for OpenAiCodexProvider {
    fn default() -> Self {
        Self::new()
    }
}

/// Environment variable for the ChatGPT Plus session token.
const CODEX_TOKEN_ENV: &str = "OPENAI_CODEX_TOKEN";

/// Fallback: also check the standard OpenAI API key env var,
/// since some setups may use the same key for both.
const FALLBACK_ENV: &str = "OPENAI_API_KEY";

#[async_trait]
impl Provider for OpenAiCodexProvider {
    async fn stream<'a>(
        &'a self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> Result<EventStream<'a>, ThetaError> {
        let token = get_codex_token().ok_or_else(|| ThetaError::MissingApiKey {
            provider: crate::types::Provider::OpenAiCodex,
        })?;

        let request_body = build_codex_request_body(model, context, options)?;
        let url = format!(
            "{}/v1/chat/completions",
            model.base_url.trim_end_matches('/')
        );

        tracing::debug!(
            "POST {} (codex) with {} messages",
            url,
            context.messages.len(),
        );

        let response = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            // 401 often means expired session token.
            if status.as_u16() == 401 {
                return Err(ThetaError::ApiError {
                    status: 401,
                    message: format!(
                        "ChatGPT session token expired or invalid. \
                         Re-extract your __Secure-next-auth.session-token \
                         cookie from chatgpt.com. Details: {}",
                        body
                    ),
                });
            }
            return Err(ThetaError::ApiError {
                status: status.as_u16(),
                message: body,
            });
        }

        let stream = response
            .bytes_stream()
            .map(|result| match result {
                Ok(bytes) => {
                    let line = String::from_utf8_lossy(&bytes).to_string();
                    parse_sse_line(&line)
                }
                Err(e) => Some(AssistantMessageEvent::Error {
                    code: "stream".into(),
                    message: e.to_string(),
                }),
            })
            .filter_map(|opt| async move { opt })
            .chain(futures::stream::once(async {
                AssistantMessageEvent::Done {
                    stop_reason: StopReason::Stop,
                    usage: None,
                }
            }));

        Ok(Box::pin(stream))
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
            include_usage: false,
            ..Default::default()
        };
        self.stream(model, context, &stream_opts).await
    }
}

/// Get the ChatGPT Plus session token from environment.
fn get_codex_token() -> Option<String> {
    std::env::var(CODEX_TOKEN_ENV)
        .ok()
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var(FALLBACK_ENV).ok().filter(|s| !s.is_empty()))
}

/// Build the request body for the codex API.
/// Uses the same OpenAI-compatible format but targets chatgpt.com.
fn build_codex_request_body(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
) -> Result<Value, ThetaError> {
    let mut body = serde_json::json!({
        "model": model.id,
        "stream": true,
    });

    // Messages (reuse the shared conversion).
    let messages = convert_messages(model, context);
    body["messages"] = messages;

    // System prompt
    if let Some(system_blocks) = &context.system {
        let system_text: String = system_blocks
            .iter()
            .filter_map(|b| match b {
                ContentBlock::Text { text } => Some(text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");

        if !system_text.is_empty() {
            let mut msgs = vec![serde_json::json!({
                "role": model.system_role(),
                "content": system_text,
            })];
            if let Some(existing) = body["messages"].as_array() {
                msgs.extend(existing.clone());
            }
            body["messages"] = Value::Array(msgs);
        }
    }

    // Tools
    if !context.tools.is_empty() {
        let tools: Vec<Value> = context
            .tools
            .iter()
            .map(|tool| {
                serde_json::json!({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                })
            })
            .collect();
        body["tools"] = Value::Array(tools);
    }

    // Max tokens
    if let Some(max_tokens) = options.max_tokens {
        body[model.max_tokens_field_name()] = serde_json::json!(max_tokens);
    }

    // Temperature
    if let Some(temp) = options.temperature {
        body["temperature"] = serde_json::json!(temp);
    }

    // Top-p
    if let Some(top_p) = options.top_p {
        body["top_p"] = serde_json::json!(top_p);
    }

    // Stop sequences
    if let Some(stop) = &options.stop {
        body["stop"] = serde_json::json!(stop);
    }

    // JSON mode
    if options.json_mode {
        body["response_format"] = serde_json::json!({"type": "json_object"});
    }

    // Thinking
    if let Some(level) = options.thinking_level {
        apply_thinking_params(&mut body, model, level);
    }

    Ok(body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_codex_token_not_set() {
        // Ensure test environment doesn't leak a real token.
        unsafe {
            std::env::remove_var(CODEX_TOKEN_ENV);
            std::env::remove_var(FALLBACK_ENV);
        }
        assert!(get_codex_token().is_none());
    }

    #[test]
    fn test_get_codex_token_from_codex_env() {
        unsafe {
            std::env::set_var(CODEX_TOKEN_ENV, "test-codex-token");
        }
        assert_eq!(get_codex_token(), Some("test-codex-token".into()));
        unsafe {
            std::env::remove_var(CODEX_TOKEN_ENV);
        }
    }

    #[test]
    fn test_get_codex_token_falls_back_to_openai_key() {
        unsafe {
            // Ensure codex env is not leaking from other tests
            std::env::remove_var(CODEX_TOKEN_ENV);
            std::env::set_var(FALLBACK_ENV, "fallback-key");
        }
        assert_eq!(get_codex_token(), Some("fallback-key".into()));
        unsafe {
            std::env::remove_var(FALLBACK_ENV);
        }
    }
}
