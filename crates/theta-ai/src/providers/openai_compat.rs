//! OpenAI-compatible provider implementation.
//!
//! Handles OpenAI, DeepSeek, and OpenCode through a single provider
//! with per-model compatibility flags. All three speak OpenAI's
//! `/v1/chat/completions` API with SSE streaming.

use async_trait::async_trait;
use futures::StreamExt;
use reqwest::Client;
use serde_json::{Value, json};
use tracing;

use crate::error::ThetaError;
use crate::event::AssistantMessageEvent;
use crate::model::Model;
use crate::provider::{EventStream, Provider};
use crate::types::{
    ContentBlock, Context, Message, SimpleStreamOptions, StopReason, StreamOptions, Usage,
};

/// The single OpenAI-compatible provider.
pub struct OpenAiCompatProvider {
    client: Client,
}

impl OpenAiCompatProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }
}

impl Default for OpenAiCompatProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for OpenAiCompatProvider {
    async fn stream<'a>(
        &'a self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> Result<EventStream<'a>, ThetaError> {
        let request_body = build_request_body(model, context, options, true)?;
        let url = format!(
            "{}/v1/chat/completions",
            model.base_url.trim_end_matches('/')
        );

        tracing::debug!(
            "POST {} with {} messages and {} tools",
            url,
            context.messages.len(),
            context.tools.len(),
        );

        let response = self
            .client
            .post(&url)
            .header(
                "Authorization",
                format!(
                    "Bearer {}",
                    std::env::var(api_key_env(model.provider)).unwrap_or_default()
                ),
            )
            .header("Content-Type", "application/json")
            .json(&request_body)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
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

/// Get the environment variable name for a provider's API key.
fn api_key_env(provider: crate::types::Provider) -> &'static str {
    match provider {
        crate::types::Provider::OpenAI => "OPENAI_API_KEY",
        crate::types::Provider::OpenAiCodex => "OPENAI_CODEX_TOKEN",
        crate::types::Provider::DeepSeek => "DEEPSEEK_API_KEY",
        crate::types::Provider::OpenCode => "OPENCODE_API_KEY",
        crate::types::Provider::OpenCodeGo => "OPENCODE_API_KEY",
    }
}

/// Build the OpenAI-compatible JSON request body.
fn build_request_body(
    model: &Model,
    context: &Context,
    options: &StreamOptions,
    include_tools: bool,
) -> Result<Value, ThetaError> {
    let mut body = json!({
        "model": model.id,
        "stream": true,
    });

    // Messages
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
            use serde_json::json;
            body["messages"] = {
                let mut msgs = vec![json!({
                    "role": model.system_role(),
                    "content": system_text,
                })];
                msgs.append(&mut body["messages"].as_array().cloned().unwrap_or_default());
                Value::Array(msgs)
            };
        }
    }

    // Tools
    if include_tools && !context.tools.is_empty() {
        let tools: Vec<Value> = context
            .tools
            .iter()
            .map(|tool| {
                json!({
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
        body[model.max_tokens_field_name()] = json!(max_tokens);
    }

    // Temperature
    if let Some(temp) = options.temperature {
        body["temperature"] = json!(temp);
    }

    // Top-p
    if let Some(top_p) = options.top_p {
        body["top_p"] = json!(top_p);
    }

    // Stop sequences
    if let Some(stop) = &options.stop {
        body["stop"] = json!(stop);
    }

    // Seed
    if let Some(seed) = options.seed {
        body["seed"] = json!(seed);
    }

    // JSON mode
    if options.json_mode {
        body["response_format"] = json!({"type": "json_object"});
    }

    // Streaming options (include usage)
    if model.compat.supports_usage_in_streaming {
        body["stream_options"] = json!({"include_usage": options.include_usage});
    }

    // Thinking / reasoning
    if let Some(level) = options.thinking_level {
        apply_thinking_params(&mut body, model, level);
    }

    Ok(body)
}

/// Apply thinking/reasoning parameters based on the model's thinking format.
pub fn apply_thinking_params(body: &mut Value, model: &Model, level: crate::types::ThinkingLevel) {
    let level_str = model.thinking_param(level);

    match model.compat.thinking_format {
        Some(crate::model::ThinkingFormat::DeepSeek) => {
            if let Some(s) = level_str {
                // DeepSeek: thinking { type: "enabled" } block
                body["thinking"] = json!({
                    "type": "enabled",
                    "reasoning_effort": s,
                });
            } else {
                // Explicitly disable thinking
                body["thinking"] = json!({"type": "disabled"});
            }
        }
        _ => {
            // OpenAI / OpenCode: reasoning_effort field
            if let Some(s) = level_str {
                body["reasoning_effort"] = json!(s);
            }
        }
    }
}

/// Convert our Message types to OpenAI-compatible message JSON.
pub fn convert_messages(model: &Model, context: &Context) -> Value {
    let messages: Vec<Value> = context
        .messages
        .iter()
        .filter_map(|msg| convert_message(model, msg))
        .collect();

    Value::Array(messages)
}

fn convert_message(model: &Model, msg: &Message) -> Option<Value> {
    match msg {
        Message::User { content, .. } => {
            let text = blocks_to_text(content);
            Some(json!({
                "role": "user",
                "content": text,
            }))
        }
        Message::Assistant {
            content,
            api: _,
            provider: _,
            model: _,
            usage: _,
            stop_reason: _,
            error_message: _,
            timestamp: _,
        } => {
            // Must return None instead of `Some(...)` when content is empty;
            // DeepSeek requires at least `reasoning_content: ""` on replayed msgs.
            let mut msg_json = json!({
                "role": "assistant",
                "content": Value::Null,
            });

            let mut has_content = false;

            // Add text blocks
            let text_parts: Vec<&str> = content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect();

            if !text_parts.is_empty() {
                msg_json["content"] = json!(text_parts.join("\n"));
                has_content = true;
            }

            // Add tool calls
            let tool_calls: Vec<Value> = content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::ToolCall {
                        id,
                        name,
                        arguments,
                    } => Some(json!({
                        "id": id,
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": serde_json::to_string(arguments).unwrap_or_default(),
                        }
                    })),
                    _ => None,
                })
                .collect();

            if !tool_calls.is_empty() {
                msg_json["tool_calls"] = json!(tool_calls);
                has_content = true;
            }

            // Add thinking content
            let thinking_blocks: Vec<&str> = content
                .iter()
                .filter_map(|b| match b {
                    ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
                    _ => None,
                })
                .collect();

            if !thinking_blocks.is_empty() {
                msg_json["reasoning_content"] = json!(thinking_blocks.join("\n"));
                has_content = true;
            } else if model.requires_reasoning_on_replay() {
                // DeepSeek requires empty reasoning_content on replayed assistant messages.
                msg_json["reasoning_content"] = json!("");
            }

            // If nothing added, return null content
            if !has_content && !model.requires_reasoning_on_replay() {
                msg_json["content"] = json!("");
            }

            Some(msg_json)
        }
        Message::ToolResult {
            tool_call_id,
            content,
            is_error,
            ..
        } => {
            let text = blocks_to_text(content);
            Some(json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": text,
                "is_error": is_error,
            }))
        }
        // Skip model/thinking change events — not sent to LLM.
        Message::ModelChange { .. } | Message::ThinkingLevelChange { .. } => None,
    }
}

/// Extract text from content blocks.
fn blocks_to_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Parse one SSE data line into an AssistantMessageEvent (or None for comments/empty).
pub fn parse_sse_line(line: &str) -> Option<AssistantMessageEvent> {
    // SSE format: "data: <json>\n\n"
    let line = line.trim();

    // Skip empty lines and comments.
    if line.is_empty() || line.starts_with(':') {
        return None;
    }

    let data = line.strip_prefix("data: ")?;
    if data == "[DONE]" {
        return None; // done handled by stream end
    }

    match serde_json::from_str::<Value>(data) {
        Ok(chunk) => parse_chunk(&chunk),
        Err(e) => {
            tracing::warn!("Failed to parse SSE data: {} — {}", data, e);
            None
        }
    }
}

/// Parse a single SSE JSON chunk into events.
fn parse_chunk(chunk: &Value) -> Option<AssistantMessageEvent> {
    // Check for top-level error
    if let Some(error) = chunk.get("error") {
        return Some(AssistantMessageEvent::Error {
            code: error
                .get("code")
                .and_then(|c| c.as_str())
                .unwrap_or("unknown")
                .into(),
            message: error
                .get("message")
                .and_then(|m| m.as_str())
                .unwrap_or("unknown error")
                .into(),
        });
    }

    // Check for usage info
    if let Some(usage) = chunk.get("usage") {
        return Some(AssistantMessageEvent::Usage {
            usage: parse_usage(usage),
        });
    }

    let choices = chunk.get("choices")?.as_array()?;
    let choice = choices.first()?;
    let delta = choice.get("delta")?;

    // Check finish reason
    let finish_reason = choice.get("finish_reason").and_then(|r| r.as_str());

    // Tool call delta
    if let Some(tool_calls) = delta.get("tool_calls")
        && let Some(tool_call_array) = tool_calls.as_array()
    {
        for tc_delta in tool_call_array {
            let _index = tc_delta.get("index").and_then(|i| i.as_u64()).unwrap_or(0);
            let id = tc_delta.get("id").and_then(|i| i.as_str()).unwrap_or("");
            let function = tc_delta.get("function")?;

            // New tool call starting
            if !id.is_empty() {
                let name = function.get("name").and_then(|n| n.as_str()).unwrap_or("");
                if !name.is_empty() {
                    return Some(AssistantMessageEvent::ToolCallStart {
                        id: id.to_string(),
                        name: name.to_string(),
                    });
                }
            }

            // Tool call arguments delta
            if let Some(args) = function.get("arguments").and_then(|a| a.as_str()) {
                return Some(AssistantMessageEvent::ToolCallDelta {
                    id: id.to_string(),
                    arguments: args.to_string(),
                });
            }
        }
    }

    // Reasoning/thinking content (DeepSeek and o-series)
    if let Some(reasoning) = delta.get("reasoning_content")
        && let Some(text) = reasoning.as_str()
        && !text.is_empty()
    {
        return Some(AssistantMessageEvent::ThinkingDelta {
            thinking: text.to_string(),
        });
    }

    // Regular text content
    if let Some(content) = delta.get("content")
        && let Some(text) = content.as_str()
        && !text.is_empty()
    {
        return Some(AssistantMessageEvent::TextDelta {
            text: text.to_string(),
        });
    }

    // Handle finish reason
    if let Some(reason) = finish_reason {
        if reason == "tool_calls" {
            return Some(AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            });
        }
        if reason == "length" {
            return Some(AssistantMessageEvent::Done {
                stop_reason: StopReason::Length,
                usage: None,
            });
        }
        if reason == "stop" {
            return Some(AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            });
        }
    }

    None
}

/// Parse a usage object from the stream.
fn parse_usage(usage: &Value) -> Usage {
    Usage {
        input_tokens: usage
            .get("prompt_tokens")
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32,
        output_tokens: usage
            .get("completion_tokens")
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32,
        cache_write_tokens: usage
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cache_creation_tokens"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32,
        cache_read_tokens: usage
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cache_read_tokens"))
            .and_then(|t| t.as_u64())
            .unwrap_or(0) as u32,
        cost: None, // calculated later
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_env() {
        assert_eq!(
            api_key_env(crate::types::Provider::OpenAI),
            "OPENAI_API_KEY"
        );
        assert_eq!(
            api_key_env(crate::types::Provider::DeepSeek),
            "DEEPSEEK_API_KEY"
        );
        assert_eq!(
            api_key_env(crate::types::Provider::OpenCode),
            "OPENCODE_API_KEY"
        );
    }

    #[test]
    fn test_parse_sse_empty() {
        assert!(parse_sse_line("").is_none());
        assert!(parse_sse_line(": heartbeat").is_none());
        assert!(parse_sse_line("data: [DONE]").is_none());
    }

    #[test]
    fn test_parse_text_delta() {
        let event =
            parse_sse_line(r#"data: {"choices":[{"delta":{"content":"Hello"},"index":0}]}"#);
        assert!(event.is_some());
        if let Some(AssistantMessageEvent::TextDelta { text }) = event {
            assert_eq!(text, "Hello");
        } else {
            panic!("Expected TextDelta");
        }
    }

    #[test]
    fn test_parse_thinking_delta() {
        let event = parse_sse_line(
            r#"data: {"choices":[{"delta":{"reasoning_content":"Let me think..."},"index":0}]}"#,
        );
        assert!(event.is_some());
        if let Some(AssistantMessageEvent::ThinkingDelta { thinking }) = event {
            assert_eq!(thinking, "Let me think...");
        } else {
            panic!("Expected ThinkingDelta");
        }
    }

    #[test]
    fn test_parse_tool_call_start() {
        let event = parse_sse_line(
            r#"data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"read","arguments":""}}]},"index":0}]}"#,
        );
        assert!(event.is_some());
        if let Some(AssistantMessageEvent::ToolCallStart { id, name }) = event {
            assert_eq!(id, "call_123");
            assert_eq!(name, "read");
        } else {
            panic!("Expected ToolCallStart");
        }
    }

    #[test]
    fn test_parse_finish_tool_calls() {
        let event = parse_sse_line(
            r#"data: {"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
        );
        assert!(event.is_some());
        if let Some(AssistantMessageEvent::Done { stop_reason, .. }) = event {
            assert_eq!(stop_reason, StopReason::ToolUse);
        } else {
            panic!("Expected Done with ToolUse");
        }
    }

    #[test]
    fn test_parse_error() {
        let event = parse_sse_line(
            r#"data: {"error":{"code":"rate_limit","message":"Too many requests"}}"#,
        );
        assert!(event.is_some());
        if let Some(AssistantMessageEvent::Error { code, message }) = event {
            assert_eq!(code, "rate_limit");
            assert_eq!(message, "Too many requests");
        } else {
            panic!("Expected Error event");
        }
    }
}
