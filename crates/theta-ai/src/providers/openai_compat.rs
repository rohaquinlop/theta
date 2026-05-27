//! OpenAI-compatible provider implementation.
//!
//! Handles OpenAI, DeepSeek, and OpenCode through a single provider
//! with per-model compatibility flags. All three speak OpenAI's
//! `/v1/chat/completions` API with SSE streaming.

use std::sync::RwLock;

use async_trait::async_trait;
use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::Client;
use serde_json::{Value, json};
use tracing;

use crate::error::ThetaError;
use crate::event::AssistantMessageEvent;
use crate::model::Model;
use crate::provider::{EventStream, Provider};
#[cfg(test)]
use crate::replay::sanitize_messages_for_replay;
use crate::types::{
    ContentBlock, Context, Message, SimpleStreamOptions, StopReason, StreamOptions, Usage,
};

/// The single OpenAI-compatible provider.
pub struct OpenAiCompatProvider {
    client: Client,
    api_key: RwLock<Option<String>>,
}

impl OpenAiCompatProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_key: RwLock::new(None),
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

        let api_key = self
            .api_key
            .read()
            .ok()
            .and_then(|key| key.clone())
            .or_else(|| std::env::var(api_key_env(model.provider)).ok())
            .ok_or(ThetaError::MissingApiKey {
                provider: model.provider,
            })?;

        let mut request = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {api_key}"))
            .header("Content-Type", "application/json")
            .json(&request_body);

        if let Some(timeout_ms) = options.timeout_ms {
            request = request.timeout(std::time::Duration::from_millis(timeout_ms));
        }

        let response = request.send().await?;

        let status = response.status();
        if !status.is_success() {
            let retry_ms = response
                .headers()
                .get("retry-after-ms")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.parse().ok());
            let body = response.text().await.unwrap_or_default();
            return Err(ThetaError::ApiError {
                status: status.as_u16(),
                message: body,
                retry_after_ms: retry_ms,
            });
        }

        let mut parser = OpenAiCompatStreamParser::new();
        let stream = response
            .bytes_stream()
            .eventsource()
            .map(move |result| match result {
                Ok(event) => parser.parse_data(&event.data),
                Err(e) => vec![AssistantMessageEvent::Error {
                    code: "stream".into(),
                    message: e.to_string(),
                }],
            })
            .flat_map(futures::stream::iter);

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

    fn set_token(&self, token: &str) {
        if let Ok(mut api_key) = self.api_key.write() {
            *api_key = Some(token.to_string());
        }
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

    // Messages — context.messages are already sanitized by build_context().
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
    } else if include_tools && has_tool_history(&context.messages) {
        body["tools"] = Value::Array(vec![]);
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
            } else if level == crate::types::ThinkingLevel::Off {
                // Explicitly disable thinking
                body["thinking"] = json!({"type": "disabled"});
            }
            // Unmapped non-Off levels (minimal/low/medium on DeepSeek):
            // do nothing — let model use its default behavior.
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
    let mut messages: Vec<Value> = Vec::new();
    let mut last_role_was_tool = false;

    for msg in &context.messages {
        if model.compat.requires_assistant_after_tool_result
            && last_role_was_tool
            && matches!(msg, Message::User { .. })
        {
            messages.push(json!({
                "role": "assistant",
                "content": "I have processed the tool results.",
            }));
            last_role_was_tool = false;
        }

        if let Some(converted) = convert_message(model, msg) {
            last_role_was_tool = converted.get("role").and_then(|r| r.as_str()) == Some("tool");
            messages.push(converted);
        }
    }

    Value::Array(messages)
}

fn has_tool_history(messages: &[Message]) -> bool {
    messages.iter().any(|msg| match msg {
        Message::ToolResult { .. } => true,
        Message::Assistant { content, .. } => content
            .iter()
            .any(|b| matches!(b, ContentBlock::ToolCall { .. })),
        _ => false,
    })
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
                "content": "",
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
                msg_json["content"] = json!(text_parts.join(""));
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
                msg_json["reasoning_content"] = json!(thinking_blocks.join("\n\n"));
                has_content = true;
            } else if model.requires_reasoning_on_replay() {
                // DeepSeek requires empty reasoning_content on replayed assistant messages.
                msg_json["reasoning_content"] = json!("");
            }

            // Skip empty assistant replay messages unless provider explicitly
            // requires reasoning_content to be present.
            if !has_content
                && msg_json.get("tool_calls").is_none()
                && !model.requires_reasoning_on_replay()
            {
                return None;
            }

            Some(msg_json)
        }
        Message::ToolResult {
            tool_call_id,
            content,
            is_error: _,
            ..
        } => {
            let text = blocks_to_text(content);
            Some(json!({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": text,
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
#[cfg(test)]
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
        Ok(chunk) => OpenAiCompatStreamParser::new()
            .parse_chunk(&chunk)
            .into_iter()
            .next(),
        Err(e) => {
            tracing::warn!("Failed to parse SSE data: {} — {}", data, e);
            None
        }
    }
}

#[derive(Debug, Default)]
struct OpenAiCompatStreamParser {
    tool_calls: Vec<OpenAiToolCallState>,
    /// Tracks whether ThinkingStart has been emitted for the current
    /// reasoning_content stream. Reset on ThinkingEnd or finish reason.
    thinking_started: bool,
}

#[derive(Debug, Default)]
struct OpenAiToolCallState {
    index: usize,
    id: String,
    name: String,
    arguments: String,
    emitted_start: bool,
    emitted_end: bool,
}

fn upsert_tool_call_state_by_id_or_index(
    parser: &mut OpenAiCompatStreamParser,
    index: usize,
    id: Option<&str>,
) -> usize {
    if let Some(non_empty_id) = id.filter(|v| !v.is_empty())
        && let Some(existing_idx) = parser
            .tool_calls
            .iter()
            .position(|tc| !tc.id.is_empty() && tc.id == non_empty_id)
    {
        parser.tool_calls[existing_idx].index = index;
        return existing_idx;
    }

    parser
        .tool_calls
        .iter()
        .position(|tc| tc.index == index)
        .unwrap_or_else(|| {
            parser.tool_calls.push(OpenAiToolCallState {
                index,
                ..Default::default()
            });
            parser.tool_calls.len() - 1
        })
}

impl OpenAiCompatStreamParser {
    fn new() -> Self {
        Self::default()
    }

    fn parse_data(&mut self, data: &str) -> Vec<AssistantMessageEvent> {
        if data.trim().is_empty() || data == "[DONE]" {
            return Vec::new();
        }

        match serde_json::from_str::<Value>(data) {
            Ok(chunk) => self.parse_chunk(&chunk),
            Err(e) => {
                tracing::warn!("Failed to parse SSE data: {} — {}", data, e);
                Vec::new()
            }
        }
    }

    /// Parse a single SSE JSON chunk into zero or more events.
    fn parse_chunk(&mut self, chunk: &Value) -> Vec<AssistantMessageEvent> {
        parse_chunk(self, chunk)
    }
}

/// Parse a single SSE JSON chunk into events.
fn parse_chunk(parser: &mut OpenAiCompatStreamParser, chunk: &Value) -> Vec<AssistantMessageEvent> {
    let mut events = Vec::new();

    // Check for top-level error
    if let Some(error) = chunk.get("error") {
        events.push(AssistantMessageEvent::Error {
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
        return events;
    }

    // Check for usage info
    if let Some(usage) = chunk.get("usage") {
        events.push(AssistantMessageEvent::Usage {
            usage: parse_usage(usage),
        });
    }

    let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) else {
        return events;
    };
    let Some(choice) = choices.first() else {
        return events;
    };

    // Check finish reason
    let finish_reason = choice.get("finish_reason").and_then(|r| r.as_str());

    let delta = choice.get("delta");

    // Tool call delta
    if let Some(tool_calls) = delta.and_then(|d| d.get("tool_calls"))
        && let Some(tool_call_array) = tool_calls.as_array()
    {
        for tc_delta in tool_call_array {
            let index = tc_delta.get("index").and_then(|i| i.as_u64()).unwrap_or(0) as usize;
            let id = tc_delta.get("id").and_then(|i| i.as_str());
            let function = tc_delta.get("function");
            let name = function
                .and_then(|f| f.get("name"))
                .and_then(|n| n.as_str());

            let state_idx = upsert_tool_call_state_by_id_or_index(parser, index, id);

            let state = &mut parser.tool_calls[state_idx];
            if !state.emitted_start
                && let Some(id) = id
                && !id.is_empty()
            {
                state.id = id.to_string();
            }
            if !state.emitted_start
                && let Some(name) = name
                && !name.is_empty()
            {
                state.name = name.to_string();
            }

            if !state.emitted_start && !state.id.is_empty() && !state.name.is_empty() {
                state.emitted_start = true;
                events.push(AssistantMessageEvent::ToolCallStart {
                    id: state.id.clone(),
                    name: state.name.clone(),
                });
                if !state.arguments.is_empty() {
                    events.push(AssistantMessageEvent::ToolCallDelta {
                        id: state.id.clone(),
                        arguments: state.arguments.clone(),
                    });
                }
            }

            if let Some(args) = function
                .and_then(|f| f.get("arguments"))
                .and_then(|a| a.as_str())
                && !args.is_empty()
            {
                state.arguments.push_str(args);
                if state.emitted_start {
                    events.push(AssistantMessageEvent::ToolCallDelta {
                        id: state.id.clone(),
                        arguments: args.to_string(),
                    });
                }
            }
        }
    }

    // Legacy function_call delta (older OpenAI-compatible providers).
    if let Some(function_call) = delta.and_then(|d| d.get("function_call")) {
        let id = choice
            .get("message")
            .and_then(|m| m.get("tool_call_id"))
            .and_then(|v| v.as_str())
            .or_else(|| {
                choice
                    .get("message")
                    .and_then(|m| m.get("id"))
                    .and_then(|v| v.as_str())
            })
            .or_else(|| choice.get("id").and_then(|v| v.as_str()));
        let state_idx = upsert_tool_call_state_by_id_or_index(parser, 0, id);
        let state = &mut parser.tool_calls[state_idx];

        if !state.emitted_start
            && let Some(id) = id
            && !id.is_empty()
        {
            state.id = id.to_string();
        }
        if !state.emitted_start
            && let Some(name) = function_call.get("name").and_then(|n| n.as_str())
            && !name.is_empty()
        {
            state.name = name.to_string();
        }

        if !state.emitted_start && !state.id.is_empty() && !state.name.is_empty() {
            state.emitted_start = true;
            events.push(AssistantMessageEvent::ToolCallStart {
                id: state.id.clone(),
                name: state.name.clone(),
            });
            if !state.arguments.is_empty() {
                events.push(AssistantMessageEvent::ToolCallDelta {
                    id: state.id.clone(),
                    arguments: state.arguments.clone(),
                });
            }
        }

        if let Some(args) = function_call.get("arguments").and_then(|a| a.as_str())
            && !args.is_empty()
        {
            state.arguments.push_str(args);
            if state.emitted_start {
                events.push(AssistantMessageEvent::ToolCallDelta {
                    id: state.id.clone(),
                    arguments: args.to_string(),
                });
            }
        }
    }

    // Reasoning/thinking content (DeepSeek and o-series)
    if let Some(reasoning) = delta.and_then(|d| d.get("reasoning_content"))
        && let Some(text) = reasoning.as_str()
        && !text.is_empty()
    {
        if !parser.thinking_started {
            parser.thinking_started = true;
            events.push(AssistantMessageEvent::ThinkingStart);
        }
        events.push(AssistantMessageEvent::ThinkingDelta {
            thinking: text.to_string(),
        });
    }

    // Transition from reasoning to text: emit ThinkingEnd.
    if parser.thinking_started
        && let Some(content) = delta.and_then(|d| d.get("content"))
        && let Some(text) = content.as_str()
        && !text.is_empty()
    {
        parser.thinking_started = false;
        events.push(AssistantMessageEvent::ThinkingEnd);
    }

    // Regular text content
    if let Some(content) = delta.and_then(|d| d.get("content"))
        && let Some(text) = content.as_str()
        && !text.is_empty()
    {
        events.push(AssistantMessageEvent::TextDelta {
            text: text.to_string(),
        });
    }

    // Handle finish reason
    if let Some(reason) = finish_reason {
        if reason == "tool_calls" || reason == "function_call" {
            for state in &mut parser.tool_calls {
                if state.id.is_empty() {
                    state.id = format!("tool_call_{}", state.index);
                }
                if !state.emitted_start && !state.id.is_empty() {
                    state.emitted_start = true;
                    events.push(AssistantMessageEvent::ToolCallStart {
                        id: state.id.clone(),
                        name: if state.name.is_empty() {
                            "unknown".to_string()
                        } else {
                            state.name.clone()
                        },
                    });
                    if !state.arguments.is_empty() {
                        events.push(AssistantMessageEvent::ToolCallDelta {
                            id: state.id.clone(),
                            arguments: state.arguments.clone(),
                        });
                    }
                }
                if state.emitted_start && !state.emitted_end {
                    state.emitted_end = true;
                    events.push(AssistantMessageEvent::ToolCallEnd {
                        id: state.id.clone(),
                    });
                }
            }
        }

        // If thinking was in progress, end it before finish reason.
        if parser.thinking_started {
            parser.thinking_started = false;
            events.push(AssistantMessageEvent::ThinkingEnd);
        }

        if reason == "tool_calls" {
            events.push(AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            });
        } else if reason == "length" {
            events.push(AssistantMessageEvent::Done {
                stop_reason: StopReason::Length,
                usage: None,
            });
        } else if reason == "content_filter" || reason == "insufficient_system_resource" {
            events.push(AssistantMessageEvent::Done {
                stop_reason: StopReason::Error,
                usage: None,
            });
        } else if reason == "stop" {
            events.push(AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            });
        } else if reason == "function_call" {
            // Backward-compat finish reason used by some OpenAI-compatible providers.
            events.push(AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                usage: None,
            });
        }
    }

    events
}

/// Parse a usage object from the stream.
///
/// Handles two response formats:
/// - DeepSeek: `usage.prompt_cache_hit_tokens` / `usage.prompt_cache_miss_tokens` (top-level)
/// - OpenAI / OpenCode: `usage.prompt_tokens_details.cached_tokens` / `.cache_creation_tokens`
fn parse_usage(usage: &Value) -> Usage {
    let input_tokens = usage
        .get("prompt_tokens")
        .and_then(|t| t.as_u64())
        .unwrap_or(0) as u32;
    let output_tokens = usage
        .get("completion_tokens")
        .and_then(|t| t.as_u64())
        .unwrap_or(0) as u32;

    // Try DeepSeek top-level fields first, then fall back to OpenAI nested format.
    let cache_hit = usage
        .get("prompt_cache_hit_tokens")
        .and_then(|t| t.as_u64())
        .or_else(|| {
            usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cached_tokens"))
                .and_then(|t| t.as_u64())
        })
        .or_else(|| {
            usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cache_read_tokens"))
                .and_then(|t| t.as_u64())
        })
        .unwrap_or(0) as u32;

    let cache_miss_or_write = usage
        .get("prompt_cache_miss_tokens")
        .and_then(|t| t.as_u64())
        .or_else(|| {
            usage
                .get("prompt_tokens_details")
                .and_then(|d| d.get("cache_creation_tokens"))
                .and_then(|t| t.as_u64())
        })
        .unwrap_or(0) as u32;

    Usage {
        input_tokens,
        output_tokens,
        cache_write_tokens: cache_miss_or_write,
        cache_read_tokens: cache_hit,
        cost: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::EventAccumulator;

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
        let mut parser = OpenAiCompatStreamParser::new();
        let events = parser.parse_data(
            r#"{"choices":[{"delta":{"reasoning_content":"Let me think..."},"index":0}]}"#,
        );
        assert_eq!(events.len(), 2, "expected ThinkingStart + ThinkingDelta");
        assert!(
            matches!(&events[0], AssistantMessageEvent::ThinkingStart),
            "first event should be ThinkingStart"
        );
        if let AssistantMessageEvent::ThinkingDelta { thinking } = &events[1] {
            assert_eq!(thinking, "Let me think...");
        } else {
            panic!("Expected ThinkingDelta as second event");
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

    #[test]
    fn test_parse_streamed_tool_call_arguments_by_index() {
        let mut parser = OpenAiCompatStreamParser::new();
        let mut accumulator = EventAccumulator::new();

        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_123","type":"function","function":{"name":"read","arguments":""}}]},"index":0}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\""}}]},"index":0}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\"Cargo.toml\"}"}}]},"index":0}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
        ];

        let events: Vec<AssistantMessageEvent> = chunks
            .iter()
            .flat_map(|chunk| parser.parse_data(chunk))
            .collect();

        assert!(matches!(
            events.first(),
            Some(AssistantMessageEvent::ToolCallStart { id, name })
                if id == "call_123" && name == "read"
        ));
        assert!(matches!(
            events.iter().rev().nth(1),
            Some(AssistantMessageEvent::ToolCallEnd { id }) if id == "call_123"
        ));
        assert!(matches!(
            events.last(),
            Some(AssistantMessageEvent::Done {
                stop_reason: StopReason::ToolUse,
                ..
            })
        ));

        for event in &events {
            accumulator.feed(event);
        }

        let blocks = accumulator.content_blocks();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "call_123");
                assert_eq!(name, "read");
                assert_eq!(arguments["path"], "Cargo.toml");
            }
            other => panic!("expected tool call, got {other:?}"),
        }
        assert_eq!(accumulator.stop_reason(), Some(StopReason::ToolUse));
    }

    #[test]
    fn test_parse_tool_call_arguments_before_id_is_retained() {
        let mut parser = OpenAiCompatStreamParser::new();
        let mut accumulator = EventAccumulator::new();

        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"path\""}}]},"index":0}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"read","arguments":":\"Cargo.toml\"}"}}]},"index":0}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
        ];

        let events: Vec<AssistantMessageEvent> = chunks
            .iter()
            .flat_map(|chunk| parser.parse_data(chunk))
            .collect();

        for event in &events {
            accumulator.feed(event);
        }

        let blocks = accumulator.content_blocks();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolCall {
                id,
                name,
                arguments,
            } => {
                assert_eq!(id, "call_abc");
                assert_eq!(name, "read");
                assert_eq!(arguments["path"], "Cargo.toml");
            }
            other => panic!("expected tool call, got {other:?}"),
        }
    }

    #[test]
    fn test_parse_tool_call_id_is_stable_when_id_arrives_late() {
        let mut parser = OpenAiCompatStreamParser::new();

        let chunks = [
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"read","arguments":"{\"path\""}}]},"index":0}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_real","function":{"arguments":":\"Cargo.toml\"}"}}]},"index":0}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
        ];

        let events: Vec<AssistantMessageEvent> = chunks
            .iter()
            .flat_map(|chunk| parser.parse_data(chunk))
            .collect();

        let start_ids: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                AssistantMessageEvent::ToolCallStart { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect();
        let end_ids: Vec<String> = events
            .iter()
            .filter_map(|e| match e {
                AssistantMessageEvent::ToolCallEnd { id } => Some(id.clone()),
                _ => None,
            })
            .collect();

        assert_eq!(start_ids, vec!["call_real".to_string()]);
        assert_eq!(end_ids, vec!["call_real".to_string()]);
    }

    #[test]
    fn test_parse_usage_and_done_same_chunk() {
        let mut parser = OpenAiCompatStreamParser::new();
        let chunk = r#"{
            "choices":[{"delta":{"content":""},"finish_reason":"stop","index":0}],
            "usage":{"prompt_tokens":10,"completion_tokens":2}
        }"#;
        let events = parser.parse_data(chunk);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, AssistantMessageEvent::Usage { .. }))
        );
        assert!(events.iter().any(|e| matches!(
            e,
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                ..
            }
        )));
    }

    #[test]
    fn test_tool_result_conversion_omits_non_openai_fields() {
        let model = Model {
            id: "test-model".into(),
            name: "Test Model".into(),
            api: crate::types::Api::OpenAiCompletions,
            provider: crate::types::Provider::OpenAI,
            base_url: "https://api.openai.com".into(),
            reasoning: false,
            thinking_level_map: Default::default(),
            input: vec![crate::types::Modality::Text],
            cost: Default::default(),
            context_window: 128_000,
            max_tokens: 16_384,
            compat: crate::model::ModelCompat::for_openai(),
        };
        let msg = Message::ToolResult {
            tool_call_id: "call_123".into(),
            tool_name: "read".into(),
            content: vec![ContentBlock::Text {
                text: "done".into(),
            }],
            details: None,
            is_error: false,
            timestamp: 0,
        };

        let converted = convert_message(&model, &msg).expect("tool result converts");
        assert_eq!(converted["role"], "tool");
        assert_eq!(converted["tool_call_id"], "call_123");
        assert_eq!(converted["content"], "done");
        assert!(converted.get("is_error").is_none());
    }

    #[test]
    fn test_transform_synthesizes_missing_tool_result() {
        let model = Model {
            id: "gpt-5.5".into(),
            name: "OpenAI".into(),
            api: crate::types::Api::OpenAiCompletions,
            provider: crate::types::Provider::OpenAI,
            base_url: "https://api.openai.com".into(),
            reasoning: false,
            thinking_level_map: Default::default(),
            input: vec![crate::types::Modality::Text],
            cost: Default::default(),
            context_window: 128_000,
            max_tokens: 16_384,
            compat: crate::model::ModelCompat::for_openai(),
        };
        let messages = vec![
            Message::User {
                content: vec![ContentBlock::text("do thing")],
                timestamp: 1,
            },
            Message::Assistant {
                content: vec![ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "read".into(),
                    arguments: json!({"path":"Cargo.toml"}),
                }],
                api: Some(crate::types::Api::OpenAiCompletions),
                provider: Some(crate::types::Provider::OpenAI),
                model: Some("gpt-5.5".into()),
                usage: None,
                stop_reason: Some(StopReason::ToolUse),
                error_message: None,
                timestamp: 2,
            },
            Message::User {
                content: vec![ContentBlock::text("next")],
                timestamp: 3,
            },
        ];

        let (transformed, _stats) = sanitize_messages_for_replay(&messages, &model);
        assert!(
            transformed.iter().any(|m| matches!(
                m,
                Message::ToolResult {
                    tool_call_id,
                    is_error,
                    ..
                } if tool_call_id == "call_1" && *is_error
            )),
            "expected synthetic tool result for orphan tool call"
        );
    }

    #[test]
    fn test_transform_drops_aborted_assistant_message() {
        let model = Model {
            id: "gpt-5.5".into(),
            name: "OpenAI".into(),
            api: crate::types::Api::OpenAiCompletions,
            provider: crate::types::Provider::OpenAI,
            base_url: "https://api.openai.com".into(),
            reasoning: false,
            thinking_level_map: Default::default(),
            input: vec![crate::types::Modality::Text],
            cost: Default::default(),
            context_window: 128_000,
            max_tokens: 16_384,
            compat: crate::model::ModelCompat::for_openai(),
        };
        let messages = vec![
            Message::User {
                content: vec![ContentBlock::text("hi")],
                timestamp: 1,
            },
            Message::Assistant {
                content: vec![ContentBlock::text("partial")],
                api: Some(crate::types::Api::OpenAiCompletions),
                provider: Some(crate::types::Provider::OpenAI),
                model: Some("gpt-5.5".into()),
                usage: None,
                stop_reason: Some(StopReason::Aborted),
                error_message: Some("aborted".into()),
                timestamp: 2,
            },
            Message::User {
                content: vec![ContentBlock::text("continue")],
                timestamp: 3,
            },
        ];

        let (transformed, _stats) = sanitize_messages_for_replay(&messages, &model);
        assert_eq!(
            transformed
                .iter()
                .filter(|m| matches!(m, Message::Assistant { .. }))
                .count(),
            0,
            "aborted assistant message should not be replayed"
        );
    }

    #[test]
    fn test_bridge_assistant_inserted_after_tool_before_user() {
        let mut model = Model {
            id: "deepseek-v4-pro".into(),
            name: "DeepSeek".into(),
            api: crate::types::Api::OpenAiCompletions,
            provider: crate::types::Provider::DeepSeek,
            base_url: "https://api.deepseek.com".into(),
            reasoning: true,
            thinking_level_map: Default::default(),
            input: vec![crate::types::Modality::Text],
            cost: Default::default(),
            context_window: 128_000,
            max_tokens: 16_384,
            compat: crate::model::ModelCompat::for_deepseek(),
        };
        model.compat.requires_assistant_after_tool_result = true;
        let ctx = Context {
            system: None,
            messages: vec![
                Message::ToolResult {
                    tool_call_id: "call_1".into(),
                    tool_name: "read".into(),
                    content: vec![ContentBlock::text("ok")],
                    details: None,
                    is_error: false,
                    timestamp: 1,
                },
                Message::User {
                    content: vec![ContentBlock::text("next")],
                    timestamp: 2,
                },
            ],
            tools: vec![],
            thinking_level: None,
        };
        let msgs = convert_messages(&model, &ctx);
        let arr = msgs.as_array().expect("messages array");
        assert_eq!(arr.len(), 3);
        assert_eq!(arr[0]["role"], "tool");
        assert_eq!(arr[1]["role"], "assistant");
        assert_eq!(arr[2]["role"], "user");
    }

    #[test]
    fn test_tools_empty_sent_when_tool_history_present() {
        let model = Model {
            id: "gpt-5.5".into(),
            name: "OpenAI".into(),
            api: crate::types::Api::OpenAiCompletions,
            provider: crate::types::Provider::OpenAI,
            base_url: "https://api.openai.com".into(),
            reasoning: true,
            thinking_level_map: Default::default(),
            input: vec![crate::types::Modality::Text],
            cost: Default::default(),
            context_window: 128_000,
            max_tokens: 16_384,
            compat: crate::model::ModelCompat::for_openai(),
        };
        let ctx = Context {
            system: None,
            messages: vec![
                Message::Assistant {
                    content: vec![ContentBlock::ToolCall {
                        id: "call_1".into(),
                        name: "read".into(),
                        arguments: serde_json::json!({"path":"Cargo.toml"}),
                    }],
                    api: Some(crate::types::Api::OpenAiCompletions),
                    provider: Some(crate::types::Provider::OpenAI),
                    model: Some("gpt-5.5".into()),
                    usage: None,
                    stop_reason: Some(StopReason::ToolUse),
                    error_message: None,
                    timestamp: 1,
                },
                Message::ToolResult {
                    tool_call_id: "call_1".into(),
                    tool_name: "read".into(),
                    content: vec![ContentBlock::text("ok")],
                    details: None,
                    is_error: false,
                    timestamp: 2,
                },
            ],
            tools: vec![],
            thinking_level: None,
        };
        let body = build_request_body(&model, &ctx, &StreamOptions::default(), true).unwrap();
        assert!(body.get("tools").is_some());
        assert_eq!(
            body["tools"]
                .as_array()
                .map(|a| a.len())
                .unwrap_or_default(),
            0
        );
    }

    #[test]
    fn test_deepseek_non_off_thinking_falls_back_to_high() {
        let model = Model {
            id: "deepseek-v4-pro".into(),
            name: "DeepSeek".into(),
            api: crate::types::Api::OpenAiCompletions,
            provider: crate::types::Provider::DeepSeek,
            base_url: "https://api.deepseek.com".into(),
            reasoning: true,
            thinking_level_map: Default::default(),
            input: vec![crate::types::Modality::Text],
            cost: Default::default(),
            context_window: 1_000_000,
            max_tokens: 384_000,
            compat: crate::model::ModelCompat::for_deepseek(),
        };
        let mut body = json!({});
        apply_thinking_params(&mut body, &model, crate::types::ThinkingLevel::Medium);
        // Medium is unmapped on DeepSeek — no thinking params sent, model default.
        assert!(
            body.get("thinking").is_none(),
            "unmapped level should not send thinking params"
        );

        let mut body_off = json!({});
        apply_thinking_params(&mut body_off, &model, crate::types::ThinkingLevel::Off);
        assert_eq!(body_off["thinking"]["type"], "disabled");
    }

    #[test]
    fn test_deepseek_request_body_drops_orphan_tool_result_from_aborted_turn() {
        let model = Model {
            id: "deepseek-v4-pro".into(),
            name: "DeepSeek".into(),
            api: crate::types::Api::OpenAiCompletions,
            provider: crate::types::Provider::DeepSeek,
            base_url: "https://api.deepseek.com".into(),
            reasoning: true,
            thinking_level_map: Default::default(),
            input: vec![crate::types::Modality::Text],
            cost: Default::default(),
            context_window: 1_000_000,
            max_tokens: 384_000,
            compat: crate::model::ModelCompat::for_deepseek(),
        };
        // Dirty input: aborted assistant + orphan tool result. In production,
        // build_context() sanitizes before build_request_body() sees it. Test
        // that sanitize_messages_for_replay drops the orphan, then verify the
        // sanitized output converts cleanly.
        let dirty = vec![
            Message::Assistant {
                content: vec![ContentBlock::ToolCall {
                    id: "call_1".into(),
                    name: "read".into(),
                    arguments: serde_json::json!({"path":"Cargo.toml"}),
                }],
                api: Some(crate::types::Api::OpenAiCompletions),
                provider: Some(crate::types::Provider::DeepSeek),
                model: Some("deepseek-v4-pro".into()),
                usage: None,
                stop_reason: Some(StopReason::Aborted),
                error_message: Some("aborted".into()),
                timestamp: 1,
            },
            Message::ToolResult {
                tool_call_id: "call_1".into(),
                tool_name: "read".into(),
                content: vec![ContentBlock::text("ok")],
                details: None,
                is_error: false,
                timestamp: 2,
            },
            Message::User {
                content: vec![ContentBlock::text("continue")],
                timestamp: 3,
            },
        ];
        let (sanitized, _) = sanitize_messages_for_replay(&dirty, &model);
        let ctx = Context {
            system: None,
            messages: sanitized,
            tools: vec![],
            thinking_level: None,
        };
        let body = build_request_body(&model, &ctx, &StreamOptions::default(), true).unwrap();
        let messages = body["messages"].as_array().expect("messages array");
        assert!(
            !messages
                .iter()
                .any(|m| m.get("role").and_then(|r| r.as_str()) == Some("tool")),
            "orphan tool messages must not be replayed"
        );
    }

    #[test]
    fn test_parse_legacy_function_call_stream_to_tool_call() {
        let mut parser = OpenAiCompatStreamParser::new();
        let mut accumulator = EventAccumulator::new();
        let chunks = [
            r#"{"choices":[{"delta":{"function_call":{"name":"read","arguments":"{\"path\""}},"index":0}]}"#,
            r#"{"choices":[{"delta":{"function_call":{"arguments":":\"Cargo.toml\"}"}},"index":0}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"function_call","index":0}]}"#,
        ];

        let events: Vec<AssistantMessageEvent> = chunks
            .iter()
            .flat_map(|chunk| parser.parse_data(chunk))
            .collect();

        for event in &events {
            accumulator.feed(event);
        }

        let blocks = accumulator.content_blocks();
        assert_eq!(blocks.len(), 1);
        match &blocks[0] {
            ContentBlock::ToolCall {
                id: _,
                name,
                arguments,
            } => {
                assert_eq!(name, "read");
                assert_eq!(arguments["path"], "Cargo.toml");
            }
            other => panic!("expected tool call, got {other:?}"),
        }
        assert_eq!(accumulator.stop_reason(), Some(StopReason::ToolUse));
    }

    #[test]
    fn test_parse_mixed_function_call_and_tool_calls_yields_single_call() {
        let mut parser = OpenAiCompatStreamParser::new();
        let mut accumulator = EventAccumulator::new();
        let chunks = [
            r#"{"choices":[{"delta":{"function_call":{"name":"read","arguments":"{\"path\""}},"index":0}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_real","type":"function","function":{"arguments":":\"Cargo.toml\"}"}}]},"index":0}]}"#,
            r#"{"choices":[{"delta":{},"finish_reason":"tool_calls","index":0}]}"#,
        ];

        let events: Vec<AssistantMessageEvent> = chunks
            .iter()
            .flat_map(|chunk| parser.parse_data(chunk))
            .collect();

        for event in &events {
            accumulator.feed(event);
        }
        let blocks = accumulator.content_blocks();
        let tool_calls: Vec<&ContentBlock> = blocks
            .iter()
            .filter(|b| matches!(b, ContentBlock::ToolCall { .. }))
            .collect();
        assert_eq!(tool_calls.len(), 1);
    }
}
