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
use crate::types::{
    ContentBlock, Context, Message, SimpleStreamOptions, StopReason, StreamOptions, Usage,
};

/// The single OpenAI-compatible provider.
pub struct OpenAiCompatProvider {
    client: Client,
    api_key: RwLock<Option<String>>,
    /// User-selected MiMo cluster URL override (from latency test).
    mimo_base_url: RwLock<Option<String>>,
}

impl OpenAiCompatProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
            api_key: RwLock::new(None),
            mimo_base_url: RwLock::new(None),
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

        let api_key = self
            .api_key
            .read()
            .ok()
            .and_then(|key| key.clone())
            .or_else(|| std::env::var(api_key_env(model.provider)).ok())
            .ok_or(ThetaError::MissingApiKey {
                provider: model.provider,
            })?;

        // Detect base URL from API key prefix for MiMo:
        //   sk-* → pay-as-you-go → model.base_url (api.xiaomimimo.com)
        //   tp-* → token plan    → MIMO_BASE_URL or token-plan-sgp.xiaomimimo.com
        let base_url = if model.provider == crate::types::Provider::XiaomiMiMo {
            resolve_mimo_base_url(&api_key, &self.mimo_base_url)
        } else {
            model.base_url.to_string()
        };
        let url = format!("{}/v1/chat/completions", base_url.trim_end_matches('/'));

        // Debug: log first/last 4 chars of key and its length
        let key_preview = if api_key.len() <= 8 {
            "***too-short***".to_string()
        } else {
            format!(
                "{}...{} (len={})",
                &api_key[..4],
                &api_key[api_key.len() - 4..],
                api_key.len()
            )
        };
        tracing::debug!(
            api_key_preview = %key_preview,
            provider = ?model.provider,
            url = %url,
            uses_api_key_header = model.compat.uses_api_key_header,
            "sending request"
        );

        let mut request = self
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .json(&request_body);

        if model.compat.uses_api_key_header {
            request = request.header("api-key", &api_key);
        } else {
            request = request.header("Authorization", format!("Bearer {api_key}"));
        }

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
            tracing::warn!(
                status = %status,
                response_body = %body,
                "MiMo/OpenAI compat API returned non-success"
            );
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl OpenAiCompatProvider {
    /// Set the user-selected MiMo cluster URL (from latency test).
    /// Overrides auto-detection and env vars.
    pub fn set_mimo_base_url(&self, url: &str) {
        if let Ok(mut u) = self.mimo_base_url.write() {
            *u = Some(url.to_string());
        }
    }
}

/// Get the environment variable name for a provider's API key.
pub fn api_key_env(provider: crate::types::Provider) -> &'static str {
    match provider {
        crate::types::Provider::OpenAI => "OPENAI_API_KEY",
        crate::types::Provider::OpenAiCodex => "OPENAI_CODEX_TOKEN",
        crate::types::Provider::DeepSeek => "DEEPSEEK_API_KEY",
        crate::types::Provider::OpenCode => "OPENCODE_API_KEY",
        crate::types::Provider::OpenCodeGo => "OPENCODE_API_KEY",
        crate::types::Provider::XiaomiMiMo => "MIMO_API_KEY",
    }
}

fn resolve_mimo_base_url(api_key: &str, override_url: &RwLock<Option<String>>) -> String {
    // 1. User-selected cluster override (from latency test modal).
    //    Always honored regardless of key prefix.
    if let Ok(Some(url)) = override_url.read().map(|u| u.clone()) {
        return url;
    }
    // 2. MIMO_BASE_URL env var.
    if let Ok(url) = std::env::var("MIMO_BASE_URL") {
        return url;
    }
    // 3. Key-prefix-based default.
    if api_key.starts_with("tp-") {
        "https://token-plan-sgp.xiaomimimo.com".to_string()
    } else {
        "https://api.xiaomimimo.com".to_string()
    }
}

/// Build the OpenAI-compatible JSON request body.
pub fn build_request_body(
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
    // Never send reasoning params to models that don't support reasoning.
    if !model.reasoning {
        return;
    }
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
        Some(crate::model::ThinkingFormat::XiaomiMiMo) => {
            // MiMo: binary on/off only, no reasoning_effort field.
            if level == crate::types::ThinkingLevel::Off {
                body["thinking"] = json!({"type": "disabled"});
            } else if level_str.is_some() {
                body["thinking"] = json!({"type": "enabled"});
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

pub fn convert_message(model: &Model, msg: &Message) -> Option<Value> {
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
#[allow(dead_code)]
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
pub struct OpenAiCompatStreamParser {
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn parse_data(&mut self, data: &str) -> Vec<AssistantMessageEvent> {
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
    }
}
