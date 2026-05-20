//! OpenAI Codex provider — ChatGPT Plus subscription authentication.
//!
//! Codex uses the **Responses API** at `chatgpt.com/backend-api/codex/responses`.
//! Transport: WebSocket (primary) with SSE fallback.
//!
//! ## Headers
//! - `OpenAI-Beta: responses=experimental` — required
//! - `chatgpt-account-id` — extracted from JWT
//! - `originator: theta` — request origin
//! - `x-client-request-id` — per-request trace ID

use async_trait::async_trait;
use futures::StreamExt;
use futures_util::SinkExt;
use serde_json::Value;

use crate::error::ThetaError;
use crate::event::AssistantMessageEvent;
use crate::model::Model;
use crate::provider::{EventStream, Provider};
use crate::types::{
    ContentBlock, Context, SimpleStreamOptions, StopReason, StreamOptions, ThinkingLevel, Tool,
};

const CODEX_TOKEN_ENV: &str = "OPENAI_CODEX_TOKEN";

pub struct OpenAiCodexProvider {
    client: reqwest::Client,
    token: tokio::sync::RwLock<Option<String>>,
}

impl OpenAiCodexProvider {
    pub fn new() -> Self {
        let env_token = std::env::var(CODEX_TOKEN_ENV).ok();
        Self {
            client: reqwest::Client::new(),
            token: tokio::sync::RwLock::new(env_token),
        }
    }
}

impl Default for OpenAiCodexProvider {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Provider for OpenAiCodexProvider {
    fn set_token(&mut self, token: &str) {
        self.token = tokio::sync::RwLock::new(Some(token.to_string()));
    }

    async fn stream<'a>(
        &'a self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> Result<EventStream<'a>, ThetaError> {
        let token = self.token.read().await.clone().ok_or_else(|| {
            ThetaError::MissingApiKey {
                provider: crate::types::Provider::OpenAiCodex,
            }
        })?;

        let http_url = codex_url(&model.base_url);
        let ws_url = codex_ws_url(&model.base_url);
        let body = build_request_body(model, context, options);
        let account_id = extract_account_id(&token).unwrap_or_default();

        tracing::debug!(
            "Codex model={} messages={}",
            model.id,
            context.messages.len(),
        );

        // Try WebSocket first (lower latency), fall back to SSE.
        match ws_stream(&ws_url, &body, &account_id, &token).await {
            Ok(stream) => return Ok(stream),
            Err(e) => {
                tracing::warn!("Codex WebSocket failed, falling back to SSE: {e}");
            }
        }

        sse_stream(&self.client, &http_url, &body, &account_id, &token).await
    }

    async fn stream_simple<'a>(
        &'a self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> Result<EventStream<'a>, ThetaError> {
        let opts = StreamOptions {
            max_tokens: options.max_tokens,
            temperature: options.temperature,
            top_p: None,
            stop: None,
            thinking_level: None,
            include_usage: false,
            json_mode: false,
            seed: None,
            service_tier: None,
        };
        self.stream(model, context, &opts).await
    }
}

// ---------------------------------------------------------------------------
// URL construction
// ---------------------------------------------------------------------------

fn codex_url(base_url: &str) -> String {
    let base = base_url.trim_end_matches('/');
    if base.ends_with("/codex/responses") {
        return base.to_string();
    }
    if base.ends_with("/codex") {
        return format!("{base}/responses");
    }
    format!("{base}/codex/responses")
}

fn codex_ws_url(base_url: &str) -> String {
    let http = codex_url(base_url);
    if let Some(rest) = http.strip_prefix("https://") {
        format!("wss://{rest}")
    } else if let Some(rest) = http.strip_prefix("http://") {
        format!("ws://{rest}")
    } else {
        http
    }
}

// ---------------------------------------------------------------------------
// SSE transport (fallback)
// ---------------------------------------------------------------------------

async fn sse_stream(
    client: &reqwest::Client,
    url: &str,
    body: &Value,
    account_id: &str,
    token: &str,
) -> Result<EventStream<'static>, ThetaError> {
    let mut req = client
        .post(url)
        .header("Authorization", format!("Bearer {token}"))
        .header("Content-Type", "application/json")
        .header("OpenAI-Beta", "responses=experimental")
        .header("accept", "text/event-stream")
        .header("originator", "theta");

    if !account_id.is_empty() {
        req = req.header("chatgpt-account-id", account_id);
    }

    let response = req.json(body).send().await?;

    let status = response.status();
    if !status.is_success() {
        let retry_ms = response
            .headers()
            .get("retry-after-ms")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse().ok());
        let body_text = response.text().await.unwrap_or_default();
        if status.as_u16() == 401 {
            return Err(ThetaError::ApiError {
                status: 401,
                message: "ChatGPT session token expired. Re-authenticate with `theta login`."
                    .into(),
                retry_after_ms: None,
            });
        }
        return Err(ThetaError::ApiError {
            status: status.as_u16(),
            message: body_text,
            retry_after_ms: retry_ms,
        });
    }

    let stream = byte_stream_to_events(response.bytes_stream());
    Ok(stream)
}

// ---------------------------------------------------------------------------
// WebSocket transport (primary)
// ---------------------------------------------------------------------------

async fn ws_stream(
    url: &str,
    body: &Value,
    account_id: &str,
    token: &str,
) -> Result<EventStream<'static>, ThetaError> {
    use tokio_tungstenite::tungstenite::protocol::Message;

    let mut req_builder = http::Request::builder()
        .uri(url)
        .header("Authorization", format!("Bearer {token}"))
        .header("OpenAI-Beta", "responses=experimental")
        .header("originator", "theta");

    if !account_id.is_empty() {
        req_builder = req_builder.header("chatgpt-account-id", account_id);
    }

    let req = req_builder.body(()).unwrap();

    let (ws_stream, _) = tokio_tungstenite::connect_async(req)
        .await
        .map_err(|e| ThetaError::ApiError {
            status: 500,
            message: format!("WebSocket connect failed: {e}"),
            retry_after_ms: None,
        })?;
    let (mut write, read) = ws_stream.split();

    // Send the JSON body as a text frame.
    let payload = serde_json::to_string(body)
        .map_err(ThetaError::Json)?;
    write
        .send(Message::Text(payload.into()))
        .await
        .map_err(|e| ThetaError::ApiError {
            status: 500,
            message: format!("WebSocket send failed: {e}"),
            retry_after_ms: None,
        })?;

    // Read frames — each text frame is a complete JSON event.
    let events = read
        .filter_map(|msg| async move {
            match msg {
                Ok(Message::Text(text)) => {
                    let text = text.to_string();
                    parse_codex_json(&text)
                }
                Ok(Message::Close(_)) => {
                    Some(AssistantMessageEvent::Done {
                        stop_reason: StopReason::Stop,
                        usage: None,
                    })
                }
                Err(e) => Some(AssistantMessageEvent::Error {
                    code: "ws".into(),
                    message: e.to_string(),
                }),
                _ => None,
            }
        })
        .chain(futures::stream::once(async {
            AssistantMessageEvent::Done {
                stop_reason: StopReason::Stop,
                usage: None,
            }
        }))
        .boxed();

    Ok(events)
}

// ---------------------------------------------------------------------------
// Shared stream processing
// ---------------------------------------------------------------------------

/// Convert a byte stream into buffered, newline-delimited events.
fn byte_stream_to_events(
    byte_stream: impl futures::Stream<Item = Result<bytes::Bytes, reqwest::Error>>
        + Send
        + Unpin
        + 'static,
) -> EventStream<'static> {
    futures::stream::unfold(
        (byte_stream, String::new(), false),
        |(mut stream, mut buf, mut exhausted)| async move {
            if exhausted {
                return None;
            }
            loop {
                while let Some(pos) = buf.find('\n') {
                    let line = buf[..pos].to_string();
                    buf = buf[pos + 1..].to_string();
                    if let Some(event) = parse_codex_sse(&line) {
                        return Some((event, (stream, buf, exhausted)));
                    }
                }
                match stream.next().await {
                    Some(Ok(bytes)) => {
                        buf.push_str(&String::from_utf8_lossy(&bytes));
                        continue;
                    }
                    Some(Err(e)) => {
                        exhausted = true;
                        return Some((
                            AssistantMessageEvent::Error {
                                code: "stream".into(),
                                message: e.to_string(),
                            },
                            (stream, buf, exhausted),
                        ));
                    }
                    None => {
                        exhausted = true;
                        return Some((
                            AssistantMessageEvent::Done {
                                stop_reason: StopReason::Stop,
                                usage: None,
                            },
                            (stream, buf, exhausted),
                        ));
                    }
                }
            }
        },
    )
    .boxed()
}

// ---------------------------------------------------------------------------
// Request building
// ---------------------------------------------------------------------------

fn build_request_body(model: &Model, context: &Context, options: &StreamOptions) -> Value {
    let messages = convert_messages(model, context);
    let instructions = extract_system_text(&context.system)
        .unwrap_or_else(|| "You are a helpful assistant.".to_string());

    let mut body = serde_json::json!({
        "model": model.id,
        "store": false,
        "stream": true,
        "instructions": instructions,
        "input": messages,
        "text": { "verbosity": "low" },
        "include": ["reasoning.encrypted_content"],
        "tool_choice": "auto",
        "parallel_tool_calls": true,
    });

    if !context.tools.is_empty() {
        body["tools"] = serde_json::json!(convert_tools(&context.tools));
    }
    if let Some(temp) = options.temperature {
        body["temperature"] = serde_json::json!(temp);
    }
    if let Some(ref tier) = options.service_tier {
        body["service_tier"] = serde_json::json!(tier);
    }

    let effort_level = options.thinking_level.or(context.thinking_level);
    if let Some(level) = effort_level {
        let effort = resolve_reasoning_effort(model, level);
        if let Some(e) = effort {
            body["reasoning"] = serde_json::json!({
                "effort": e,
                "summary": "auto",
            });
        }
    }

    body
}

fn extract_system_text(system: &Option<Vec<ContentBlock>>) -> Option<String> {
    system.as_ref().and_then(|blocks| {
        blocks.iter().find_map(|b| match b {
            ContentBlock::Text { text } => Some(text.clone()),
            _ => None,
        })
    })
}

fn resolve_reasoning_effort(model: &Model, level: ThinkingLevel) -> Option<String> {
    if level == ThinkingLevel::Off {
        return None;
    }
    if let Some(mapped) = model.thinking_level_map.get(&level) {
        return mapped.clone();
    }
    let fallback = map_default_effort(level);
    if fallback.is_empty() { None } else { Some(fallback) }
}

fn map_default_effort(level: ThinkingLevel) -> String {
    match level {
        ThinkingLevel::Off => String::new(),
        ThinkingLevel::Minimal => "minimal".into(),
        ThinkingLevel::Low => "low".into(),
        ThinkingLevel::Medium => "medium".into(),
        ThinkingLevel::High => "high".into(),
        ThinkingLevel::XHigh => "max".into(),
    }
}

fn convert_messages(model: &Model, context: &Context) -> Vec<Value> {
    let mut items = Vec::new();

    if model.compat.supports_developer_role
        && let Some(sp) = extract_system_text(&context.system)
    {
        items.push(serde_json::json!({
            "type": "message",
            "role": "developer",
            "content": sp,
        }));
    }

    for msg in &context.messages {
        match msg {
            crate::types::Message::User { content, .. } => {
                let text = blocks_to_text(content);
                items.push(serde_json::json!({
                    "role": "user",
                    "content": [{ "type": "input_text", "text": text }],
                }));
            }
            crate::types::Message::Assistant { content, .. } => {
                for block in content {
                    match block {
                        ContentBlock::Text { text } => {
                            items.push(serde_json::json!({
                                "type": "message",
                                "role": "assistant",
                                "content": [{
                                    "type": "output_text",
                                    "text": text,
                                    "annotations": [],
                                }],
                                "status": "completed",
                            }));
                        }
                        ContentBlock::ToolCall { id, name, arguments } => {
                            let (call_id, item_id) = split_tool_call_id(id);
                            let args_str =
                                serde_json::to_string(arguments).unwrap_or_else(|_| "{}".into());
                            items.push(serde_json::json!({
                                "type": "function_call",
                                "id": item_id,
                                "call_id": call_id,
                                "name": name,
                                "arguments": args_str,
                            }));
                        }
                        ContentBlock::Thinking { .. } => {}
                        _ => {}
                    }
                }
            }
            crate::types::Message::ToolResult {
                tool_call_id, content, ..
            } => {
                let text = blocks_to_text(content);
                let call_id = tool_call_id.split('|').next().unwrap_or(tool_call_id);
                items.push(serde_json::json!({
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": text,
                }));
            }
            _ => {}
        }
    }

    items
}

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

fn split_tool_call_id(id: &str) -> (&str, &str) {
    let parts: Vec<&str> = id.splitn(2, '|').collect();
    if parts.len() == 2 { (parts[0], parts[1]) } else { (id, "") }
}

fn convert_tools(tools: &[Tool]) -> Vec<Value> {
    tools
        .iter()
        .map(|tool| {
            serde_json::json!({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
                "strict": false,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// SSE / JSON parsing
// ---------------------------------------------------------------------------

fn parse_codex_sse(line: &str) -> Option<AssistantMessageEvent> {
    let line = line.trim();
    if let Some(data) = line.strip_prefix("data: ") {
        if data == "[DONE]" {
            return None;
        }
        parse_codex_json(data)
    } else {
        None
    }
}

fn parse_codex_json(json_str: &str) -> Option<AssistantMessageEvent> {
    let data: Value = serde_json::from_str(json_str).ok()?;
    parse_codex_event(&data)
}

fn parse_codex_event(data: &Value) -> Option<AssistantMessageEvent> {
    let event_type = data["type"].as_str()?;

    match event_type {
        "response.created" => None,

        "response.output_item.added" => {
            let item_type = data["item"]["type"].as_str()?;
            match item_type {
                "reasoning" => Some(AssistantMessageEvent::ThinkingStart),
                "message" => Some(AssistantMessageEvent::TextStart),
                "function_call" => {
                    let item = &data["item"];
                    let name = item["name"].as_str().unwrap_or("unknown");
                    let call_id = item["call_id"].as_str().unwrap_or("");
                    let item_id = item["id"].as_str().unwrap_or("");
                    Some(AssistantMessageEvent::ToolCallStart {
                        id: format!("{call_id}|{item_id}"),
                        name: name.to_string(),
                    })
                }
                _ => None,
            }
        }

        "response.output_text.delta" => {
            let delta = data["delta"].as_str().unwrap_or("");
            Some(AssistantMessageEvent::TextDelta { text: delta.to_string() })
        }

        "response.reasoning_text.delta"
        | "response.reasoning_summary_text.delta" => {
            let delta = data["delta"].as_str().unwrap_or("");
            Some(AssistantMessageEvent::ThinkingDelta { thinking: delta.to_string() })
        }

        "response.function_call_arguments.delta" => {
            let item = &data;
            let delta = item["delta"].as_str().unwrap_or("");
            let call_id = item["call_id"].as_str().unwrap_or("");
            Some(AssistantMessageEvent::ToolCallDelta {
                id: call_id.to_string(),
                arguments: delta.to_string(),
            })
        }

        "response.function_call_arguments.done" => {
            let item = data;
            let call_id = item["call_id"].as_str().unwrap_or("");
            let item_id = item["id"].as_str().unwrap_or("");
            Some(AssistantMessageEvent::ToolCallEnd {
                id: format!("{call_id}|{item_id}"),
            })
        }

        "response.output_item.done" => {
            let item_type = data["item"]["type"].as_str();
            match item_type {
                Some("reasoning") => Some(AssistantMessageEvent::ThinkingEnd),
                Some("message") => Some(AssistantMessageEvent::TextEnd),
                _ => None,
            }
        }

        "response.completed" | "response.done" => {
            None // Done emitted by stream end handler
        }

        "error" => {
            let code = data["code"].as_str().unwrap_or("unknown");
            let message = data["message"].as_str().unwrap_or("unknown error");
            Some(AssistantMessageEvent::Error {
                code: code.to_string(),
                message: message.to_string(),
            })
        }

        _ => None,
    }
}

// ---------------------------------------------------------------------------
// JWT account ID
// ---------------------------------------------------------------------------

fn extract_account_id(token: &str) -> Option<String> {
    let parts: Vec<&str> = token.splitn(3, '.').collect();
    if parts.len() != 3 {
        return None;
    }
    let payload = base64_url_decode(parts[1])?;
    let parsed: Value = serde_json::from_str(&payload).ok()?;
    parsed
        .get("https://api.openai.com/auth")?
        .get("chatgpt_account_id")?
        .as_str()
        .map(|s| s.to_string())
}

fn base64_url_decode(input: &str) -> Option<String> {
    let mut padded = input.to_string();
    while !padded.len().is_multiple_of(4) {
        padded.push('=');
    }
    let standard = padded.replace('-', "+").replace('_', "/");
    use base64::Engine;
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(&standard)
        .ok()?;
    String::from_utf8(bytes).ok()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codex_url_resolution() {
        assert_eq!(codex_url("https://chatgpt.com/backend-api"),
            "https://chatgpt.com/backend-api/codex/responses");
        assert_eq!(codex_url("https://chatgpt.com/backend-api/"),
            "https://chatgpt.com/backend-api/codex/responses");
        assert_eq!(codex_url("https://chatgpt.com/backend-api/codex"),
            "https://chatgpt.com/backend-api/codex/responses");
        assert_eq!(codex_url("https://chatgpt.com/backend-api/codex/responses"),
            "https://chatgpt.com/backend-api/codex/responses");
    }

    #[test]
    fn test_codex_ws_url() {
        assert_eq!(codex_ws_url("https://chatgpt.com/backend-api"),
            "wss://chatgpt.com/backend-api/codex/responses");
    }

    #[test]
    fn test_split_tool_call_id() {
        assert_eq!(split_tool_call_id("a|b"), ("a", "b"));
        assert_eq!(split_tool_call_id("abc"), ("abc", ""));
        assert_eq!(split_tool_call_id("a|b|c"), ("a", "b|c"));
    }

    #[test]
    fn test_parse_sse_done() {
        let event = parse_codex_sse("data: [DONE]");
        assert!(event.is_none());
    }
}
