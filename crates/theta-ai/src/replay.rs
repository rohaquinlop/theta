//! Conversation replay sanitization utilities.
//!
//! Normalizes transcript shape before replaying history to providers.

use std::collections::{HashMap, HashSet};

use crate::{ContentBlock, Message, Model, StopReason};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReplaySanitizationStats {
    pub dropped_assistant_messages: u32,
    pub synthesized_tool_results: u32,
    pub normalized_tool_call_ids: u32,
}

impl ReplaySanitizationStats {
    pub fn changed(&self) -> bool {
        self.dropped_assistant_messages > 0
            || self.synthesized_tool_results > 0
            || self.normalized_tool_call_ids > 0
    }
}

fn normalize_tool_call_id_for_model(id: &str, model: &Model) -> String {
    if id.contains('|') {
        let call_id = id.split('|').next().unwrap_or(id);
        return call_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .take(40)
            .collect();
    }

    if model.provider == crate::Provider::OpenAI && id.len() > 40 {
        id.chars().take(40).collect()
    } else {
        id.to_string()
    }
}

pub fn sanitize_messages_for_replay(
    messages: &[Message],
    model: &Model,
) -> (Vec<Message>, ReplaySanitizationStats) {
    let mut stats = ReplaySanitizationStats::default();

    // Pass 1: normalize tool-call IDs in assistant messages, remap tool results.
    let mut id_map: HashMap<String, String> = HashMap::new();
    let mut first_pass: Vec<Message> = Vec::with_capacity(messages.len());
    for msg in messages {
        match msg {
            Message::Assistant {
                content,
                api,
                provider,
                model: msg_model,
                usage,
                stop_reason,
                error_message,
                timestamp,
            } => {
                let is_same_model = provider == &Some(model.provider)
                    && api == &Some(model.api)
                    && msg_model.as_deref() == Some(model.id.as_str());
                let mut changed = false;
                let mut new_content = Vec::with_capacity(content.len());
                for block in content {
                    match block {
                        ContentBlock::ToolCall {
                            id,
                            name,
                            arguments,
                        } => {
                            if is_same_model {
                                new_content.push(block.clone());
                                continue;
                            }
                            let normalized = normalize_tool_call_id_for_model(id, model);
                            if normalized != *id {
                                id_map.insert(id.clone(), normalized.clone());
                                stats.normalized_tool_call_ids += 1;
                                changed = true;
                                new_content.push(ContentBlock::ToolCall {
                                    id: normalized,
                                    name: name.clone(),
                                    arguments: arguments.clone(),
                                });
                            } else {
                                new_content.push(block.clone());
                            }
                        }
                        _ => new_content.push(block.clone()),
                    }
                }
                if changed {
                    first_pass.push(Message::Assistant {
                        content: new_content,
                        api: *api,
                        provider: *provider,
                        model: msg_model.clone(),
                        usage: usage.clone(),
                        stop_reason: *stop_reason,
                        error_message: error_message.clone(),
                        timestamp: *timestamp,
                    });
                } else {
                    first_pass.push(msg.clone());
                }
            }
            Message::ToolResult {
                tool_call_id,
                tool_name,
                content,
                details,
                is_error,
                timestamp,
            } => {
                let remapped = id_map
                    .get(tool_call_id)
                    .cloned()
                    .unwrap_or_else(|| tool_call_id.clone());
                first_pass.push(Message::ToolResult {
                    tool_call_id: remapped,
                    tool_name: tool_name.clone(),
                    content: content.clone(),
                    details: details.clone(),
                    is_error: *is_error,
                    timestamp: *timestamp,
                });
            }
            _ => first_pass.push(msg.clone()),
        }
    }

    // Pass 2: drop errored/aborted assistant messages + synthesize missing tool results.
    let mut out = Vec::with_capacity(first_pass.len());
    let mut pending_tool_calls: Vec<(String, String, u64)> = Vec::new();
    let mut existing_results: HashSet<String> = HashSet::new();

    fn flush_pending_tool_calls(
        out: &mut Vec<Message>,
        pending_tool_calls: &mut Vec<(String, String, u64)>,
        existing_results: &mut HashSet<String>,
        stats: &mut ReplaySanitizationStats,
    ) {
        if pending_tool_calls.is_empty() {
            return;
        }
        for (id, name, ts) in pending_tool_calls.drain(..) {
            if !existing_results.contains(&id) {
                out.push(Message::ToolResult {
                    tool_call_id: id,
                    tool_name: name,
                    content: vec![ContentBlock::text("No result provided")],
                    details: None,
                    is_error: true,
                    timestamp: ts,
                });
                stats.synthesized_tool_results += 1;
            }
        }
        existing_results.clear();
    }

    for msg in first_pass {
        match &msg {
            Message::Assistant {
                content,
                stop_reason,
                timestamp,
                ..
            } => {
                flush_pending_tool_calls(
                    &mut out,
                    &mut pending_tool_calls,
                    &mut existing_results,
                    &mut stats,
                );
                if matches!(stop_reason, Some(StopReason::Error | StopReason::Aborted)) {
                    stats.dropped_assistant_messages += 1;
                    continue;
                }
                let tool_calls: Vec<(String, String, u64)> = content
                    .iter()
                    .filter_map(|b| match b {
                        ContentBlock::ToolCall { id, name, .. } => {
                            Some((id.clone(), name.clone(), *timestamp))
                        }
                        _ => None,
                    })
                    .collect();
                if !tool_calls.is_empty() {
                    pending_tool_calls = tool_calls;
                    existing_results.clear();
                }
                out.push(msg);
            }
            Message::ToolResult { tool_call_id, .. } => {
                existing_results.insert(tool_call_id.clone());
                out.push(msg);
            }
            Message::User { .. } => {
                flush_pending_tool_calls(
                    &mut out,
                    &mut pending_tool_calls,
                    &mut existing_results,
                    &mut stats,
                );
                out.push(msg);
            }
            _ => out.push(msg),
        }
    }
    flush_pending_tool_calls(
        &mut out,
        &mut pending_tool_calls,
        &mut existing_results,
        &mut stats,
    );
    (out, stats)
}
