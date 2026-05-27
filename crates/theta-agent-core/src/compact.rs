//! Context compaction: trim old messages to fit within the model's context window.
//!
//! Uses approximate token counting (4 chars/token) to decide when truncation
//! is needed, then drops the oldest user/assistant/tool-result triples first.
//! The system prompt is never dropped. The most recent user message is always kept.

use theta_ai::Message;

use crate::types::{CompactionConfig, CompactionStrategy};

/// Result of a compaction pass.
#[derive(Debug, Clone)]
pub struct CompactionResult {
    /// The compacted message list (subset of the input).
    pub messages: Vec<Message>,
    /// How many messages were trimmed.
    pub trimmed_count: u32,
    /// Approximate tokens before compaction.
    pub tokens_before: u32,
    /// Approximate tokens after compaction.
    pub tokens_after: u32,
}

/// Compact the given messages to fit within `context_window - reserve_tokens`,
/// accounting for the system prompt token count. Returns the subset and stats.
pub fn compact_messages(
    messages: &[Message],
    system_prompt_tokens: u32,
    context_window: u32,
    config: &CompactionConfig,
) -> CompactionResult {
    if !config.enabled {
        let tokens = total_tokens(messages);
        return CompactionResult {
            messages: messages.to_vec(),
            trimmed_count: 0,
            tokens_before: tokens,
            tokens_after: tokens,
        };
    }

    let available = context_window.saturating_sub(config.reserve_tokens + system_prompt_tokens);
    let tokens_before = total_tokens(messages);

    if tokens_before <= available {
        return CompactionResult {
            messages: messages.to_vec(),
            trimmed_count: 0,
            tokens_before,
            tokens_after: tokens_before,
        };
    }

    // Walk from the end (newest) backwards, accumulating tokens.
    // Stop when we're under the budget, but never drop the last user message.
    // This keeps the most recent context intact.
    let mut kept: Vec<&Message> = Vec::new();
    let mut running_tokens: u32 = 0;
    let mut last_user_seen = false;

    for msg in messages.iter().rev() {
        let token_cost = msg_token_cost(msg);

        // Always keep the newest user message — it's the prompt we're answering.
        if !last_user_seen && matches!(msg, Message::User { .. }) {
            last_user_seen = true;
            running_tokens += token_cost;
            kept.push(msg);
            continue;
        }

        if running_tokens + token_cost > available {
            // We'll trim this and everything older.
            break;
        }

        running_tokens += token_cost;
        kept.push(msg);
    }

    // Reverse back to oldest-first order and clone to owned.
    kept.reverse();
    let mut kept_owned: Vec<Message> = kept.into_iter().cloned().collect();
    let trimmed_count = messages.len().saturating_sub(kept_owned.len()) as u32;
    if trimmed_count > 0 && config.strategy == CompactionStrategy::Textual {
        let trimmed_len = messages.len().saturating_sub(kept_owned.len());
        if let Some(summary) = compacted_summary(&messages[..trimmed_len], trimmed_count) {
            kept_owned.insert(0, summary);
        }
    }
    let tokens_after = total_tokens(&kept_owned);

    CompactionResult {
        messages: kept_owned,
        trimmed_count,
        tokens_before,
        tokens_after,
    }
}

fn compacted_summary(trimmed: &[Message], trimmed_count: u32) -> Option<Message> {
    let mut user_lines = Vec::new();
    let mut assistant_lines = Vec::new();

    for msg in trimmed.iter().rev() {
        match msg {
            Message::User { content, .. } if user_lines.len() < 3 => {
                if let Some(text) = content_text(content) {
                    user_lines.push(text);
                }
            }
            Message::Assistant { content, .. } if assistant_lines.len() < 3 => {
                if let Some(text) = content_text(content) {
                    assistant_lines.push(text);
                }
            }
            _ => {}
        }
        if user_lines.len() >= 3 && assistant_lines.len() >= 3 {
            break;
        }
    }

    if user_lines.is_empty() && assistant_lines.is_empty() {
        return None;
    }

    user_lines.reverse();
    assistant_lines.reverse();
    let mut text = format!("Context compacted: {trimmed_count} older messages were summarized.");
    if !user_lines.is_empty() {
        text.push_str("\nRecent trimmed user messages:");
        for line in user_lines {
            text.push_str("\n- ");
            text.push_str(&truncate_chars(&line, 180));
        }
    }
    if !assistant_lines.is_empty() {
        text.push_str("\nRecent trimmed assistant messages:");
        for line in assistant_lines {
            text.push_str("\n- ");
            text.push_str(&truncate_chars(&line, 180));
        }
    }

    Some(Message::Assistant {
        content: vec![theta_ai::ContentBlock::text(truncate_chars(&text, 1200))],
        api: None,
        provider: None,
        model: None,
        usage: None,
        stop_reason: None,
        error_message: None,
        timestamp: 0,
    })
}

fn content_text(content: &[theta_ai::ContentBlock]) -> Option<String> {
    let text = content
        .iter()
        .filter_map(|block| match block {
            theta_ai::ContentBlock::Text { text } => Some(text.as_str()),
            theta_ai::ContentBlock::Thinking { thinking, .. } => Some(thinking.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ");
    if text.trim().is_empty() {
        None
    } else {
        Some(text)
    }
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

/// Count approximate tokens for all messages.
fn total_tokens(messages: &[Message]) -> u32 {
    messages.iter().map(msg_token_cost).sum()
}

/// Approximate token cost for a single message.
fn msg_token_cost(msg: &Message) -> u32 {
    match msg {
        Message::User { .. } | Message::Assistant { .. } | Message::ToolResult { .. } => {
            msg.token_count()
        }
        Message::ModelChange { .. } | Message::ThinkingLevelChange { .. } => 0,
    }
}
