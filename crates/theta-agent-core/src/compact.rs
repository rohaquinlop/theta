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

#[cfg(test)]
mod tests {
    use super::*;
    use theta_ai::ContentBlock;

    fn user(text: &str) -> Message {
        Message::User {
            content: vec![ContentBlock::text(text)],
            timestamp: 0,
        }
    }

    fn assistant(text: &str) -> Message {
        Message::Assistant {
            content: vec![ContentBlock::text(text)],
            api: None,
            provider: None,
            model: None,
            usage: None,
            stop_reason: None,
            error_message: None,
            timestamp: 0,
        }
    }

    #[test]
    fn test_no_compaction_when_under_budget() {
        let msgs = vec![user("hello"), assistant("hi there")];
        let result = compact_messages(
            &msgs,
            0,
            100,
            &CompactionConfig {
                enabled: true,
                reserve_tokens: 0,
                strategy: CompactionStrategy::Textual,
                summary_max_tokens: 512,
            },
        );
        assert_eq!(result.trimmed_count, 0);
        assert_eq!(result.messages.len(), 2);
    }

    #[test]
    fn test_compaction_trims_oldest() {
        let msgs = vec![
            user("a very long message that takes many tokens to represent"),
            assistant("reply 1"),
            user("another very long message with lots of content"),
            assistant("reply 2"),
            user("current question"),
            assistant("current answer"),
        ];
        // Very small budget — only room for ~last pair
        let result = compact_messages(
            &msgs,
            0,
            20,
            &CompactionConfig {
                enabled: true,
                reserve_tokens: 0,
                strategy: CompactionStrategy::Textual,
                summary_max_tokens: 512,
            },
        );
        assert!(result.trimmed_count > 0);
        assert!(result.messages.len() < msgs.len());
        // Last user message must always be kept.
        let has_user = result
            .messages
            .iter()
            .any(|m| matches!(m, Message::User { .. }));
        assert!(has_user);
    }

    #[test]
    fn test_disabled_compaction() {
        let msgs = vec![
            user("message 1"),
            assistant("reply 1"),
            user("message 2"),
            assistant("reply 2"),
        ];
        let result = compact_messages(
            &msgs,
            0,
            2, // tiny budget
            &CompactionConfig {
                enabled: false,
                reserve_tokens: 0,
                strategy: CompactionStrategy::Textual,
                summary_max_tokens: 512,
            },
        );
        assert_eq!(result.trimmed_count, 0);
        assert_eq!(result.messages.len(), 4);
    }

    #[test]
    fn test_reserve_tokens_reduces_available() {
        let msgs = vec![user("short"), assistant("ok")];
        // 100 token budget, reserve 95 → only 5 available
        let result = compact_messages(
            &msgs,
            0,
            100,
            &CompactionConfig {
                enabled: true,
                reserve_tokens: 95,
                strategy: CompactionStrategy::Textual,
                summary_max_tokens: 512,
            },
        );
        // Short messages should still fit under 5 tokens.
        assert_eq!(result.trimmed_count, 0);
    }

    #[test]
    fn test_system_prompt_accounted() {
        let msgs = vec![
            user("a long introduction message that takes up space"),
            assistant("brief reply"),
            user("latest question"),
        ];
        // 20 token budget, system prompt takes 10 → only 10 available
        let result = compact_messages(
            &msgs,
            10,
            20,
            &CompactionConfig {
                enabled: true,
                reserve_tokens: 0,
                strategy: CompactionStrategy::Textual,
                summary_max_tokens: 512,
            },
        );
        // Should trim at least the first pair.
        assert!(result.trimmed_count > 0);
    }

    #[test]
    fn test_compaction_inserts_summary() {
        let msgs = vec![
            user("old requirement about files"),
            assistant("old answer about files"),
            user("latest question"),
        ];
        let result = compact_messages(
            &msgs,
            0,
            8,
            &CompactionConfig {
                enabled: true,
                reserve_tokens: 0,
                strategy: CompactionStrategy::Textual,
                summary_max_tokens: 512,
            },
        );
        assert!(result.trimmed_count > 0);
        let Message::Assistant { content, .. } = &result.messages[0] else {
            panic!("expected summary assistant message");
        };
        assert!(matches!(
            &content[0],
            theta_ai::ContentBlock::Text { text } if text.contains("Context compacted")
        ));
    }
}
