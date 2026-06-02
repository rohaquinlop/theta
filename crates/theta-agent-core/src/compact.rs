//! Context compaction: trim old messages to fit within the model's context window.
//!
//! Prefix-preserving design: keeps the earliest messages (system prompt + prefix)
//! byte-stable in their original positions, summarizes the middle region, and
//! keeps the recent tail verbatim. This preserves DeepSeek/MiMo prefix cache
//! across compaction — the early prefix never shifts.

use theta_ai::Message;

use crate::types::{CompactionConfig, CompactionStrategy};

#[derive(Debug, Clone)]
pub struct CompactionResult {
    pub messages: Vec<Message>,
    pub trimmed_count: u32,
    pub tokens_before: u32,
    pub tokens_after: u32,
    /// Index of the first trimmed message in the ORIGINAL messages slice.
    /// Only set when trimmed_count > 0. Combined with trimmed_count, this
    /// identifies the exact region that was summarized: [trim_start..trim_start+trimmed_count].
    pub trim_start: usize,
}

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
            trim_start: 0,
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
            trim_start: 0,
        };
    }

    // Keep the most recent messages verbatim (tail), bounded by keep_recent_tokens.
    // Walk newest-to-oldest until the token budget is exhausted.
    let mut tail_start = messages.len();
    let mut tail_tokens: u32 = 0;
    for i in (0..messages.len()).rev() {
        let cost = msg_token_cost(&messages[i]);
        if tail_start < messages.len() && tail_tokens + cost > config.keep_recent_tokens {
            break;
        }
        tail_tokens += cost;
        tail_start = i;
    }

    // Align tail_start off any tool result so the tail never begins with an
    // orphan whose assistant tool_calls were summarized away.
    while tail_start > 0 && matches!(messages[tail_start], Message::ToolResult { .. }) {
        tail_start -= 1;
    }

    // Compute how many messages to keep as prefix (before the summarized region).
    // The prefix + summary + tail must fit in `available`.
    let summary_overhead = 50; // approximate tokens for the summary message wrapper
    let prefix_budget = available.saturating_sub(tail_tokens + summary_overhead);

    let mut head = 0;
    let mut prefix_tokens: u32 = 0;
    while head < tail_start {
        let cost = msg_token_cost(&messages[head]);
        if prefix_tokens + cost > prefix_budget {
            break;
        }
        prefix_tokens += cost;
        head += 1;
    }

    // Ensure head doesn't split a tool-call/result pair.
    // If head lands on a ToolResult, back up to before the corresponding assistant.
    while head > 0 && head < tail_start && matches!(messages[head], Message::ToolResult { .. }) {
        head -= 1;
    }

    let trimmed_count = tail_start.saturating_sub(head) as u32;
    if trimmed_count == 0 {
        // Nothing meaningful to compact — the tail already covers everything.
        return CompactionResult {
            messages: messages.to_vec(),
            trimmed_count: 0,
            tokens_before,
            tokens_after: tokens_before,
            trim_start: 0,
        };
    }

    // Build prefix-preserving output: [0..head] + [summary] + [tail_start..]
    let mut output: Vec<Message> = Vec::with_capacity(head + 1 + (messages.len() - tail_start));
    output.extend_from_slice(&messages[..head]);

    // Deterministic summary of the trimmed middle region.
    let summary = if config.strategy == CompactionStrategy::Textual {
        compacted_summary(&messages[head..tail_start], trimmed_count)
    } else {
        // Llm strategy: insert a placeholder that the caller will replace
        // with an LLM-generated summary.
        None
    };

    if let Some(summary_msg) = summary {
        output.push(summary_msg);
    } else if config.strategy == CompactionStrategy::Llm {
        // Placeholder — the caller (build_context) will call summarize_compacted_messages
        // and replace this with a proper LLM summary.
        output.push(Message::User {
            content: vec![theta_ai::ContentBlock::text(
                "[Compacted context — summarizing...]",
            )],
            timestamp: 0,
        });
    }

    output.extend_from_slice(&messages[tail_start..]);
    let tokens_after = total_tokens(&output);

    CompactionResult {
        messages: output,
        trimmed_count,
        tokens_before,
        tokens_after,
        trim_start: head,
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

pub(crate) fn total_tokens(messages: &[Message]) -> u32 {
    messages.iter().map(msg_token_cost).sum()
}

pub(crate) fn msg_token_cost(msg: &Message) -> u32 {
    match msg {
        Message::User { .. } | Message::Assistant { .. } | Message::ToolResult { .. } => {
            msg.token_count()
        }
        Message::ModelChange { .. } | Message::ThinkingLevelChange { .. } => 0,
    }
}
