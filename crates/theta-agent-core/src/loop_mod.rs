//! Agent loop: the core turn execution engine.
//!
//! Implements the nested loop pattern:
//! - Outer loop: handles follow-up turns (until shouldStopAfterTurn or queue empty)
//! - Inner loop: handles LLM call + tool execution cycle

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tracing;

use futures::StreamExt;
use theta_ai::event::EventAccumulator;
use theta_ai::{
    AssistantMessageEvent, ContentBlock, Context, LlmProvider, Message, StopReason, StreamOptions,
    ThinkingLevel,
};

use crate::error::AgentError;
use crate::events::{AgentEvent, TurnDecisionReason};
use crate::hooks::Hooks;
use crate::state::AgentState;
use crate::tools;
use crate::types::{AgentIntent, AgentLoopConfig, CompactionStrategy, ToolCall};

/// Drain all messages from a shared queue and add them to state.
fn drain_queue(queue: &Arc<Mutex<Vec<(Message, u64)>>>, state: &mut AgentState) -> bool {
    let mut guard = queue.lock().expect("queue lock poisoned");
    if guard.is_empty() {
        return false;
    }
    for (msg, _ts) in guard.drain(..) {
        state.messages.push(msg);
    }
    true
}

/// Check if the abort token has been triggered, accounting for steering.
macro_rules! check_abort {
    ($token:expr, $steering:expr) => {
        let has_steering = !$steering.lock().expect("lock").is_empty();
        if !has_steering {
            if let Some(ref token) = $token {
                if token.is_cancelled() {
                    return Err(AgentError::Aborted);
                }
            }
        }
    };
}

/// Run the outer agent loop for a `prompt` call.
/// Always emits AgentStart and AgentEnd regardless of success/error.
#[allow(clippy::too_many_arguments)]
pub async fn run_prompt_loop(
    state: &mut AgentState,
    provider: &dyn LlmProvider,
    hooks: &Arc<dyn Hooks>,
    config: &AgentLoopConfig,
    event_tx: &broadcast::Sender<AgentEvent>,
    abort_token: Option<CancellationToken>,
    steering_abort: Arc<AtomicBool>,
    steering_queue: Arc<Mutex<Vec<(Message, u64)>>>,
    follow_up_queue: Arc<Mutex<Vec<(Message, u64)>>>,
) -> Result<(), AgentError> {
    let _ = event_tx.send(AgentEvent::AgentStart);

    let result = run_outer_loop(
        state,
        provider,
        hooks,
        config,
        event_tx,
        abort_token,
        steering_abort,
        steering_queue,
        follow_up_queue,
    )
    .await;

    let aborted = matches!(result, Err(AgentError::Aborted));
    let _ = event_tx.send(AgentEvent::AgentEnd { aborted });

    result
}

#[allow(clippy::too_many_arguments)]
async fn run_outer_loop(
    state: &mut AgentState,
    provider: &dyn LlmProvider,
    hooks: &Arc<dyn Hooks>,
    config: &AgentLoopConfig,
    event_tx: &broadcast::Sender<AgentEvent>,
    abort_token: Option<CancellationToken>,
    steering_abort: Arc<AtomicBool>,
    steering_queue: Arc<Mutex<Vec<(Message, u64)>>>,
    follow_up_queue: Arc<Mutex<Vec<(Message, u64)>>>,
) -> Result<(), AgentError> {
    let mut turn_index: u32 = 0;

    loop {
        check_abort!(abort_token, steering_queue);

        run_single_turn(
            state,
            provider,
            hooks,
            config,
            event_tx,
            turn_index,
            abort_token.clone(),
            steering_abort.clone(),
            &steering_queue,
        )
        .await?;

        let _ = event_tx.send(AgentEvent::TurnEnd { turn_index });

        // Check hooks: should we stop?
        if hooks.should_stop_after_turn(state).await {
            tracing::debug!("hooks.should_stop_after_turn returned true");
            break;
        }

        // Check if there are more follow-ups or steering queued.
        let has_follow = !follow_up_queue.lock().expect("lock").is_empty();
        let has_steer = !steering_queue.lock().expect("lock").is_empty();
        if !has_follow && !has_steer {
            break;
        }

        // Drain them into state for the next turn.
        drain_queue(&follow_up_queue, state);
        drain_queue(&steering_queue, state);

        turn_index += 1;
    }

    Ok(())
}

/// Run a single turn of the agent loop (one LLM call, possible tool cycles).
#[allow(clippy::too_many_arguments)]
async fn run_single_turn(
    state: &mut AgentState,
    provider: &dyn LlmProvider,
    hooks: &Arc<dyn Hooks>,
    config: &AgentLoopConfig,
    event_tx: &broadcast::Sender<AgentEvent>,
    turn_index: u32,
    abort_token: Option<CancellationToken>,
    steering_abort: Arc<AtomicBool>,
    steering_queue: &Arc<Mutex<Vec<(Message, u64)>>>,
) -> Result<(), AgentError> {
    let _ = event_tx.send(AgentEvent::TurnStart { turn_index });

    // Inject any prepare-next-turn messages.
    let prepend = hooks.prepare_next_turn(state).await;
    for msg in prepend {
        state.messages.push(msg);
    }

    // Inner loop: LLM call + tool execution.
    let mut tool_round: u32 = 0;
    let mut empty_assistant_retries: u32 = 0;
    let mut consecutive_noop_rounds: u32 = 0;
    let mut executed_tools_in_turn = false;
    let mut repeated_tool_signature_counts: HashMap<String, u32> = HashMap::new();
    let max_same_tool_signature_repeats = config.max_same_tool_call_repeats.unwrap_or(6);

    loop {
        // Drain any steering messages — they interrupt the current turn.
        drain_queue(steering_queue, state);

        // Only check abort if no steering messages are pending.
        check_abort!(abort_token, steering_queue);

        if let Some(max_rounds) = config.max_tool_rounds
            && tool_round >= max_rounds
        {
            tracing::warn!("max tool rounds reached ({max_rounds})");
            let _ = event_tx.send(AgentEvent::TurnDecision {
                reason: TurnDecisionReason::MaxRounds,
                details: format!(
                    "stopped after reaching max tool rounds ({max_rounds}); likely provider/tool-call loop"
                ),
                turn: turn_index,
                round: tool_round,
            });
            break;
        }
        let intent = infer_intent(latest_user_text(state).as_deref().unwrap_or_default());

        // Build the LLM context from current state, with compaction.
        let (context, compaction_stats, replay_stats) =
            build_context(state, provider, config, event_tx).await;

        // Emit compaction event if messages were trimmed.
        if let Some(stats) = compaction_stats {
            let _ = event_tx.send(AgentEvent::ContextCompacted {
                trimmed_count: stats.trimmed_count,
                tokens_before: stats.tokens_before,
                tokens_after: stats.tokens_after,
            });
        }
        if let Some(stats) = replay_stats {
            let _ = event_tx.send(AgentEvent::ReplaySanitized {
                dropped_assistant_messages: stats.dropped_assistant_messages,
                synthesized_tool_results: stats.synthesized_tool_results,
                normalized_tool_call_ids: stats.normalized_tool_call_ids,
            });
        }

        tracing::debug!(
            turn = turn_index,
            round = tool_round,
            messages = context.messages.len(),
            tools = context.tools.len(),
            "calling LLM",
        );

        // Notify that we're starting a message.
        let _ = event_tx.send(AgentEvent::MessageStart);
        state.is_streaming = true;

        // Build stream options.
        let stream_options = StreamOptions {
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            thinking_level: Some(state.thinking_level),
            include_usage: config.include_usage,
            timeout_ms: config.provider_timeout_ms,
            ..Default::default()
        };

        // Call the LLM provider and consume the stream.
        match run_llm_stream(
            state,
            provider,
            &context,
            &stream_options,
            config,
            event_tx,
            abort_token.clone(),
            steering_abort.clone(),
            true,
        )
        .await
        {
            Ok((assistant_msg, stop_reason, unresolved_tool_calls)) => {
                state.is_streaming = false;

                let assistant_has_text = match &assistant_msg {
                    Message::Assistant { content, .. } => content.iter().any(|block| {
                        matches!(block, ContentBlock::Text { text } if !text.trim().is_empty())
                    }),
                    _ => false,
                };

                state.add_assistant_message(assistant_msg.clone());

                let _ = event_tx.send(AgentEvent::MessageEnd {
                    message: assistant_msg,
                });

                let tool_calls =
                    ToolCall::from_message(state.messages.last().expect("just pushed"));
                let has_tool_calls = !tool_calls.is_empty();

                if !assistant_has_text && !has_tool_calls {
                    if empty_assistant_retries < 2 {
                        empty_assistant_retries += 1;
                        let _ = event_tx.send(AgentEvent::Error {
                            message: "empty assistant response; retrying same turn".to_string(),
                        });
                        state.messages.push(Message::User {
                            content: vec![ContentBlock::text(
                                "Previous response was empty. Continue and provide the requested answer or emit required tool calls now.",
                            )],
                            timestamp: now_ms(),
                        });
                        tool_round += 1;
                        continue;
                    }

                    let _ = event_tx.send(AgentEvent::Error {
                        message: "assistant produced no text and no tool calls after retries"
                            .to_string(),
                    });
                    break;
                }

                empty_assistant_retries = 0;

                let unresolved_tool_use = unresolved_tool_calls > 0
                    && !has_tool_calls
                    && stop_reason == Some(StopReason::ToolUse);

                if unresolved_tool_use {
                    if assistant_has_text {
                        tracing::warn!(
                            unresolved_tool_calls,
                            "ignoring unresolved tool-call state because assistant returned text"
                        );
                    } else {
                        let _ = event_tx.send(AgentEvent::Error {
                            message: format!(
                                "tool-call parsing incomplete: {unresolved_tool_calls} unresolved tool call(s); requesting tool-call replay"
                            ),
                        });
                        state.messages.push(Message::User {
                            content: vec![ContentBlock::text(
                                "Previous turn indicated tool use, but tool-call payload was incomplete. Re-emit the tool call(s) now using function-calling only, no prose.",
                            )],
                            timestamp: now_ms(),
                        });
                        tool_round += 1;
                        continue;
                    }
                }

                if !has_tool_calls {
                    let assistant_text =
                        assistant_text_opt(&state.messages[state.messages.len() - 1])
                            .unwrap_or_default();

                    if intent == AgentIntent::PlanOnly || intent == AgentIntent::Clarify {
                        break;
                    }

                    if intent == AgentIntent::Execute
                        && !executed_tools_in_turn
                        && (consecutive_noop_rounds < 1
                            || looks_like_execution_promise(&assistant_text))
                    {
                        consecutive_noop_rounds += 1;
                        let _ = event_tx.send(AgentEvent::TurnDecision {
                            reason: TurnDecisionReason::NoopRetry,
                            details: "execute intent produced no tool calls; retrying once"
                                .to_string(),
                            turn: turn_index,
                            round: tool_round,
                        });
                        state.messages.push(Message::User {
                            content: vec![ContentBlock::text(VALIDATION_RETRY_PROMPT)],
                            timestamp: now_ms(),
                        });
                        tool_round += 1;
                        continue;
                    }
                    if intent == AgentIntent::Execute
                        && !executed_tools_in_turn
                        && classify_action_blocker(&assistant_text)
                    {
                        let _ = event_tx.send(AgentEvent::TurnDecision {
                            reason: TurnDecisionReason::BlockedNoop,
                            details: "execute intent stopped due to explicit blocker".to_string(),
                            turn: turn_index,
                            round: tool_round,
                        });
                    }

                    if intent == AgentIntent::Inspect && !executed_tools_in_turn {
                        consecutive_noop_rounds += 1;
                        if consecutive_noop_rounds <= 1 {
                            state.messages.push(Message::User {
                                content: vec![ContentBlock::text(
                                    "This is an inspection request. Use read-only tools now and report findings.",
                                )],
                                timestamp: now_ms(),
                            });
                            tool_round += 1;
                            continue;
                        }
                    }
                    break;
                }
                consecutive_noop_rounds = 0;

                if stop_reason != Some(StopReason::ToolUse) {
                    let _ = event_tx.send(AgentEvent::Error {
                        message: "tool calls detected without tool_use stop reason; executing tools via fallback".to_string(),
                    });
                }

                for tc in &tool_calls {
                    let signature = format!("{}:{}", tc.name, tc.arguments);
                    let count = repeated_tool_signature_counts
                        .entry(signature.clone())
                        .and_modify(|c| *c += 1)
                        .or_insert(1);
                    if *count > max_same_tool_signature_repeats {
                        let _ = event_tx.send(AgentEvent::Error {
                            message: format!(
                                "agent stopped repeated identical tool call loop: '{}' exceeded {} repeats",
                                signature, max_same_tool_signature_repeats
                            ),
                        });
                        return Ok(());
                    }
                }

                if intent == AgentIntent::AnalyzeOnly {
                    let mut allowed = Vec::new();
                    for tc in &tool_calls {
                        if is_read_only_tool_call(tc) {
                            allowed.push(tc.clone());
                        } else {
                            let _ = event_tx.send(AgentEvent::TurnDecision {
                                reason: TurnDecisionReason::AnalyzeOnlyRejectedTool,
                                details: format!(
                                    "blocked mutating tool call '{}' during analyze-only turn",
                                    tc.name
                                ),
                                turn: turn_index,
                                round: tool_round,
                            });
                        }
                    }
                    if allowed.is_empty() {
                        break;
                    }
                    tools::execute_tool_calls(
                        state,
                        &allowed,
                        abort_token.clone(),
                        event_tx,
                        hooks,
                    )
                    .await?;
                    executed_tools_in_turn = true;
                } else {
                    tools::execute_tool_calls(
                        state,
                        &tool_calls,
                        abort_token.clone(),
                        event_tx,
                        hooks,
                    )
                    .await?;
                    executed_tools_in_turn = true;
                }

                tool_round += 1;
            }
            Err(AgentError::Aborted) => {
                state.is_streaming = false;
                // If steering messages are queued, the abort was intentional
                // to interrupt for steering. Reset the flag, drain, and continue.
                if drain_queue(steering_queue, state) {
                    steering_abort.store(false, Ordering::SeqCst);
                    tracing::debug!("aborted for steering, continuing turn");
                    continue;
                }
                // Otherwise propagate the abort.
                return Err(AgentError::Aborted);
            }
            Err(e) => {
                state.is_streaming = false;
                return Err(e);
            }
        }
    }

    Ok(())
}

const VALIDATION_RETRY_PROMPT: &str = "This is an execution request. Call the relevant tools now. If blocked, state the blocker clearly and stop.";

fn classify_action_blocker(text: &str) -> bool {
    let t = text.to_lowercase();
    let missing_info = [
        "need more detail",
        "what should i implement",
        "provide the target",
        "please provide",
        "missing info",
        "which file",
    ]
    .iter()
    .any(|kw| t.contains(kw));
    let permission = [
        "permission denied",
        "not permitted",
        "need approval",
        "requires approval",
        "access denied",
    ]
    .iter()
    .any(|kw| t.contains(kw));
    let runtime = [
        "no such file or directory",
        "path not found",
        "sandbox",
        "token expired",
        "authentication",
        "network",
        "timeout",
        "cannot access",
        "blocked",
    ]
    .iter()
    .any(|kw| t.contains(kw));
    missing_info || permission || runtime
}

fn tokenize_words(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

fn contains_token_sequence(tokens: &[String], phrase_tokens: &[&str]) -> bool {
    if phrase_tokens.is_empty() {
        return false;
    }
    if phrase_tokens.len() == 1 {
        return tokens.iter().any(|tok| tok == phrase_tokens[0]);
    }
    tokens.windows(phrase_tokens.len()).any(|window| {
        window
            .iter()
            .map(String::as_str)
            .eq(phrase_tokens.iter().copied())
    })
}

fn is_read_only_tool_call(tc: &ToolCall) -> bool {
    if matches!(tc.name.as_str(), "read" | "ls" | "find" | "grep") {
        return true;
    }
    if tc.name != "bash" {
        return false;
    }
    let Some(cmd) = tc.arguments.get("command").and_then(|v| v.as_str()) else {
        return false;
    };
    is_read_only_bash_command(cmd)
}

fn is_read_only_bash_command(command: &str) -> bool {
    for segment in split_shell_command_segments(command) {
        let tokens = tokenize_shell_segment(segment);
        if tokens.is_empty() {
            continue;
        }
        if !is_read_only_tokenized_command(&tokens) {
            return false;
        }
    }
    true
}

fn is_read_only_tokenized_command(tokens: &[String]) -> bool {
    let first = tokens[0].as_str();
    match first {
        "cat" | "head" | "tail" | "wc" | "ls" | "rg" | "grep" | "find" => true,
        "sed" => !tokens.iter().any(|t| t == "-i" || t.starts_with("-i")),
        "git" => {
            let sub = tokens.get(1).map(String::as_str).unwrap_or_default();
            matches!(
                sub,
                "status" | "diff" | "show" | "log" | "rev-parse" | "branch" | "remote" | "ls-files"
            )
        }
        _ => false,
    }
}

fn split_shell_command_segments(command: &str) -> Vec<&str> {
    command
        .split(&[';', '|', '&'][..])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

fn tokenize_shell_segment(segment: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    for ch in segment.chars() {
        match ch {
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            c if c.is_whitespace() && !in_single && !in_double => {
                if !cur.is_empty() {
                    tokens.push(cur.clone());
                    cur.clear();
                }
            }
            _ => cur.push(ch),
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}

fn assistant_text_opt(message: &Message) -> Option<String> {
    match message {
        Message::Assistant { content, .. } => {
            let text = content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if text.trim().is_empty() {
                None
            } else {
                Some(text)
            }
        }
        _ => None,
    }
}

fn latest_user_text(state: &AgentState) -> Option<String> {
    state.messages.iter().rev().find_map(|msg| match msg {
        Message::User { content, .. } => {
            let text = content
                .iter()
                .filter_map(|block| match block {
                    ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("\n");
            if text.trim().is_empty() {
                None
            } else {
                Some(text)
            }
        }
        _ => None,
    })
}

fn looks_like_execution_request(text: &str) -> bool {
    let t = text.to_lowercase();
    let tokens = tokenize_words(&t);
    [
        &["implement"][..],
        &["fix"],
        &["patch"],
        &["edit"],
        &["modify"],
        &["update", "code"],
        &["change", "code"],
        &["add", "code"],
        &["remove", "code"],
        &["refactor"],
        &["commit"],
        &["push"],
        &["run", "git"],
        &["run", "it"],
        &["do", "it"],
    ]
    .iter()
    .any(|seq| contains_token_sequence(&tokens, seq))
}

fn looks_like_execution_promise(text: &str) -> bool {
    let t = text.to_lowercase();
    [
        "on it",
        "i'll implement",
        "i will implement",
        "i'll patch",
        "i will patch",
        "starting code changes",
    ]
    .iter()
    .any(|kw| t.contains(kw))
}

fn infer_intent(text: &str) -> AgentIntent {
    let t = text.to_lowercase();
    if t.trim().is_empty() {
        return AgentIntent::Default;
    }
    if (t.contains("plan only")
        || t.contains("just plan")
        || t.contains("brainstorm")
        || t.contains("do not implement")
        || t.contains("don't implement"))
        && !looks_like_execution_request(&t)
    {
        return AgentIntent::PlanOnly;
    }
    if t.trim() == "do it" || t.trim() == "fix it" {
        return AgentIntent::Clarify;
    }
    if t.contains("review")
        || t.contains("analyze")
        || t.contains("analyse")
        || t.contains("architecture")
    {
        return AgentIntent::AnalyzeOnly;
    }
    if t.contains("commit") || t.contains("push") || t.contains("apply patch") {
        return AgentIntent::Execute;
    }
    if t.contains("inspect") || t.contains("check") || t.contains("what changed") {
        return AgentIntent::Inspect;
    }
    if looks_like_execution_request(&t) {
        return AgentIntent::Execute;
    }
    AgentIntent::Default
}

/// Consume an LLM stream, emitting AgentEvents and accumulating content.
/// Returns the assembled assistant message and stop reason.
/// Includes retry logic with exponential backoff for transient provider errors.
#[allow(clippy::too_many_arguments)]
async fn run_llm_stream(
    state: &AgentState,
    provider: &dyn LlmProvider,
    context: &Context,
    options: &StreamOptions,
    config: &AgentLoopConfig,
    event_tx: &broadcast::Sender<AgentEvent>,
    abort_token: Option<CancellationToken>,
    steering_abort: Arc<AtomicBool>,
    emit_events: bool,
) -> Result<(Message, Option<StopReason>, usize), AgentError> {
    let retry = &config.retry;

    // Retry loop for provider.stream().
    let mut stream;
    let mut attempt: u32 = 0;

    loop {
        match provider.stream(&state.model, context, options).await {
            Ok(s) => {
                stream = s;
                break;
            }
            Err(e) => {
                let msg = e.to_string();
                if !retry.is_retryable(&msg) || attempt >= retry.max_retries {
                    return Err(AgentError::Llm(e));
                }
                attempt += 1;
                let delay_ms = retry
                    .base_delay_ms
                    .saturating_mul(2u64.pow(attempt.saturating_sub(1)));
                if emit_events {
                    let _ = event_tx.send(AgentEvent::Retrying { attempt, delay_ms });
                }
                tracing::warn!(
                    attempt = attempt,
                    delay_ms = delay_ms,
                    error = %msg,
                    "provider call failed, retrying"
                );
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
        }
    }

    let mut accumulator = EventAccumulator::new();

    // Consume the stream, emitting events as we go.
    while let Some(event) = stream.next().await {
        // Check permanent abort.
        if let Some(ref token) = abort_token
            && token.is_cancelled()
        {
            return Err(AgentError::Aborted);
        }
        // Check per-stream steering abort.
        if steering_abort.load(Ordering::SeqCst) {
            return Err(AgentError::Aborted);
        }

        accumulator.feed(&event);

        // Emit corresponding agent events.
        match &event {
            AssistantMessageEvent::TextDelta { text } if emit_events => {
                let _ = event_tx.send(AgentEvent::TextDelta { text: text.clone() });
            }
            AssistantMessageEvent::ThinkingDelta { thinking } if emit_events => {
                let _ = event_tx.send(AgentEvent::ThinkingDelta {
                    thinking: thinking.clone(),
                });
            }
            AssistantMessageEvent::ToolCallStart { id, name } if emit_events => {
                let _ = event_tx.send(AgentEvent::ToolCallStart {
                    id: id.clone(),
                    name: name.clone(),
                });
            }
            AssistantMessageEvent::ToolCallDelta { id, arguments } if emit_events => {
                let _ = event_tx.send(AgentEvent::ToolCallDelta {
                    id: id.clone(),
                    arguments: arguments.clone(),
                });
            }
            AssistantMessageEvent::ToolCallEnd { id } if emit_events => {
                let _ = event_tx.send(AgentEvent::ToolCallEnd { id: id.clone() });
            }
            AssistantMessageEvent::Done { .. } | AssistantMessageEvent::Error { .. } => {}
            _ => {}
        }
    }

    // Build the assistant message from accumulated events.
    let assistant_msg = Message::Assistant {
        content: accumulator.content_blocks(),
        api: Some(state.model.api),
        provider: Some(state.model.provider),
        model: Some(state.model.id.clone()),
        usage: accumulator.usage().cloned(),
        stop_reason: accumulator.stop_reason(),
        error_message: accumulator.error_message().map(|s| s.to_string()),
        timestamp: now_ms(),
    };

    Ok((
        assistant_msg,
        accumulator.stop_reason(),
        accumulator.unresolved_tool_call_count(),
    ))
}

/// Build the LLM Context from the current agent state, with optional
/// context compaction to stay within the model's context window.
struct CompactionStats {
    trimmed_count: u32,
    tokens_before: u32,
    tokens_after: u32,
}

async fn build_context(
    state: &AgentState,
    provider: &dyn LlmProvider,
    config: &AgentLoopConfig,
    event_tx: &broadcast::Sender<AgentEvent>,
) -> (
    Context,
    Option<CompactionStats>,
    Option<theta_ai::ReplaySanitizationStats>,
) {
    let system = if state.system_prompt.is_empty() {
        None
    } else {
        Some(state.system_prompt.clone())
    };

    // Approximate system prompt tokens.
    let sys_tokens: u32 = system
        .as_ref()
        .map(|blocks| {
            blocks
                .iter()
                .map(|b| {
                    theta_ai::approximate_token_count(&serde_json::to_string(b).unwrap_or_default())
                })
                .sum()
        })
        .unwrap_or(0);

    let all_messages = state.llm_messages();
    let all_slice: Vec<theta_ai::Message> = all_messages.into_iter().cloned().collect();
    let (sanitized_messages, replay_stats) =
        theta_ai::sanitize_messages_for_replay(&all_slice, &state.model);

    let mut compact_result = crate::compact::compact_messages(
        &sanitized_messages,
        sys_tokens,
        state.model.context_window,
        &config.compaction,
    );

    if compact_result.trimmed_count > 0 && config.compaction.strategy == CompactionStrategy::Llm {
        let trimmed_len = (compact_result.trimmed_count as usize).min(all_slice.len());
        match summarize_compacted_messages(
            state,
            provider,
            &all_slice[..trimmed_len],
            config,
            event_tx,
        )
        .await
        {
            Ok(summary) => {
                if let Some(first) = compact_result.messages.first_mut() {
                    *first = summary;
                } else {
                    compact_result.messages.insert(0, summary);
                }
                compact_result.tokens_after = compact_result
                    .messages
                    .iter()
                    .map(|message| message.token_count())
                    .sum();
            }
            Err(error) => {
                tracing::warn!(error = %error, "LLM compaction summary failed; using deterministic summary");
            }
        }
    }

    let tools: Vec<theta_ai::Tool> = state
        .tools
        .iter()
        .map(|t| theta_ai::Tool {
            name: t.name().to_string(),
            description: t.description().to_string(),
            parameters: t.parameters(),
        })
        .collect();

    let stats = (compact_result.trimmed_count > 0).then_some(CompactionStats {
        trimmed_count: compact_result.trimmed_count,
        tokens_before: compact_result.tokens_before,
        tokens_after: compact_result.tokens_after,
    });

    (
        Context {
            system,
            messages: compact_result.messages,
            tools,
            thinking_level: Some(state.thinking_level),
        },
        stats,
        replay_stats.changed().then_some(replay_stats),
    )
}

async fn summarize_compacted_messages(
    state: &AgentState,
    provider: &dyn LlmProvider,
    trimmed: &[Message],
    config: &AgentLoopConfig,
    event_tx: &broadcast::Sender<AgentEvent>,
) -> Result<Message, AgentError> {
    let transcript = trimmed
        .iter()
        .map(message_to_summary_text)
        .collect::<Vec<_>>()
        .join("\n\n");

    let context = Context {
        system: Some(vec![ContentBlock::text(
            "Summarize compacted coding-agent conversation context. Preserve concrete user goals, files, commands, tool results, decisions, unresolved tasks, constraints, and current project state. Be concise but specific. Do not invent facts.",
        )]),
        messages: vec![Message::User {
            content: vec![ContentBlock::text(format!(
                "Summarize this older transcript for future context:\n\n{transcript}"
            ))],
            timestamp: now_ms(),
        }],
        tools: Vec::new(),
        thinking_level: Some(ThinkingLevel::Off),
    };
    let options = StreamOptions {
        max_tokens: Some(config.compaction.summary_max_tokens),
        temperature: Some(0.2),
        thinking_level: Some(ThinkingLevel::Off),
        include_usage: false,
        timeout_ms: config.provider_timeout_ms,
        ..Default::default()
    };

    let (message, _, _) = run_llm_stream(
        state,
        provider,
        &context,
        &options,
        config,
        event_tx,
        None,
        Arc::new(AtomicBool::new(false)),
        false,
    )
    .await?;

    Ok(Message::Assistant {
        content: vec![ContentBlock::text(format!(
            "Context compacted by LLM summary:\n{}",
            assistant_text(&message)
        ))],
        api: Some(state.model.api),
        provider: Some(state.model.provider),
        model: Some(state.model.id.clone()),
        usage: None,
        stop_reason: None,
        error_message: None,
        timestamp: now_ms(),
    })
}

fn message_to_summary_text(message: &Message) -> String {
    match message {
        Message::User { content, .. } => format!("User: {}", content_to_text(content)),
        Message::Assistant { content, .. } => format!("Assistant: {}", content_to_text(content)),
        Message::ToolResult {
            tool_name,
            content,
            is_error,
            ..
        } => format!(
            "ToolResult({tool_name}, error={is_error}): {}",
            content_to_text(content)
        ),
        Message::ModelChange { model_id, .. } => {
            format!("Model changed to {}", model_id.as_deref().unwrap_or("?"))
        }
        Message::ThinkingLevelChange { level, .. } => {
            format!("Thinking level changed to {level:?}")
        }
    }
}

fn assistant_text(message: &Message) -> String {
    match message {
        Message::Assistant { content, .. } => content_to_text(content),
        _ => String::new(),
    }
}

fn content_to_text(content: &[ContentBlock]) -> String {
    content
        .iter()
        .map(|block| match block {
            ContentBlock::Text { text } => text.clone(),
            ContentBlock::Thinking { thinking, .. } => thinking.clone(),
            ContentBlock::ToolCall {
                name, arguments, ..
            } => format!("tool_call {name} {arguments}"),
            ContentBlock::ToolResult {
                tool_name, content, ..
            } => format!("tool_result {tool_name}: {}", content_to_text(content)),
            ContentBlock::Image { .. } => "[image]".to_string(),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Get current time in milliseconds since epoch.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infer_intent_does_not_treat_provider_path_as_commit_ops() {
        let intent = infer_intent(
            "Use grep tool on crates/theta-ai/src/providers/openai_compat.rs and report lines",
        );
        assert_ne!(intent, AgentIntent::Execute);
    }

    #[test]
    fn infer_intent_detects_analyze_only() {
        let intent = infer_intent("Please review the architecture and analyze the current design.");
        assert_eq!(intent, AgentIntent::AnalyzeOnly);
    }

    #[test]
    fn token_sequence_matching_requires_boundaries() {
        let tokens = tokenize_words("providers openai_compat");
        assert!(!contains_token_sequence(&tokens, &["pr"]));
    }

    #[test]
    fn read_only_tool_call_detects_common_read_only_bash_commands() {
        let tc = ToolCall {
            id: "1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command":"sed -n '1,20p' Cargo.toml && git show HEAD~1"}),
        };
        assert!(is_read_only_tool_call(&tc));
    }

    #[test]
    fn read_only_tool_call_rejects_mutating_git_commands() {
        let tc = ToolCall {
            id: "1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command":"git commit -m test"}),
        };
        assert!(!is_read_only_tool_call(&tc));
    }

    #[test]
    fn read_only_tool_call_rejects_sed_in_place() {
        let tc = ToolCall {
            id: "1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command":"sed -i '' 's/a/b/g' file.txt"}),
        };
        assert!(!is_read_only_tool_call(&tc));
    }
}
