//! Agent loop: the core turn execution engine.
//!
//! Implements the nested loop pattern:
//! - Outer loop: handles follow-up turns (until shouldStopAfterTurn or queue empty)
//! - Inner loop: handles LLM call + tool execution cycle

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
use crate::events::AgentEvent;
use crate::hooks::Hooks;
use crate::state::AgentState;
use crate::tools;
use crate::types::{AgentLoopConfig, ToolCall};

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
    hooks: &dyn Hooks,
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
    hooks: &dyn Hooks,
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
    hooks: &dyn Hooks,
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
    let max_rounds = config.max_tool_rounds.unwrap_or(20);
    let mut empty_assistant_retries: u32 = 0;
    let mut action_noop_retries: u32 = 0;
    let mut inspection_noop_retries: u32 = 0;
    let mut commit_ops_noop_retries: u32 = 0;
    let mut validation_noop_retries: u32 = 0;
    let mut reproduction_noop_retries: u32 = 0;
    let mut promised_execution_noop_retries: u32 = 0;
    let mut executed_tools_in_turn = false;
    let mut flags = recompute_turn_flags(state);
    let mut executed_inspection_tools_in_turn = false;
    let mut executed_git_tools_in_turn = false;
    let mut executed_validation_tools_in_turn = false;

    loop {
        // Drain any steering messages — they interrupt the current turn.
        drain_queue(steering_queue, state);

        // Only check abort if no steering messages are pending.
        check_abort!(abort_token, steering_queue);

        if tool_round >= max_rounds {
            tracing::warn!("max tool rounds reached ({max_rounds})");
            let _ = event_tx.send(AgentEvent::Error {
                message: format!(
                    "agent stopped after reaching max tool rounds ({max_rounds}); likely provider/tool-call loop"
                ),
            });
            break;
        }

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
                        flags = recompute_turn_flags(state);
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
                        flags = recompute_turn_flags(state);
                        tool_round += 1;
                        continue;
                    }
                }

                if !has_tool_calls {
                    let assistant_text =
                        assistant_text_opt(&state.messages[state.messages.len() - 1])
                            .unwrap_or_default();

                    if looks_like_execution_promise(&assistant_text)
                        && promised_execution_noop_retries < 1
                    {
                        promised_execution_noop_retries += 1;
                        let _ = event_tx.send(AgentEvent::Error {
                            message: "assistant promised execution but emitted no tool calls; retrying same turn".to_string(),
                        });
                        state.messages.push(Message::User {
                            content: vec![ContentBlock::text(ACTION_RETRY_PROMPT)],
                            timestamp: now_ms(),
                        });
                        flags = recompute_turn_flags(state);
                        tool_round += 1;
                        continue;
                    }

                    if flags.requires_plan_only {
                        break;
                    }

                    if flags.requires_inspection
                        && !executed_inspection_tools_in_turn
                        && inspection_noop_retries < 1
                    {
                        inspection_noop_retries += 1;
                        let _ = event_tx.send(AgentEvent::Error {
                            message: "inspection turn produced no inspection tool calls; retrying same turn".to_string(),
                        });
                        state.messages.push(Message::User {
                            content: vec![ContentBlock::text(INSPECTION_RETRY_PROMPT)],
                            timestamp: now_ms(),
                        });
                        flags = recompute_turn_flags(state);
                        tool_round += 1;
                        continue;
                    }

                    if flags.requires_commit_ops
                        && !executed_git_tools_in_turn
                        && commit_ops_noop_retries < 1
                    {
                        commit_ops_noop_retries += 1;
                        let _ = event_tx.send(AgentEvent::Error {
                            message: "git turn produced no git tool calls; retrying same turn"
                                .to_string(),
                        });
                        state.messages.push(Message::User {
                            content: vec![ContentBlock::text(COMMIT_OPS_RETRY_PROMPT)],
                            timestamp: now_ms(),
                        });
                        flags = recompute_turn_flags(state);
                        tool_round += 1;
                        continue;
                    }

                    if flags.requires_reproduction
                        && !executed_inspection_tools_in_turn
                        && reproduction_noop_retries < 1
                    {
                        reproduction_noop_retries += 1;
                        let _ = event_tx.send(AgentEvent::Error {
                            message: "reproduction turn produced no evidence-gathering tool calls; retrying same turn".to_string(),
                        });
                        state.messages.push(Message::User {
                            content: vec![ContentBlock::text(REPRODUCTION_RETRY_PROMPT)],
                            timestamp: now_ms(),
                        });
                        flags = recompute_turn_flags(state);
                        tool_round += 1;
                        continue;
                    }

                    if flags.requires_clarification {
                        let _ = event_tx.send(AgentEvent::Error {
                            message: "request appears underspecified; assistant should ask one precise clarification".to_string(),
                        });
                    }

                    if flags.requires_action && !executed_tools_in_turn {
                        let blocker = classify_action_blocker(&assistant_text);

                        if blocker == ActionBlocker::None && action_noop_retries < 1 {
                            action_noop_retries += 1;
                            let _ = event_tx.send(AgentEvent::Error {
                                message: "action turn produced no tool calls and no explicit blocker; retrying same turn".to_string(),
                            });
                            state.messages.push(Message::User {
                                content: vec![ContentBlock::text(ACTION_RETRY_PROMPT)],
                                timestamp: now_ms(),
                            });
                            flags = recompute_turn_flags(state);
                            tool_round += 1;
                            continue;
                        }

                        if blocker != ActionBlocker::None {
                            let _ = event_tx.send(AgentEvent::Error {
                                message: format!(
                                    "action turn ended without tool calls due to explicit blocker ({})",
                                    blocker.as_str()
                                ),
                            });
                        } else if action_noop_retries >= 1 {
                            let _ = event_tx.send(AgentEvent::Error {
                                message: "action turn still produced no tool calls after retry; ending turn".to_string(),
                            });
                        } else if looks_like_execution_promise(&assistant_text) {
                            let _ = event_tx.send(AgentEvent::Error {
                                message: "assistant promised execution but emitted no tool calls"
                                    .to_string(),
                            });
                        } else {
                            let _ = event_tx.send(AgentEvent::Error {
                                message: "action turn produced no tool calls; ending turn"
                                    .to_string(),
                            });
                        }
                    }

                    if flags.requires_validation
                        && executed_tools_in_turn
                        && !executed_validation_tools_in_turn
                        && validation_noop_retries < 1
                    {
                        validation_noop_retries += 1;
                        let _ = event_tx.send(AgentEvent::Error {
                            message: "action turn completed without validation commands; retrying for validation".to_string(),
                        });
                        state.messages.push(Message::User {
                            content: vec![ContentBlock::text(VALIDATION_RETRY_PROMPT)],
                            timestamp: now_ms(),
                        });
                        flags = recompute_turn_flags(state);
                        tool_round += 1;
                        continue;
                    }

                    break;
                }

                if stop_reason != Some(StopReason::ToolUse) {
                    let _ = event_tx.send(AgentEvent::Error {
                        message: "tool calls detected without tool_use stop reason; executing tools via fallback".to_string(),
                    });
                }

                tools::execute_tool_calls(state, &tool_calls, abort_token.clone(), event_tx)
                    .await?;
                executed_tools_in_turn = true;
                if tool_calls.iter().any(is_inspection_tool_call) {
                    executed_inspection_tools_in_turn = true;
                }
                if tool_calls.iter().any(is_git_tool_call) {
                    executed_git_tools_in_turn = true;
                }
                if tool_calls.iter().any(is_validation_tool_call) {
                    executed_validation_tools_in_turn = true;
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

const ACTION_RETRY_PROMPT: &str = "This is an action request. Execute now by calling required tools first. If blocked, state the exact blocker briefly.";
const INSPECTION_RETRY_PROMPT: &str = "This is an inspection request. Run relevant read-only tools now and report findings; do not ask for confirmation.";
const COMMIT_OPS_RETRY_PROMPT: &str =
    "This requires git operations. Run the required git commands now via tools and report results.";
const VALIDATION_RETRY_PROMPT: &str =
    "Run validation commands now (tests/build/lint as appropriate) and report outcomes.";
const REPRODUCTION_RETRY_PROMPT: &str =
    "Reproduce or gather evidence first using tools (logs/status/tests), then report findings.";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActionBlocker {
    MissingInfo,
    Permission,
    RuntimeConstraint,
    None,
}

impl ActionBlocker {
    fn as_str(self) -> &'static str {
        match self {
            Self::MissingInfo => "missing_info",
            Self::Permission => "permission",
            Self::RuntimeConstraint => "runtime_constraint",
            Self::None => "none",
        }
    }
}

fn classify_action_blocker(text: &str) -> ActionBlocker {
    let t = text.to_lowercase();
    if [
        "need more detail",
        "what should i implement",
        "provide the target",
        "please provide",
        "missing info",
        "which file",
    ]
    .iter()
    .any(|kw| t.contains(kw))
    {
        return ActionBlocker::MissingInfo;
    }

    if [
        "permission denied",
        "not permitted",
        "need approval",
        "requires approval",
        "access denied",
    ]
    .iter()
    .any(|kw| t.contains(kw))
    {
        return ActionBlocker::Permission;
    }

    if [
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
    .any(|kw| t.contains(kw))
    {
        return ActionBlocker::RuntimeConstraint;
    }

    ActionBlocker::None
}

#[derive(Debug, Clone, Copy, Default)]
struct TurnFlags {
    requires_action: bool,
    requires_inspection: bool,
    requires_validation: bool,
    requires_reproduction: bool,
    requires_commit_ops: bool,
    requires_plan_only: bool,
    requires_clarification: bool,
}

fn determine_turn_flags(text: &str) -> TurnFlags {
    let t = text.to_lowercase();
    let tokens = tokenize_words(&t);

    let requires_plan_only = [
        &["plan", "only"][..],
        &["just", "plan"],
        &["brainstorm"],
        &["do", "not", "implement"],
        &["don't", "implement"],
    ]
    .iter()
    .any(|seq| contains_token_sequence(&tokens, seq));
    let requires_action = looks_like_execution_request(&t) && !requires_plan_only;
    let requires_inspection = [
        &["review"][..],
        &["inspect"],
        &["analyze"],
        &["analyse"],
        &["what", "changed"],
        &["current", "changes"],
        &["changes", "impact"],
        &["impact", "project"],
        &["uncommitted"],
        &["uncommited"],
        &["uncommitted", "changes"],
        &["uncommited", "changes"],
        &["git", "diff"],
        &["show", "diff"],
        &["git", "status"],
        &["check", "file"],
        &["check", "files"],
        &["check", "contents"],
    ]
    .iter()
    .any(|seq| contains_token_sequence(&tokens, seq));
    let requires_validation = [
        &["validate"][..],
        &["verify"],
        &["make", "sure"],
        &["ensure"],
        &["run", "tests"],
        &["test", "it"],
        &["cargo", "test"],
        &["pytest"],
    ]
    .iter()
    .any(|seq| contains_token_sequence(&tokens, seq));
    let requires_reproduction = [
        &["reproduce"][..],
        &["bug"],
        &["fails"],
        &["failure"],
        &["error"],
        &["issue"],
    ]
    .iter()
    .any(|seq| contains_token_sequence(&tokens, seq));
    let requires_commit_ops = [
        &["git"][..],
        &["commit"],
        &["push"],
        &["pull", "request"],
        &["stash"],
    ]
    .iter()
    .any(|seq| contains_token_sequence(&tokens, seq));
    let requires_clarification = [&["do", "it"][..], &["implement", "it"], &["fix", "it"]]
        .iter()
        .any(|seq| contains_token_sequence(&tokens, seq))
        && tokens.len() <= 2;

    tracing::debug!(
        "intent matched: action={} inspection={} validation={} reproduction={} git={} plan_only={} clarification={}",
        requires_action,
        requires_inspection,
        requires_validation,
        requires_reproduction,
        requires_commit_ops,
        requires_plan_only,
        requires_clarification
    );

    TurnFlags {
        requires_action,
        requires_inspection,
        requires_validation,
        requires_reproduction,
        requires_commit_ops,
        requires_plan_only,
        requires_clarification,
    }
}

fn recompute_turn_flags(state: &AgentState) -> TurnFlags {
    latest_user_text(state)
        .map(|text| determine_turn_flags(&text))
        .unwrap_or_default()
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

fn is_inspection_tool_call(tc: &ToolCall) -> bool {
    matches!(tc.name.as_str(), "read" | "ls" | "find" | "grep")
        || (tc.name == "bash"
            && tc
                .arguments
                .get("command")
                .and_then(|v| v.as_str())
                .is_some_and(|cmd| {
                    let c = cmd.to_lowercase();
                    c.contains("git status")
                        || c.contains("git diff")
                        || c.contains("cat ")
                        || c.contains("ls ")
                        || c.contains("rg ")
                }))
}

fn is_git_tool_call(tc: &ToolCall) -> bool {
    tc.name == "bash"
        && tc
            .arguments
            .get("command")
            .and_then(|v| v.as_str())
            .is_some_and(|cmd| cmd.trim_start().starts_with("git "))
}

fn is_validation_tool_call(tc: &ToolCall) -> bool {
    tc.name == "bash"
        && tc
            .arguments
            .get("command")
            .and_then(|v| v.as_str())
            .is_some_and(|cmd| {
                let c = cmd.to_lowercase();
                c.contains("cargo test")
                    || c.contains("cargo check")
                    || c.contains("cargo clippy")
                    || c.contains("cargo fmt")
                    || c.contains("npm test")
                    || c.contains("pytest")
                    || c.contains("go test")
            })
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
        "i'll",
        "i will",
        "starting now",
        "let me",
        "i can implement",
        "i can patch",
    ]
    .iter()
    .any(|kw| t.contains(kw))
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
                let _ = event_tx.send(AgentEvent::Retrying { attempt, delay_ms });
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
            AssistantMessageEvent::TextDelta { text } => {
                let _ = event_tx.send(AgentEvent::TextDelta { text: text.clone() });
            }
            AssistantMessageEvent::ThinkingDelta { thinking } => {
                let _ = event_tx.send(AgentEvent::ThinkingDelta {
                    thinking: thinking.clone(),
                });
            }
            AssistantMessageEvent::ToolCallStart { id, name } => {
                let _ = event_tx.send(AgentEvent::ToolCallStart {
                    id: id.clone(),
                    name: name.clone(),
                });
            }
            AssistantMessageEvent::ToolCallDelta { id, arguments } => {
                let _ = event_tx.send(AgentEvent::ToolCallDelta {
                    id: id.clone(),
                    arguments: arguments.clone(),
                });
            }
            AssistantMessageEvent::ToolCallEnd { id } => {
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

    if compact_result.trimmed_count > 0 && config.compaction.summarize_with_llm {
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

    let (message, _) =
        run_silent_llm_stream(state, provider, &context, &options, config, event_tx).await?;

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

async fn run_silent_llm_stream(
    state: &AgentState,
    provider: &dyn LlmProvider,
    context: &Context,
    options: &StreamOptions,
    config: &AgentLoopConfig,
    event_tx: &broadcast::Sender<AgentEvent>,
) -> Result<(Message, Option<StopReason>), AgentError> {
    let retry = &config.retry;
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
                let _ = event_tx.send(AgentEvent::Retrying { attempt, delay_ms });
                tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
            }
        }
    }

    let mut accumulator = EventAccumulator::new();
    while let Some(event) = stream.next().await {
        accumulator.feed(&event);
    }

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

    Ok((assistant_msg, accumulator.stop_reason()))
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
    fn turn_flags_do_not_treat_providers_as_pr() {
        let flags = determine_turn_flags(
            "Use grep tool on crates/theta-ai/src/providers/openai_compat.rs and report lines",
        );
        assert!(
            !flags.requires_commit_ops,
            "providers path should not trigger git intent"
        );
    }

    #[test]
    fn turn_flags_detect_explicit_commit_ops() {
        let flags = determine_turn_flags("Please commit changes and push after git status.");
        assert!(flags.requires_commit_ops);
    }

    #[test]
    fn turn_flags_check_file_contents_is_inspection_not_validation_or_git() {
        let flags = determine_turn_flags("check file contents in Cargo.toml");
        assert!(flags.requires_inspection);
        assert!(!flags.requires_validation);
        assert!(!flags.requires_commit_ops);
    }

    #[test]
    fn token_sequence_matching_requires_boundaries() {
        let tokens = tokenize_words("providers openai_compat");
        assert!(!contains_token_sequence(&tokens, &["pr"]));
    }
}
