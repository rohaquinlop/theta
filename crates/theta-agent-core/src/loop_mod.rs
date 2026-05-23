//! Agent loop: the core turn execution engine.
//!
//! Implements the nested loop pattern:
//! - Outer loop: handles follow-up turns (until shouldStopAfterTurn or queue empty)
//! - Inner loop: handles LLM call + tool execution cycle

use std::collections::HashMap;
use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tracing;

use futures::StreamExt;
use theta_ai::event::EventAccumulator;
use theta_ai::{
    AssistantMessageEvent, ContentBlock, Context, ErrorClass, LlmProvider, Message, Model,
    StopReason, StreamOptions, ThinkingLevel,
};

use crate::error::AgentError;
use crate::events::{AgentEvent, TurnDecisionReason};
use crate::hooks::Hooks;
use crate::state::AgentState;
use crate::tools;
use crate::types::{
    AgentIntent, AgentLoopConfig, CompactionStrategy, RunReport, SafetyDecisionKind, ToolCall,
    TurnEndReason, TurnMode,
};
use crate::{command_policy, command_policy::SafetyDecision};

#[derive(Debug, Clone)]
struct BreakerState {
    consecutive_failures: u32,
    opened_at: Option<Instant>,
}

impl BreakerState {
    fn new() -> Self {
        Self {
            consecutive_failures: 0,
            opened_at: None,
        }
    }
}

static CIRCUIT_BREAKERS: LazyLock<Mutex<HashMap<String, BreakerState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

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
    let run_id = format!("run-{}", now_ms());
    state.current_run_id = Some(run_id.clone());
    state.current_run_report = Some(RunReport {
        run_id: run_id.clone(),
        started_at_ms: now_ms(),
        finished_at_ms: None,
        outcome: None,
        events: Vec::new(),
    });
    state.push_run_event(
        "agent_start",
        [
            ("run_id".to_string(), run_id.clone()),
            ("model".to_string(), state.model.id.clone()),
            (
                "provider".to_string(),
                format!("{:?}", state.model.provider),
            ),
        ],
    );

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
    if let Some(mut report) = state.current_run_report.take() {
        report.finished_at_ms = Some(now_ms());
        report.outcome = if aborted {
            Some(TurnEndReason::AbortedByUser)
        } else {
            state.last_turn_end_reason
        };
        report.events.push(crate::types::RunReportEvent {
            ts_ms: now_ms(),
            kind: "agent_end".to_string(),
            fields: std::collections::BTreeMap::from([
                ("run_id".to_string(), report.run_id.clone()),
                ("model".to_string(), state.model.id.clone()),
                (
                    "provider".to_string(),
                    format!("{:?}", state.model.provider),
                ),
                ("aborted".to_string(), aborted.to_string()),
                (
                    "outcome".to_string(),
                    format!("{:?}", report.outcome.unwrap_or(TurnEndReason::Completed)),
                ),
            ]),
        });
        state.last_run_report = Some(report);
    }
    state.current_run_id = None;
    state.current_turn_id = None;
    state.executed_tool_call_ids_in_turn.clear();

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
    let turn_id = format!("turn-{}-{turn_index}", now_ms());
    state.current_turn_id = Some(turn_id.clone());
    state.executed_tool_call_ids_in_turn.clear();
    state.push_run_event(
        "turn_start",
        [
            ("turn".to_string(), turn_index.to_string()),
            ("turn_id".to_string(), turn_id),
        ],
    );
    let (turn_mode, mode_source) = resolve_turn_mode(state);
    state.last_turn_mode = Some(turn_mode);
    let _ = event_tx.send(AgentEvent::TurnModeResolved {
        turn_index,
        mode: turn_mode,
        source: mode_source.to_string(),
    });
    state.push_run_event(
        "turn_mode_resolved",
        [
            ("turn".to_string(), turn_index.to_string()),
            ("mode".to_string(), format!("{turn_mode:?}")),
            ("source".to_string(), mode_source.to_string()),
        ],
    );

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
    let mut terminated: Option<(TurnEndReason, String, u32)> = None;

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
            state.push_run_event(
                "turn_decision",
                [
                    ("turn".to_string(), turn_index.to_string()),
                    ("round".to_string(), tool_round.to_string()),
                    ("reason".to_string(), "MaxRounds".to_string()),
                ],
            );
            terminated = Some((
                TurnEndReason::MaxToolRounds,
                format!("reached max tool rounds ({max_rounds})"),
                tool_round,
            ));
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
                deduped_tool_results: stats.deduped_tool_results,
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

                    if matches!(turn_mode, TurnMode::PlanOnly | TurnMode::Clarify) {
                        terminated = Some((
                            TurnEndReason::Completed,
                            "mode-complete".to_string(),
                            tool_round,
                        ));
                        break;
                    }

                    if turn_mode == TurnMode::Execute
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
                        state.push_run_event(
                            "turn_decision",
                            [
                                ("turn".to_string(), turn_index.to_string()),
                                ("round".to_string(), tool_round.to_string()),
                                ("reason".to_string(), "NoopRetry".to_string()),
                            ],
                        );
                        state.messages.push(Message::User {
                            content: vec![ContentBlock::text(VALIDATION_RETRY_PROMPT)],
                            timestamp: now_ms(),
                        });
                        tool_round += 1;
                        continue;
                    }
                    if turn_mode == TurnMode::Execute
                        && !executed_tools_in_turn
                        && classify_action_blocker(&assistant_text).is_some()
                    {
                        let reason = classify_action_blocker(&assistant_text)
                            .unwrap_or(TurnEndReason::BlockedRuntimeConstraint);
                        let _ = event_tx.send(AgentEvent::TurnDecision {
                            reason: TurnDecisionReason::BlockedNoop,
                            details: "execute intent stopped due to explicit blocker".to_string(),
                            turn: turn_index,
                            round: tool_round,
                        });
                        state.push_run_event(
                            "turn_decision",
                            [
                                ("turn".to_string(), turn_index.to_string()),
                                ("round".to_string(), tool_round.to_string()),
                                ("reason".to_string(), "BlockedNoop".to_string()),
                            ],
                        );
                        terminated = Some((reason, assistant_text, tool_round));
                    }

                    if turn_mode == TurnMode::Inspect && !executed_tools_in_turn {
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
                    if turn_mode == TurnMode::AnalyzeOnly && !executed_tools_in_turn {
                        consecutive_noop_rounds += 1;
                        if consecutive_noop_rounds <= 1 {
                            let _ = event_tx.send(AgentEvent::TurnDecision {
                                reason: TurnDecisionReason::NoopRetry,
                                details:
                                    "analyze-only intent produced no tool calls; retrying once"
                                        .to_string(),
                                turn: turn_index,
                                round: tool_round,
                            });
                            state.push_run_event(
                                "turn_decision",
                                [
                                    ("turn".to_string(), turn_index.to_string()),
                                    ("round".to_string(), tool_round.to_string()),
                                    ("reason".to_string(), "NoopRetry".to_string()),
                                ],
                            );
                            state.messages.push(Message::User {
                                content: vec![ContentBlock::text(ANALYZE_RETRY_PROMPT)],
                                timestamp: now_ms(),
                            });
                            tool_round += 1;
                            continue;
                        }
                    }
                    if terminated.is_none() {
                        terminated = Some((
                            TurnEndReason::Completed,
                            "completed".to_string(),
                            tool_round,
                        ));
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

                if matches!(turn_mode, TurnMode::AnalyzeOnly | TurnMode::Inspect) {
                    let mut allowed = Vec::new();
                    let latest_user_text = latest_user_text(state).unwrap_or_default();
                    for tc in &tool_calls {
                        let SafetyDecision { decision, details } =
                            command_policy::evaluate_tool_call(
                                turn_mode,
                                tc,
                                config.command_policy_strict,
                            );
                        if decision == SafetyDecisionKind::Allowed
                            && let Some(class) = command_policy::required_user_authorization(tc)
                            && !user_authorizes_action_class(&latest_user_text, class)
                        {
                            let details = format!(
                                "{class:?} blocked: user did not explicitly request this action in latest message"
                            );
                            let _ = event_tx.send(AgentEvent::SafetyDecision {
                                decision: SafetyDecisionKind::Rejected,
                                mode: turn_mode,
                                tool_name: tc.name.clone(),
                                details: details.clone(),
                            });
                            let _ = event_tx.send(AgentEvent::TurnDecision {
                                reason: TurnDecisionReason::AnalyzeOnlyRejectedTool,
                                details,
                                turn: turn_index,
                                round: tool_round,
                            });
                        } else if decision == SafetyDecisionKind::Allowed {
                            let _ = event_tx.send(AgentEvent::SafetyDecision {
                                decision,
                                mode: turn_mode,
                                tool_name: tc.name.clone(),
                                details,
                            });
                            state.push_run_event(
                                "safety_decision",
                                [
                                    ("turn".to_string(), turn_index.to_string()),
                                    ("round".to_string(), tool_round.to_string()),
                                    ("tool_name".to_string(), tc.name.clone()),
                                    ("decision".to_string(), "Allowed".to_string()),
                                ],
                            );
                            allowed.push(tc.clone());
                        } else {
                            let _ = event_tx.send(AgentEvent::SafetyDecision {
                                decision,
                                mode: turn_mode,
                                tool_name: tc.name.clone(),
                                details: details.clone(),
                            });
                            let _ = event_tx.send(AgentEvent::TurnDecision {
                                reason: TurnDecisionReason::AnalyzeOnlyRejectedTool,
                                details: format!(
                                    "blocked mutating tool call '{}' during {turn_mode:?} turn",
                                    tc.name,
                                ),
                                turn: turn_index,
                                round: tool_round,
                            });
                        }
                    }
                    if allowed.is_empty() {
                        terminated = Some((
                            TurnEndReason::SafetyRejected,
                            format!("all tool calls rejected by {turn_mode:?} policy"),
                            tool_round,
                        ));
                        break;
                    }
                    tools::execute_tool_calls(
                        state,
                        &allowed,
                        abort_token.clone(),
                        event_tx,
                        hooks,
                        &config.tool_watchdog,
                    )
                    .await?;
                    state.push_run_event(
                        "tool_batch_executed",
                        [
                            ("turn".to_string(), turn_index.to_string()),
                            ("round".to_string(), tool_round.to_string()),
                            ("tool_count".to_string(), allowed.len().to_string()),
                        ],
                    );
                    executed_tools_in_turn = true;
                } else {
                    let mut allowed = Vec::new();
                    let latest_user_text = latest_user_text(state).unwrap_or_default();
                    for tc in &tool_calls {
                        let SafetyDecision { decision, details } =
                            command_policy::evaluate_tool_call(
                                turn_mode,
                                tc,
                                config.command_policy_strict,
                            );
                        let _ = event_tx.send(AgentEvent::SafetyDecision {
                            decision,
                            mode: turn_mode,
                            tool_name: tc.name.clone(),
                            details: details.clone(),
                        });
                        state.push_run_event(
                            "safety_decision",
                            [
                                ("turn".to_string(), turn_index.to_string()),
                                ("round".to_string(), tool_round.to_string()),
                                ("tool_name".to_string(), tc.name.clone()),
                                ("decision".to_string(), format!("{decision:?}")),
                            ],
                        );
                        if decision == SafetyDecisionKind::Allowed
                            && let Some(class) = command_policy::required_user_authorization(tc)
                            && !user_authorizes_action_class(&latest_user_text, class)
                        {
                            let details = format!(
                                "{class:?} blocked: user did not explicitly request this action in latest message"
                            );
                            let _ = event_tx.send(AgentEvent::SafetyDecision {
                                decision: SafetyDecisionKind::Rejected,
                                mode: turn_mode,
                                tool_name: tc.name.clone(),
                                details: details.clone(),
                            });
                            let _ = event_tx.send(AgentEvent::TurnDecision {
                                reason: TurnDecisionReason::AnalyzeOnlyRejectedTool,
                                details,
                                turn: turn_index,
                                round: tool_round,
                            });
                        } else if decision == SafetyDecisionKind::Allowed {
                            allowed.push(tc.clone());
                        }
                    }
                    if allowed.is_empty() {
                        terminated = Some((
                            TurnEndReason::SafetyRejected,
                            "all tool calls rejected by policy".to_string(),
                            tool_round,
                        ));
                        break;
                    }
                    tools::execute_tool_calls(
                        state,
                        &allowed,
                        abort_token.clone(),
                        event_tx,
                        hooks,
                        &config.tool_watchdog,
                    )
                    .await?;
                    state.push_run_event(
                        "tool_batch_executed",
                        [
                            ("turn".to_string(), turn_index.to_string()),
                            ("round".to_string(), tool_round.to_string()),
                            ("tool_count".to_string(), allowed.len().to_string()),
                        ],
                    );
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
                state.last_turn_end_reason = Some(TurnEndReason::AbortedByUser);
                let _ = event_tx.send(AgentEvent::TurnTerminated {
                    reason: TurnEndReason::AbortedByUser,
                    details: "aborted by user".to_string(),
                    turn: turn_index,
                    round: tool_round,
                });
                state.push_run_event(
                    "turn_terminated",
                    [
                        ("turn".to_string(), turn_index.to_string()),
                        ("round".to_string(), tool_round.to_string()),
                        ("reason".to_string(), "AbortedByUser".to_string()),
                    ],
                );
                return Err(AgentError::Aborted);
            }
            Err(e) => {
                state.is_streaming = false;
                state.last_turn_end_reason = Some(TurnEndReason::ProviderFailure);
                let _ = event_tx.send(AgentEvent::TurnTerminated {
                    reason: TurnEndReason::ProviderFailure,
                    details: e.to_string(),
                    turn: turn_index,
                    round: tool_round,
                });
                state.push_run_event(
                    "turn_terminated",
                    [
                        ("turn".to_string(), turn_index.to_string()),
                        ("round".to_string(), tool_round.to_string()),
                        ("reason".to_string(), "ProviderFailure".to_string()),
                    ],
                );
                return Err(e);
            }
        }
    }

    let (reason, details, round) = terminated.unwrap_or((
        TurnEndReason::Completed,
        "completed".to_string(),
        tool_round,
    ));
    state.last_turn_end_reason = Some(reason);
    let _ = event_tx.send(AgentEvent::TurnTerminated {
        reason,
        details,
        turn: turn_index,
        round,
    });
    state.push_run_event(
        "turn_terminated",
        [
            ("turn".to_string(), turn_index.to_string()),
            ("round".to_string(), round.to_string()),
            ("reason".to_string(), format!("{reason:?}")),
        ],
    );

    Ok(())
}

const VALIDATION_RETRY_PROMPT: &str = "This is an execution request. Call the relevant tools now. If blocked, state the blocker clearly and stop.";
const ANALYZE_RETRY_PROMPT: &str = "This is an analysis request. Use read-only tools now (read/grep/find/ls or read-only bash) and report findings.";

fn classify_action_blocker(text: &str) -> Option<TurnEndReason> {
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
    if missing_info {
        Some(TurnEndReason::BlockedMissingInfo)
    } else if permission {
        Some(TurnEndReason::BlockedPermission)
    } else if runtime {
        Some(TurnEndReason::BlockedRuntimeConstraint)
    } else {
        None
    }
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

fn user_authorizes_action_class(text: &str, class: command_policy::AuthorizationClass) -> bool {
    let t = text.to_lowercase();
    let tokens = tokenize_words(&t);
    match class {
        command_policy::AuthorizationClass::Commit => {
            if contains_token_sequence(&tokens, &["do", "not", "commit"])
                || contains_token_sequence(&tokens, &["dont", "commit"])
                || contains_token_sequence(&tokens, &["without", "commit"])
            {
                return false;
            }
            contains_token_sequence(&tokens, &["commit"])
                || contains_token_sequence(&tokens, &["create", "commit"])
                || contains_token_sequence(&tokens, &["make", "commit"])
                || contains_token_sequence(&tokens, &["git", "commit"])
        }
        command_policy::AuthorizationClass::VcsMutation => {
            if contains_token_sequence(&tokens, &["do", "not", "change", "branch"])
                || contains_token_sequence(&tokens, &["do", "not", "push"])
                || contains_token_sequence(&tokens, &["do", "not", "merge"])
            {
                return false;
            }
            contains_token_sequence(&tokens, &["git"])
                || contains_token_sequence(&tokens, &["branch"])
                || contains_token_sequence(&tokens, &["tag"])
                || contains_token_sequence(&tokens, &["rebase"])
                || contains_token_sequence(&tokens, &["merge"])
                || contains_token_sequence(&tokens, &["push"])
                || contains_token_sequence(&tokens, &["checkout"])
                || contains_token_sequence(&tokens, &["switch"])
                || contains_token_sequence(&tokens, &["cherry", "pick"])
        }
        command_policy::AuthorizationClass::DependencyMutation => {
            if contains_token_sequence(&tokens, &["do", "not", "install"])
                || contains_token_sequence(&tokens, &["without", "install"])
            {
                return false;
            }
            contains_token_sequence(&tokens, &["install"])
                || contains_token_sequence(&tokens, &["dependency"])
                || contains_token_sequence(&tokens, &["dependencies"])
                || contains_token_sequence(&tokens, &["package"])
                || contains_token_sequence(&tokens, &["packages"])
                || contains_token_sequence(&tokens, &["add", "package"])
                || contains_token_sequence(&tokens, &["add", "dependency"])
        }
        command_policy::AuthorizationClass::FileMutation => {
            if contains_token_sequence(&tokens, &["do", "not", "modify"])
                || contains_token_sequence(&tokens, &["do", "not", "change"])
                || contains_token_sequence(&tokens, &["do", "not", "edit"])
                || contains_token_sequence(&tokens, &["read", "only"])
            {
                return false;
            }
            contains_token_sequence(&tokens, &["implement"])
                || contains_token_sequence(&tokens, &["fix"])
                || contains_token_sequence(&tokens, &["patch"])
                || contains_token_sequence(&tokens, &["edit"])
                || contains_token_sequence(&tokens, &["modify"])
                || contains_token_sequence(&tokens, &["change"])
                || contains_token_sequence(&tokens, &["update"])
                || contains_token_sequence(&tokens, &["refactor"])
                || contains_token_sequence(&tokens, &["write"])
                || contains_token_sequence(&tokens, &["create"])
                || contains_token_sequence(&tokens, &["delete"])
                || contains_token_sequence(&tokens, &["remove"])
        }
    }
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
    if t.contains("inspect")
        || t.contains("check")
        || t.contains("validate")
        || t.contains("validation")
        || t.contains("what changed")
    {
        return AgentIntent::Inspect;
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
    if looks_like_execution_request(&t) {
        return AgentIntent::Execute;
    }
    AgentIntent::Default
}

fn resolve_turn_mode(state: &AgentState) -> (TurnMode, &'static str) {
    if let Some(mode) = state.turn_mode_override {
        return (mode, "runtime_override");
    }
    if let Some(mode) = infer_turn_mode_hint(state) {
        return (mode, "context_hint");
    }
    let text = latest_user_text(state).unwrap_or_default();
    (TurnMode::from(infer_intent(&text)), "fallback_classifier")
}

fn infer_turn_mode_hint(state: &AgentState) -> Option<TurnMode> {
    let latest_user = latest_user_text(state).unwrap_or_default().to_lowercase();
    if let Some(mode) = parse_mode_hint(&latest_user) {
        return Some(mode);
    }
    let system_text = state
        .system_prompt
        .iter()
        .filter_map(|b| match b {
            ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
        .to_lowercase();
    parse_mode_hint(&system_text)
}

fn parse_mode_hint(text: &str) -> Option<TurnMode> {
    let pairs = [
        ("mode:execute", TurnMode::Execute),
        ("mode:inspect", TurnMode::Inspect),
        ("mode:analyze", TurnMode::AnalyzeOnly),
        ("mode:analyzeonly", TurnMode::AnalyzeOnly),
        ("mode:plan", TurnMode::PlanOnly),
        ("mode:clarify", TurnMode::Clarify),
    ];
    pairs
        .iter()
        .find_map(|(needle, mode)| text.contains(needle).then_some(*mode))
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
    let fallback_models = resolve_fallback_models(state, config);
    let mut stream = None;
    let mut selected_model = state.model.clone();
    let mut last_error: Option<theta_ai::ThetaError> = None;

    for (idx, candidate_model) in fallback_models.iter().enumerate() {
        if idx > 0 && emit_events {
            let _ = event_tx.send(AgentEvent::ProviderFallback {
                from_model: selected_model.id.clone(),
                to_model: candidate_model.id.clone(),
                reason: last_error
                    .as_ref()
                    .map(ToString::to_string)
                    .unwrap_or_else(|| "fallback requested".to_string()),
            });
        }
        selected_model = candidate_model.clone();
        let key = format!("{:?}:{}", selected_model.provider, selected_model.id);
        if let Some(retry_in_ms) = breaker_retry_in_ms(&key, config) {
            if emit_events {
                let _ = event_tx.send(AgentEvent::ProviderCircuitOpen { key, retry_in_ms });
            }
            continue;
        }

        let mut attempt: u32 = 0;
        loop {
            match provider.stream(&selected_model, context, options).await {
                Ok(s) => {
                    breaker_record_success(&key);
                    stream = Some(s);
                    break;
                }
                Err(e) => {
                    last_error = Some(e);
                    let err = last_error.as_ref().expect("set above");
                    if !retry.is_retryable(err) || attempt >= retry.max_retries {
                        breaker_record_failure(&key, err.class(), config);
                        break;
                    }
                    attempt += 1;
                    let delay_ms = err.retry_after_ms().unwrap_or_else(|| {
                        retry
                            .base_delay_ms
                            .saturating_mul(2u64.pow(attempt.saturating_sub(1)))
                    });
                    if emit_events {
                        let _ = event_tx.send(AgentEvent::Retrying { attempt, delay_ms });
                    }
                    tracing::warn!(
                        model = %selected_model.id,
                        attempt = attempt,
                        delay_ms = delay_ms,
                        error = %err,
                        "provider call failed, retrying"
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
                }
            }
        }
        if stream.is_some() {
            break;
        }
    }

    let Some(mut stream) = stream else {
        return Err(AgentError::Llm(
            last_error.unwrap_or(theta_ai::ThetaError::StreamEndedEarly),
        ));
    };

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
        api: Some(selected_model.api),
        provider: Some(selected_model.provider),
        model: Some(selected_model.id.clone()),
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

fn resolve_fallback_models(state: &AgentState, config: &AgentLoopConfig) -> Vec<Model> {
    let mut models = vec![state.model.clone()];
    for model_id in &config.provider_fallback_chain {
        if let Some(found) = state.available_models.iter().find(|m| &m.id == model_id) {
            models.push(found.clone());
        }
    }
    models
}

fn breaker_retry_in_ms(key: &str, config: &AgentLoopConfig) -> Option<u64> {
    let mut guard = CIRCUIT_BREAKERS.lock().expect("breaker lock poisoned");
    let state = guard
        .entry(key.to_string())
        .or_insert_with(BreakerState::new);
    let opened_at = state.opened_at?;
    let elapsed = opened_at.elapsed().as_millis() as u64;
    if elapsed >= config.provider_circuit_breaker.open_cooldown_ms {
        state.opened_at = None;
        None
    } else {
        Some(config.provider_circuit_breaker.open_cooldown_ms - elapsed)
    }
}

fn breaker_record_success(key: &str) {
    let mut guard = CIRCUIT_BREAKERS.lock().expect("breaker lock poisoned");
    let state = guard
        .entry(key.to_string())
        .or_insert_with(BreakerState::new);
    state.consecutive_failures = 0;
    state.opened_at = None;
}

fn breaker_record_failure(key: &str, class: ErrorClass, config: &AgentLoopConfig) {
    if !matches!(class, ErrorClass::Transient) {
        return;
    }
    let mut guard = CIRCUIT_BREAKERS.lock().expect("breaker lock poisoned");
    let state = guard
        .entry(key.to_string())
        .or_insert_with(BreakerState::new);
    state.consecutive_failures = state.consecutive_failures.saturating_add(1);
    if state.consecutive_failures >= config.provider_circuit_breaker.failure_threshold {
        state.opened_at = Some(Instant::now());
    }
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
    fn infer_intent_treats_validation_review_as_inspection() {
        let intent = infer_intent("Validate the current changes and review for inconsistencies.");
        assert_eq!(intent, AgentIntent::Inspect);
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
        let decision = crate::command_policy::evaluate_tool_call(TurnMode::AnalyzeOnly, &tc, true);
        assert_eq!(decision.decision, SafetyDecisionKind::Allowed);
    }

    #[test]
    fn command_policy_allows_mode_mismatched_git_commands() {
        let tc = ToolCall {
            id: "1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command":"git commit -m test"}),
        };
        let decision = crate::command_policy::evaluate_tool_call(TurnMode::AnalyzeOnly, &tc, true);
        assert_eq!(decision.decision, SafetyDecisionKind::Allowed);
    }

    #[test]
    fn command_policy_allows_mode_mismatched_sed_in_place() {
        let tc = ToolCall {
            id: "1".into(),
            name: "bash".into(),
            arguments: serde_json::json!({"command":"sed -i '' 's/a/b/g' file.txt"}),
        };
        let decision = crate::command_policy::evaluate_tool_call(TurnMode::AnalyzeOnly, &tc, true);
        assert_eq!(decision.decision, SafetyDecisionKind::Allowed);
    }

    #[test]
    fn parse_mode_hint_detects_known_hints() {
        assert_eq!(
            parse_mode_hint("please run mode:inspect now"),
            Some(TurnMode::Inspect)
        );
        assert_eq!(
            parse_mode_hint("system says mode:analyze"),
            Some(TurnMode::AnalyzeOnly)
        );
        assert_eq!(parse_mode_hint("mode:plan"), Some(TurnMode::PlanOnly));
        assert_eq!(parse_mode_hint("no hint"), None);
    }

    #[test]
    fn user_authorizes_action_classes() {
        assert!(user_authorizes_action_class(
            "please commit these changes",
            command_policy::AuthorizationClass::Commit
        ));
        assert!(!user_authorizes_action_class(
            "review this diff only",
            command_policy::AuthorizationClass::Commit
        ));
        assert!(user_authorizes_action_class(
            "install dependencies and run tests",
            command_policy::AuthorizationClass::DependencyMutation
        ));
        assert!(!user_authorizes_action_class(
            "inspect the architecture",
            command_policy::AuthorizationClass::FileMutation
        ));
        assert!(user_authorizes_action_class(
            "fix and update the implementation",
            command_policy::AuthorizationClass::FileMutation
        ));
    }

    #[test]
    fn user_authorization_prompt_matrix_ambiguous_cases() {
        let cases = [
            (
                "check and fix if needed",
                command_policy::AuthorizationClass::FileMutation,
                true,
            ),
            (
                "prepare a commit message but do not commit",
                command_policy::AuthorizationClass::Commit,
                false,
            ),
            (
                "inspect and summarize only; do not modify files",
                command_policy::AuthorizationClass::FileMutation,
                false,
            ),
            (
                "run install only if missing and then test",
                command_policy::AuthorizationClass::DependencyMutation,
                true,
            ),
            (
                "review the diff and mention git branch status",
                command_policy::AuthorizationClass::VcsMutation,
                true,
            ),
            (
                "analyze architecture and provide recommendations",
                command_policy::AuthorizationClass::DependencyMutation,
                false,
            ),
        ];
        for (prompt, class, expected) in cases {
            assert_eq!(
                user_authorizes_action_class(prompt, class),
                expected,
                "prompt='{prompt}' class={class:?}"
            );
        }
    }
}
