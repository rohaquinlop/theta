//! Agent loop: the core turn execution engine.
//!
//! Follows Pi's approach: the loop is dumb and universal — it calls the LLM,
//! executes tools, feeds results back, and repeats until the model stops
//! emitting tool calls. Intelligence lives in the system prompt, not in
//! heuristic classifiers. Command-policy safety checks are always-on.
//!
//! Outer loop: follow-up turns (until shouldStopAfterTurn or queue empty).
//! Inner loop: LLM call → tool execution → tool results → repeat.

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

use crate::command_policy;
use crate::error::AgentError;
use crate::events::{AgentEvent, TurnDecisionReason};
use crate::hooks::Hooks;
use crate::state::AgentState;
use crate::tools;
use crate::types::{AgentLoopConfig, CompactionStrategy, RunReport, ToolCall, TurnEndReason};

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

/// Run a single turn: LLM call → tool execution → repeat until no more tools.
///
/// The loop is intentionally dumb — it does not classify intent, infer modes,
/// or selectively filter tools. All tool filtering is handled by the always-on
/// command policy (which checks for destructive operations like `rm -rf`,
/// `git push --force`, etc.) and by the system prompt (which guides the model
/// on when to analyze vs implement).
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

    // Inject any prepare-next-turn messages.
    let prepend = hooks.prepare_next_turn(state).await;
    for msg in prepend {
        state.messages.push(msg);
    }

    // Inner loop: LLM call + tool execution.
    let mut tool_round: u32 = 0;
    let mut empty_assistant_retries: u32 = 0;
    let mut repeated_tool_signature_counts: HashMap<String, u32> = HashMap::new();
    let max_same_tool_signature_repeats = config.max_same_tool_call_repeats.unwrap_or(6);

    loop {
        // Drain any steering messages — they interrupt the current turn.
        drain_queue(steering_queue, state);
        check_abort!(abort_token, steering_queue);

        if let Some(max_rounds) = config.max_tool_rounds
            && tool_round >= max_rounds
        {
            tracing::warn!("max tool rounds reached ({max_rounds})");
            let _ = event_tx.send(AgentEvent::TurnDecision {
                reason: TurnDecisionReason::MaxRounds,
                details: format!("stopped after reaching max tool rounds ({max_rounds})"),
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
            state.last_turn_end_reason = Some(TurnEndReason::MaxToolRounds);
            let _ = event_tx.send(AgentEvent::TurnTerminated {
                reason: TurnEndReason::MaxToolRounds,
                details: format!("reached max tool rounds ({max_rounds})"),
                turn: turn_index,
                round: tool_round,
            });
            return Ok(());
        }

        // Build context (with compaction).
        let (context, compaction_stats, replay_stats) =
            build_context(state, provider, config, event_tx).await;

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
            "calling LLM",
        );

        // Call the LLM.
        let _ = event_tx.send(AgentEvent::MessageStart);
        state.is_streaming = true;

        let stream_options = StreamOptions {
            max_tokens: config.max_tokens,
            temperature: config.temperature,
            thinking_level: Some(state.thinking_level),
            include_usage: config.include_usage,
            timeout_ms: config.provider_timeout_ms,
            ..Default::default()
        };

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

                // Empty response: retry up to twice, then stop.
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

                // Unresolved tool-call state: replay request.
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
                                "tool-call parsing incomplete: {unresolved_tool_calls} unresolved tool call(s)"
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

                // No tool calls — turn is complete.
                if !has_tool_calls {
                    state.last_turn_end_reason = Some(TurnEndReason::Completed);
                    let _ = event_tx.send(AgentEvent::TurnTerminated {
                        reason: TurnEndReason::Completed,
                        details: "completed".to_string(),
                        turn: turn_index,
                        round: tool_round,
                    });
                    state.push_run_event(
                        "turn_terminated",
                        [
                            ("turn".to_string(), turn_index.to_string()),
                            ("round".to_string(), tool_round.to_string()),
                            ("reason".to_string(), "Completed".to_string()),
                        ],
                    );
                    return Ok(());
                }

                if stop_reason != Some(StopReason::ToolUse) {
                    let _ = event_tx.send(AgentEvent::Error {
                        message: "tool calls detected without tool_use stop reason; executing tools via fallback".to_string(),
                    });
                }

                // Repeated tool signature detection.
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
                        state.last_turn_end_reason = Some(TurnEndReason::Completed);
                        let _ = event_tx.send(AgentEvent::TurnTerminated {
                            reason: TurnEndReason::Completed,
                            details: "stopped due to repeated tool calls".to_string(),
                            turn: turn_index,
                            round: tool_round,
                        });
                        return Ok(());
                    }
                }

                // Execute tools. Command policy runs as always-on safety net:
                // it blocks destructive operations regardless of "mode".
                // The system prompt is responsible for guiding the model on
                // when to use mutation tools vs read-only tools.
                let mut allowed = Vec::new();
                for tc in &tool_calls {
                    let decision =
                        command_policy::evaluate_tool_call(tc, config.command_policy_strict);
                    let _ = event_tx.send(AgentEvent::SafetyDecision {
                        decision: decision.decision,
                        tool_name: tc.name.clone(),
                        details: decision.details.clone(),
                    });
                    state.push_run_event(
                        "safety_decision",
                        [
                            ("turn".to_string(), turn_index.to_string()),
                            ("round".to_string(), tool_round.to_string()),
                            ("tool_name".to_string(), tc.name.clone()),
                            ("decision".to_string(), format!("{:?}", decision.decision)),
                        ],
                    );
                    if decision.decision == crate::types::SafetyDecisionKind::Allowed {
                        allowed.push(tc.clone());
                    }
                }

                if allowed.is_empty() {
                    state.last_turn_end_reason = Some(TurnEndReason::SafetyRejected);
                    let _ = event_tx.send(AgentEvent::TurnTerminated {
                        reason: TurnEndReason::SafetyRejected,
                        details: "all tool calls rejected by command policy".to_string(),
                        turn: turn_index,
                        round: tool_round,
                    });
                    state.push_run_event(
                        "turn_terminated",
                        [
                            ("turn".to_string(), turn_index.to_string()),
                            ("round".to_string(), tool_round.to_string()),
                            ("reason".to_string(), "SafetyRejected".to_string()),
                        ],
                    );
                    return Ok(());
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

                tool_round += 1;
            }
            Err(AgentError::Aborted) => {
                state.is_streaming = false;
                if drain_queue(steering_queue, state) {
                    steering_abort.store(false, Ordering::SeqCst);
                    tracing::debug!("aborted for steering, continuing turn");
                    continue;
                }
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
                let details = format_agent_error_chain(&e);
                let _ = event_tx.send(AgentEvent::TurnTerminated {
                    reason: TurnEndReason::ProviderFailure,
                    details,
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

    state.last_turn_end_reason = Some(TurnEndReason::Completed);
    let _ = event_tx.send(AgentEvent::TurnTerminated {
        reason: TurnEndReason::Completed,
        details: "completed".to_string(),
        turn: turn_index,
        round: tool_round,
    });
    state.push_run_event(
        "turn_terminated",
        [
            ("turn".to_string(), turn_index.to_string()),
            ("round".to_string(), tool_round.to_string()),
            ("reason".to_string(), "Completed".to_string()),
        ],
    );

    Ok(())
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
            AssistantMessageEvent::ThinkingStart if emit_events => {
                let _ = event_tx.send(AgentEvent::ThinkingStart);
            }
            AssistantMessageEvent::ThinkingEnd if emit_events => {
                let _ = event_tx.send(AgentEvent::ThinkingEnd);
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

    // Approximate system prompt tokens — uses cached value from AgentState.
    let sys_tokens: u32 = state.system_prompt_tokens;

    // Account for resource context tokens in compaction budget.
    let res_tokens: u32 = state.resource_context_tokens();
    let effective_sys_tokens = sys_tokens + res_tokens;

    let (sanitized_messages, replay_stats) =
        theta_ai::sanitize_messages_for_replay(&state.messages, &state.model);

    // Compaction: only call compact_messages when enabled. When disabled,
    // use sanitized_messages directly — no clone or token scan needed.
    let (mut messages, compaction_stats) = if config.compaction.enabled {
        let mut compact_result = crate::compact::compact_messages(
            &sanitized_messages,
            effective_sys_tokens,
            state.model.context_window,
            &config.compaction,
        );

        if compact_result.trimmed_count > 0 && config.compaction.strategy == CompactionStrategy::Llm
        {
            let trimmed_len = (compact_result.trimmed_count as usize).min(state.messages.len());
            match summarize_compacted_messages(
                state,
                provider,
                &state.messages[..trimmed_len],
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

        let stats = (compact_result.trimmed_count > 0).then_some(CompactionStats {
            trimmed_count: compact_result.trimmed_count,
            tokens_before: compact_result.tokens_before,
            tokens_after: compact_result.tokens_after,
        });
        (compact_result.messages, stats)
    } else {
        (sanitized_messages, None)
    };

    // Prepend resource context (skills, extensions) — never subject to compaction.
    if let Some(ref res_ctx) = state.resource_context
        && !res_ctx.is_empty()
    {
        messages.insert(
            0,
            Message::User {
                content: res_ctx.clone(),
                timestamp: 0,
            },
        );
    }

    let tools: Vec<theta_ai::Tool> = state.theta_ai_tools.clone();

    (
        Context {
            system,
            messages,
            tools,
            thinking_level: Some(state.thinking_level),
        },
        compaction_stats,
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

fn format_agent_error_chain(error: &AgentError) -> String {
    let mut out = error.to_string();
    let mut source = std::error::Error::source(error);
    while let Some(err) = source {
        out.push_str("\ncaused by: ");
        out.push_str(&err.to_string());
        source = err.source();
    }
    out
}

#[cfg(test)]
mod tests {
    // No intent-classification tests. The loop follows Pi's approach:
    // system prompt guides behavior, code does not infer intent.
}
