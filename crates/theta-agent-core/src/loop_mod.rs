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
use theta_ai::{AssistantMessageEvent, Context, LlmProvider, Message, StopReason, StreamOptions};

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

    loop {
        // Drain any steering messages — they interrupt the current turn.
        drain_queue(steering_queue, state);

        // Only check abort if no steering messages are pending.
        check_abort!(abort_token, steering_queue);

        if tool_round >= max_rounds {
            tracing::warn!("max tool rounds reached ({max_rounds})");
            break;
        }

        // Build the LLM context from current state.
        let context = build_context(state);

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
            ..Default::default()
        };

        // Call the LLM provider and consume the stream.
        match run_llm_stream(
            state,
            provider,
            &context,
            &stream_options,
            event_tx,
            abort_token.clone(),
            steering_abort.clone(),
        )
        .await
        {
            Ok((assistant_msg, stop_reason)) => {
                state.is_streaming = false;
                let has_tool_calls = stop_reason == Some(StopReason::ToolUse);

                state.add_assistant_message(assistant_msg.clone());

                let _ = event_tx.send(AgentEvent::MessageEnd {
                    message: assistant_msg,
                });

                if !has_tool_calls {
                    break;
                }

                let tool_calls =
                    ToolCall::from_message(state.messages.last().expect("just pushed"));

                if tool_calls.is_empty() {
                    break;
                }

                tools::execute_tool_calls(state, &tool_calls, abort_token.clone(), event_tx)
                    .await?;

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

/// Consume an LLM stream, emitting AgentEvents and accumulating content.
/// Returns the assembled assistant message and stop reason.
async fn run_llm_stream(
    state: &AgentState,
    provider: &dyn LlmProvider,
    context: &Context,
    options: &StreamOptions,
    event_tx: &broadcast::Sender<AgentEvent>,
    abort_token: Option<CancellationToken>,
    steering_abort: Arc<AtomicBool>,
) -> Result<(Message, Option<StopReason>), AgentError> {
    let mut accumulator = EventAccumulator::new();
    let mut stream = provider.stream(&state.model, context, options).await?;

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

    Ok((assistant_msg, accumulator.stop_reason()))
}

/// Build the LLM Context from the current agent state.
fn build_context(state: &AgentState) -> Context {
    let system = if state.system_prompt.is_empty() {
        None
    } else {
        Some(state.system_prompt.clone())
    };

    let messages: Vec<Message> = state.llm_messages().into_iter().cloned().collect();

    let tools: Vec<theta_ai::Tool> = state
        .tools
        .iter()
        .map(|t| theta_ai::Tool {
            name: t.name().to_string(),
            description: t.description().to_string(),
            parameters: t.parameters(),
        })
        .collect();

    Context {
        system,
        messages,
        tools,
        thinking_level: Some(state.thinking_level),
    }
}

/// Get current time in milliseconds since epoch.
fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}
