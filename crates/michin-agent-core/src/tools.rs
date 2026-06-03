//! Tool execution: sequential and parallel batching.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use tokio::sync::broadcast;
use tokio_util::sync::CancellationToken;
use tracing;

use crate::error::AgentError;
use crate::events::AgentEvent;
use crate::hooks::Hooks;
use crate::state::AgentState;
use crate::types::{ToolCall, ToolExecutionMode, ToolResult, ToolUpdate, ToolWatchdogConfig};

/// Execute a batch of tool calls.
///
/// Tool errors are converted to error ToolResult messages, never
/// propagated as Err — a single tool failure should not abort the turn.
///
/// All tools (parallel and sequential) go through before/after hooks.
pub async fn execute_tool_calls(
    state: &mut AgentState,
    tool_calls: &[ToolCall],
    abort_token: Option<CancellationToken>,
    event_tx: &broadcast::Sender<AgentEvent>,
    hooks: &Arc<dyn Hooks>,
    watchdog: &ToolWatchdogConfig,
) -> Result<(), AgentError> {
    let watchdog_cfg = watchdog.clone();
    let mut deduped_calls: Vec<&ToolCall> = Vec::new();
    for tc in tool_calls {
        if state.executed_tool_call_ids_in_turn.insert(tc.id.clone()) {
            deduped_calls.push(tc);
        } else {
            let _ = event_tx.send(AgentEvent::Error {
                message: format!(
                    "skipping duplicate tool call id '{}' in current turn",
                    tc.id
                ),
            });
        }
    }
    // Partition by execution mode.
    let mut parallel: Vec<&ToolCall> = Vec::new();
    let mut sequential: Vec<&ToolCall> = Vec::new();

    for tc in deduped_calls {
        let tool = state.tools.iter().find(|t| t.name() == tc.name);
        let mode = tool
            .map(|t| t.execution_mode())
            .unwrap_or(ToolExecutionMode::Parallel);

        match mode {
            ToolExecutionMode::Parallel => parallel.push(tc),
            ToolExecutionMode::Sequential => sequential.push(tc),
        }
    }

    // Execute parallel tools concurrently.
    if !parallel.is_empty() {
        let event_tx = event_tx.clone();
        let abort = abort_token.clone();
        let hooks: Arc<dyn Hooks> = Arc::clone(hooks);
        let state_arc = Arc::new(state.clone());

        let handles: Vec<_> = parallel
            .iter()
            .map(|tc| {
                let state = Arc::clone(&state_arc);
                let event_tx = event_tx.clone();
                let abort = abort.clone();
                let hooks = Arc::clone(&hooks);
                let tc = (*tc).clone();
                let watchdog = watchdog_cfg.clone();
                tokio::spawn(async move {
                    execute_one(&state, &tc, abort, &event_tx, &*hooks, &watchdog).await
                })
            })
            .collect();

        for (handle, tc) in handles.into_iter().zip(parallel.iter()) {
            let result = match handle.await {
                Ok(Ok(result)) => result,
                Ok(Err(e)) => {
                    let _ = event_tx.send(AgentEvent::Error {
                        message: e.to_string(),
                    });
                    ToolResult {
                        tool_call_id: tc.id.clone(),
                        tool_name: tc.name.clone(),
                        content: vec![michin_ai::ContentBlock::text(format!("Error: {e}"))],
                        details: None,
                        is_error: true,
                    }
                }
                Err(join_err) => {
                    let msg = format!("tool task panicked: {join_err}");
                    let _ = event_tx.send(AgentEvent::ToolExecutionEnd {
                        result: ToolResult {
                            tool_call_id: tc.id.clone(),
                            tool_name: tc.name.clone(),
                            content: vec![michin_ai::ContentBlock::text(msg.clone())],
                            details: None,
                            is_error: true,
                        },
                    });
                    let _ = event_tx.send(AgentEvent::Error {
                        message: msg.clone(),
                    });
                    ToolResult {
                        tool_call_id: tc.id.clone(),
                        tool_name: tc.name.clone(),
                        content: vec![michin_ai::ContentBlock::text(msg)],
                        details: None,
                        is_error: true,
                    }
                }
            };
            state.push_run_event(
                "tool_execution_end",
                [
                    ("tool_call_id".to_string(), tc.id.clone()),
                    ("tool_name".to_string(), tc.name.clone()),
                    (
                        "outcome".to_string(),
                        if result.is_error { "error" } else { "ok" }.to_string(),
                    ),
                ],
            );
            let msg = result_to_message(&result);
            state.add_tool_result(msg);
        }
    }

    // Execute sequential tools one at a time.
    for tc in &sequential {
        let result =
            match execute_one(state, tc, abort_token.clone(), event_tx, &**hooks, watchdog).await {
                Ok(r) => r,
                Err(e) => {
                    let _ = event_tx.send(AgentEvent::Error {
                        message: e.to_string(),
                    });
                    ToolResult {
                        tool_call_id: tc.id.clone(),
                        tool_name: tc.name.clone(),
                        content: vec![michin_ai::ContentBlock::text(format!("Error: {e}"))],
                        details: None,
                        is_error: true,
                    }
                }
            };
        state.push_run_event(
            "tool_execution_end",
            [
                ("tool_call_id".to_string(), tc.id.clone()),
                ("tool_name".to_string(), tc.name.clone()),
                (
                    "outcome".to_string(),
                    if result.is_error { "error" } else { "ok" }.to_string(),
                ),
            ],
        );
        let msg = result_to_message(&result);
        state.add_tool_result(msg);
    }

    Ok(())
}

/// Execute a single tool call with before/after hooks.
/// This is the unified path for all tools — parallel and sequential.
async fn execute_one(
    state: &AgentState,
    tool_call: &ToolCall,
    abort_token: Option<CancellationToken>,
    event_tx: &broadcast::Sender<AgentEvent>,
    hooks: &dyn Hooks,
    watchdog: &ToolWatchdogConfig,
) -> Result<ToolResult, AgentError> {
    hooks
        .before_tool_call(state, tool_call)
        .await
        .map_err(|e| {
            tracing::warn!(
                tool_name = %tool_call.name,
                error = %e,
                "before_tool_call blocked execution"
            );
            AgentError::ToolExecution {
                tool_name: tool_call.name.clone(),
                message: e.to_string(),
            }
        })?;

    let result = run_tool(state, tool_call, abort_token, event_tx, watchdog).await;

    if let Ok(ref r) = result {
        let _ = hooks
            .after_tool_call(state, tool_call, r)
            .await
            .map_err(|e| {
                tracing::warn!(
                    tool_name = %tool_call.name,
                    error = %e,
                    "after_tool_call hook error"
                );
            });
    }

    result
}

/// Core tool execution logic (no hooks).
async fn run_tool(
    state: &AgentState,
    tool_call: &ToolCall,
    abort_token: Option<CancellationToken>,
    event_tx: &tokio::sync::broadcast::Sender<AgentEvent>,
    watchdog: &ToolWatchdogConfig,
) -> Result<ToolResult, AgentError> {
    let tool = state
        .tools
        .iter()
        .find(|t| t.name() == tool_call.name)
        .ok_or_else(|| AgentError::ToolNotFound {
            tool_name: tool_call.name.clone(),
        })?;

    tracing::info!(
        tool_name = %tool_call.name,
        tool_call_id = %tool_call.id,
        "executing tool"
    );

    let _ = event_tx.send(AgentEvent::ToolExecutionStart {
        tool_call_id: tool_call.id.clone(),
        tool_name: tool_call.name.clone(),
        arguments: Some(tool_call.arguments.clone()),
    });

    let last_progress_ms = Arc::new(AtomicU64::new(now_ms()));
    let watchdog_finished = Arc::new(AtomicBool::new(false));
    let watchdog_tx = event_tx.clone();
    let watchdog_call_id = tool_call.id.clone();
    let watchdog_tool_name = tool_call.name.clone();
    let watchdog_last = Arc::clone(&last_progress_ms);
    let watchdog_done = Arc::clone(&watchdog_finished);
    let stall_warning_ms = watchdog.stall_warning_ms.max(1_000);
    let watchdog_handle = tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_millis(stall_warning_ms)).await;
            if watchdog_done.load(Ordering::Relaxed) {
                break;
            }
            let last = watchdog_last.load(Ordering::Relaxed);
            let now = now_ms();
            let stalled_ms = now.saturating_sub(last);
            if stalled_ms >= stall_warning_ms {
                let _ = watchdog_tx.send(AgentEvent::ToolWatchdogWarning {
                    tool_call_id: watchdog_call_id.clone(),
                    tool_name: watchdog_tool_name.clone(),
                    stalled_ms,
                });
            }
        }
    });

    let tx = event_tx.clone();
    let cid = tool_call.id.clone();
    let progress_clock = Arc::clone(&last_progress_ms);
    let on_update: crate::types::ToolUpdateSender = Arc::new(move |update: ToolUpdate| {
        progress_clock.store(now_ms(), Ordering::Relaxed);
        if let Some(output) = update.output {
            let _ = tx.send(AgentEvent::ToolExecutionProgress {
                tool_call_id: cid.clone(),
                output,
            });
        }
    });

    let result = tool
        .execute(
            &tool_call.id,
            tool_call.arguments.clone(),
            abort_token,
            Some(on_update),
        )
        .await;
    watchdog_finished.store(true, Ordering::Relaxed);
    watchdog_handle.abort();

    match &result {
        Ok(r) => {
            tracing::info!(
                tool_name = %tool_call.name,
                tool_call_id = %tool_call.id,
                is_error = r.is_error,
                "tool completed"
            );
        }
        Err(e) => {
            tracing::error!(
                tool_name = %tool_call.name,
                tool_call_id = %tool_call.id,
                error = %e,
                "tool failed"
            );
        }
    }

    let final_result = match result {
        Ok(r) => r,
        Err(e) => ToolResult {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            content: vec![michin_ai::ContentBlock::text(format!("Error: {e}"))],
            details: None,
            is_error: true,
        },
    };

    if final_result.is_error {
        let _ = event_tx.send(AgentEvent::ToolWatchdogWarning {
            tool_call_id: tool_call.id.clone(),
            tool_name: tool_call.name.clone(),
            stalled_ms: watchdog.stall_warning_ms,
        });
    }

    let _ = event_tx.send(AgentEvent::ToolExecutionEnd {
        result: final_result.clone(),
    });

    Ok(final_result)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Convert a ToolResult to a Message::ToolResult for the transcript.
fn result_to_message(result: &ToolResult) -> michin_ai::Message {
    michin_ai::Message::ToolResult {
        tool_call_id: result.tool_call_id.clone(),
        tool_name: result.tool_name.clone(),
        content: result.content.clone(),
        details: result.details.clone(),
        is_error: result.is_error,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64,
    }
}
