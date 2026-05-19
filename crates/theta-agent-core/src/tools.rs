//! Tool execution: sequential and parallel batching.

use std::sync::Arc;

use tokio_util::sync::CancellationToken;
use tracing;

use crate::error::AgentError;
use crate::events::AgentEvent;
use crate::state::AgentState;
use crate::types::{AgentTool, ToolCall, ToolExecutionMode, ToolResult, ToolUpdate};

/// Execute a batch of tool calls. Handles ordering:
/// 1. All parallel tools run concurrently.
/// 2. Sequential tools run one at a time after parallel tools finish.
///
/// Tool results are added to state.messages as ToolResult messages.
/// Events are emitted via the provided sender.
pub async fn execute_tool_calls(
    state: &mut AgentState,
    tool_calls: &[ToolCall],
    abort_token: Option<CancellationToken>,
    event_tx: &tokio::sync::broadcast::Sender<AgentEvent>,
) -> Result<(), AgentError> {
    // Partition by execution mode.
    let mut parallel: Vec<&ToolCall> = Vec::new();
    let mut sequential: Vec<&ToolCall> = Vec::new();

    for tc in tool_calls {
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
        let handles: Vec<_> = parallel
            .iter()
            .map(|tc| {
                let state_snapshot = state.tools.clone();
                let event_tx = event_tx.clone();
                let abort = abort_token.clone();
                let tc = (*tc).clone();
                tokio::spawn(
                    async move { execute_one(&state_snapshot, &tc, abort, &event_tx).await },
                )
            })
            .collect();

        for handle in handles {
            match handle.await {
                Ok(Ok(result)) => {
                    let msg = result_to_message(&result);
                    state.add_tool_result(msg);
                }
                Ok(Err(e)) => {
                    let _ = event_tx.send(AgentEvent::Error {
                        message: e.to_string(),
                    });
                    return Err(e);
                }
                Err(e) => {
                    let msg = format!("tool task panicked: {e}");
                    let _ = event_tx.send(AgentEvent::Error {
                        message: msg.clone(),
                    });
                    return Err(AgentError::Other(msg));
                }
            }
        }
    }

    // Execute sequential tools one at a time.
    for tc in &sequential {
        match execute_one(&state.tools, tc, abort_token.clone(), event_tx).await {
            Ok(result) => {
                let msg = result_to_message(&result);
                state.add_tool_result(msg);
            }
            Err(e) => {
                let _ = event_tx.send(AgentEvent::Error {
                    message: e.to_string(),
                });
                return Err(e);
            }
        }
    }

    Ok(())
}

/// Execute a single tool call.
async fn execute_one(
    tools: &[Arc<dyn AgentTool>],
    tool_call: &ToolCall,
    abort_token: Option<CancellationToken>,
    event_tx: &tokio::sync::broadcast::Sender<AgentEvent>,
) -> Result<ToolResult, AgentError> {
    let tool = tools
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
    });

    // Progress callback.
    let tx = event_tx.clone();
    let cid = tool_call.id.clone();
    let on_update: crate::types::ToolUpdateSender = Arc::new(move |update: ToolUpdate| {
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
            content: vec![theta_ai::ContentBlock::text(format!("Error: {e}"))],
            details: None,
            is_error: true,
        },
    };

    let _ = event_tx.send(AgentEvent::ToolExecutionEnd {
        result: final_result.clone(),
    });

    Ok(final_result)
}

/// Convert a ToolResult to a Message::ToolResult for the transcript.
fn result_to_message(result: &ToolResult) -> theta_ai::Message {
    theta_ai::Message::ToolResult {
        tool_call_id: result.tool_call_id.clone(),
        tool_name: result.tool_name.clone(),
        content: result.content.clone(),
        details: result.details.clone(),
        is_error: result.is_error,
        timestamp: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64,
    }
}
