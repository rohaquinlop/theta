//! bash tool: executes shell commands.

use std::process::Stdio;

use async_trait::async_trait;
use michin_agent_core::error::AgentError;
use michin_agent_core::types::{
    AgentTool, ToolExecutionMode, ToolResult, ToolUpdate, ToolUpdateSender, ToolUpdateStatus,
};
use michin_ai::ContentBlock;
use tokio::process::Command;
use tokio_util::sync::CancellationToken;

use super::{ToolContext, TruncationLimits, truncate_output};

pub struct BashTool {
    ctx: ToolContext,
}

impl BashTool {
    pub fn new(ctx: ToolContext) -> Self {
        Self { ctx }
    }
}

#[async_trait]
impl AgentTool for BashTool {
    fn name(&self) -> &str {
        "bash"
    }

    fn description(&self) -> &str {
        "Execute a bash command. The working directory is already set — do NOT prefix \
         commands with `cd`. Returns stdout and stderr. Output is truncated to last 2000 \
         lines or 50KB (whichever is hit first). If truncated, full output is saved to a \
         temp file. Do NOT use this for reading, writing, or editing files — use the \
         `read`, `write`, and `edit` tools for all file operations."
    }

    fn label(&self) -> &str {
        "bash"
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "required": ["command"],
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute"
                }
            }
        })
    }

    fn execution_mode(&self) -> ToolExecutionMode {
        ToolExecutionMode::Sequential
    }

    async fn execute(
        &self,
        tool_call_id: &str,
        args: serde_json::Value,
        signal: Option<CancellationToken>,
        on_update: Option<ToolUpdateSender>,
    ) -> Result<ToolResult, AgentError> {
        let command = args["command"]
            .as_str()
            .ok_or_else(|| AgentError::ToolExecution {
                tool_name: "bash".into(),
                message: "missing required 'command' parameter".into(),
            })?;
        let raw_command = command.to_string();

        // Strip redundant leading `cd <working_dir> &&/;` — the tool already sets cwd.
        let cwd_str = self.ctx.working_dir.to_string_lossy();
        let stripped = raw_command
            .strip_prefix(&format!("cd {} && ", cwd_str))
            .or_else(|| raw_command.strip_prefix(&format!("cd {}; ", cwd_str)))
            .or_else(|| raw_command.strip_prefix(&format!("cd {}\n", cwd_str)))
            .or_else(|| raw_command.strip_prefix(&format!("cd {} &&", cwd_str)))
            .unwrap_or(&raw_command);
        let clean_command = stripped.trim();

        if let Some(ref update_sender) = on_update {
            update_sender(ToolUpdate {
                tool_call_id: tool_call_id.into(),
                tool_name: "bash".into(),
                status: ToolUpdateStatus::Running,
                output: Some(format!("running: {clean_command}")),
            });
        }

        let child = Command::new("bash")
            .arg("-c")
            .arg(clean_command)
            .current_dir(&self.ctx.working_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .process_group(0)
            .spawn()
            .map_err(|e| AgentError::ToolExecution {
                tool_name: "bash".into(),
                message: format!("failed to spawn command: {e}"),
            })?;

        let pid = child.id();

        // Spawn an abort watcher that kills the entire process group when the
        // cancel token fires. process_group(0) above sets the child as its
        // own process group leader, so -pid targets the whole tree.
        let abort_handle = if let (Some(token), Some(pid)) = (signal.clone(), pid) {
            Some(tokio::spawn(async move {
                token.cancelled().await;
                // Graceful termination first — allows temp-file cleanup and lock release.
                let _ = Command::new("sh")
                    .arg("-c")
                    .arg(format!("kill -15 -{pid} 2>/dev/null || true"))
                    .output()
                    .await;
                // Give processes a grace period to exit cleanly.
                tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                // Force-kill any stragglers.
                let _ = Command::new("sh")
                    .arg("-c")
                    .arg(format!("kill -9 -{pid} 2>/dev/null || true"))
                    .output()
                    .await;
            }))
        } else {
            None
        };

        let output = child.wait_with_output().await;

        // Abort watcher is no longer needed (child completed).
        if let Some(handle) = abort_handle {
            handle.abort();
        }

        let output = output.map_err(|e| AgentError::ToolExecution {
            tool_name: "bash".into(),
            message: format!("failed to wait for command: {e}"),
        })?;

        // Check if we were aborted (cancel token fired during execution).
        if let Some(ref token) = signal
            && token.is_cancelled()
        {
            return Err(AgentError::Aborted);
        }

        let exit_code = output.status.code();
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let combined = if stderr.is_empty() {
            stdout
        } else if stdout.is_empty() {
            format!("[stderr]\n{stderr}")
        } else {
            format!("{stdout}\n[stderr]\n{stderr}")
        };

        let mut result = ToolResult {
            tool_call_id: tool_call_id.into(),
            tool_name: "bash".into(),
            content: vec![ContentBlock::Text { text: combined }],
            details: Some(serde_json::json!({
                "exit_code": exit_code,
                "command": clean_command
            })),
            is_error: exit_code.is_some_and(|c| c != 0),
        };

        truncate_output(&mut result, &TruncationLimits::default());

        Ok(result)
    }
}
