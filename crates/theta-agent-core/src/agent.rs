//! Agent: the main entry point for the agent runtime.
//!
//! Owns state, provider, hooks, and runs the prompt/continue loops.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{RwLock, broadcast};
use tokio_util::sync::CancellationToken;
use tracing;

use theta_ai::Provider as ProviderKind;
use theta_ai::providers::ProviderRegistry;
use theta_ai::{ContentBlock, Message, Model};

use crate::error::AgentError;
use crate::events::AgentEvent;
use crate::hooks::{Hooks, NoopHooks};
use crate::loop_mod;
use crate::state::AgentState;
use crate::types::{AgentLoopConfig, AgentTool, RunReport};

/// The agent: holds all state and orchestrates LLM interaction.
pub struct Agent {
    /// Mutable agent state (transcript, tools, config).
    state: RwLock<AgentState>,

    /// Event broadcaster for TUI/RPC subscribers.
    event_tx: broadcast::Sender<AgentEvent>,

    /// Provider registry for LLM calls.
    provider: Arc<ProviderRegistry>,

    /// Lifecycle hooks.
    hooks: Arc<dyn Hooks>,

    /// Active run handle (ensures only one prompt/continue at a time).
    active_run: Mutex<Option<ActiveRun>>,

    /// Loop configuration.
    config: AgentLoopConfig,

    /// Follow-up messages queued for after the current turn.
    follow_up_queue: Arc<Mutex<Vec<(Message, u64)>>>,

    /// Steering messages that interrupt the current turn mid-stream.
    steering_queue: Arc<Mutex<Vec<(Message, u64)>>>,
}

/// Handle for an in-progress agent run.
struct ActiveRun {
    /// Permanent abort: user cancels the entire run.
    abort_token: CancellationToken,
    /// Per-stream abort for steering. Set when `steer()` interrupts.
    steering_abort: Arc<AtomicBool>,
}

impl Agent {
    /// Create a new agent with the given model, provider, and available models.
    pub fn new(
        model: Model,
        provider: Arc<ProviderRegistry>,
        available_models: Vec<Model>,
    ) -> Self {
        Self {
            state: RwLock::new(AgentState::new(model, available_models)),
            event_tx: broadcast::channel(8192).0,
            provider,
            hooks: Arc::new(NoopHooks),
            active_run: Mutex::new(None),
            config: AgentLoopConfig::default(),
            follow_up_queue: Arc::new(Mutex::new(Vec::new())),
            steering_queue: Arc::new(Mutex::new(Vec::new())),
        }
    }

    // ── Configuration ──────────────────────────────────────────

    /// Set lifecycle hooks.
    pub fn set_hooks(&mut self, hooks: Arc<dyn Hooks>) {
        self.hooks = hooks;
    }

    /// Get a reference to the hooks.
    pub fn hooks(&self) -> &Arc<dyn Hooks> {
        &self.hooks
    }

    /// Set the agent loop configuration.
    pub fn set_config(&mut self, config: AgentLoopConfig) {
        self.config = config;
    }

    /// Get a reference to the loop config.
    pub fn config(&self) -> &AgentLoopConfig {
        &self.config
    }

    // ── State access ───────────────────────────────────────────

    /// Get a read lock on agent state.
    pub async fn state(&self) -> tokio::sync::RwLockReadGuard<'_, AgentState> {
        self.state.read().await
    }

    /// Register a tool with the agent.
    pub async fn add_tool(&self, tool: Arc<dyn AgentTool>) {
        let mut state = self.state.write().await;
        state.tools.push(tool);
    }

    /// Add or replace an authentication token for a provider.
    pub fn set_api_key(&self, provider: ProviderKind, key: impl Into<String>) {
        self.provider.set_api_key(provider, key);
    }

    /// Set the system prompt.
    pub async fn set_system_prompt(&self, prompt: Vec<ContentBlock>) {
        let mut state = self.state.write().await;
        state.system_prompt = prompt;
    }

    /// Set the resource context (skills, extensions, startup skills).
    /// This is injected as a synthetic user message, NOT in the system prompt.
    pub async fn set_resource_context(&self, prompt: Vec<ContentBlock>) {
        let mut state = self.state.write().await;
        state.resource_context = Some(prompt);
    }

    /// Switch the active model.
    /// Set the model and record a ModelChange in the transcript.
    pub async fn set_model(&self, model: Model) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let mut state = self.state.write().await;
        state.messages.push(Message::ModelChange {
            provider: Some(model.provider),
            model_id: Some(model.id.clone()),
            timestamp: now_ms,
        });
        state.model = model;
    }

    /// Set the thinking level and record a ThinkingLevelChange in the transcript.
    pub async fn set_thinking_level(&self, level: theta_ai::ThinkingLevel) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64;
        let mut state = self.state.write().await;
        state.messages.push(Message::ThinkingLevelChange {
            level,
            timestamp: now_ms,
        });
        state.thinking_level = level;
    }

    /// Load past messages from a session (for continue/resume).
    pub async fn load_messages(&self, messages: Vec<Message>) {
        let mut state = self.state.write().await;
        let (sanitized, stats) = theta_ai::sanitize_messages_for_replay(&messages, &state.model);
        state.load_messages(sanitized);
        if stats.changed() {
            let _ = self.event_tx.send(AgentEvent::ReplaySanitized {
                dropped_assistant_messages: stats.dropped_assistant_messages,
                synthesized_tool_results: stats.synthesized_tool_results,
                normalized_tool_call_ids: stats.normalized_tool_call_ids,
                deduped_tool_results: stats.deduped_tool_results,
            });
        }
    }

    /// Get the number of messages currently in the transcript.
    pub async fn message_count(&self) -> usize {
        self.state.read().await.messages.len()
    }

    /// Live context stats for the TUI.
    pub async fn context_stats(&self) -> (usize, u32, Option<u32>) {
        let state = self.state.read().await;
        let resource_tokens: u32 = state.resource_context_tokens();
        (
            state.messages.len(),
            state.token_count() + resource_tokens,
            state.last_real_input_tokens(),
        )
    }

    /// Last completed run report, if available.
    pub async fn last_run_report(&self) -> Option<RunReport> {
        self.state.read().await.last_run_report.clone()
    }

    /// Manual compaction: trim old messages without running a loop.
    /// Forces compaction regardless of the `enabled` config flag.
    /// Sends ContextCompacted event and returns how many were trimmed.
    pub async fn compact_context(&self) -> Result<u32, AgentError> {
        let (result, _compaction_config) = {
            let state = self.state.read().await;
            let system_tokens: u32 = state
                .system_prompt
                .iter()
                .map(|b| {
                    theta_ai::approximate_token_count(&serde_json::to_string(b).unwrap_or_default())
                })
                .sum();
            let res_tokens: u32 = state.resource_context_tokens();
            let llm_msgs: Vec<theta_ai::Message> = state.llm_messages();
            // Force-enable compaction for manual trigger (ignore config.enabled).
            let mut force_config = self.config.compaction.clone();
            force_config.enabled = true;

            // Manual compact: preserve keep_recent_tokens of recent
            // conversation, summarize everything older. Keep a fixed
            // useful window (default 20K tokens) rather than basing
            // it on the model's context window.
            // available = context_window - reserve_tokens - system_tokens - resource_tokens
            // We set context_window so that available = keep_recent_tokens.
            let effective_window = self
                .config
                .compaction
                .keep_recent_tokens
                .saturating_add(self.config.compaction.reserve_tokens)
                .saturating_add(system_tokens)
                .saturating_add(res_tokens);
            (
                crate::compact::compact_messages(
                    &llm_msgs,
                    system_tokens + res_tokens,
                    effective_window,
                    &force_config,
                ),
                self.config.compaction.clone(),
            )
        };

        let trimmed = result.trimmed_count;
        if trimmed > 0 {
            let _ = self.event_tx.send(AgentEvent::ContextCompacted {
                trimmed_count: trimmed,
                tokens_before: result.tokens_before,
                tokens_after: result.tokens_after,
            });

            // Apply compaction: keep ModelChange/ThinkingLevelChange + compacted result.
            let mut state = self.state.write().await;
            let meta_entries: Vec<_> = state
                .messages
                .iter()
                .filter(|m| {
                    matches!(
                        m,
                        theta_ai::Message::ModelChange { .. }
                            | theta_ai::Message::ThinkingLevelChange { .. }
                    )
                })
                .cloned()
                .collect();
            // Prepend meta entries (oldest first), then compacted messages.
            let mut new_messages = meta_entries;
            // Sort compacted messages by timestamp to keep chronological order.
            let mut compacted = result.messages;
            compacted.sort_by_key(|m| m.timestamp());
            new_messages.extend(compacted);
            state.messages = new_messages;
        }

        Ok(trimmed)
    }

    // ── Run control ───────────────────────────────────────────

    /// Start a new agent run with a user prompt.
    ///
    /// Adds the user message to the transcript and runs the full
    /// agent loop (LLM calls + tool execution).
    pub async fn prompt(&self, content: Vec<ContentBlock>) -> Result<(), AgentError> {
        let timestamp = now_ms();

        // Acquire the active run lock.
        let (abort_token, steering_abort) = {
            let mut run = self.active_run.lock().expect("active_run lock poisoned");
            if run.is_some() {
                return Err(AgentError::AlreadyRunning);
            }
            let steering_abort = Arc::new(AtomicBool::new(false));
            let token = CancellationToken::new();
            *run = Some(ActiveRun {
                abort_token: token.clone(),
                steering_abort: steering_abort.clone(),
            });
            (token, steering_abort)
        };

        // Add user message to state.
        {
            let mut state = self.state.write().await;
            state.add_user_message(content, timestamp);
        }

        // Run the agent loop.
        let result = self
            .run_loop(Some(abort_token.clone()), steering_abort)
            .await;

        // Release the active run.
        {
            let mut run = self.active_run.lock().expect("active_run lock poisoned");
            *run = None;
        }

        result
    }

    /// Continue the conversation (e.g., after user abort or from a loaded session).
    pub async fn continue_(&self) -> Result<(), AgentError> {
        let (abort_token, steering_abort) = {
            let mut run = self.active_run.lock().expect("active_run lock poisoned");
            if run.is_some() {
                return Err(AgentError::AlreadyRunning);
            }
            let steering_abort = Arc::new(AtomicBool::new(false));
            let token = CancellationToken::new();
            *run = Some(ActiveRun {
                abort_token: token.clone(),
                steering_abort: steering_abort.clone(),
            });
            (token, steering_abort)
        };

        let result = self
            .run_loop(Some(abort_token.clone()), steering_abort)
            .await;

        {
            let mut run = self.active_run.lock().expect("active_run lock poisoned");
            *run = None;
        }

        result
    }

    /// Run the full outer agent loop.
    async fn run_loop(
        &self,
        abort_token: Option<CancellationToken>,
        steering_abort: Arc<AtomicBool>,
    ) -> Result<(), AgentError> {
        // Get the provider for the current model (read lock, then drop).
        let provider_api = {
            let state = self.state.read().await;
            state.model.api
        };

        let provider =
            self.provider
                .get(&provider_api)
                .ok_or_else(|| theta_ai::ThetaError::ApiError {
                    status: 500,
                    message: format!("no provider registered for API {provider_api:?}"),
                    retry_after_ms: None,
                })?;

        // Pass shared queue references into the loop so steer()
        // and the loop see the same queues.
        let follow_up_queue = self.follow_up_queue.clone();
        let steering_queue = self.steering_queue.clone();

        {
            let mut state = self.state.write().await;

            loop_mod::run_prompt_loop(
                &mut state,
                provider,
                &self.hooks,
                &self.config,
                &self.event_tx,
                abort_token,
                steering_abort,
                steering_queue,
                follow_up_queue,
            )
            .await?;
        }

        Ok(())
    }

    /// Abort the currently running agent loop.
    pub fn abort(&self) -> Result<(), AgentError> {
        let run = self.active_run.lock().expect("active_run lock poisoned");
        match run.as_ref() {
            Some(active) => {
                active.abort_token.cancel();
                tracing::info!("agent aborted");
                Ok(())
            }
            None => Err(AgentError::NotRunning),
        }
    }

    // ── Steering ──────────────────────────────────────────────

    /// Inject a steering message that interrupts the current turn.
    ///
    /// If the agent is running, the message is added to the transcript
    /// before the next inner-loop iteration. The current LLM stream
    /// is aborted via the per-stream steering flag.
    ///
    /// If the agent is not running, the message is queued for the next
    /// `prompt()` or `continue_()` call.
    pub fn steer(&self, content: Vec<ContentBlock>) {
        let msg = Message::User {
            content,
            timestamp: now_ms(),
        };
        {
            let mut queue = self
                .steering_queue
                .lock()
                .expect("steering_queue lock poisoned");
            queue.push((msg, now_ms()));
        }

        // Signal the per-stream abort so the inner loop restarts.
        if let Ok(run) = self.active_run.lock()
            && let Some(ref active) = *run
        {
            active.steering_abort.store(true, Ordering::SeqCst);
            tracing::debug!("steering abort signaled");
        }
    }

    /// Queue a follow-up message.
    pub fn follow_up(&self, content: Vec<ContentBlock>) {
        let msg = Message::User {
            content,
            timestamp: now_ms(),
        };
        let mut queue = self
            .follow_up_queue
            .lock()
            .expect("follow_up_queue lock poisoned");
        queue.push((msg, now_ms()));
    }

    /// Return pending queue lengths: (steering, follow-up).
    pub fn queue_lengths(&self) -> (usize, usize) {
        let steering = self
            .steering_queue
            .lock()
            .expect("steering_queue lock poisoned");
        let follow_up = self
            .follow_up_queue
            .lock()
            .expect("follow_up_queue lock poisoned");
        (steering.len(), follow_up.len())
    }

    // ── Events ────────────────────────────────────────────────

    /// Subscribe to agent events. Returns a receiver that will get all
    /// future events.
    pub fn subscribe(&self) -> broadcast::Receiver<AgentEvent> {
        self.event_tx.subscribe()
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}
