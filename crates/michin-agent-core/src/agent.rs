//! Agent: the main entry point for the agent runtime.
//!
//! Owns state, provider, hooks, and runs the prompt/continue loops.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{RwLock, broadcast};
use tokio_util::sync::CancellationToken;
use tracing;

use michin_ai::Provider as ProviderKind;
use michin_ai::providers::ProviderRegistry;
use michin_ai::{ContentBlock, Message, Model};

use crate::error::AgentError;
use crate::events::AgentEvent;
use crate::hooks::{Hooks, NoopHooks};
use crate::loop_mod;
use crate::state::AgentState;
use crate::types::{AgentLoopConfig, AgentTool, RunReport};

pub struct Agent {
    state: RwLock<AgentState>,
    event_tx: broadcast::Sender<AgentEvent>,
    provider: Arc<ProviderRegistry>,
    hooks: Arc<dyn Hooks>,
    /// Active run handle (ensures only one prompt/continue at a time).
    active_run: Mutex<Option<ActiveRun>>,
    config: AgentLoopConfig,
    follow_up_queue: Arc<Mutex<Vec<(Message, u64)>>>,
    steering_queue: Arc<Mutex<Vec<(Message, u64)>>>,
    /// Fast-path flag: true when steering_queue has items. Avoids locking the
    /// mutex on every inner-loop iteration just to check for pending steering.
    steering_has_items: Arc<AtomicBool>,
}

struct ActiveRun {
    abort_token: CancellationToken,
    /// Per-stream abort for steering. Set when `steer()` interrupts.
    steering_abort: Arc<AtomicBool>,
}

impl Agent {
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
            steering_has_items: Arc::new(AtomicBool::new(false)),
        }
    }

    // ── Configuration ──────────────────────────────────────────

    pub fn set_hooks(&mut self, hooks: Arc<dyn Hooks>) {
        self.hooks = hooks;
    }

    pub fn hooks(&self) -> &Arc<dyn Hooks> {
        &self.hooks
    }

    pub fn set_config(&mut self, config: AgentLoopConfig) {
        self.config = config;
    }

    pub fn config(&self) -> &AgentLoopConfig {
        &self.config
    }

    // ── State access ───────────────────────────────────────────

    pub async fn state(&self) -> tokio::sync::RwLockReadGuard<'_, AgentState> {
        self.state.read().await
    }

    pub async fn add_tool(&self, tool: Arc<dyn AgentTool>) {
        let mut state = self.state.write().await;
        state.tools.push(tool);
        state.rebuild_michin_ai_tools();
    }

    pub fn set_api_key(&self, provider: ProviderKind, key: impl Into<String>) {
        self.provider.set_api_key(provider, key);
    }

    pub fn set_mimo_base_url(&self, url: &str) {
        let url = url.to_string();
        self.provider.set_mimo_base_url(&url);
    }

    pub fn provider_key(&self, provider: ProviderKind) -> Option<String> {
        self.provider.get_api_key(provider)
    }

    pub async fn set_system_prompt(&self, prompt: Vec<ContentBlock>) {
        let mut state = self.state.write().await;
        state.system_prompt = prompt;
        state.update_cached_tokens();
    }

    /// Set the resource context (skills, extensions).
    /// Appended to the system prompt before each LLM call.
    pub async fn set_resource_context(&self, prompt: Vec<ContentBlock>) {
        let mut state = self.state.write().await;
        state.resource_context = Some(prompt);
        state.update_cached_tokens();
    }

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

    pub async fn set_thinking_level(&self, level: michin_ai::ThinkingLevel) {
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

    /// Toggle plan mode on or off.
    ///
    /// When enabled: mutations are blocked by command policy, the system prompt
    /// guides the model toward plan-only exploration.
    pub async fn set_plan_mode(&self, enabled: bool) {
        let mut state = self.state.write().await;
        state.plan_mode = enabled;
        drop(state);
        let _ = self.event_tx.send(AgentEvent::PlanModeToggled { enabled });
    }

    /// Returns whether plan mode is currently active.
    pub async fn plan_mode(&self) -> bool {
        self.state.read().await.plan_mode
    }

    /// Set the caveman communication mode level.
    ///
    /// `None` disables caveman mode. `Some("full")` enables at that level.
    pub async fn set_caveman_mode(&self, level: Option<String>) {
        let mut state = self.state.write().await;
        state.caveman_mode = level.clone();
        drop(state);
        let _ = self.event_tx.send(AgentEvent::CavemanModeToggled { level });
    }

    /// Returns the current caveman mode level, if active.
    pub async fn caveman_mode(&self) -> Option<String> {
        self.state.read().await.caveman_mode.clone()
    }

    /// Set the model to escalate to when the assistant requests it.
    pub async fn set_escalation_model(&self, model: Option<Model>) {
        let mut state = self.state.write().await;
        state.escalation_model = model;
    }

    pub async fn escalation_model(&self) -> Option<Model> {
        self.state.read().await.escalation_model.clone()
    }

    /// Set volatile overlays — content blocks appended to system context at request time.
    pub async fn set_volatile_overlays(&self, overlays: Vec<ContentBlock>) {
        let mut state = self.state.write().await;
        state.volatile_overlays = overlays;
    }

    pub async fn load_messages(&self, messages: Vec<Message>) {
        let mut state = self.state.write().await;
        let (sanitized, stats) = michin_ai::sanitize_messages_for_replay(&messages, &state.model);
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

    pub async fn message_count(&self) -> usize {
        self.state.read().await.messages.len()
    }

    /// Load persisted cache stats from a session index.
    pub async fn load_cache_stats(
        &self,
        stats: std::collections::HashMap<String, crate::CacheStats>,
    ) {
        if stats.is_empty() {
            return;
        }
        let mut state = self.state.write().await;
        for (key, val) in stats {
            if let Ok(provider) = serde_json::from_str::<michin_ai::Provider>(&format!("\"{key}\""))
            {
                state.cache_stats.insert(provider, val);
            }
        }
    }

    pub async fn context_stats(&self) -> (usize, u32, Option<u32>) {
        let state = self.state.read().await;
        let resource_tokens: u32 = state.resource_context_tokens();
        (
            state.messages.len(),
            state.token_count() + resource_tokens,
            state.last_real_input_tokens(),
        )
    }

    pub async fn last_run_report(&self) -> Option<RunReport> {
        self.state.read().await.last_run_report.clone()
    }

    /// Manual compaction: trim old messages without running a loop.
    /// Forces compaction regardless of the `enabled` config flag.
    pub async fn compact_context(&self) -> Result<u32, AgentError> {
        let (result, _compaction_config) = {
            let state = self.state.read().await;
            let system_tokens: u32 = state
                .system_prompt
                .iter()
                .map(|b| {
                    michin_ai::approximate_token_count(
                        &serde_json::to_string(b).unwrap_or_default(),
                    )
                })
                .sum();
            let res_tokens: u32 = state.resource_context_tokens();
            let llm_msgs: Vec<michin_ai::Message> = state.llm_messages();
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
                        michin_ai::Message::ModelChange { .. }
                            | michin_ai::Message::ThinkingLevelChange { .. }
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

    pub async fn prompt(&self, content: Vec<ContentBlock>) -> Result<(), AgentError> {
        let timestamp = now_ms();

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

        {
            let mut state = self.state.write().await;
            state.add_user_message(content, timestamp);
        }

        let result = self
            .run_loop(Some(abort_token.clone()), steering_abort)
            .await;

        {
            let mut run = self.active_run.lock().expect("active_run lock poisoned");
            *run = None;
        }

        result
    }

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

    async fn run_loop(
        &self,
        abort_token: Option<CancellationToken>,
        steering_abort: Arc<AtomicBool>,
    ) -> Result<(), AgentError> {
        // Get the provider for the current model, then drop the read lock.
        let provider_api = {
            let state = self.state.read().await;
            state.model.api
        };

        let provider =
            self.provider
                .get(&provider_api)
                .ok_or_else(|| michin_ai::MichiNError::ApiError {
                    status: 500,
                    message: format!("no provider registered for API {provider_api:?}"),
                    retry_after_ms: None,
                })?;

        let follow_up_queue = self.follow_up_queue.clone();
        let steering_queue = self.steering_queue.clone();
        let steering_has_items = self.steering_has_items.clone();

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
                steering_has_items,
                follow_up_queue,
            )
            .await?;
        }

        Ok(())
    }

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
        self.steering_has_items.store(true, Ordering::Relaxed);

        if let Ok(run) = self.active_run.lock()
            && let Some(ref active) = *run
        {
            active.steering_abort.store(true, Ordering::SeqCst);
            tracing::debug!("steering abort signaled");
        }
    }

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
