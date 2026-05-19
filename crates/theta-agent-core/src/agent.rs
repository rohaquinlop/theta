//! Agent: the main entry point for the agent runtime.
//!
//! Owns state, provider, hooks, and runs the prompt/continue loops.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tokio::sync::{RwLock, broadcast};
use tokio_util::sync::CancellationToken;
use tracing;

use theta_ai::providers::ProviderRegistry;
use theta_ai::{ContentBlock, Message, Model, ModelCatalog};

use crate::error::AgentError;
use crate::events::AgentEvent;
use crate::hooks::{Hooks, NoopHooks};
use crate::loop_mod;
use crate::state::AgentState;
use crate::types::{AgentLoopConfig, AgentTool};

/// The agent: holds all state and orchestrates LLM interaction.
pub struct Agent {
    /// Mutable agent state (transcript, tools, config).
    state: RwLock<AgentState>,

    /// Event broadcaster for TUI/RPC subscribers.
    event_tx: broadcast::Sender<AgentEvent>,

    /// Provider registry for LLM calls.
    provider: Arc<ProviderRegistry>,

    /// Model catalog for model lookup.
    #[allow(dead_code)]
    model_catalog: Arc<dyn ModelCatalog>,

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
    /// Create a new agent with the given model, provider, and model catalog.
    pub fn new(
        model: Model,
        provider: Arc<ProviderRegistry>,
        model_catalog: Arc<dyn ModelCatalog>,
    ) -> Self {
        Self {
            state: RwLock::new(AgentState::new(model)),
            event_tx: broadcast::channel(256).0,
            provider,
            model_catalog,
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

    /// Set the system prompt.
    pub async fn set_system_prompt(&self, prompt: Vec<ContentBlock>) {
        let mut state = self.state.write().await;
        state.system_prompt = prompt;
    }

    /// Switch the active model.
    pub async fn set_model(&self, model: Model) {
        let mut state = self.state.write().await;
        state.model = model;
    }

    /// Set the thinking level.
    pub async fn set_thinking_level(&self, level: theta_ai::ThinkingLevel) {
        let mut state = self.state.write().await;
        state.thinking_level = level;
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
                &*self.hooks,
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
        .unwrap()
        .as_millis() as u64
}
