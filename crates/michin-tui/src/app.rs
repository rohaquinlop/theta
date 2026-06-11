//! Application — main TUI event loop and layout management.

use std::collections::{HashMap, HashSet, VecDeque};

use crossterm::event::EventStream;
use fff_search::shared::SharedFilePicker;
use futures::StreamExt;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem},
};
use tokio::sync::mpsc;

use crate::components::CommandEntry;
use crate::components::caveman_selector::CavemanSelector;
use crate::components::chat::{Chat, ChatMessage, ChatRole};
use crate::components::editor::Editor;
use crate::components::login_flow::{LoginFlow, known_providers};
use crate::components::mimo_cluster::{MimoClusterEntry, MimoClusterSelector};
use crate::components::model_selector::{ModelEntry, ModelSelector};
use crate::components::session_picker::{SessionInfo, SessionPicker};
use crate::components::settings_selector::{SettingsSelector, SettingsView};
use crate::components::status::StatusBar;
use crate::components::theme_selector::ThemeSelector;
use crate::components::thinking_selector::ThinkingSelector;
use crate::components::{Action, Component};
use crate::keybinding::{Keybinding, default_bindings, resolve_event};
use crate::terminal;
use crate::theme::Theme;

/// Commands sent from the TUI back to the interactive handler.
#[derive(Debug, Clone)]
pub enum TuiAction {
    /// Switch to a different model.
    SwitchModel {
        model_id: String,
        provider: Option<String>,
        request_id: u64,
    },
    /// Change thinking level.
    SetThinking {
        level: String,
        request_id: u64,
    },
    /// Fork the current session.
    ForkSession,
    /// Login result from the login flow.
    LoginResult {
        provider: String,
        token: String,
    },
    /// Resume a specific session by ID.
    ResumeSession(String),
    /// Create a new session (dismiss session picker).
    NewSession,
    /// Show the session picker.
    ShowSessions,
    Steer(String),
    FollowUp(String),
    /// Start Codex OAuth flow (triggered from login_flow).
    StartCodexOAuth,
    /// Request live session info (token counts, context window, etc.).
    ShowSessionInfo,
    /// Show latest compact run timeline report.
    ShowRunTimeline,
    /// Request valid thinking levels for the current model.
    ShowThinkingSelector,
    /// Manually compact context now (even if under the auto-compaction threshold).
    CompactContext,
    /// Abort the currently running agent turn.
    AbortAgent,
    /// Toggle the selected model in the favorites list.
    ToggleFavoriteModel {
        model_id: String,
    },
    /// Show the MiMo cluster selector (latency test).
    ShowMimoClusters,
    /// Select a MiMo cluster URL.
    SelectMimoCluster {
        url: String,
    },
    /// Persist a theme selection to config.toml.
    SetTheme {
        name: String,
    },
    /// Persist settings changes to settings.json and apply live.
    UpdateSettings {
        steering_mode: String,
        follow_up_mode: String,
        transport_preference: String,
        show_thinking: bool,
        show_tool_diffs: bool,
        tool_progress_hz: u64,
        enter_behavior: String,
        max_context_window: Option<u32>,
        auto_escalate: bool,
    },
    /// Toggle plan mode on or off.
    TogglePlanMode,
    /// Toggle caveman compression mode. None = off, Some("full") = level.
    ToggleCavemanMode {
        level: Option<String>,
    },
    /// Toggle automatic escalation flash→pro.
    ToggleAutoEscalate,
    /// Set escalation target model.
    SetEscalationModel {
        model_id: String,
    },
    /// Open filtered model picker for escalation model selection.
    ShowEscalationSelector,
}

/// Events sent from the agent loop to the TUI.
#[derive(Debug, Clone)]
pub enum TuiEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ThinkingStart,
    ThinkingEnd,
    ToolCallPrepared {
        name: String,
        id: String,
    },
    ToolStart {
        name: String,
        id: String,
        args: Option<String>,
    },
    ToolProgress {
        name: String,
        message: String,
    },
    ToolEnd {
        id: String,
        name: String,
        is_error: bool,
        summary: String,
        /// Tool execution details (path, changes, diff, etc.).
        details: Option<serde_json::Value>,
    },
    TurnStart,
    TurnEnd {
        stop_reason: String,
    },
    TurnDecision {
        reason: String,
        details: String,
    },
    /// Fires when LLM streaming completes and tool execution is about to
    /// begin. The TUI uses this to know that the current assistant message
    /// is complete (no more TextDelta) and any subsequent events are tool
    /// execution phase.
    MessageEnd,
    /// Fires when the agent run completes (or is aborted).
    AgentEnd {
        aborted: bool,
    },
    ContextCompacted {
        trimmed_count: u32,
        tokens_before: u32,
        tokens_after: u32,
    },
    CompactionPaused {
        context_window: u32,
        reserve_tokens: u32,
    },
    Retrying {
        attempt: u32,
        delay_ms: u64,
    },
    /// Show the session picker with the given sessions.
    SessionPicker(Vec<SessionInfo>),
    /// A new session was created lazily on first message.
    SessionCreated {
        id: String,
        model: String,
    },
    /// Clear the chat display (used for /new and /clear).
    ClearChat,
    /// Informational system message (not an error).
    Info(String),
    /// Set the agent state in the status bar (e.g. "compacting").
    SetAgentState(String),
    /// Live session info: token counts, context window, compaction status.
    SessionInfo {
        message_count: usize,
        approx_tokens: u32,
        real_input_tokens: Option<u32>,
        context_window: u32,
        compaction_enabled: bool,
        reserve_tokens: u32,
        keep_recent_tokens: u32,
        model_id: String,
        provider: String,
    },
    Error(String),
    /// Load session history into the chat display.
    LoadHistory(Vec<HistoryEntry>),
    /// Refresh available models (e.g. after login).
    UpdateModels(Vec<ModelEntry>),
    /// Active model was switched successfully.
    ModelSwitched {
        model: String,
        provider: String,
    },
    /// Open filtered model picker for escalation model selection.
    /// Provider context is sent to filter to same provider.
    ShowEscalationSelector {
        provider: String,
    },
    /// Model self-escalated within a turn or restored.
    ModelEscalated {
        from: String,
        to: String,
        is_escalation: bool,
    },
    /// A config action (model/thinking) has been applied or rejected.
    ActionAck {
        request_id: u64,
    },
    QueueStatus {
        steer: usize,
        follow_up: usize,
    },
    /// Extension status line update from Rhai scripts.
    ExtensionStatus(ExtensionStatusPayload),
    /// Real token usage from the last API call (input_tokens from usage).
    ContextTokens {
        tokens: u32,
        pct: u32,
    },
    /// Valid thinking levels for the current model.
    ThinkingLevels {
        levels: Vec<String>,
        current: String,
        /// When true, keep the selector visible after updating entries
        /// (used after model switch with unknown thinking level).
        show_selector: bool,
    },
    /// Thinking level was applied by backend.
    ThinkingSet {
        level: String,
    },
    /// Plan mode was toggled on or off.
    PlanModeToggled {
        enabled: bool,
    },
    /// Caveman mode level changed. None = off, Some("full") = active.
    CavemanModeToggled {
        level: Option<String>,
    },
    /// A skill was activated by the agent.
    SkillActivated {
        name: String,
    },
    /// Favorites list was updated — model selector needs to rebuild.
    ModelFavoritesUpdated {
        favorites: Vec<String>,
        toggled_model: String,
        is_favorite: bool,
    },
    /// MiMo cluster latency results (from async measurement).
    /// Includes the currently configured cluster URL for pre-selection.
    MimoClusterResults {
        clusters: Vec<MimoClusterEntry>,
        current_url: Option<String>,
    },
    /// Settings were saved and should be applied live to the TUI.
    SettingsApplied {
        steering_mode: String,
        follow_up_mode: String,
        show_thinking: bool,
        show_tool_diffs: bool,
        tool_progress_hz: u64,
    },
}

/// Structured status-bar data from extensions (Rhai scripts).
#[derive(Debug, Clone)]
pub struct ExtensionStatusPayload {
    /// Lines by row index (0 = primary row, the one rendered at bottom).
    pub rows: Vec<crate::components::status::StatusRow>,
    /// Number of rows from tui.row() callbacks that need their own visual row
    /// (excludes status lines merged into row 0).
    pub extension_row_count: usize,
}

/// A historical message entry for session resume.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub role: String, // "user", "assistant", "tool", "system"
    pub text: String,
}

#[derive(Debug, Clone)]
pub struct SettingsPayload {
    pub steering_mode: String,
    pub follow_up_mode: String,
    pub transport_preference: String,
    pub show_thinking: bool,
    pub show_tool_diffs: bool,
    pub tool_progress_hz: u64,
    pub enter_behavior: String,
    pub max_context_window: Option<u32>,
    pub auto_escalate: bool,
}

/// Which view is currently active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppMode {
    Chat,
    SessionPicker,
}

/// A queued steer or follow-up message shown in the floating queue box above
/// the editor during agent streaming.
#[derive(Debug, Clone)]
struct SteerEntry {
    text: String,
    kind: SteerKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SteerKind {
    Steer,
    FollowUp,
}

/// The main TUI application.
pub struct App {
    chat: Chat,
    editor: Editor,
    status: StatusBar,
    working_dir: std::path::PathBuf,
    session_picker: Option<SessionPicker>,
    model_selector: ModelSelector,

    thinking_selector: ThinkingSelector,
    caveman_selector: CavemanSelector,
    theme_selector: ThemeSelector,
    mimo_cluster_selector: MimoClusterSelector,
    settings_selector: SettingsSelector,
    commands: Vec<CommandEntry>,
    skill_commands: Vec<String>,
    keybindings: Vec<Keybinding>,
    running: bool,
    mode: AppMode,
    /// Send user messages to the agent.
    pub message_tx: mpsc::UnboundedSender<String>,
    /// Send structured actions back to the interactive handler.
    pub action_tx: mpsc::UnboundedSender<TuiAction>,
    /// Receive TUI events from the agent.
    pub event_rx: mpsc::UnboundedReceiver<TuiEvent>,
    theme: Theme,
    theme_idx: usize,
    current_theme_name: String,
    user_themes: HashMap<String, Theme>,
    streaming: bool,
    /// Set to true when MessageEnd fires — transition from LLM streaming
    /// to tool execution phase. ToolStart events after this flag show a visual cue.
    tool_exec_phase: bool,
    current_tool: Option<String>,
    /// Tool tracking maps keyed by tool_call_id (not tool_name).
    tool_display_text: HashMap<String, String>,
    tool_started_at: HashMap<String, std::time::Instant>,
    tool_last_tick_sec: HashMap<String, u64>,
    /// Maps tool_call_id → tool_name for display in update_tool_elapsed.
    tool_id_to_name: HashMap<String, String>,
    tools_in_turn: usize,
    retries_in_turn: u32,
    turn_index: u32,
    turn_intent: String,
    diag_enabled: bool,
    /// Steer/follow-up messages queued during streaming, rendered as a
    /// floating box above the editor. Flushed into chat on AgentEnd.
    queued_messages: VecDeque<SteerEntry>,
    /// Selection index into queued_messages for Ctrl+Up/Ctrl+Down navigation.
    /// Clamped on access; defaults to last entry when queue is non-empty.
    queued_selection: usize,
    /// True while the user has popped a queued message into the editor and is
    /// editing it. Suppresses queue flush on AgentEnd.
    editing_queued: bool,
    /// Set when AgentEnd fires while editing_queued was true. When the user
    /// finishes editing and re-queues, the queue is flushed immediately.
    deferred_flush: bool,
    /// True when the queue box has keyboard focus (Ctrl+U toggles).
    /// Arrow keys navigate the queue; Enter pops to editor; Delete discards.
    queue_focused: bool,
    show_thinking: bool,
    show_tool_diffs: bool,
    steering_mode: String,
    follow_up_mode: String,
    tool_progress_hz: u64,
    /// Active login flow (replaces chat+editor when set).
    login_flow: Option<LoginFlow>,
    window_title: Option<String>,
    pending_config_actions: HashSet<u64>,
    next_config_request_id: u64,
    /// Plan mode — controlled by `/plan` and tracked for status badge.
    plan_mode: bool,
    /// True when escalation model picker is active.
    escalation_picker_active: bool,
    queued_pending_messages: VecDeque<String>,
    /// Set to true when user presses Cancel while idle — second Cancel quits.
    quit_confirmation: bool,
    /// Timestamp of the first cancel press — if too old, resets the confirm.
    quit_confirm_at: Option<std::time::Instant>,
}

impl App {
    #[allow(clippy::too_many_arguments)]
    /// Activate the login flow from outside the app (e.g. on startup when auth is missing).
    pub fn start_login_flow(
        &mut self,
        providers: Vec<crate::components::login_flow::ProviderEntry>,
    ) {
        self.login_flow = Some(LoginFlow::new(self.theme.clone(), providers));
    }

    /// Update the session ID in the status bar (for lazy session creation).
    pub fn set_session_id(&mut self, id: String) {
        self.status.session_id = id;
    }

    /// Clear quit confirmation if the 2-second window has expired.
    /// Update streaming tool messages with elapsed runtime.
    /// Called from the 30fps tick when tools are executing.
    /// Only updates once per second to avoid excessive cache rebuilds.
    fn update_tool_elapsed(&mut self) {
        if self.tool_started_at.is_empty() {
            return;
        }
        let ids: Vec<String> = self.tool_started_at.keys().cloned().collect();
        for call_id in ids {
            let Some(start) = self.tool_started_at.get(&call_id) else {
                continue;
            };
            let elapsed = start.elapsed().as_secs();
            if elapsed < 1 {
                continue;
            }
            let last = self
                .tool_last_tick_sec
                .get(&call_id)
                .copied()
                .unwrap_or(u64::MAX);
            if elapsed == last {
                continue;
            }
            let display = self
                .tool_display_text
                .get(&call_id)
                .cloned()
                .unwrap_or_default();
            let tool_name = self
                .tool_id_to_name
                .get(&call_id)
                .cloned()
                .unwrap_or_else(|| call_id.clone());
            const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let spinner = SPINNER[(elapsed as usize) % 10];
            let text = if display.is_empty() {
                format!("{tool_name} {spinner} {elapsed}s")
            } else {
                format!("{display} {spinner} {elapsed}s")
            };
            self.chat.upsert_tool_message(&call_id, &text, true, false);
            self.tool_last_tick_sec.insert(call_id.clone(), elapsed);
        }
    }

    fn clear_expired_quit_confirmation(&mut self) {
        if !self.quit_confirmation {
            return;
        }
        if let Some(at) = self.quit_confirm_at
            && at.elapsed().as_millis() >= 2000
        {
            self.quit_confirmation = false;
            self.quit_confirm_at = None;
            if self.status.detail == "Press Esc/Ctrl+C again to exit" {
                self.status.set_detail("");
            }
        }
    }

    /// Inject an initial user message (used for startup prompt text).
    pub fn send_initial_message(&mut self, text: String) {
        if text.trim().is_empty() {
            return;
        }
        if self.queue_message_while_config_pending(Some(text.clone())) {
            return;
        }
        self.send_user_message_now(text);
        self.status.set_detail("sending initial prompt");
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        theme: Theme,
        user_themes: HashMap<String, Theme>,
        initial_theme_name: &str,
        model: &str,
        session_id: &str,
        thinking: &str,
        settings: SettingsPayload,
        models: Vec<ModelEntry>,
        favorites: Vec<String>,
        commands: Vec<CommandEntry>,
        working_dir: std::path::PathBuf,
        event_rx: mpsc::UnboundedReceiver<TuiEvent>,
        message_tx: mpsc::UnboundedSender<String>,
        action_tx: mpsc::UnboundedSender<TuiAction>,
        window_title: Option<String>,
        fff_picker: Option<SharedFilePicker>,
    ) -> Self {
        let mut status = StatusBar::new(theme.clone());
        status.model = model.to_string();
        status.session_id = session_id.to_string();
        status.thinking = thinking.to_string();
        status.set_agent_state("idle");

        let enter_behavior = settings.enter_behavior.clone();
        let mut editor = Editor::new(
            theme.clone(),
            working_dir.clone(),
            commands.iter().map(|c| c.name.clone()).collect(),
            enter_behavior,
            fff_picker,
        );
        let skill_commands = commands
            .iter()
            .filter_map(|c| c.name.strip_prefix("skill:").map(str::to_string))
            .collect();
        editor.focus(true); // Editor starts focused.

        let theme_names = Theme::all_names(&user_themes);

        Self {
            working_dir,
            chat: Chat::new(theme.clone()),
            editor,
            status,
            theme: theme.clone(),
            theme_idx: theme_names
                .iter()
                .position(|n| n == initial_theme_name)
                .unwrap_or(0),
            current_theme_name: initial_theme_name.to_string(),
            theme_selector: ThemeSelector::new(theme_names, user_themes.clone()),
            user_themes,
            session_picker: None,
            model_selector: ModelSelector::new(models, favorites, theme.clone()),
            thinking_selector: ThinkingSelector::new(theme.clone()),
            caveman_selector: CavemanSelector::new(theme.clone()),
            mimo_cluster_selector: MimoClusterSelector::new(),
            settings_selector: SettingsSelector::new(
                theme.clone(),
                SettingsView {
                    steering_mode: settings.steering_mode.clone(),
                    follow_up_mode: settings.follow_up_mode.clone(),
                    transport_preference: settings.transport_preference.clone(),
                    show_thinking: settings.show_thinking,
                    show_tool_diffs: settings.show_tool_diffs,
                    tool_progress_hz: settings.tool_progress_hz,
                    enter_behavior: settings.enter_behavior.clone(),
                    max_context_window: settings.max_context_window,
                    auto_escalate: settings.auto_escalate,
                },
            ),
            commands,
            skill_commands,
            keybindings: default_bindings(),
            running: true,
            mode: AppMode::Chat,
            message_tx,
            action_tx,
            event_rx,
            streaming: false,
            tool_exec_phase: false,
            current_tool: None,
            tool_display_text: HashMap::new(),
            tool_started_at: HashMap::new(),
            tool_last_tick_sec: HashMap::new(),
            tool_id_to_name: HashMap::new(),
            tools_in_turn: 0,
            retries_in_turn: 0,
            turn_index: 0,
            turn_intent: "chat".to_string(),
            diag_enabled: false,
            queued_messages: VecDeque::new(),
            queued_selection: 0,
            editing_queued: false,
            deferred_flush: false,
            queue_focused: false,
            show_thinking: settings.show_thinking,
            show_tool_diffs: settings.show_tool_diffs,
            steering_mode: settings.steering_mode,
            follow_up_mode: settings.follow_up_mode,
            tool_progress_hz: settings.tool_progress_hz.max(1),
            login_flow: None,
            window_title,
            pending_config_actions: HashSet::new(),
            next_config_request_id: 1,
            plan_mode: false,
            escalation_picker_active: false,
            queued_pending_messages: VecDeque::new(),
            quit_confirmation: false,
            quit_confirm_at: None,
        }
    }

    /// Run the TUI event loop.
    pub async fn run(&mut self) -> anyhow::Result<()> {
        if let Some(ref title) = self.window_title {
            terminal::setup_with_title(title)?;
        } else {
            terminal::setup()?;
        }
        let mut term = terminal::create_terminal()?;

        let result = self.run_loop(&mut term).await;

        terminal::restore()?;
        result
    }

    async fn run_loop(
        &mut self,
        term: &mut ratatui::Terminal<ratatui::backend::CrosstermBackend<std::io::Stdout>>,
    ) -> anyhow::Result<()> {
        let mut reader = EventStream::new();
        // Two render triggers:
        // 1. Fixed 30fps tick for scrolling, cursor blink, spinner animation
        // 2. Immediate render after draining an agent event burst
        //
        // The tick uses biased selection to prefer crossterm/event branches
        // over the timer. This prevents term.draw() (5-16ms) from blocking
        // event ingestion — events are always processed before rendering.
        let mut redraw_tick = tokio::time::interval(std::time::Duration::from_millis(33));
        redraw_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        while self.running {
            tokio::select! {
                biased;
                crossterm_event = reader.next() => {
                    if let Some(Ok(event)) = crossterm_event {
                        self.handle_input_event(&event);
                    }
                    let _ = term.draw(|frame| self.draw(frame));
                }
                Some(event) = self.event_rx.recv() => {
                    // Process ONE event, then render immediately.
                    // No drain loop — rendering after each event means
                    // intermediate states (tool preparing → running →
                    // done) are always visible before moving on. This
                    // trades away the drain optimization (useful for
                    // TextDelta bursts from LLM streaming) in favor of
                    // always showing the user the current state.
                    self.handle_agent_event(event);
                    let _ = term.draw(|frame| self.draw(frame));
                }
                _ = redraw_tick.tick() => {
                    self.clear_expired_quit_confirmation();
                    self.status.clear_expired_detail();
                    self.update_tool_elapsed();
                    let _ = term.draw(|frame| self.draw(frame));
                }
            }
        }
        Ok(())
    }

    fn draw(&mut self, frame: &mut Frame) {
        let area = frame.area();

        // Fill root background so themes with non-default bg take effect.
        frame.render_widget(
            Block::default().style(Style::default().bg(self.theme.bg)),
            area,
        );

        // Model selector overlay — renders on top of everything.
        self.model_selector.render(area, frame);
        if self.model_selector.visible {
            return;
        }
        self.thinking_selector.render(area, frame);
        if self.thinking_selector.visible {
            return;
        }
        self.caveman_selector.render(area, frame);
        if self.caveman_selector.visible {
            return;
        }
        self.theme_selector.render(area, frame, &self.theme);
        if self.theme_selector.visible {
            return;
        }
        self.mimo_cluster_selector.render(area, frame, &self.theme);
        if self.mimo_cluster_selector.visible {
            return;
        }
        self.settings_selector.render(area, frame);
        if self.settings_selector.visible {
            return;
        }

        // Session picker mode.
        if self.mode == AppMode::SessionPicker
            && let Some(ref mut picker) = self.session_picker
        {
            let status_h = self.status.desired_height();
            let main = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(status_h), Constraint::Min(8)])
                .split(area);
            self.status.render(main[0], frame);
            picker.render(main[1], frame);
            return;
        }

        if let Some(ref mut login) = self.login_flow {
            let status_h = self.status.desired_height();
            let main = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(status_h), Constraint::Min(8)])
                .split(area);
            self.status.render(main[0], frame);
            login.render(main[1], frame);
            return;
        }

        // Editor grows with content, capped at ~1/3 of terminal height
        // (min 6, max 15 rows) so chat always has room.
        let max_editor = (area.height / 3).clamp(6, 15);
        let editor_height = self.editor.desired_height(area.width as usize, max_editor);
        let status_height = self.status.desired_height();

        // Queued steer/follow-up messages get a floating box between chat and editor.
        let queue_height = if self.queued_messages.is_empty() {
            0
        } else {
            // 1 row border-top + N entries + 1 row border-bottom.
            // Cap at 6 entries to avoid stealing too much space from chat.
            (self.queued_messages.len() as u16 + 2).min(8)
        };

        let main = if queue_height > 0 {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(3),
                    Constraint::Length(queue_height),
                    Constraint::Length(editor_height),
                    Constraint::Length(status_height),
                ])
                .split(area)
        } else {
            Layout::default()
                .direction(Direction::Vertical)
                .constraints([
                    Constraint::Min(3),
                    Constraint::Length(editor_height),
                    Constraint::Length(status_height),
                ])
                .split(area)
        };

        self.chat.render(main[0], frame);
        if queue_height > 0 {
            self.render_queue_box(main[1], frame);
            self.editor.render(main[2], frame);
            self.status.render(main[3], frame);
        } else {
            self.editor.render(main[1], frame);
            self.status.render(main[2], frame);
        }

        // Render autocomplete popup on top (after editor so it overlays).
        let editor_idx = if queue_height > 0 { 2 } else { 1 };
        if self.editor.autocomplete_active() {
            self.render_autocomplete(main[editor_idx], frame);
        }
    }

    /// Render the floating queue box above the editor showing queued steer/follow-up messages.
    fn render_queue_box(&mut self, area: ratatui::layout::Rect, frame: &mut Frame) {
        self.clamp_queue_selection();
        let (title, border_style) = if self.queue_focused {
            (
                " Queued ◄► (Enter to edit, Del to discard, Esc) ",
                Style::default().fg(self.theme.accent),
            )
        } else if self.editing_queued {
            (
                " Queued (editing…) ",
                Style::default().fg(self.theme.warning),
            )
        } else {
            (
                " Queued (Ctrl+U to focus) ",
                Style::default().fg(self.theme.warning),
            )
        };
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(border_style)
            .title(title);
        let inner = block.inner(area);
        frame.render_widget(
            Block::default().style(Style::default().bg(self.theme.bg)),
            area,
        );
        frame.render_widget(block, area);
        if inner.height == 0 || self.queued_messages.is_empty() {
            return;
        }

        let visible_count = (inner.height as usize).min(self.queued_messages.len());
        let sel = self.queued_selection;
        let items: Vec<ListItem> = self
            .queued_messages
            .iter()
            .take(visible_count)
            .enumerate()
            .map(|(i, entry)| {
                let kind_label = match entry.kind {
                    SteerKind::Steer => "steer",
                    SteerKind::FollowUp => "follow-up",
                };
                let text = format!("{} [{}] {}", i + 1, kind_label, entry.text);
                let style = if i == sel {
                    Style::default().fg(self.theme.bg).bg(self.theme.warning)
                } else {
                    Style::default().fg(self.theme.warning)
                };
                ListItem::new(Line::from(Span::styled(text, style)))
            })
            .collect();

        let list = List::new(items);
        frame.render_widget(list, inner);
    }

    fn render_autocomplete(&mut self, editor_area: ratatui::layout::Rect, frame: &mut Frame) {
        let items = self.editor.autocomplete_items();
        if items.is_empty() {
            return;
        }
        let selected = self.editor.autocomplete_selected();

        let popup_height = (items.len() as u16 + 2).min(8);
        let popup_width = editor_area.width.min(50);
        // Position directly above the current editor area.
        let y = editor_area.y.saturating_sub(popup_height);
        let popup = ratatui::layout::Rect {
            x: editor_area.x,
            y,
            width: popup_width,
            height: popup_height,
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .style(Style::default().bg(self.theme.bg))
            .border_style(Style::default().fg(self.theme.border));
        let inner = block.inner(popup);
        frame.render_widget(
            Block::default().style(Style::default().bg(self.theme.bg)),
            popup,
        );
        frame.render_widget(block, popup);

        let visible_count = inner.height as usize;
        let start = selected.saturating_sub(visible_count.saturating_sub(1));
        let end = (start + visible_count).min(items.len());

        let list_items: Vec<ListItem> = items[start..end]
            .iter()
            .enumerate()
            .map(|(i, item)| {
                let idx = start + i;
                let prefix = if idx == selected { "> " } else { "  " };
                let text = format!("{prefix}{item}");
                let style = if idx == selected {
                    Style::default()
                        .fg(self.theme.accent)
                        .bg(self.theme.highlight)
                } else {
                    Style::default().fg(self.theme.fg).bg(self.theme.bg)
                };
                // Pad to full row width so background fills entire row.
                let padded = format!("{:<width$}", text, width = inner.width as usize);
                ListItem::new(Line::from(Span::styled(padded, style)))
            })
            .collect();

        let list = List::new(list_items).highlight_style(Style::default().bg(self.theme.bg));
        frame.render_widget(list, inner);
    }

    fn handle_input_event(&mut self, event: &crossterm::event::Event) {
        // Clear expired quit confirmation before processing.
        self.clear_expired_quit_confirmation();

        // Model selector mode — handle keys exclusively.
        if self.model_selector.visible {
            if let crossterm::event::Event::Key(key) = event {
                match key.code {
                    crossterm::event::KeyCode::Esc => {
                        if self.escalation_picker_active {
                            self.model_selector.restore_all_models();
                            self.escalation_picker_active = false;
                        }
                        self.model_selector.hide();
                    }
                    crossterm::event::KeyCode::Enter => {
                        if self.escalation_picker_active {
                            if let Some(entry) = self.model_selector.selected_model() {
                                let model_id = entry.id.clone();
                                let _ = self.action_tx.send(TuiAction::SetEscalationModel {
                                    model_id: model_id.clone(),
                                });
                                self.chat.add_message(ChatMessage {
                                    role: ChatRole::System,
                                    text: format!("Escalation model set to {model_id}"),
                                    tool_call_id: None,
                                    is_streaming: false,
                                    is_error: false,
                                });
                            }
                            self.model_selector.restore_all_models();
                            self.model_selector.hide();
                            self.escalation_picker_active = false;
                            return;
                        }
                        let is_xiaomi_and_cluster_needed;
                        if let Some(entry) = self.model_selector.selected_model() {
                            let model_id = entry.id.clone();
                            let provider = entry.provider.clone();
                            is_xiaomi_and_cluster_needed = entry.provider == "xiaomi";
                            let request_id = self.begin_config_action();
                            if !self.send_config_action(
                                request_id,
                                TuiAction::SwitchModel {
                                    model_id: model_id.clone(),
                                    provider: Some(provider),
                                    request_id,
                                },
                                "model",
                            ) {
                                self.model_selector.hide();
                                return;
                            }
                            self.chat.add_message(ChatMessage {
                                role: ChatRole::System,
                                text: format!("Switching model to {model_id}..."),
                                tool_call_id: None,
                                is_streaming: false,
                                is_error: false,
                            });
                            // For MiMo models without a selected cluster, show
                            // the cluster selector so the user can pick an endpoint.
                            if is_xiaomi_and_cluster_needed {
                                let _ = self.action_tx.send(TuiAction::ShowMimoClusters);
                                self.mimo_cluster_selector.start_measuring();
                            }
                        }
                        self.model_selector.hide();
                    }
                    crossterm::event::KeyCode::Up => {
                        self.model_selector.select_up();
                    }
                    crossterm::event::KeyCode::Down => {
                        self.model_selector.select_down();
                    }
                    crossterm::event::KeyCode::Backspace => {
                        self.model_selector.pop_query();
                    }
                    crossterm::event::KeyCode::Char('f')
                        if key
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL) =>
                    {
                        if let Some(entry) = self.model_selector.selected_model() {
                            let model_id = entry.id.clone();
                            let _ = self
                                .action_tx
                                .send(TuiAction::ToggleFavoriteModel { model_id });
                        }
                    }
                    crossterm::event::KeyCode::Char(c) => {
                        self.model_selector.push_query(c);
                    }
                    _ => {}
                }
            }
            return;
        }

        // Thinking selector mode — handle keys exclusively.
        if self.thinking_selector.visible {
            if let crossterm::event::Event::Key(key) = event {
                match key.code {
                    crossterm::event::KeyCode::Esc => {
                        self.thinking_selector.hide();
                    }
                    crossterm::event::KeyCode::Enter => {
                        if let Some(level) = self.thinking_selector.selected_level() {
                            let level = level.to_string();
                            let request_id = self.begin_config_action();
                            if !self.send_config_action(
                                request_id,
                                TuiAction::SetThinking {
                                    level: level.clone(),
                                    request_id,
                                },
                                "thinking",
                            ) {
                                self.thinking_selector.hide();
                                return;
                            }
                            self.chat.add_message(ChatMessage {
                                role: ChatRole::System,
                                text: format!("Setting thinking level to {level}..."),
                                tool_call_id: None,
                                is_streaming: false,
                                is_error: false,
                            });
                        }
                        self.thinking_selector.hide();
                    }
                    crossterm::event::KeyCode::Up => {
                        self.thinking_selector.select_up();
                    }
                    crossterm::event::KeyCode::Down => {
                        self.thinking_selector.select_down();
                    }
                    _ => {}
                }
            }
            return;
        }

        // Caveman selector mode — handle keys exclusively.
        if self.caveman_selector.visible {
            if let crossterm::event::Event::Key(key) = event {
                match key.code {
                    crossterm::event::KeyCode::Esc => {
                        self.caveman_selector.hide();
                    }
                    crossterm::event::KeyCode::Enter => {
                        if let Some(level) = self.caveman_selector.selected_level() {
                            let level = level.to_string();
                            let _ = self.action_tx.send(TuiAction::ToggleCavemanMode {
                                level: if level == "off" { None } else { Some(level) },
                            });
                        }
                        self.caveman_selector.hide();
                    }
                    crossterm::event::KeyCode::Up => {
                        self.caveman_selector.select_up();
                    }
                    crossterm::event::KeyCode::Down => {
                        self.caveman_selector.select_down();
                    }
                    _ => {}
                }
            }
            return;
        }

        // Theme selector mode — handle keys exclusively.
        if self.theme_selector.visible {
            if let crossterm::event::Event::Key(key) = event {
                match key.code {
                    crossterm::event::KeyCode::Esc => {
                        self.theme_selector.hide();
                    }
                    crossterm::event::KeyCode::Enter => {
                        if let Some(name) = self.theme_selector.selected_theme() {
                            let name = name.to_string();
                            if let Some(theme) = self.theme_selector.resolve_selected() {
                                self.apply_theme(theme, &name);
                            }
                            let _ = self
                                .action_tx
                                .send(TuiAction::SetTheme { name: name.clone() });
                            self.chat.add_message(ChatMessage {
                                role: ChatRole::System,
                                text: format!("Theme: {name}"),
                                tool_call_id: None,
                                is_streaming: false,
                                is_error: false,
                            });
                        }
                        self.theme_selector.hide();
                    }
                    crossterm::event::KeyCode::Up => {
                        self.theme_selector.select_up();
                    }
                    crossterm::event::KeyCode::Down => {
                        self.theme_selector.select_down();
                    }
                    _ => {}
                }
            }
            return;
        }

        // MiMo cluster selector mode — handle keys exclusively.
        if self.mimo_cluster_selector.visible {
            // Delegate navigation to the component.
            self.mimo_cluster_selector.handle_event(event);
            if let crossterm::event::Event::Key(key) = event {
                match key.code {
                    crossterm::event::KeyCode::Esc => {
                        self.mimo_cluster_selector.close();
                    }
                    crossterm::event::KeyCode::Enter => {
                        if let Some(url) = self
                            .mimo_cluster_selector
                            .selected_url()
                            .map(|s| s.to_string())
                        {
                            let _ = self.action_tx.send(TuiAction::SelectMimoCluster { url });
                        }
                        self.mimo_cluster_selector.close();
                    }
                    _ => {}
                }
            }
            return;
        }

        // Settings selector mode — handle keys exclusively.
        if self.settings_selector.visible {
            if let crossterm::event::Event::Key(key) = event {
                match key.code {
                    crossterm::event::KeyCode::Esc => {
                        let view = self.settings_selector.current_view();
                        let _ = self.action_tx.send(TuiAction::UpdateSettings {
                            steering_mode: view.steering_mode,
                            follow_up_mode: view.follow_up_mode,
                            transport_preference: view.transport_preference,
                            show_thinking: view.show_thinking,
                            show_tool_diffs: view.show_tool_diffs,
                            tool_progress_hz: view.tool_progress_hz,
                            enter_behavior: view.enter_behavior,
                            max_context_window: view.max_context_window,
                            auto_escalate: view.auto_escalate,
                        });
                        self.settings_selector.hide();
                    }
                    _ => {
                        self.settings_selector.handle_event(event);
                    }
                }
            }
            return;
        }
        // Session picker mode — handle picker keys exclusively.
        if self.mode == AppMode::SessionPicker {
            if let crossterm::event::Event::Key(key) = event {
                match key.code {
                    crossterm::event::KeyCode::Char('j') | crossterm::event::KeyCode::Down => {
                        if let Some(ref mut picker) = self.session_picker {
                            picker.select_down();
                        }
                    }
                    crossterm::event::KeyCode::Char('k') | crossterm::event::KeyCode::Up => {
                        if let Some(ref mut picker) = self.session_picker {
                            picker.select_up();
                        }
                    }
                    crossterm::event::KeyCode::Enter => {
                        if let Some(ref picker) = self.session_picker
                            && let Some(info) = picker.selected_session()
                        {
                            self.chat.clear_messages();
                            let _ = self
                                .action_tx
                                .send(TuiAction::ResumeSession(info.id.clone()));
                        } else {
                            let _ = self.action_tx.send(TuiAction::NewSession);
                        }
                        self.mode = AppMode::Chat;
                        self.session_picker = None;
                    }
                    crossterm::event::KeyCode::Char('n') | crossterm::event::KeyCode::Esc => {
                        let _ = self.action_tx.send(TuiAction::NewSession);
                        self.mode = AppMode::Chat;
                        self.session_picker = None;
                    }
                    crossterm::event::KeyCode::Char('s') => {
                        if let Some(ref mut picker) = self.session_picker {
                            picker.cycle_sort_mode();
                        }
                    }
                    _ => {}
                }
            }
            return;
        }

        // If login flow is active, handle its events exclusively.
        if let Some(ref mut login) = self.login_flow {
            let _ = login.handle_event(event);
            if login.is_done() {
                if !login.is_cancelled()
                    && let Some((provider, token)) = login.take_result()
                {
                    if token == "oauth" {
                        // Subscription provider: trigger OAuth flow in action handler.
                        // The action handler will show progress messages.
                        let _ = self.action_tx.send(TuiAction::StartCodexOAuth);
                    } else {
                        let _ = self
                            .action_tx
                            .send(TuiAction::LoginResult { provider, token });
                        self.chat.add_message(ChatMessage {
                            role: ChatRole::System,
                            text: "Token saved successfully.".into(),
                            tool_call_id: None,
                            is_streaming: false,
                            is_error: false,
                        });
                    }
                }
                self.login_flow = None;
            }
            return;
        }

        // ── Queue navigation mode (Ctrl+U toggles) ──
        if self.queue_focused {
            if let crossterm::event::Event::Key(key) = event {
                match key.code {
                    crossterm::event::KeyCode::Esc => {
                        self.queue_focused = false;
                        self.status.set_detail("");
                    }
                    crossterm::event::KeyCode::Up => {
                        if self.queued_selection > 0 {
                            self.queued_selection -= 1;
                        } else {
                            self.queued_selection = self.queued_messages.len() - 1;
                        }
                    }
                    crossterm::event::KeyCode::Down => {
                        if self.queued_selection + 1 < self.queued_messages.len() {
                            self.queued_selection += 1;
                        } else {
                            self.queued_selection = 0;
                        }
                    }
                    crossterm::event::KeyCode::Enter => {
                        // Pop selected entry to editor.
                        if let Some(entry) = self.queued_messages.remove(self.queued_selection) {
                            self.editor.set_text(&entry.text);
                            self.editing_queued = true;
                            self.status.set_detail("popped queued message to editor");
                        }
                        self.queue_focused = false;
                        if !self.queued_messages.is_empty() {
                            self.clamp_queue_selection();
                        }
                    }
                    crossterm::event::KeyCode::Delete | crossterm::event::KeyCode::Backspace => {
                        // Discard the selected queued message.
                        self.queued_messages.remove(self.queued_selection);
                        self.status.set_detail("discarded queued message");
                        if self.queued_messages.is_empty() {
                            self.queue_focused = false;
                            self.status.set_detail("");
                        } else {
                            self.clamp_queue_selection();
                        }
                    }
                    _ => {}
                }
            }
            return;
        }

        if let Some(action) = resolve_event(event, &self.keybindings) {
            self.handle_action(action);
            return;
        }

        if matches!(event, crossterm::event::Event::Mouse(_)) {
            if let Some(action) = self.chat.handle_event(event) {
                self.handle_action(action);
            }
            if let Some(action) = self.editor.handle_event(event) {
                self.handle_action(action);
            }
            return;
        }

        if let Some(action) = self.editor.handle_event(event) {
            self.handle_action(action);
        }
    }

    fn begin_config_action(&mut self) -> u64 {
        let request_id = self.next_config_request_id;
        self.next_config_request_id = self.next_config_request_id.wrapping_add(1);
        self.pending_config_actions.insert(request_id);
        request_id
    }

    fn send_config_action(&mut self, request_id: u64, action: TuiAction, kind: &str) -> bool {
        if self.action_tx.send(action).is_ok() {
            return true;
        }
        self.pending_config_actions.remove(&request_id);
        self.status.set_detail("failed to dispatch config update");
        self.chat.add_message(ChatMessage {
            role: ChatRole::System,
            text: format!("Failed to dispatch {kind} update. Please retry."),
            tool_call_id: None,
            is_streaming: false,
            is_error: false,
        });
        false
    }

    fn queue_message_while_config_pending(&mut self, text: Option<String>) -> bool {
        if self.pending_config_actions.is_empty() {
            return false;
        }
        if let Some(text) = text
            && !text.trim().is_empty()
        {
            self.queued_pending_messages.push_back(text);
        }
        let pending = self.pending_config_actions.len();
        self.status.set_detail(&format!(
            "waiting for model/thinking update ({pending} pending)"
        ));
        true
    }

    /// Clamp queued_selection to a valid index for the current queue.
    fn clamp_queue_selection(&mut self) {
        if self.queued_messages.is_empty() {
            self.queued_selection = 0;
        } else if self.queued_selection >= self.queued_messages.len() {
            self.queued_selection = self.queued_messages.len() - 1;
        }
    }

    /// Pop the next queued message from the FIFO queue, add it to chat as a
    /// User message, and send it to the agent as a new prompt. Returns true
    /// if a message was processed.
    fn process_next_queued(&mut self) -> bool {
        if self.editing_queued {
            self.deferred_flush = true;
            return false;
        }
        self.deferred_flush = false;
        let Some(entry) = self.queued_messages.pop_front() else {
            return false;
        };
        if !self.queued_messages.is_empty() {
            self.clamp_queue_selection();
        }
        self.chat.add_message(ChatMessage {
            role: ChatRole::User,
            text: entry.text.clone(),
            tool_call_id: None,
            is_streaming: false,
            is_error: false,
        });
        self.status.set_agent_state("streaming");
        self.streaming = true;
        let _ = self.message_tx.send(entry.text);
        true
    }

    fn send_user_message_now(&mut self, text: String) {
        self.quit_confirmation = false;
        self.quit_confirm_at = None;
        self.chat.add_message(ChatMessage {
            role: ChatRole::User,
            text: text.clone(),
            tool_call_id: None,
            is_streaming: false,
            is_error: false,
        });
        self.status.set_agent_state("streaming");
        self.streaming = true;
        // Don't set detail — [stream] badge already shows the mode.
        let _ = self.message_tx.send(text);
    }

    fn flush_queued_message_if_ready(&mut self) {
        if !self.pending_config_actions.is_empty() || self.streaming {
            return;
        }
        if let Some(text) = self.queued_pending_messages.pop_front() {
            self.send_user_message_now(text);
        }
    }

    fn handle_action(&mut self, action: Action) {
        match action {
            Action::SendMessage(text) => {
                // Intercept slash commands before sending to agent.
                if let Some(slash) = text.strip_prefix('/') {
                    self.handle_slash_command(slash);
                    return;
                }
                if self.streaming {
                    // Queue the message for processing after the current turn
                    // completes. It does NOT steer the current turn — instead it
                    // will be sent as a new prompt when AgentEnd fires.
                    let was_editing = self.editing_queued;
                    self.editing_queued = false;
                    if self.steering_mode == "follow-up" {
                        self.queued_messages.push_back(SteerEntry {
                            text: text.clone(),
                            kind: SteerKind::FollowUp,
                        });
                    } else {
                        self.queued_messages.push_back(SteerEntry {
                            text: text.clone(),
                            kind: SteerKind::Steer,
                        });
                    }
                    self.queued_selection = self.queued_messages.len() - 1;
                    self.status.set_detail("queued message");
                    // If AgentEnd fired while we were editing, process the next
                    // queued message now that the edited text was re-queued.
                    if was_editing && self.deferred_flush {
                        self.process_next_queued();
                    }
                } else {
                    if self.queue_message_while_config_pending(Some(text.clone())) {
                        return;
                    }
                    self.send_user_message_now(text);
                }
            }
            Action::Quit => {
                self.running = false;
            }
            Action::Cancel => {
                if self.streaming {
                    // Cancel the current agent turn.
                    let _ = self.action_tx.send(TuiAction::AbortAgent);
                    self.queued_messages.clear();
                    self.editing_queued = false;
                    self.status.set_detail("cancelling...");
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Cancelling current agent execution...".into(),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                } else {
                    // Idle: quit confirmation (Esc twice, or Ctrl+C twice).
                    let now = std::time::Instant::now();
                    let within_timeout = self
                        .quit_confirm_at
                        .map(|t| now.duration_since(t).as_millis() < 2000)
                        .unwrap_or(false);
                    if self.quit_confirmation && within_timeout {
                        // Second press within 2s — confirm quit.
                        self.running = false;
                    } else {
                        // First press — show confirmation.
                        self.quit_confirmation = true;
                        self.quit_confirm_at = Some(now);
                        self.status.set_detail("Press Esc/Ctrl+C again to exit");
                    }
                }
            }
            Action::ShowModelSelector => {
                self.model_selector.show();
            }
            Action::FollowUpMessage(text) => {
                if self.streaming {
                    let was_editing = self.editing_queued;
                    self.editing_queued = false;
                    if self.follow_up_mode == "steer" {
                        self.queued_messages.push_back(SteerEntry {
                            text: text.clone(),
                            kind: SteerKind::Steer,
                        });
                    } else {
                        self.queued_messages.push_back(SteerEntry {
                            text: text.clone(),
                            kind: SteerKind::FollowUp,
                        });
                    }
                    self.queued_selection = self.queued_messages.len() - 1;
                    self.status.set_detail("queued message");
                    if was_editing && self.deferred_flush {
                        self.process_next_queued();
                    }
                } else if self.follow_up_mode == "steer" {
                    let _ = self.action_tx.send(TuiAction::Steer(text));
                } else {
                    let _ = self.action_tx.send(TuiAction::FollowUp(text));
                }
            }
            Action::EditQueuedMessage => {
                // Toggle queue focus mode. When focused, arrow keys navigate
                // the queue, Enter pops the selected entry to the editor,
                // and Delete/Backspace discards the selected entry.
                if self.queued_messages.is_empty() {
                    return;
                }
                self.queue_focused = !self.queue_focused;
                if self.queue_focused {
                    self.clamp_queue_selection();
                    self.queued_selection = self.queued_messages.len() - 1;
                    self.status
                        .set_detail("queue: arrows navigate, Enter to edit, Del to discard");
                } else {
                    self.status.set_detail("");
                }
            }
            Action::TogglePlanMode => {
                let _ = self.action_tx.send(TuiAction::TogglePlanMode);
            }
            Action::CycleTheme => {
                self.cycle_theme();
            }
            Action::CopySelection(text) => {
                if copy_to_clipboard(&text).is_ok() {
                    self.status.set_detail("copied selection");
                } else {
                    self.status.set_detail("copy failed");
                }
            }
            Action::OpenUrl(url) => {
                if open::that(&url).is_ok() {
                    self.status.set_detail("opened link");
                } else {
                    self.status.set_detail("open link failed");
                }
            }
            _ => {}
        }
    }

    fn cycle_theme(&mut self) {
        let names = Theme::all_names(&self.user_themes);
        if names.is_empty() {
            return;
        }
        self.theme_idx = (self.theme_idx + 1) % names.len();
        let name = names[self.theme_idx].clone();
        let theme = Theme::named_with_users(&name, &self.user_themes);
        self.apply_theme(theme, &name);
        self.chat.add_message(ChatMessage {
            role: ChatRole::System,
            text: format!("Theme: {name}"),
            tool_call_id: None,
            is_streaming: false,
            is_error: false,
        });
    }

    /// Apply a theme to all components and update the active theme name.
    fn apply_theme(&mut self, theme: Theme, name: &str) {
        self.theme = theme.clone();
        self.current_theme_name = name.to_string();
        self.chat.set_theme(theme.clone());
        self.editor.set_theme(theme.clone());
        self.status.set_theme(theme.clone());
        self.model_selector.set_theme(theme.clone());
        self.theme_selector.set_themes(
            Theme::all_names(&self.user_themes),
            self.user_themes.clone(),
        );
        if let Some(ref mut picker) = self.session_picker {
            picker.set_theme(theme.clone());
        }
        if let Some(ref mut login) = self.login_flow {
            login.set_theme(theme);
        }
    }

    /// Handle a slash command (text after the initial `/`).
    fn handle_slash_command(&mut self, cmd: &str) {
        let parts: Vec<&str> = cmd.splitn(2, ' ').collect();
        let command = parts[0];
        let arg = parts.get(1).copied().unwrap_or("");

        match command {
            "help" | "h" => {
                let help_text = [
                    "Slash commands:",
                    "  /model          Open model picker (available models)",
                    "  /thinking [lvl] Open selector or set thinking level (off/enabled/minimal/low/medium/high/xhigh)",
                    "  /effort [lvl]   Alias for /thinking",
                    "  /plan           Toggle plan mode (explore/analyze without changing source code)",
                    "  /caveman [lvl]  Open selector or set caveman level (off/lite/full/ultra/wenyan-lite/wenyan-full/wenyan-ultra)",
                    "  /clear          Clear the chat display",
                    "  /session        Show session info (tokens, context window, compaction)",
                    "  /status         Show live runtime status snapshot",
                    "  /timeline       Show compact timeline from latest run report",
                    "  /compact        Manually compact context to fit in context window",
                    "  /fork           Fork the current session",
                    "  /new            Start a new unsaved session",
                    "  /sessions       List recent sessions (in picker press s to sort)",
                    "  /resume         Alias for /sessions",
                    "  /themes         Open theme picker with live preview",
                    "  /skills         List available skills",
                    "  /settings       Open settings panel (UI behavior, prefs)",
                    "  /model <id>     Switch model directly by id",
                    "  /diag on|off    Toggle diagnostic event stream in chat",
                    "  /tools-rate <hz> Set tool progress update rate (1-60)",
                    "  /skill:<name>   Invoke a skill",
                    "  /cancel         Cancel current agent execution",
                    "  /auto-escalate  Toggle auto-escalation between models",
                    "  /escalation-model Set or pick target escalation model",
                    "  /login          Log in to a provider",
                    "  /keys           Show keyboard shortcuts",
                    "  /exit           Exit MichiN",
                    "  /help           Show this help",
                ]
                .join("\n");
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: help_text,
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
            }
            "keys" | "bindings" => {
                let keys_text = [
                    "Global shortcuts:",
                    "  Ctrl+P         Open model picker",
                    "  Ctrl+T         Cycle theme (default / monokai / user themes)",
                    "  Ctrl+C         Cancel agent execution / quit confirm (press twice)",
                    "  Ctrl+Click     Open URL under cursor in browser",
                    "  Esc            Same as Ctrl+C",
                    "  Ctrl+U         Focus queued messages (arrows, Enter, Del)",
                    "  Tab            Switch focus between editor and chat",
                    "",
                    "Editor:",
                    "  Enter          Send message",
                    "  Alt+Enter      Queue follow-up (or insert newline in newline mode)",
                    "  Ctrl+Enter     Send as follow-up message",
                    "  Shift+Enter    Insert newline",
                    "  Ctrl+J         Insert newline",
                    "  Up/Down        Move cursor (history at first/last line)",
                    "  Alt+Up/Down    Browse send history",
                    "  Left/Right     Move cursor one char",
                    "  Alt+Left/Right Move cursor one word",
                    "  Super+Left     Jump to start of text",
                    "  Super+Right    Jump to end of text",
                    "  Super+Up       Jump to start of text",
                    "  Super+Down     Jump to end of text",
                    "  Home/End       Jump to start/end of visual line",
                    "  PageUp/Down    Move cursor one page (history at boundary)",
                    "  Tab            Insert 2 spaces",
                    "  Backspace/Del  Delete character",
                    "  @              Trigger file autocomplete (fuzzy)",
                    "  /              Trigger command autocomplete",
                    "",
                    "Overlays (model picker, etc.):",
                    "  Up/Down        Navigate list",
                    "  Enter          Select",
                    "  Esc            Close",
                    "  Backspace      Pop query char (model picker)",
                    "  Ctrl+F         Toggle favorite (model picker)",
                ]
                .join("\n");
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: keys_text,
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
            }
            "model" => {
                if arg.is_empty() {
                    self.model_selector.show();
                } else {
                    let request_id = self.begin_config_action();
                    if !self.send_config_action(
                        request_id,
                        TuiAction::SwitchModel {
                            model_id: arg.to_string(),
                            provider: None,
                            request_id,
                        },
                        "model",
                    ) {
                        return;
                    }
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: format!("Switching model to {arg}..."),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                }
            }
            "thinking" | "effort" => {
                if arg.is_empty() {
                    // No arg — open the modal selector.
                    if self.thinking_selector.has_levels() {
                        self.thinking_selector.visible = true;
                    } else {
                        // Ask the handler for valid levels.
                        let _ = self.action_tx.send(TuiAction::ShowThinkingSelector);
                        self.chat.add_message(ChatMessage {
                            role: ChatRole::System,
                            text: "Loading thinking levels...".into(),
                            tool_call_id: None,
                            is_streaming: false,
                            is_error: false,
                        });
                    }
                } else {
                    let request_id = self.begin_config_action();
                    if !self.send_config_action(
                        request_id,
                        TuiAction::SetThinking {
                            level: arg.to_string(),
                            request_id,
                        },
                        "thinking",
                    ) {
                        return;
                    }
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: format!("Setting thinking level to {arg}..."),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                }
            }
            "clear" => {
                self.chat.clear_messages();
            }
            "session" | "s" => {
                let _ = self.action_tx.send(TuiAction::ShowSessionInfo);
            }
            "status" => {
                let detail = if self.status.detail.trim().is_empty() {
                    "(none)".to_string()
                } else {
                    self.status.detail.clone()
                };
                let current_tool = self.current_tool.as_deref().unwrap_or("(none)");
                let steer_count = self
                    .queued_messages
                    .iter()
                    .filter(|e| e.kind == SteerKind::Steer)
                    .count();
                let follow_up_count = self
                    .queued_messages
                    .iter()
                    .filter(|e| e.kind == SteerKind::FollowUp)
                    .count();
                let snapshot = format!(
                    "Runtime status:\nState: {}\nDetail: {}\nTurn: {}\nStreaming: {}\nCurrent tool: {}\nTools in turn: {}\nRetries in turn: {}\nSteer queue: {}\nFollow-up queue: {}\nLast turn decision: {}\nLast end reason: {}",
                    self.status.agent_state,
                    detail,
                    self.turn_index,
                    self.streaming,
                    current_tool,
                    self.tools_in_turn,
                    self.retries_in_turn,
                    steer_count,
                    follow_up_count,
                    if self.status.last_turn_decision.is_empty() {
                        "(none)".to_string()
                    } else {
                        self.status.last_turn_decision.clone()
                    },
                    self.status.last_end_reason,
                );
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: snapshot,
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
            }
            "timeline" => {
                let _ = self.action_tx.send(TuiAction::ShowRunTimeline);
            }
            "compact" | "comp" => {
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: "Compacting context...".into(),
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
                let _ = self.action_tx.send(TuiAction::CompactContext);
            }
            "plan" => {
                // Single toggle: /plan flips plan mode on ↔ off.
                let _ = self.action_tx.send(TuiAction::TogglePlanMode);
                let new_state = !self.plan_mode;
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: if new_state {
                        "Plan mode on. Explore and plan — ask me to save the plan to a file when ready.".into()
                    } else {
                        "Plan mode off.".into()
                    },
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
            }
            "caveman" => {
                if arg.is_empty() {
                    self.caveman_selector
                        .show(self.status.caveman_mode.as_deref());
                } else {
                    let level = arg.to_lowercase();
                    let _ = self
                        .action_tx
                        .send(TuiAction::ToggleCavemanMode { level: Some(level) });
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: format!("Caveman mode: {arg}"),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                }
            }
            "login" => {
                // Start the login flow.
                let providers = known_providers(false, false, false, false, false);
                self.login_flow = Some(LoginFlow::new(self.theme.clone(), providers));
            }
            "new" => {
                // Clear chat immediately in the UI — agent reset happens async.
                self.chat.clear_messages();
                let _ = self.action_tx.send(TuiAction::NewSession);
            }
            "sessions" | "resume" => {
                let _ = self.action_tx.send(TuiAction::ShowSessions);
            }
            "themes" | "theme" => {
                self.theme_selector.show(&self.current_theme_name);
            }
            "skills" => {
                if self.skill_commands.is_empty() {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "No skills found in ~/.michin/skills, ~/.agents/skills, ./.michin/skills, or ./.agents/skills".into(),
                        tool_call_id: None,
                        is_streaming: false,
                    is_error: false,
                });
                } else {
                    let mut rows = vec!["Available skills:".to_string()];
                    for command_name in &self.skill_commands {
                        if let Some(entry) = self
                            .commands
                            .iter()
                            .find(|c| c.name == format!("skill:{command_name}"))
                        {
                            rows.push(format!("  /skill:{} - {}", command_name, entry.description));
                        } else {
                            rows.push(format!("  /skill:{}", command_name));
                        }
                    }
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: rows.join("\n"),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                }
            }
            "settings" => {
                self.settings_selector.show();
            }
            "diag" => match arg {
                "on" => {
                    self.diag_enabled = true;
                    self.status.set_show_diagnostics(true);
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Diagnostics stream enabled.".into(),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                }
                "off" => {
                    self.diag_enabled = false;
                    self.status.set_show_diagnostics(false);
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Diagnostics stream hidden (critical failures still shown).".into(),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                }
                _ => {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Usage: /diag <on|off>".into(),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                }
            },
            "tools-rate" => {
                let parsed = arg.trim().parse::<u64>().ok();
                let Some(hz) = parsed else {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Usage: /tools-rate <1-60>".into(),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                    return;
                };
                if !(1..=60).contains(&hz) {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Usage: /tools-rate <1-60>".into(),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                    return;
                }
                self.tool_progress_hz = hz;
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: format!("Tool progress rate set to {hz} Hz."),
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
            }
            "fork" => {
                let _ = self.action_tx.send(TuiAction::ForkSession);
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: "Forking session...".into(),
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
            }
            "exit" => {
                self.running = false;
            }
            "cancel" => {
                if self.streaming {
                    let _ = self.action_tx.send(TuiAction::AbortAgent);
                    self.queued_messages.clear();
                    self.editing_queued = false;
                    self.status.set_detail("cancelling...");
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Cancelling current agent execution...".into(),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                } else {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "No agent execution to cancel.".into(),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                }
            }
            "auto-escalate" => {
                let _ = self.action_tx.send(TuiAction::ToggleAutoEscalate);
            }
            "escalation-model" => {
                if arg.is_empty() {
                    let _ = self.action_tx.send(TuiAction::ShowEscalationSelector);
                } else {
                    let _ = self.action_tx.send(TuiAction::SetEscalationModel {
                        model_id: arg.to_string(),
                    });
                }
            }
            _ if command.starts_with("skill:") => {
                let Some(skill_name) = command.strip_prefix("skill:") else {
                    return;
                };
                if self.skill_commands.iter().any(|n| n == skill_name) {
                    let full = if arg.is_empty() {
                        format!("/{command}")
                    } else {
                        format!("/{command} {arg}")
                    };
                    if self.queue_message_while_config_pending(Some(full.clone())) {
                        return;
                    }
                    self.send_user_message_now(full);
                } else {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: format!(
                            "Unknown skill: {skill_name}. Type /skills to list available skills."
                        ),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                }
            }
            _ => {
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: format!(
                        "Unknown command: /{command}. Type /help for available commands."
                    ),
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
            }
        }
    }

    fn handle_agent_event(&mut self, event: TuiEvent) {
        match event {
            TuiEvent::TextDelta(text) => {
                if self.streaming {
                    self.chat.update_last(&text, ChatRole::Assistant, true);
                }
                if self.current_tool.is_none() {
                    self.status.set_agent_state("ModelCall");
                }
                // Clear lingering retry detail once the model starts responding.
                if self.status.detail.starts_with("retry attempt") {
                    self.status.set_detail("");
                }
            }
            TuiEvent::ThinkingDelta(text) => {
                self.status.set_agent_state("thinking");
                if self.show_thinking {
                    self.chat.update_last(&text, ChatRole::Thinking, true);
                }
                // Clear lingering retry detail once the model starts responding.
                if self.status.detail.starts_with("retry attempt") {
                    self.status.set_detail("");
                }
            }
            TuiEvent::ThinkingStart => {
                self.status.set_agent_state("thinking");
                if self.show_thinking {
                    self.chat.update_last("", ChatRole::Thinking, true);
                }
            }
            TuiEvent::ThinkingEnd => {
                if self.show_thinking {
                    self.chat.finish_last(ChatRole::Thinking);
                }
            }
            TuiEvent::ToolCallPrepared { name, id } => {
                // Show a visual cue that a tool call is being prepared
                // during LLM streaming (before execution). This creates a
                // tool message immediately so the user sees the transition
                // from text streaming to tool preparation.
                self.chat
                    .upsert_tool_message(&id, &format!("{name} preparing..."), true, false);
            }
            TuiEvent::ToolStart { name, id, args, .. } => {
                self.current_tool = Some(name.clone());
                self.status.set_agent_state("ToolExec");
                if self.status.detail.starts_with("retry attempt") {
                    self.status.set_detail("");
                }
                self.tools_in_turn += 1;
                self.tool_exec_phase = true;
                // Extract command from args for display.
                let display_text =
                    tool_display_text(&name, &args, &self.working_dir.to_string_lossy());
                self.tool_display_text
                    .insert(id.clone(), display_text.clone());
                self.tool_started_at
                    .insert(id.clone(), std::time::Instant::now());
                self.tool_id_to_name.insert(id.clone(), name.clone());
                self.chat
                    .upsert_tool_message(&id, &display_text, true, false);
            }
            TuiEvent::ToolProgress {
                name: _,
                message: _,
            } => {
                // Rate-limited progress — do not show tool output in chat.
            }
            TuiEvent::ToolEnd {
                id,
                name,
                is_error,
                summary,
                details,
            } => {
                if self.current_tool.as_deref() == Some(name.as_str()) {
                    self.current_tool = None;
                }
                self.tool_display_text.remove(&id);
                self.tool_started_at.remove(&id);
                self.tool_last_tick_sec.remove(&id);
                self.tool_id_to_name.remove(&id);
                let mut final_text = if summary.is_empty() {
                    name.to_string()
                } else {
                    summary
                };
                // Conditionally append diff when enabled.
                if self.show_tool_diffs
                    && name == "edit"
                    && let Some(ref d) = details
                    && let Some(diff) = d.get("diff").and_then(|v| v.as_str())
                    && !diff.is_empty()
                {
                    use std::fmt::Write;
                    let _ = write!(final_text, "\n```diff\n{diff}```");
                }
                self.chat.complete_tool_compact(&id, &final_text, is_error);
                if is_error {
                    self.status.set_agent_state(&format!("tool error: {name}"));
                }
                // Don't set "Completed" here — AgentEnd handles the final state.
            }
            TuiEvent::TurnStart => {
                self.streaming = true;
                self.turn_index += 1;
                self.tools_in_turn = 0;
                self.retries_in_turn = 0;
                self.turn_intent = "chat".to_string();
                self.status.set_turn_index(self.turn_index);
                self.status.set_agent_state("ModelCall");
                self.status.set_detail("");
            }
            TuiEvent::TurnEnd { stop_reason } => {
                self.chat.finish_last(ChatRole::Assistant);
                self.chat.finish_last(ChatRole::Thinking);
                // Don't set streaming = false here! If TurnEnd arrives in the
                // same burst as subsequent tool events (ToolStart, TextDelta for
                // next turn), setting streaming=false causes TextDelta events to
                // be silently dropped. Only AgentEnd marks the true end of output.
                let turn_result = if stop_reason == "error" {
                    "failed"
                } else {
                    "success"
                };
                if self.diag_enabled {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: format!(
                            "Turn #{} · intent: {} · tools: {} · retries: {} · result: {}",
                            self.turn_index,
                            self.turn_intent,
                            self.tools_in_turn,
                            self.retries_in_turn,
                            turn_result
                        ),
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                }
                self.status.last_end_reason = stop_reason.clone();
                let normalized = stop_reason.to_lowercase();
                if normalized.contains("blocked") || normalized.contains("rejected") {
                    self.status.set_agent_state("Blocked");
                } else if normalized.contains("error") || normalized.contains("failure") {
                    self.status.set_agent_state("Failed");
                }
                // Don't set detail on TurnEnd — the agent loop will emit
                // AgentEnd with the final idle state when truly finished.
            }
            TuiEvent::TurnDecision { reason, details } => {
                self.status.last_turn_decision = reason;
                self.status.set_detail(&truncate_status_text(&details, 100));
            }
            TuiEvent::MessageEnd => {
                // LLM streaming is done. Finish the assistant/thinking messages
                // so the cursor is removed and subsequent tool events render cleanly.
                self.chat.finish_last(ChatRole::Assistant);
                self.chat.finish_last(ChatRole::Thinking);
                // Don't set detail — agent state already conveys the mode.
            }
            TuiEvent::ContextCompacted {
                trimmed_count,
                tokens_before,
                tokens_after,
            } => {
                self.status.set_agent_state("compacting");
                if trimmed_count == 1 {
                    self.status.set_detail(&format!(
                        "trimmed {trimmed_count} old message (~{tokens_before}→~{tokens_after} tok)"
                    ));
                } else {
                    self.status.set_detail(&format!(
                        "trimmed {trimmed_count} old messages (~{tokens_before}→~{tokens_after} tok)"
                    ));
                }
            }
            TuiEvent::CompactionPaused {
                context_window,
                reserve_tokens: _,
            } => {
                self.status.set_agent_state("compaction paused");
                self.status.set_detail(&format!(
                    "context window {context_window} too small; auto-compaction paused"
                ));
            }
            TuiEvent::Retrying {
                attempt,
                delay_ms: _,
            } => {
                self.retries_in_turn = self.retries_in_turn.max(attempt);
                self.status.set_agent_state("Retrying");
                self.status.set_detail(&format!("retry attempt {attempt}"));
            }
            TuiEvent::SessionPicker(sessions) => {
                self.session_picker = Some(SessionPicker::new(sessions, self.theme.clone()));
                self.mode = AppMode::SessionPicker;
            }
            TuiEvent::SessionCreated { id, model } => {
                self.status.session_id = id;
                self.status.model = model;
                self.status.context_tokens = 0;
                self.status.ctx_pct = 0;
                self.status.turn_index = 0;
                self.status.set_detail("session created");
            }
            TuiEvent::AgentEnd { aborted } => {
                self.chat.finish_last(ChatRole::Assistant);
                self.chat.finish_last(ChatRole::Thinking);
                for msg in &mut self.chat.messages {
                    if msg.role == ChatRole::Tool && msg.is_streaming {
                        msg.is_streaming = false;
                    }
                }
                // Flush queued steer/follow-up messages into chat now that
                // the model has finished. They appear after the assistant
                // response, where they logically belong.
                if !aborted {
                    self.process_next_queued();
                } else {
                    self.queued_messages.clear();
                    self.editing_queued = false;
                }
                // Don't invalidate render cache! The incremental cache is
                // already up-to-date after processing all tool events. Invalidating
                // here forces a full rebuild on the next render, wasting the
                // progressive cache updates that made streaming work.
                self.streaming = false;
                self.tool_exec_phase = false;
                self.current_tool = None;
                // Clear tool tracking state to prevent stuck spinners if a
                // ToolEnd event was dropped (e.g. due to broadcast channel lag).
                self.tool_started_at.clear();
                self.tool_display_text.clear();
                self.tool_last_tick_sec.clear();
                self.tool_id_to_name.clear();
                if aborted {
                    self.status.set_agent_state("Cancelled");
                    self.status.set_detail("execution cancelled");
                } else if self.status.agent_state != "Blocked"
                    && self.status.agent_state != "Failed"
                {
                    self.status.set_agent_state("idle");
                    self.status.set_detail("");
                }
            }
            TuiEvent::ClearChat => {
                self.chat.clear_messages();
            }
            TuiEvent::SetAgentState(state) => {
                self.status.set_agent_state(&state);
            }
            TuiEvent::Info(msg) => {
                if self.diag_enabled || !is_diagnostic_message(&msg) {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: msg,
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                }
            }
            TuiEvent::SessionInfo {
                message_count,
                approx_tokens,
                real_input_tokens,
                context_window,
                compaction_enabled,
                reserve_tokens,
                keep_recent_tokens,
                model_id,
                provider,
            } => {
                let avail = context_window.saturating_sub(reserve_tokens);
                let display_tokens = real_input_tokens.unwrap_or(approx_tokens);
                let token_source = if real_input_tokens.is_some() {
                    "(API)"
                } else {
                    "(est)"
                };
                let pct = if avail > 0 {
                    (display_tokens as f64 / avail as f64 * 100.0) as u32
                } else {
                    0
                };
                let comp_state = if compaction_enabled {
                    format!("on (keep recent: {keep_recent_tokens}, reserve: {reserve_tokens})")
                } else {
                    "off".into()
                };
                let info = format!(
                    "Session: {}\nModel: {model_id} ({provider})\nMessages: {message_count}\nContext tokens: ~{display_tokens} {token_source} / {avail} available ({pct}%)\nContext window: {context_window} tokens\nAuto-compaction: {comp_state}\n\nUse /compact to trim context now.",
                    self.status.session_id
                );
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: info,
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
            }
            TuiEvent::Error(msg) => {
                self.turn_intent = infer_intent_from_error(&msg).to_string();
                // Internal execution-enforcement diagnostics are noisy in
                // normal mode; keep them in diagnostics-only mode.
                if !self.diag_enabled && is_diagnostic_message(&msg) {
                    self.status.set_agent_state("streaming");
                    return;
                }
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: format!("Error: {msg}"),
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
                self.status.set_agent_state("error");
                self.status.set_detail(&truncate_status_text(&msg, 80));
            }
            TuiEvent::LoadHistory(entries) => {
                self.chat.clear_messages();
                for entry in entries {
                    if entry.role == "thinking" && !self.show_thinking {
                        continue;
                    }
                    let role = match entry.role.as_str() {
                        "user" => ChatRole::User,
                        "assistant" => ChatRole::Assistant,
                        "thinking" => ChatRole::Thinking,
                        "tool" => ChatRole::Tool,
                        _ => ChatRole::System,
                    };
                    self.chat.add_message(ChatMessage {
                        role,
                        text: entry.text,
                        tool_call_id: None,
                        is_streaming: false,
                        is_error: false,
                    });
                }
            }
            TuiEvent::UpdateModels(models) => {
                self.model_selector.set_models(models);
            }
            TuiEvent::ModelSwitched { model, provider } => {
                self.status.model = model;
                self.status.model_provider = provider;
            }
            TuiEvent::ActionAck { request_id } => {
                self.pending_config_actions.remove(&request_id);
                if self.pending_config_actions.is_empty() {
                    if self
                        .status
                        .detail
                        .starts_with("waiting for model/thinking update")
                    {
                        self.status.set_detail("settings updated");
                    }
                    self.flush_queued_message_if_ready();
                }
            }
            TuiEvent::QueueStatus { .. } => {
                // TUI queue is now independent of the agent's steer/follow-up
                // queue. Messages queued during streaming are processed locally
                // as new prompts via process_next_queued on AgentEnd.
            }
            TuiEvent::ExtensionStatus(payload) => {
                self.status.set_extension_rows(payload.rows);
                self.status
                    .set_extension_row_count(payload.extension_row_count);
            }
            TuiEvent::ContextTokens { tokens, pct } => {
                self.status.context_tokens = tokens;
                self.status.ctx_pct = pct;
            }
            TuiEvent::ThinkingLevels {
                levels,
                current,
                show_selector,
            } => {
                self.status.thinking = current.clone();
                let was_visible = self.thinking_selector.visible;
                let entries = levels
                    .into_iter()
                    .map(|id| {
                        let label = match id.as_str() {
                            "off" => "Disabled".to_string(),
                            "enabled" => "Enabled".to_string(),
                            "minimal" => "Minimal".to_string(),
                            "low" => "Low".to_string(),
                            "medium" => "Medium".to_string(),
                            "high" => "High".to_string(),
                            "xhigh" => "X-High (Max)".to_string(),
                            _ => id.clone(),
                        };
                        crate::components::thinking_selector::ThinkingLevelEntry { id, label }
                    })
                    .collect::<Vec<_>>();
                self.thinking_selector.show(entries, Some(&current));
                // Keep selector visible if it was already open or explicitly requested.
                if !show_selector && !was_visible {
                    self.thinking_selector.hide();
                }
            }
            TuiEvent::ThinkingSet { level } => {
                self.status.thinking = level.clone();
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: format!("Thinking level set to {level}"),
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
            }
            TuiEvent::PlanModeToggled { enabled } => {
                self.plan_mode = enabled;
                self.status.plan_mode = enabled;
            }
            TuiEvent::CavemanModeToggled { level } => {
                self.status.set_caveman_mode(level);
            }
            TuiEvent::SkillActivated { name } => {
                self.chat.add_message(ChatMessage {
                    role: ChatRole::Skill,
                    text: name,
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
            }
            TuiEvent::ModelFavoritesUpdated {
                favorites,
                toggled_model,
                is_favorite,
            } => {
                self.model_selector.set_favorites(favorites);
                let msg = if is_favorite {
                    format!("Added {toggled_model} to favorites")
                } else {
                    format!("Removed {toggled_model} from favorites")
                };
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: msg,
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
            }
            TuiEvent::MimoClusterResults {
                clusters,
                current_url,
            } => {
                if clusters.is_empty() {
                    self.mimo_cluster_selector.close();
                } else {
                    self.mimo_cluster_selector
                        .open(clusters, current_url.as_deref());
                }
            }
            TuiEvent::SettingsApplied {
                steering_mode,
                follow_up_mode,
                show_thinking,
                show_tool_diffs,
                tool_progress_hz,
            } => {
                self.steering_mode = steering_mode;
                self.follow_up_mode = follow_up_mode;
                self.show_thinking = show_thinking;
                self.show_tool_diffs = show_tool_diffs;
                self.tool_progress_hz = tool_progress_hz;
            }
            TuiEvent::ShowEscalationSelector { provider } => {
                self.model_selector.show_filtered_for_provider(&provider);
                self.escalation_picker_active = true;
            }
            TuiEvent::ModelEscalated {
                from: _,
                to,
                is_escalation,
            } => {
                self.status.model = to.clone();
                self.status.model_provider = "deepseek".into();
                let msg = if is_escalation {
                    format!("Escalated to {to}")
                } else {
                    format!("Restored to {to}")
                };
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: msg,
                    tool_call_id: None,
                    is_streaming: false,
                    is_error: false,
                });
            }
        }
    }
}

/// Build the chat display text for a tool call.
/// Shows the tool name plus its key argument (path for file tools, command for bash).
fn tool_display_text(name: &str, args: &Option<String>, cwd: &str) -> String {
    let Some(args_str) = args else {
        return name.to_string();
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(args_str) else {
        return name.to_string();
    };
    match name {
        "bash" => {
            if let Some(cmd) = json.get("command").and_then(|v| v.as_str()) {
                format!("{name}: {}", strip_cd_prefix(cmd, cwd))
            } else {
                name.to_string()
            }
        }
        "write" | "edit" | "read" => {
            if let Some(path) = json.get("path").and_then(|v| v.as_str()) {
                format!("{name}: {path}")
            } else {
                name.to_string()
            }
        }
        _ => name.to_string(),
    }
}

/// Strip a redundant leading `cd <cwd> &&`/`;`/`\n` prefix from a bash command.
/// Only strips when the cd target matches the current working directory.
fn strip_cd_prefix<'a>(cmd: &'a str, cwd: &str) -> &'a str {
    let trimmed = cmd.trim_start();
    if let Some(rest) = trimmed.strip_prefix("cd ") {
        // Check if the path after `cd ` starts with the working directory.
        if rest.starts_with(cwd) {
            // Find the separator after the path: &&, ;, or newline
            let sep_offset = rest
                .find(" &&")
                .or_else(|| rest.find(';'))
                .or_else(|| rest.find('\n'));
            if let Some(idx) = sep_offset {
                let after_sep = &rest[idx..];
                // Skip past the separator (&&, ;, or newline)
                let skip = if after_sep.starts_with(" &&") { 3 } else { 1 };
                let rest_after = &after_sep[skip..];
                let result = rest_after.trim_start();
                if !result.is_empty() {
                    return result;
                }
            }
        }
    }
    cmd
}
fn is_diagnostic_message(msg: &str) -> bool {
    msg.contains("produced no")
        || msg.contains("retrying same turn")
        || msg.contains("validation")
        || msg.contains("tool calls detected")
        || msg.contains("assistant promised execution but emitted no tool calls")
        || msg.contains("Execution gap:")
        || msg.starts_with("LLM error")
}

fn infer_intent_from_error(msg: &str) -> &'static str {
    if msg.contains("inspection") {
        "inspection"
    } else if msg.contains("git turn") || msg.contains("git tool") {
        "git"
    } else if msg.contains("validation") {
        "validation"
    } else if msg.contains("action turn") {
        "action"
    } else {
        "chat"
    }
}

fn truncate_status_text(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}

fn copy_to_clipboard(text: &str) -> anyhow::Result<()> {
    #[cfg(target_os = "macos")]
    {
        use std::io::Write;
        let mut child = std::process::Command::new("pbcopy")
            .stdin(std::process::Stdio::piped())
            .spawn()?;
        if let Some(mut stdin) = child.stdin.take() {
            stdin.write_all(text.as_bytes())?;
        }
        let status = child.wait()?;
        if status.success() {
            return Ok(());
        }
    }

    // OSC52 fallback for terminals that support clipboard escape sequences.
    let b64 = base64_encode(text.as_bytes());
    print!("\x1b]52;c;{}\x07", b64);
    use std::io::Write;
    std::io::stdout().flush()?;
    Ok(())
}

fn base64_encode(bytes: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut i = 0;
    while i + 3 <= bytes.len() {
        let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8) | (bytes[i + 2] as u32);
        out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
        out.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
        out.push(TABLE[(n & 0x3f) as usize] as char);
        i += 3;
    }
    match bytes.len() - i {
        1 => {
            let n = (bytes[i] as u32) << 16;
            out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
            out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
            out.push('=');
            out.push('=');
        }
        2 => {
            let n = ((bytes[i] as u32) << 16) | ((bytes[i + 1] as u32) << 8);
            out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
            out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
            out.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
            out.push('=');
        }
        _ => {}
    }
    out
}
