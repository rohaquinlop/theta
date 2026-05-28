//! Application — main TUI event loop and layout management.

use std::collections::{HashMap, HashSet, VecDeque};

use crossterm::event::EventStream;
use futures::StreamExt;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Clear, List, ListItem},
};
use tokio::sync::mpsc;

use crate::components::CommandEntry;
use crate::components::chat::{Chat, ChatMessage, ChatRole};
use crate::components::editor::Editor;
use crate::components::login_flow::{LoginFlow, known_providers};
use crate::components::model_selector::{ModelEntry, ModelSelector};
use crate::components::session_picker::{SessionInfo, SessionPicker};
use crate::components::status::StatusBar;
use crate::components::thinking_selector::ThinkingSelector;
use crate::components::tree_selector::{TreeFilter, TreeSelector};
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
    ShowTree(String),
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
    },
    /// A config action (model/thinking) has been applied or rejected.
    ActionAck {
        request_id: u64,
    },
    QueueStatus {
        steer: usize,
        follow_up: usize,
    },
    TreeSessions {
        sessions: Vec<SessionInfo>,
        filter: String,
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
    pub tool_progress_hz: u64,
    pub enter_behavior: String,
    pub max_context_window: Option<u32>,
}

/// Which view is currently active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppMode {
    Chat,
    SessionPicker,
    TreePicker,
}

/// The main TUI application.
pub struct App {
    chat: Chat,
    editor: Editor,
    status: StatusBar,
    working_dir: std::path::PathBuf,
    session_picker: Option<SessionPicker>,
    model_selector: ModelSelector,
    tree_selector: TreeSelector,
    thinking_selector: ThinkingSelector,
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
    #[allow(dead_code)]
    theme: Theme,
    theme_idx: usize,
    streaming: bool,
    /// Set to true when MessageEnd fires — transition from LLM streaming
    /// to tool execution phase. ToolStart events after this flag show a visual cue.
    tool_exec_phase: bool,
    current_tool: Option<String>,
    tool_display_text: HashMap<String, String>,
    tool_started_at: HashMap<String, std::time::Instant>,
    tool_last_tick_sec: HashMap<String, u64>,
    tools_in_turn: usize,
    retries_in_turn: u32,
    turn_index: u32,
    turn_intent: String,
    diag_enabled: bool,
    steer_queue_count: usize,
    follow_up_queue_count: usize,
    show_thinking: bool,
    steering_mode: String,
    follow_up_mode: String,
    tool_progress_hz: u64,
    /// Active login flow (replaces chat+editor when set).
    login_flow: Option<LoginFlow>,
    window_title: Option<String>,
    pending_config_actions: HashSet<u64>,
    next_config_request_id: u64,
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
        let names: Vec<String> = self.tool_started_at.keys().cloned().collect();
        for name in names {
            let Some(start) = self.tool_started_at.get(&name) else {
                continue;
            };
            let elapsed = start.elapsed().as_secs();
            if elapsed < 1 {
                continue;
            }
            let last = self
                .tool_last_tick_sec
                .get(&name)
                .copied()
                .unwrap_or(u64::MAX);
            if elapsed == last {
                continue;
            }
            let display = self
                .tool_display_text
                .get(&name)
                .cloned()
                .unwrap_or_default();
            const SPINNER: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];
            let spinner = SPINNER[(elapsed as usize) % 10];
            let text = if display.is_empty() {
                format!("{name} {spinner} {elapsed}s")
            } else {
                format!("{display} {spinner} {elapsed}s")
            };
            self.chat.upsert_tool_message(&name, &text, true);
            self.tool_last_tick_sec.insert(name.clone(), elapsed);
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
    ) -> Self {
        let mut status = StatusBar::new(theme.clone());
        status.model = model.to_string();
        status.session_id = session_id.to_string();
        status.thinking = thinking.to_string();
        status.set_agent_state("idle");

        let mut editor = Editor::new(
            theme.clone(),
            working_dir.clone(),
            commands.iter().map(|c| c.name.clone()).collect(),
            settings.enter_behavior,
        );
        let skill_commands = commands
            .iter()
            .filter_map(|c| c.name.strip_prefix("skill:").map(str::to_string))
            .collect();
        editor.focus(true); // Editor starts focused.

        Self {
            working_dir,
            chat: Chat::new(theme.clone()),
            editor,
            status,
            theme: theme.clone(),
            theme_idx: 0,
            session_picker: None,
            model_selector: ModelSelector::new(models, favorites, theme.clone()),
            tree_selector: TreeSelector::new(theme.clone()),
            thinking_selector: ThinkingSelector::new(theme.clone()),
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
            tools_in_turn: 0,
            retries_in_turn: 0,
            turn_index: 0,
            turn_intent: "chat".to_string(),
            diag_enabled: false,
            steer_queue_count: 0,
            follow_up_queue_count: 0,
            show_thinking: settings.show_thinking,
            steering_mode: settings.steering_mode,
            follow_up_mode: settings.follow_up_mode,
            tool_progress_hz: settings.tool_progress_hz.max(1),
            login_flow: None,
            window_title,
            pending_config_actions: HashSet::new(),
            next_config_request_id: 1,
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
                    self.update_tool_elapsed();
                    let _ = term.draw(|frame| self.draw(frame));
                }
            }
        }
        Ok(())
    }

    fn draw(&mut self, frame: &mut Frame) {
        let area = frame.area();

        // Model selector overlay — renders on top of everything.
        self.model_selector.render(area, frame);
        if self.model_selector.visible {
            return;
        }
        self.tree_selector.render(area, frame);
        if self.tree_selector.visible {
            return;
        }
        self.thinking_selector.render(area, frame);
        if self.thinking_selector.visible {
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
        let main = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(3),
                Constraint::Length(editor_height),
                Constraint::Length(status_height),
            ])
            .split(area);

        self.chat.render(main[0], frame);
        self.editor.render(main[1], frame);
        self.status.render(main[2], frame);

        // Render autocomplete popup on top (after editor so it overlays).
        if self.editor.autocomplete_active() {
            self.render_autocomplete(main[1], frame);
        }
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
            .style(Style::default().bg(Color::Black))
            .border_style(Style::default().fg(self.theme.border));
        let inner = block.inner(popup);
        frame.render_widget(Clear, popup);
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
                    Style::default().fg(self.theme.accent).bg(Color::DarkGray)
                } else {
                    Style::default().bg(Color::Black)
                };
                ListItem::new(Line::from(Span::styled(text, style)))
            })
            .collect();

        let list = List::new(list_items);
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
                        self.model_selector.hide();
                    }
                    crossterm::event::KeyCode::Enter => {
                        if let Some(entry) = self.model_selector.selected_model() {
                            let model_id = entry.id.clone();
                            let provider = entry.provider.clone();
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
                                tool_name: None,
                                is_streaming: false,
                            });
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
                                tool_name: None,
                                is_streaming: false,
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

        if self.mode == AppMode::TreePicker {
            if let crossterm::event::Event::Key(key) = event {
                match key.code {
                    crossterm::event::KeyCode::Char('j') | crossterm::event::KeyCode::Down => {
                        self.tree_selector.select_down()
                    }
                    crossterm::event::KeyCode::Char('k') | crossterm::event::KeyCode::Up => {
                        self.tree_selector.select_up()
                    }
                    crossterm::event::KeyCode::Enter => {
                        if let Some(s) = self.tree_selector.selected() {
                            self.chat.clear_messages();
                            let _ = self.action_tx.send(TuiAction::ResumeSession(s.id.clone()));
                        }
                        self.tree_selector.visible = false;
                        self.mode = AppMode::Chat;
                    }
                    crossterm::event::KeyCode::Esc => {
                        self.tree_selector.visible = false;
                        self.mode = AppMode::Chat;
                    }
                    _ => {}
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
                            tool_name: None,
                            is_streaming: false,
                        });
                    }
                }
                self.login_flow = None;
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
            tool_name: None,
            is_streaming: false,
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

    fn send_user_message_now(&mut self, text: String) {
        self.quit_confirmation = false;
        self.quit_confirm_at = None;
        self.chat.add_message(ChatMessage {
            role: ChatRole::User,
            text: text.clone(),
            tool_name: None,
            is_streaming: false,
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
                    if self.steering_mode == "follow-up" {
                        let _ = self.action_tx.send(TuiAction::FollowUp(text.clone()));
                        self.follow_up_queue_count += 1;
                        self.chat.add_message(ChatMessage {
                            role: ChatRole::User,
                            text: format!("[queued follow-up] {text}"),
                            tool_name: None,
                            is_streaming: false,
                        });
                        self.status.set_detail("queued follow-up");
                    } else {
                        let _ = self.action_tx.send(TuiAction::Steer(text.clone()));
                        self.steer_queue_count += 1;
                        self.chat.add_message(ChatMessage {
                            role: ChatRole::User,
                            text: format!("[steer] {text}"),
                            tool_name: None,
                            is_streaming: false,
                        });
                        self.status.set_detail("queued steer message");
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
                    self.status.set_detail("cancelling...");
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Cancelling current agent execution...".into(),
                        tool_name: None,
                        is_streaming: false,
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
                if self.follow_up_mode == "steer" {
                    let _ = self.action_tx.send(TuiAction::Steer(text));
                    self.steer_queue_count += 1;
                } else {
                    let _ = self.action_tx.send(TuiAction::FollowUp(text));
                    self.follow_up_queue_count += 1;
                }
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
        let names = Theme::names();
        self.theme_idx = (self.theme_idx + 1) % names.len();
        let name = names[self.theme_idx];
        let theme = Theme::named(name);
        self.theme = theme.clone();
        self.chat.set_theme(theme.clone());
        self.editor.set_theme(theme.clone());
        self.status.set_theme(theme.clone());
        self.model_selector.set_theme(theme.clone());
        if let Some(ref mut picker) = self.session_picker {
            picker.set_theme(theme.clone());
        }
        if let Some(ref mut login) = self.login_flow {
            login.set_theme(theme);
        }
        self.chat.add_message(ChatMessage {
            role: ChatRole::System,
            text: format!("Theme: {name}"),
            tool_name: None,
            is_streaming: false,
        });
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
                    "  /thinking [lvl] Open selector or set thinking level (off/minimal/low/medium/high/xhigh)",
                    "  /effort [lvl]   Alias for /thinking",
                    "  /clear          Clear the chat display",
                    "  /session        Show session info (tokens, context window, compaction)",
                    "  /status         Show live runtime status snapshot",
                    "  /timeline       Show compact timeline from latest run report",
                    "  /compact        Manually compact context to fit in context window",
                    "  /fork           Fork the current session",
                    "  /new            Start a new unsaved session",
                    "  /sessions       List recent sessions (in picker press s to sort)",
                    "  /resume         Alias for /sessions",
                    "  /tree [filter]  Open branch tree (default|no-tools|user-only|labeled-only|all)",
                    "  /skills         List available skills",
                    "  /model <id>     Switch model directly by id",
                    "  /diag on|off    Toggle diagnostic event stream in chat",
                    "  /tools-rate <hz> Set tool progress update rate (1-60)",
                    "  /skill:<name>   Invoke a skill",
                    "  /cancel         Cancel current agent execution",
                    "  /exit           Exit Theta",
                    "  /help           Show this help",
                ]
                .join("\n");
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: help_text,
                    tool_name: None,
                    is_streaming: false,
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
                        tool_name: None,
                        is_streaming: false,
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
                            tool_name: None,
                            is_streaming: false,
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
                        tool_name: None,
                        is_streaming: false,
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
                let snapshot = format!(
                    "Runtime status:\nState: {}\nDetail: {}\nTurn: {}\nStreaming: {}\nCurrent tool: {}\nTools in turn: {}\nRetries in turn: {}\nSteer queue: {}\nFollow-up queue: {}\nLast turn decision: {}\nLast end reason: {}",
                    self.status.agent_state,
                    detail,
                    self.turn_index,
                    self.streaming,
                    current_tool,
                    self.tools_in_turn,
                    self.retries_in_turn,
                    self.steer_queue_count,
                    self.follow_up_queue_count,
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
                    tool_name: None,
                    is_streaming: false,
                });
            }
            "timeline" => {
                let _ = self.action_tx.send(TuiAction::ShowRunTimeline);
            }
            "compact" | "comp" => {
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: "Compacting context...".into(),
                    tool_name: None,
                    is_streaming: false,
                });
                let _ = self.action_tx.send(TuiAction::CompactContext);
            }
            "login" => {
                // Start the login flow.
                let providers = known_providers(false, false, false, false);
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
            "tree" => {
                let filter = if arg.is_empty() { "default" } else { arg };
                let _ = self.action_tx.send(TuiAction::ShowTree(filter.to_string()));
            }
            "skills" => {
                if self.skill_commands.is_empty() {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "No skills found in ~/.agents/skills, ~/.theta/skills, ./.agents/skills, or ./.theta/skills".into(),
                        tool_name: None,
                        is_streaming: false,
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
                        tool_name: None,
                        is_streaming: false,
                    });
                }
            }
            "diag" => match arg {
                "on" => {
                    self.diag_enabled = true;
                    self.status.set_show_diagnostics(true);
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Diagnostics stream enabled.".into(),
                        tool_name: None,
                        is_streaming: false,
                    });
                }
                "off" => {
                    self.diag_enabled = false;
                    self.status.set_show_diagnostics(false);
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Diagnostics stream hidden (critical failures still shown).".into(),
                        tool_name: None,
                        is_streaming: false,
                    });
                }
                _ => {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Usage: /diag <on|off>".into(),
                        tool_name: None,
                        is_streaming: false,
                    });
                }
            },
            "tools-rate" => {
                let parsed = arg.trim().parse::<u64>().ok();
                let Some(hz) = parsed else {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Usage: /tools-rate <1-60>".into(),
                        tool_name: None,
                        is_streaming: false,
                    });
                    return;
                };
                if !(1..=60).contains(&hz) {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Usage: /tools-rate <1-60>".into(),
                        tool_name: None,
                        is_streaming: false,
                    });
                    return;
                }
                self.tool_progress_hz = hz;
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: format!("Tool progress rate set to {hz} Hz."),
                    tool_name: None,
                    is_streaming: false,
                });
            }
            "fork" => {
                let _ = self.action_tx.send(TuiAction::ForkSession);
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: "Forking session...".into(),
                    tool_name: None,
                    is_streaming: false,
                });
            }
            "exit" => {
                self.running = false;
            }
            "cancel" => {
                if self.streaming {
                    let _ = self.action_tx.send(TuiAction::AbortAgent);
                    self.status.set_detail("cancelling...");
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Cancelling current agent execution...".into(),
                        tool_name: None,
                        is_streaming: false,
                    });
                } else {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "No agent execution to cancel.".into(),
                        tool_name: None,
                        is_streaming: false,
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
                        tool_name: None,
                        is_streaming: false,
                    });
                }
            }
            _ => {
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: format!(
                        "Unknown command: /{command}. Type /help for available commands."
                    ),
                    tool_name: None,
                    is_streaming: false,
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
                // Don't set detail — [stream] badge already shows the mode.
            }
            TuiEvent::ThinkingDelta(text) => {
                self.status.set_agent_state("thinking");
                if self.show_thinking {
                    self.chat.update_last(&text, ChatRole::Thinking, true);
                }
                // Don't set detail — thinking level is already shown in the
                // [model:thinking] badge on the left.
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
            TuiEvent::ToolCallPrepared { name, .. } => {
                // Show a visual cue that a tool call is being prepared
                // during LLM streaming (before execution). This creates a
                // tool message immediately so the user sees the transition
                // from text streaming to tool preparation.
                self.chat
                    .upsert_tool_message(&name, &format!("{name} preparing..."), true);
            }
            TuiEvent::ToolStart { name, args, .. } => {
                self.current_tool = Some(name.clone());
                self.status.set_agent_state("ToolExec");
                self.tools_in_turn += 1;
                self.tool_exec_phase = true;
                // Extract command from args for display.
                let display_text =
                    tool_display_text(&name, &args, &self.working_dir.to_string_lossy());
                self.tool_display_text
                    .insert(name.clone(), display_text.clone());
                self.tool_started_at
                    .insert(name.clone(), std::time::Instant::now());
                self.chat.upsert_tool_message(&name, &display_text, true);
            }
            TuiEvent::ToolProgress {
                name: _,
                message: _,
            } => {
                // Rate-limited progress — do not show tool output in chat.
            }
            TuiEvent::ToolEnd { name, is_error, .. } => {
                if self.current_tool.as_deref() == Some(name.as_str()) {
                    self.current_tool = None;
                }
                let display_text = self.tool_display_text.remove(&name).unwrap_or_default();
                self.tool_started_at.remove(&name);
                self.tool_last_tick_sec.remove(&name);
                // Build status suffix for the command line.
                let status = if is_error { " (failed)" } else { " (done)" };
                let final_text = if display_text.is_empty() {
                    format!("{name}{status}")
                } else {
                    format!("{display_text}{status}")
                };
                self.chat.complete_tool_compact(&name, &final_text);
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
                        tool_name: None,
                        is_streaming: false,
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
            TuiEvent::Retrying { attempt, delay_ms } => {
                self.retries_in_turn = self.retries_in_turn.max(attempt);
                self.status.set_agent_state("Retrying");
                self.status
                    .set_detail(&format!("provider retry {attempt} in {delay_ms}ms"));
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: format!(
                        "Retrying provider request (attempt {attempt}, waiting {delay_ms}ms)"
                    ),
                    tool_name: None,
                    is_streaming: false,
                });
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
                if aborted {
                    self.status.set_agent_state("Cancelled");
                    self.status.set_detail("execution cancelled");
                } else if self.status.agent_state != "Completed"
                    && self.status.agent_state != "Blocked"
                    && self.status.agent_state != "Failed"
                {
                    self.status.set_agent_state("Completed");
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
                        tool_name: None,
                        is_streaming: false,
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
                    tool_name: None,
                    is_streaming: false,
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
                    tool_name: None,
                    is_streaming: false,
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
                        tool_name: None,
                        is_streaming: false,
                    });
                }
            }
            TuiEvent::UpdateModels(models) => {
                self.model_selector.set_models(models);
            }
            TuiEvent::ModelSwitched { model } => {
                self.status.model = model;
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
            TuiEvent::QueueStatus { steer, follow_up } => {
                self.steer_queue_count = steer;
                self.follow_up_queue_count = follow_up;
            }
            TuiEvent::TreeSessions { sessions, filter } => {
                self.tree_selector
                    .set_sessions(sessions, TreeFilter::parse(&filter));
                self.mode = AppMode::TreePicker;
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
                    tool_name: None,
                    is_streaming: false,
                });
            }
            TuiEvent::SkillActivated { name } => {
                self.chat.add_message(ChatMessage {
                    role: ChatRole::Skill,
                    text: name,
                    tool_name: None,
                    is_streaming: false,
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
                    tool_name: None,
                    is_streaming: false,
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
