//! Application — main TUI event loop and layout management.

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
use crate::components::tree_selector::{TreeFilter, TreeSelector};
use crate::components::{Action, Component};
use crate::keybinding::{Keybinding, default_bindings, resolve_event};
use crate::terminal;
use crate::theme::Theme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolVerbosity {
    Compact,
    Full,
}

#[derive(Debug, Clone)]
struct ToolRecord {
    id: String,
    name: String,
    summary: String,
    is_error: bool,
}

/// Commands sent from the TUI back to the interactive handler.
#[derive(Debug, Clone)]
pub enum TuiAction {
    /// Switch to a different model.
    SwitchModel {
        model_id: String,
        provider: Option<String>,
    },
    /// Change thinking level.
    SetThinking(String),
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
}

/// Events sent from the agent loop to the TUI.
#[derive(Debug, Clone)]
pub enum TuiEvent {
    TextDelta(String),
    ThinkingDelta(String),
    ToolStart {
        name: String,
        id: String,
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
    AgentEnd,
    ContextCompacted {
        trimmed_count: u32,
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
    /// Informational system message (not an error).
    Info(String),
    Error(String),
    /// Load session history into the chat display.
    LoadHistory(Vec<HistoryEntry>),
    /// Refresh available models (e.g. after login).
    UpdateModels(Vec<ModelEntry>),
    QueueStatus {
        steer: usize,
        follow_up: usize,
    },
    TreeSessions {
        sessions: Vec<SessionInfo>,
        filter: String,
    },
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
    session_picker: Option<SessionPicker>,
    model_selector: ModelSelector,
    tree_selector: TreeSelector,
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
    current_tool: Option<String>,
    tools_in_turn: usize,
    retries_in_turn: u32,
    turn_index: u32,
    turn_intent: String,
    diag_enabled: bool,
    tool_verbosity: ToolVerbosity,
    last_tool_records: Vec<ToolRecord>,
    steer_queue_count: usize,
    follow_up_queue_count: usize,
    show_thinking: bool,
    steering_mode: String,
    follow_up_mode: String,
    /// Active login flow (replaces chat+editor when set).
    login_flow: Option<LoginFlow>,
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

    /// Inject an initial user message (used for startup prompt text).
    pub fn send_initial_message(&mut self, text: String) {
        if text.trim().is_empty() {
            return;
        }
        self.chat.add_message(ChatMessage {
            role: ChatRole::User,
            text: text.clone(),
            tool_name: None,
            is_streaming: false,
        });
        self.status.set_agent_state("streaming");
        self.streaming = true;
        let _ = self.message_tx.send(text);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        theme: Theme,
        model: &str,
        session_id: &str,
        thinking: &str,
        settings: SettingsPayload,
        models: Vec<ModelEntry>,
        commands: Vec<CommandEntry>,
        working_dir: std::path::PathBuf,
        event_rx: mpsc::UnboundedReceiver<TuiEvent>,
        message_tx: mpsc::UnboundedSender<String>,
        action_tx: mpsc::UnboundedSender<TuiAction>,
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
        );
        let skill_commands = commands
            .iter()
            .filter_map(|c| c.name.strip_prefix("skill:").map(str::to_string))
            .collect();
        editor.focus(true); // Editor starts focused.

        Self {
            chat: Chat::new(theme.clone()),
            editor,
            status,
            theme: theme.clone(),
            theme_idx: 0,
            session_picker: None,
            model_selector: ModelSelector::new(models, theme.clone()),
            tree_selector: TreeSelector::new(theme.clone()),
            commands,
            skill_commands,
            keybindings: default_bindings(),
            running: true,
            mode: AppMode::Chat,
            message_tx,
            action_tx,
            event_rx,
            streaming: false,
            current_tool: None,
            tools_in_turn: 0,
            retries_in_turn: 0,
            turn_index: 0,
            turn_intent: "chat".to_string(),
            diag_enabled: false,
            tool_verbosity: ToolVerbosity::Compact,
            last_tool_records: Vec::new(),
            steer_queue_count: 0,
            follow_up_queue_count: 0,
            show_thinking: settings.show_thinking,
            steering_mode: settings.steering_mode,
            follow_up_mode: settings.follow_up_mode,
            login_flow: None,
        }
    }

    /// Run the TUI event loop.
    pub async fn run(&mut self) -> anyhow::Result<()> {
        terminal::setup()?;
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

        while self.running {
            term.draw(|frame| self.draw(frame))?;

            tokio::select! {
                crossterm_event = reader.next() => {
                    if let Some(Ok(event)) = crossterm_event {
                        self.handle_input_event(&event);
                    }
                }
                Some(event) = self.event_rx.recv() => {
                    self.handle_agent_event(event);
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

        // Session picker mode.
        if self.mode == AppMode::SessionPicker
            && let Some(ref mut picker) = self.session_picker
        {
            let main = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(1), Constraint::Min(8)])
                .split(area);
            self.status.render(main[0], frame);
            picker.render(main[1], frame);
            return;
        }

        if let Some(ref mut login) = self.login_flow {
            let main = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(1), Constraint::Min(8)])
                .split(area);
            self.status.render(main[0], frame);
            login.render(main[1], frame);
            return;
        }

        let editor_height = self.editor.desired_height(area.width as usize, 6);
        let main = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Min(3),
                Constraint::Length(editor_height),
            ])
            .split(area);

        self.status.render(main[0], frame);
        self.chat.render(main[1], frame);
        self.editor.render(main[2], frame);

        // Render autocomplete popup on top (after editor so it overlays).
        if self.editor.autocomplete_active() {
            self.render_autocomplete(main[2], frame);
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
        // Model selector mode — handle keys exclusively.
        if self.model_selector.visible {
            if let crossterm::event::Event::Key(key) = event {
                match key.code {
                    crossterm::event::KeyCode::Esc => {
                        self.model_selector.hide();
                    }
                    crossterm::event::KeyCode::Enter => {
                        if let Some(entry) = self.model_selector.selected_model() {
                            let _ = self.action_tx.send(TuiAction::SwitchModel {
                                model_id: entry.id.clone(),
                                provider: Some(entry.provider.clone()),
                            });
                            self.chat.add_message(ChatMessage {
                                role: ChatRole::System,
                                text: format!("Switching model to {}...", entry.id),
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
                    crossterm::event::KeyCode::Char(c) => {
                        self.model_selector.push_query(c);
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
                    } else {
                        let _ = self.action_tx.send(TuiAction::Steer(text.clone()));
                        self.steer_queue_count += 1;
                        self.chat.add_message(ChatMessage {
                            role: ChatRole::User,
                            text: format!("[steer] {text}"),
                            tool_name: None,
                            is_streaming: false,
                        });
                    }
                } else {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::User,
                        text: text.clone(),
                        tool_name: None,
                        is_streaming: false,
                    });
                    self.status.set_agent_state("streaming");
                    self.streaming = true;
                    let _ = self.message_tx.send(text);
                }
            }
            Action::Quit => {
                self.running = false;
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
                    self.status.set_tool_progress("copied selection");
                } else {
                    self.status.set_tool_progress("copy failed");
                }
            }
            Action::OpenUrl(url) => {
                if open::that(&url).is_ok() {
                    self.status.set_tool_progress("opened link");
                } else {
                    self.status.set_tool_progress("open link failed");
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
                    "  /thinking <lvl> Set thinking level (off, low, medium, high)",
                    "  /clear          Clear the chat display",
                    "  /session        Show current session info",
                    "  /fork           Fork the current session",
                    "  /sessions       List recent sessions (in picker press s to sort)",
                    "  /tree [filter]  Open branch tree (default|no-tools|user-only|labeled-only|all)",
                    "  /skills         List available skills",
                    "  /model <id>     Switch model directly by id",
                    "  /diag on|off    Toggle diagnostic event stream in chat",
                    "  /tools compact|full  Toggle compact/full tool output",
                    "  /expand <id|last-tool> Show full tool summary",
                    "  /skill:<name>   Invoke a skill",
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
                    let _ = self.action_tx.send(TuiAction::SwitchModel {
                        model_id: arg.to_string(),
                        provider: None,
                    });
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: format!("Switching model to {arg}..."),
                        tool_name: None,
                        is_streaming: false,
                    });
                }
            }
            "thinking" => {
                if arg.is_empty() {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Usage: /thinking <off|low|medium|high>".into(),
                        tool_name: None,
                        is_streaming: false,
                    });
                } else {
                    let _ = self.action_tx.send(TuiAction::SetThinking(arg.to_string()));
                    self.status.thinking = arg.to_string();
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: format!("Thinking level set to {arg}"),
                        tool_name: None,
                        is_streaming: false,
                    });
                }
            }
            "clear" => {
                self.chat.messages.clear();
            }
            "session" | "s" => {
                let info = format!(
                    "Session: {}\nModel: {}\nThinking: {}",
                    self.status.session_id, self.status.model, self.status.thinking
                );
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: info,
                    tool_name: None,
                    is_streaming: false,
                });
            }
            "login" => {
                // Start the login flow.
                let providers = known_providers(false, false, false, false);
                self.login_flow = Some(LoginFlow::new(self.theme.clone(), providers));
            }
            "sessions" => {
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
            "tools" => match arg {
                "compact" => {
                    self.tool_verbosity = ToolVerbosity::Compact;
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Tool output set to compact.".into(),
                        tool_name: None,
                        is_streaming: false,
                    });
                }
                "full" => {
                    self.tool_verbosity = ToolVerbosity::Full;
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Tool output set to full.".into(),
                        tool_name: None,
                        is_streaming: false,
                    });
                }
                _ => {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Usage: /tools <compact|full>".into(),
                        tool_name: None,
                        is_streaming: false,
                    });
                }
            },
            "expand" => {
                let target = arg.trim();
                if target.is_empty() {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Usage: /expand <id|last-tool>".into(),
                        tool_name: None,
                        is_streaming: false,
                    });
                    return;
                }
                let rec = if target == "last-tool" {
                    self.last_tool_records.last()
                } else {
                    self.last_tool_records.iter().find(|r| r.id == target)
                };
                if let Some(r) = rec {
                    let header = if r.is_error {
                        format!("Tool {} ({}) failed", r.name, r.id)
                    } else {
                        format!("Tool {} ({})", r.name, r.id)
                    };
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: format!("{header}\n{}", r.summary),
                        tool_name: None,
                        is_streaming: false,
                    });
                } else {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: format!("No tool summary found for `{target}`."),
                        tool_name: None,
                        is_streaming: false,
                    });
                }
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
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::User,
                        text: full.clone(),
                        tool_name: None,
                        is_streaming: false,
                    });
                    self.status.set_agent_state("streaming");
                    self.streaming = true;
                    let _ = self.message_tx.send(full);
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
                    self.status.set_agent_state("streaming (responding)");
                }
            }
            TuiEvent::ThinkingDelta(text) => {
                let summary = text.lines().next().unwrap_or("").trim();
                self.status.set_agent_state("thinking");
                if self.show_thinking {
                    self.chat.update_last(&text, ChatRole::Thinking, true);
                }
                if summary.is_empty() {
                    self.status.set_tool_progress("");
                } else {
                    self.status
                        .set_tool_progress(&truncate_status_text(summary, 80));
                }
            }
            TuiEvent::ToolStart { name, .. } => {
                self.current_tool = Some(name.clone());
                self.status.set_agent_state(&format!("tool: {name}"));
                self.status.set_tool_progress("running...");
                self.tools_in_turn += 1;
                if self.tool_verbosity == ToolVerbosity::Full {
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::Tool,
                        text: "running".into(),
                        tool_name: Some(name),
                        is_streaming: true,
                    });
                }
            }
            TuiEvent::ToolProgress { name, message } => {
                self.status
                    .set_tool_progress(&truncate_status_text(&message, 80));
                if self.tool_verbosity == ToolVerbosity::Full {
                    self.chat.update_tool(
                        &name,
                        &format!("\n{}", truncate_status_text(&message, 120)),
                        true,
                    );
                }
            }
            TuiEvent::ToolEnd {
                id,
                name,
                is_error,
                summary,
            } => {
                if self.current_tool.as_deref() == Some(name.as_str()) {
                    self.current_tool = None;
                }
                self.last_tool_records.push(ToolRecord {
                    id: id.clone(),
                    name: name.clone(),
                    summary: summary.clone(),
                    is_error,
                });
                if self.last_tool_records.len() > 100 {
                    let drop_n = self.last_tool_records.len() - 100;
                    self.last_tool_records.drain(0..drop_n);
                }
                if is_error {
                    self.status.set_agent_state(&format!("tool error: {name}"));
                    self.status.set_tool_progress("failed");
                    if self.tool_verbosity == ToolVerbosity::Full {
                        self.chat
                            .update_tool(&name, &format!("\nfailed\n{summary}"), false);
                    } else {
                        self.chat.add_message(ChatMessage {
                            role: ChatRole::Tool,
                            text: format!("[tool:{name}] failed: {}", compact_summary(&summary)),
                            tool_name: Some(name),
                            is_streaming: false,
                        });
                    }
                } else {
                    self.status.set_agent_state("streaming (post-tool)");
                    self.status.set_tool_progress(&format!("{name} done"));
                    if self.tool_verbosity == ToolVerbosity::Full {
                        let suffix = if summary.is_empty() {
                            "\ndone".to_string()
                        } else {
                            format!("\ndone\n{summary}")
                        };
                        self.chat.update_tool(&name, &suffix, false);
                    } else {
                        self.chat.add_message(ChatMessage {
                            role: ChatRole::Tool,
                            text: format!("[tool:{name}] {}", compact_summary(&summary)),
                            tool_name: Some(name),
                            is_streaming: false,
                        });
                    }
                }
            }
            TuiEvent::TurnStart => {
                self.streaming = true;
                self.turn_index += 1;
                self.tools_in_turn = 0;
                self.retries_in_turn = 0;
                self.turn_intent = "chat".to_string();
                self.status.set_turn_index(self.turn_index);
                self.status.set_agent_state("streaming (starting)");
                self.status.set_tool_progress("");
            }
            TuiEvent::TurnEnd { stop_reason } => {
                self.chat.finish_last(ChatRole::Assistant);
                self.chat.finish_last(ChatRole::Thinking);
                self.streaming = false;
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
                self.status
                    .set_agent_state(&format!("idle (stopped: {stop_reason})"));
                self.status.set_tool_progress("");
            }
            TuiEvent::ContextCompacted { trimmed_count } => {
                self.status.set_agent_state("compacting");
                if trimmed_count == 1 {
                    self.status
                        .set_tool_progress(&format!("trimmed {trimmed_count} old message"));
                } else {
                    self.status
                        .set_tool_progress(&format!("trimmed {trimmed_count} old messages"));
                }
            }
            TuiEvent::Retrying { attempt, delay_ms } => {
                self.retries_in_turn = self.retries_in_turn.max(attempt);
                self.status
                    .set_agent_state(&format!("retrying (attempt {attempt}) in {delay_ms}ms..."));
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: format!(
                        "Retrying {} ({}/1): {}",
                        self.turn_intent,
                        attempt,
                        retry_reason_hint(&self.turn_intent)
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
            }
            TuiEvent::AgentEnd => {
                self.chat.finish_last(ChatRole::Assistant);
                self.chat.finish_last(ChatRole::Thinking);
                for msg in &mut self.chat.messages {
                    if msg.role == ChatRole::Tool && msg.is_streaming {
                        msg.is_streaming = false;
                    }
                }
                self.streaming = false;
                self.current_tool = None;
                self.status.set_tool_progress("");
                self.status.set_agent_state("idle");
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
            }
            TuiEvent::LoadHistory(entries) => {
                self.chat.messages.clear();
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
            TuiEvent::QueueStatus { steer, follow_up } => {
                self.steer_queue_count = steer;
                self.follow_up_queue_count = follow_up;
                self.status
                    .set_tool_progress(&format!("queue steer:{steer} follow-up:{follow_up}"));
            }
            TuiEvent::TreeSessions { sessions, filter } => {
                self.tree_selector
                    .set_sessions(sessions, TreeFilter::parse(&filter));
                self.mode = AppMode::TreePicker;
            }
        }
    }
}

fn compact_summary(summary: &str) -> String {
    let first = summary.lines().find(|l| !l.trim().is_empty()).unwrap_or("");
    let mut s = first.trim().to_string();
    while s.starts_with("[tool:") {
        let Some(end) = s.find(']') else {
            break;
        };
        s = s[end + 1..].trim_start().to_string();
    }
    truncate_status_text(&s, 100)
}

fn is_diagnostic_message(msg: &str) -> bool {
    msg.contains("produced no")
        || msg.contains("retrying same turn")
        || msg.contains("validation")
        || msg.contains("tool calls detected")
        || msg.contains("assistant promised execution but emitted no tool calls")
        || msg.contains("Execution gap:")
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

fn retry_reason_hint(intent: &str) -> &'static str {
    match intent {
        "inspection" => "no inspection tool call",
        "git" => "no git tool call",
        "validation" => "validation command not run",
        "action" => "no action tool call",
        _ => "missing tool call",
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
