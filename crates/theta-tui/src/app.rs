//! Application — main TUI event loop and layout management.

use crossterm::event::EventStream;
use futures::StreamExt;
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem},
};
use tokio::sync::mpsc;

use crate::components::CommandEntry;
use crate::components::chat::{Chat, ChatMessage, ChatRole};
use crate::components::editor::Editor;
use crate::components::login_flow::{LoginFlow, known_providers};
use crate::components::model_selector::{ModelEntry, ModelSelector};
use crate::components::session_picker::{SessionInfo, SessionPicker};
use crate::components::status::StatusBar;
use crate::components::{Action, Component};
use crate::keybinding::{Keybinding, default_bindings, resolve_event};
use crate::terminal;
use crate::theme::Theme;

/// Commands sent from the TUI back to the interactive handler.
#[derive(Debug, Clone)]
pub enum TuiAction {
    /// Switch to a different model.
    SwitchModel(String),
    /// Change thinking level.
    SetThinking(String),
    /// Fork the current session.
    ForkSession,
    /// Login result from the login flow.
    LoginResult { provider: String, token: String },
    /// Resume a specific session by ID.
    ResumeSession(String),
    /// Create a new session (dismiss session picker).
    NewSession,
    /// Show the session picker.
    ShowSessions,
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
}

/// A historical message entry for session resume.
#[derive(Debug, Clone)]
pub struct HistoryEntry {
    pub role: String, // "user", "assistant", "tool", "system"
    pub text: String,
}

/// Which view is currently active.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AppMode {
    Chat,
    SessionPicker,
}

/// The main TUI application.
pub struct App {
    chat: Chat,
    editor: Editor,
    status: StatusBar,
    session_picker: Option<SessionPicker>,
    model_selector: ModelSelector,
    keybindings: Vec<Keybinding>,
    focus_idx: usize,
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

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        theme: Theme,
        model: &str,
        session_id: &str,
        thinking: &str,
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
        editor.focus(true); // Editor starts focused.

        Self {
            chat: Chat::new(theme.clone()),
            editor,
            status,
            theme: theme.clone(),
            theme_idx: 0,
            session_picker: None,
            model_selector: ModelSelector::new(models, theme.clone()),
            keybindings: default_bindings(),
            focus_idx: 0,
            running: true,
            mode: AppMode::Chat,
            message_tx,
            action_tx,
            event_rx,
            streaming: false,
            current_tool: None,
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

        let editor_height = self.editor.desired_height(area.width as usize, 8);
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
            self.render_autocomplete(area, frame);
        }
    }

    fn render_autocomplete(&mut self, area: ratatui::layout::Rect, frame: &mut Frame) {
        let items = self.editor.autocomplete_items();
        if items.is_empty() {
            return;
        }
        let selected = self.editor.autocomplete_selected();

        let popup_height = (items.len() as u16 + 2).min(8);
        let popup_width = area.width.min(50);
        // Position above the bottom of the screen (above the 3-line editor area).
        let y = area.height.saturating_sub(popup_height + 3);
        let popup = ratatui::layout::Rect {
            x: area.x,
            y,
            width: popup_width,
            height: popup_height,
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.theme.border));
        let inner = block.inner(popup);
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
                    Style::default()
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
                            let _ = self
                                .action_tx
                                .send(TuiAction::SwitchModel(entry.id.clone()));
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

        if let crossterm::event::Event::Key(key) = event
            && key.code == crossterm::event::KeyCode::Tab
        {
            self.focus_idx = (self.focus_idx + 1) % 2;
            self.editor.focus(self.focus_idx == 0);
            self.chat.focus(self.focus_idx == 1);
            return;
        }

        let action = if self.focus_idx == 0 {
            self.editor.handle_event(event)
        } else {
            self.chat.handle_event(event)
        };

        if let Some(action) = action {
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
            Action::Quit => {
                self.running = false;
            }
            Action::ShowModelSelector => {
                self.model_selector.show();
            }
            Action::CycleTheme => {
                self.cycle_theme();
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
                    "  /model <id>    Switch to a different model",
                    "  /thinking <lvl> Set thinking level (off, low, medium, high)",
                    "  /clear         Clear the chat display",
                    "  /session       Show current session info",
                    "  /fork          Fork the current session",
                    "  /sessions      List recent sessions to resume",
                    "  /help          Show this help",
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
                    self.chat.add_message(ChatMessage {
                        role: ChatRole::System,
                        text: "Usage: /model <model-id>".into(),
                        tool_name: None,
                        is_streaming: false,
                    });
                } else {
                    let _ = self.action_tx.send(TuiAction::SwitchModel(arg.to_string()));
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
            "fork" => {
                let _ = self.action_tx.send(TuiAction::ForkSession);
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: "Forking session...".into(),
                    tool_name: None,
                    is_streaming: false,
                });
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
            }
            TuiEvent::ThinkingDelta(text) => {
                let summary = text.lines().next().unwrap_or("").trim();
                if summary.is_empty() {
                    self.status.set_agent_state("thinking");
                    self.status.set_tool_progress("");
                } else {
                    self.status.set_agent_state("thinking");
                    self.status
                        .set_tool_progress(&truncate_status_text(summary, 80));
                }
            }
            TuiEvent::ToolStart { name, .. } => {
                self.current_tool = Some(name.clone());
                self.status.set_agent_state("tool executing");
                self.status.set_tool_progress(&format!("running {name}..."));
                self.chat.add_message(ChatMessage {
                    role: ChatRole::Tool,
                    text: "running".into(),
                    tool_name: Some(name),
                    is_streaming: true,
                });
            }
            TuiEvent::ToolProgress { name, message } => {
                self.status
                    .set_tool_progress(&truncate_status_text(&message, 80));
                self.chat.update_tool(
                    &name,
                    &format!("\n{}", truncate_status_text(&message, 120)),
                    true,
                );
            }
            TuiEvent::ToolEnd {
                id: _,
                name,
                is_error,
                summary,
            } => {
                if self.current_tool.as_deref() == Some(name.as_str()) {
                    self.current_tool = None;
                }
                if is_error {
                    self.status.set_agent_state("tool error");
                    self.status.set_tool_progress(&format!("{name} failed"));
                    self.chat
                        .update_tool(&name, &format!("\nfailed\n{summary}"), false);
                } else {
                    self.status.set_agent_state("streaming");
                    self.status.set_tool_progress(&format!("{name} done"));
                    let suffix = if summary.is_empty() {
                        "\ndone".to_string()
                    } else {
                        format!("\ndone\n{summary}")
                    };
                    self.chat.update_tool(&name, &suffix, false);
                }
            }
            TuiEvent::TurnStart => {
                self.streaming = true;
                self.status.set_agent_state("streaming");
                self.status.set_tool_progress("");
            }
            TuiEvent::TurnEnd { stop_reason } => {
                self.chat.finish_last(ChatRole::Assistant);
                self.streaming = false;
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
                self.status
                    .set_agent_state(&format!("retrying (attempt {attempt}) in {delay_ms}ms..."));
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
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: msg,
                    tool_name: None,
                    is_streaming: false,
                });
            }
            TuiEvent::Error(msg) => {
                self.chat.add_message(ChatMessage {
                    role: ChatRole::System,
                    text: format!("Error: {msg}"),
                    tool_name: None,
                    is_streaming: false,
                });
                self.status.set_agent_state("error");
            }
            TuiEvent::LoadHistory(entries) => {
                for entry in entries {
                    let role = match entry.role.as_str() {
                        "user" => ChatRole::User,
                        "assistant" => ChatRole::Assistant,
                        "tool" => ChatRole::Tool,
                        _ => ChatRole::System,
                    };
                    self.chat.messages.push(ChatMessage {
                        role,
                        text: entry.text,
                        tool_name: None,
                        is_streaming: false,
                    });
                }
            }
        }
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
