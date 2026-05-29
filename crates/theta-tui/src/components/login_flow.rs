//! Login flow component — interactive auth provider selection and token input.
//!
//! Implements Pi's two-step login flow:
//! 1. Select auth type: Subscription or API Key
//! 2. Select provider, then enter token

use crossterm::event::{Event, KeyCode, KeyModifiers};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph},
};

use crate::components::{Action, Component};
use crate::theme::Theme;

/// Auth type choice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthType {
    Subscription,
    ApiKey,
}

/// A selectable provider entry.
#[derive(Debug, Clone)]
pub struct ProviderEntry {
    pub id: String,
    pub name: String,
    pub auth_type: AuthType,
    pub is_configured: bool,
}

/// Login flow state.
#[derive(Debug, Clone, PartialEq, Eq)]
enum LoginStep {
    /// Choosing auth type: Subscription vs API Key.
    AuthType,
    /// Choosing a provider from the list.
    Provider,
    /// Entering the token/key.
    TokenInput,
}

/// Login flow component — renders selection lists and input prompts.
pub struct LoginFlow {
    theme: Theme,
    step: LoginStep,
    auth_types: Vec<(AuthType, String)>,
    selected_auth_type: usize,
    providers: Vec<ProviderEntry>,
    selected_provider: usize,
    filtered_providers: Vec<usize>,
    token_input: String,
    token_cursor: usize,
    done: bool,
    result: Option<(String, String)>, // (provider_id, token)
    cancelled: bool,
}

impl LoginFlow {
    pub fn new(theme: Theme, providers: Vec<ProviderEntry>) -> Self {
        let auth_types = vec![
            (
                AuthType::Subscription,
                "Subscription (ChatGPT Plus, etc.)".into(),
            ),
            (AuthType::ApiKey, "API Key".into()),
        ];

        let filtered: Vec<usize> = (0..providers.len()).collect();

        Self {
            theme,
            step: LoginStep::AuthType,
            auth_types,
            selected_auth_type: 0,
            providers,
            selected_provider: 0,
            filtered_providers: filtered,
            token_input: String::new(),
            token_cursor: 0,
            done: false,
            result: None,
            cancelled: false,
        }
    }

    pub fn is_done(&self) -> bool {
        self.done
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled
    }

    pub fn take_result(&mut self) -> Option<(String, String)> {
        self.result.take()
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    fn selected_auth_type(&self) -> AuthType {
        self.auth_types[self.selected_auth_type].0
    }

    fn current_filtered(&self) -> Vec<&ProviderEntry> {
        let at = self.selected_auth_type();
        self.filtered_providers
            .iter()
            .filter_map(|&i| self.providers.get(i))
            .filter(|p| p.auth_type == at)
            .collect()
    }

    fn refresh_filter(&mut self) {
        let at = self.selected_auth_type();
        self.filtered_providers = self
            .providers
            .iter()
            .enumerate()
            .filter(|(_, p)| p.auth_type == at)
            .map(|(i, _)| i)
            .collect();
        if self.selected_provider >= self.filtered_providers.len() {
            self.selected_provider = self.filtered_providers.len().saturating_sub(1);
        }
    }

    fn move_selection_up(&mut self) {
        match self.step {
            LoginStep::AuthType => {
                self.selected_auth_type = self.selected_auth_type.saturating_sub(1);
            }
            LoginStep::Provider => {
                self.selected_provider = self.selected_provider.saturating_sub(1);
            }
            _ => {}
        }
    }

    fn move_selection_down(&mut self) {
        match self.step {
            LoginStep::AuthType => {
                let max = self.auth_types.len().saturating_sub(1);
                self.selected_auth_type = (self.selected_auth_type + 1).min(max);
            }
            LoginStep::Provider => {
                let max = self.filtered_providers.len().saturating_sub(1);
                self.selected_provider = (self.selected_provider + 1).min(max);
            }
            _ => {}
        }
    }

    fn confirm(&mut self) {
        match self.step {
            LoginStep::AuthType => {
                self.refresh_filter();
                self.selected_provider = 0;
                self.step = LoginStep::Provider;
            }
            LoginStep::Provider => {
                let filtered = self.current_filtered();
                if let Some(provider) = filtered.get(self.selected_provider) {
                    if provider.auth_type == AuthType::Subscription {
                        // Subscription providers use OAuth — no manual token input.
                        // Use sentinel value "oauth" to signal the App to start OAuth.
                        self.result = Some((provider.id.clone(), "oauth".into()));
                        self.done = true;
                    } else {
                        // API key providers: open browser, then prompt for token.
                        let url = provider_token_url(&provider.id);
                        let _ = open::that(url);
                        self.step = LoginStep::TokenInput;
                    }
                }
            }
            LoginStep::TokenInput => {
                let filtered = self.current_filtered();
                if let Some(provider) = filtered.get(self.selected_provider) {
                    let token = self.token_input.trim().to_string();
                    if !token.is_empty() {
                        self.result = Some((provider.id.clone(), token));
                        self.done = true;
                    }
                }
            }
        }
    }

    fn cancel(&mut self) {
        match self.step {
            LoginStep::AuthType => {
                self.cancelled = true;
                self.done = true;
            }
            LoginStep::Provider => {
                self.selected_provider = 0;
                self.step = LoginStep::AuthType;
            }
            LoginStep::TokenInput => {
                self.token_input.clear();
                self.token_cursor = 0;
                self.step = LoginStep::Provider;
            }
        }
    }

    fn insert_token_char(&mut self, c: char) {
        self.token_input.insert(self.token_cursor, c);
        self.token_cursor += c.len_utf8();
    }

    fn insert_token_str(&mut self, s: &str) {
        if s.is_empty() {
            return;
        }
        self.token_input.insert_str(self.token_cursor, s);
        self.token_cursor += s.len();
    }

    fn token_backspace(&mut self) {
        if self.token_cursor > 0
            && let Some(prev) = self.token_input[..self.token_cursor].chars().last()
        {
            let len = prev.len_utf8();
            self.token_input
                .replace_range(self.token_cursor - len..self.token_cursor, "");
            self.token_cursor -= len;
        }
    }

    fn token_move_left(&mut self) {
        if self.token_cursor > 0
            && let Some(prev) = self.token_input[..self.token_cursor].chars().last()
        {
            self.token_cursor -= prev.len_utf8();
        }
    }

    fn token_move_right(&mut self) {
        if self.token_cursor < self.token_input.len()
            && let Some(next) = self.token_input[self.token_cursor..].chars().next()
        {
            self.token_cursor += next.len_utf8();
        }
    }
}

impl Component for LoginFlow {
    fn render(&mut self, area: Rect, frame: &mut Frame) {
        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.theme.border))
            .title(match self.step {
                LoginStep::AuthType => " Login — choose auth type ",
                LoginStep::Provider => " Login — choose provider ",
                LoginStep::TokenInput => " Login — enter token ",
            })
            .title_style(Style::default().fg(self.theme.accent));

        let inner = block.inner(area);
        frame.render_widget(block, area);

        match self.step {
            LoginStep::AuthType => self.render_auth_types(inner, frame),
            LoginStep::Provider => self.render_providers(inner, frame),
            LoginStep::TokenInput => self.render_token_input(inner, frame),
        }
    }

    fn handle_event(&mut self, event: &Event) -> Option<Action> {
        if self.step == LoginStep::TokenInput
            && let Event::Paste(pasted) = event
        {
            self.insert_token_str(pasted);
            return None;
        }
        let Event::Key(key) = event else {
            return None;
        };

        match key {
            crossterm::event::KeyEvent {
                code: KeyCode::Up, ..
            } => {
                self.move_selection_up();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                self.move_selection_down();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Enter,
                ..
            } => {
                self.confirm();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Esc, ..
            } => {
                self.cancel();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Char('c'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.cancel();
            }
            _ => {
                if self.step == LoginStep::TokenInput {
                    match key {
                        crossterm::event::KeyEvent {
                            code: KeyCode::Char('v'),
                            modifiers,
                            ..
                        } if modifiers.contains(KeyModifiers::CONTROL)
                            || modifiers.contains(KeyModifiers::SUPER) =>
                        {
                            // Paste shortcut; bracketed paste events are handled above.
                        }
                        crossterm::event::KeyEvent {
                            code: KeyCode::Insert,
                            modifiers,
                            ..
                        } if modifiers.contains(KeyModifiers::SHIFT) => {
                            // Shift+Insert paste shortcut; bracketed paste events are handled above.
                        }
                        crossterm::event::KeyEvent {
                            code: KeyCode::Char(c),
                            modifiers: KeyModifiers::NONE | KeyModifiers::SHIFT,
                            ..
                        } => {
                            self.insert_token_char(*c);
                        }
                        crossterm::event::KeyEvent {
                            code: KeyCode::Backspace,
                            ..
                        } => self.token_backspace(),
                        crossterm::event::KeyEvent {
                            code: KeyCode::Left,
                            ..
                        } => self.token_move_left(),
                        crossterm::event::KeyEvent {
                            code: KeyCode::Right,
                            ..
                        } => self.token_move_right(),
                        _ => {}
                    }
                }
            }
        }
        None
    }

    fn is_focused(&self) -> bool {
        true
    }

    fn focus(&mut self, _focused: bool) {}
}

impl LoginFlow {
    fn render_auth_types(&mut self, area: Rect, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(3)])
            .split(area);

        let hint = Paragraph::new(
            "Choose how you want to authenticate (↑↓ to move, Enter to select, Esc to cancel)",
        )
        .style(Style::default().fg(self.theme.dim));
        frame.render_widget(hint, chunks[0]);

        let lines: Vec<Line> = self
            .auth_types
            .iter()
            .enumerate()
            .map(|(i, (_at, label))| {
                let prefix = if i == self.selected_auth_type {
                    "→ "
                } else {
                    "  "
                };
                let style = if i == self.selected_auth_type {
                    Style::default().fg(self.theme.accent)
                } else {
                    Style::default().fg(self.theme.fg)
                };
                Line::from(vec![Span::styled(format!("{prefix}{label}"), style)])
            })
            .collect();

        let para = Paragraph::new(Text::from(lines));
        frame.render_widget(para, chunks[1]);
    }

    fn render_providers(&mut self, area: Rect, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Length(1), Constraint::Min(3)])
            .split(area);

        let at_label = match self.selected_auth_type() {
            AuthType::Subscription => "Subscription providers",
            AuthType::ApiKey => "API Key providers",
        };
        let hint = Paragraph::new(format!(
            "{at_label} (↑↓ to move, Enter to select, Esc to go back)"
        ))
        .style(Style::default().fg(self.theme.dim));
        frame.render_widget(hint, chunks[0]);

        let filtered = self.current_filtered();
        let lines: Vec<Line> = filtered
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let prefix = if i == self.selected_provider {
                    "→ "
                } else {
                    "  "
                };
                let configured = if p.is_configured { " [configured]" } else { "" };
                let style = if i == self.selected_provider {
                    Style::default().fg(self.theme.accent)
                } else {
                    Style::default().fg(self.theme.fg)
                };
                Line::from(vec![Span::styled(
                    format!("{prefix}{}{configured}", p.name),
                    style,
                )])
            })
            .collect();

        if lines.is_empty() {
            let empty = vec![Line::from(Span::styled(
                "No providers available for this auth type.",
                Style::default().fg(self.theme.dim),
            ))];
            let para = Paragraph::new(Text::from(empty));
            frame.render_widget(para, chunks[1]);
        } else {
            let para = Paragraph::new(Text::from(lines));
            frame.render_widget(para, chunks[1]);
        }
    }

    fn render_token_input(&mut self, area: Rect, frame: &mut Frame) {
        let filtered = self.current_filtered();
        let provider = filtered.get(self.selected_provider);
        let provider_name = provider.map(|p| p.name.as_str()).unwrap_or("unknown");
        let provider_id = provider.map(|p| p.id.as_str()).unwrap_or("");
        let is_subscription = provider
            .map(|p| p.auth_type == AuthType::Subscription)
            .unwrap_or(false);

        let instruction_lines = if is_subscription { 4 } else { 2 };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(instruction_lines as u16),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(area);

        let hint = Paragraph::new(format!(
            "Enter token for {provider_name} (Enter to submit, Esc to go back)"
        ))
        .style(Style::default().fg(self.theme.dim));
        frame.render_widget(hint, chunks[0]);

        // Different guidance per auth type.
        if is_subscription {
            let subscription_help = vec![
                Line::from(Span::styled(
                    "Subscription tokens require manual extraction from your browser:",
                    Style::default().fg(self.theme.warning),
                )),
                Line::from(Span::styled(
                    "  chatgpt.com → F12 → Application → Cookies",
                    Style::default().fg(self.theme.dim),
                )),
                Line::from(Span::styled(
                    "  Find: __Secure-next-auth.session-token",
                    Style::default().fg(self.theme.dim),
                )),
                Line::from(Span::styled(
                    "  Copy the token value and paste below",
                    Style::default().fg(self.theme.dim),
                )),
            ];
            let para = Paragraph::new(Text::from(subscription_help));
            frame.render_widget(para, chunks[1]);
        } else {
            let url = provider_token_url(provider_id);
            let api_help = vec![
                Line::from(Span::styled(
                    "Create or copy an API key from your provider dashboard:",
                    Style::default().fg(self.theme.warning),
                )),
                Line::from(Span::styled(url, Style::default().fg(self.theme.dim))),
            ];
            let para = Paragraph::new(Text::from(api_help));
            frame.render_widget(para, chunks[1]);
        }

        let input_block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.theme.accent))
            .title("Token");
        let input_area = input_block.inner(chunks[2]);
        frame.render_widget(input_block, chunks[2]);

        let mut spans = Vec::new();
        for (i, c) in self.token_input.char_indices() {
            let at_cursor = i == self.token_cursor;
            spans.push(Span::styled(
                if c == '\n' || c == '\r' {
                    " ".to_string()
                } else {
                    "•".to_string()
                },
                if at_cursor {
                    Style::default().fg(self.theme.accent).bg(Color::DarkGray)
                } else {
                    Style::default()
                },
            ));
        }
        if self.token_cursor >= self.token_input.len() {
            spans.push(Span::styled(
                " ",
                Style::default().fg(self.theme.accent).bg(Color::DarkGray),
            ));
        }

        let para = Paragraph::new(Line::from(spans));
        frame.render_widget(para, input_area);

        let help = Paragraph::new("Paste with Ctrl+Shift+V or Cmd+V")
            .style(Style::default().fg(self.theme.dim));
        frame.render_widget(help, chunks[3]);
    }
}

/// Get the token/API key page URL for a provider.
fn provider_token_url(provider: &str) -> &str {
    match provider {
        "openai" => "https://platform.openai.com/api-keys",
        "openai-codex" => "https://chatgpt.com",
        "deepseek" => "https://platform.deepseek.com/api_keys",
        "opencode" => "https://api.opencode.ai/settings",
        "xiaomi" => "https://platform.xiaomimimo.com/token-plan",
        _ => "https://google.com",
    }
}

/// Build the list of known providers for the login flow.
pub fn known_providers(
    has_openai_key: bool,
    has_codex_token: bool,
    has_deepseek_key: bool,
    has_opencode_key: bool,
    has_xiaomi_key: bool,
) -> Vec<ProviderEntry> {
    vec![
        ProviderEntry {
            id: "openai-codex".into(),
            name: "OpenAI Codex (ChatGPT Plus)".into(),
            auth_type: AuthType::Subscription,
            is_configured: has_codex_token,
        },
        ProviderEntry {
            id: "openai".into(),
            name: "OpenAI".into(),
            auth_type: AuthType::ApiKey,
            is_configured: has_openai_key,
        },
        ProviderEntry {
            id: "deepseek".into(),
            name: "DeepSeek".into(),
            auth_type: AuthType::ApiKey,
            is_configured: has_deepseek_key,
        },
        ProviderEntry {
            id: "opencode".into(),
            name: "OpenCode".into(),
            auth_type: AuthType::ApiKey,
            is_configured: has_opencode_key,
        },
        ProviderEntry {
            id: "xiaomi".into(),
            name: "Xiaomi MiMo".into(),
            auth_type: AuthType::ApiKey,
            is_configured: has_xiaomi_key,
        },
    ]
}
