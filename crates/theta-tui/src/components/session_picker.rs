//! Session picker component — shown on TUI startup to choose a session.

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};

use crate::theme::Theme;

/// Info about a session to display in the picker.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: String,
    pub title: String,
    pub model: Option<String>,
    pub branch: Option<String>,
    pub token_count: u32,
    pub created_at: u64,
    pub message_count: usize,
}

/// Session picker component.
pub struct SessionPicker {
    /// Sessions to display.
    sessions: Vec<SessionInfo>,
    /// Currently selected index.
    selected: usize,
    /// List state for rendering.
    list_state: ListState,
    /// Theme colors.
    theme: Theme,
}

impl SessionPicker {
    pub fn new(sessions: Vec<SessionInfo>, theme: Theme) -> Self {
        let mut list_state = ListState::default();
        if !sessions.is_empty() {
            list_state.select(Some(0));
        }
        Self {
            sessions,
            selected: 0,
            list_state,
            theme,
        }
    }

    /// Move selection up.
    pub fn select_up(&mut self) {
        if self.sessions.is_empty() {
            return;
        }
        self.selected = self.selected.saturating_sub(1);
        self.list_state.select(Some(self.selected));
    }

    /// Move selection down.
    pub fn select_down(&mut self) {
        if self.sessions.is_empty() {
            return;
        }
        self.selected = (self.selected + 1).min(self.sessions.len().saturating_sub(1));
        self.list_state.select(Some(self.selected));
    }

    /// Get the currently selected session, if any.
    pub fn selected_session(&self) -> Option<&SessionInfo> {
        self.sessions.get(self.selected)
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    /// Render the session picker.
    pub fn render(&mut self, area: Rect, frame: &mut Frame) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),
                Constraint::Min(0),
                Constraint::Length(3),
            ])
            .split(area);

        // Header.
        let header = Paragraph::new(Text::from(vec![
            Line::from(Span::styled(
                "Recent Sessions",
                Style::default().fg(self.theme.accent),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "j/k to navigate  Enter to resume  N to start new  Esc to start new",
                Style::default().fg(self.theme.dim),
            )),
        ]));
        frame.render_widget(header, chunks[0]);

        // Session list.
        let items: Vec<ListItem> = self
            .sessions
            .iter()
            .map(|s| {
                let title = &s.title;
                let model = s.model.as_deref().unwrap_or("unknown");
                let branch = s.branch.as_deref().unwrap_or("-");
                let when = format_relative_time(s.created_at);
                let count = s.message_count;
                let line = format!(
                    "{model}  |  {branch}  |  {when}  |  {count} msgs  |  ~{} tok  |  {title}",
                    s.token_count
                );
                ListItem::new(Span::raw(line))
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(self.theme.dim))
                    .title("Sessions"),
            )
            .highlight_style(Style::default().fg(self.theme.accent).bg(Color::DarkGray))
            .highlight_symbol("> ");

        frame.render_stateful_widget(list, chunks[1], &mut self.list_state);

        // Footer.
        let count = self.sessions.len();
        let footer_text = if count == 0 {
            "No recent sessions found. Press N or Enter to start a new session.".to_string()
        } else {
            format!("{count} session{} found", if count == 1 { "" } else { "s" })
        };
        let footer = Paragraph::new(Span::styled(
            footer_text,
            Style::default().fg(self.theme.dim),
        ));
        frame.render_widget(footer, chunks[2]);
    }
}

/// Format a unix timestamp as a relative time string.
fn format_relative_time(ts: u64) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let ts_secs = ts / 1000;
    let diff = now.saturating_sub(ts_secs);

    if diff < 60 {
        "just now".into()
    } else if diff < 3600 {
        format!("{}m ago", diff / 60)
    } else if diff < 86400 {
        format!("{}h ago", diff / 3600)
    } else if diff < 604800 {
        format!("{}d ago", diff / 86400)
    } else {
        format!("{}w ago", diff / 604800)
    }
}
