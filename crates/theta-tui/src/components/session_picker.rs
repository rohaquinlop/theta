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
    pub sessions: Vec<SessionInfo>,
    /// Currently selected index.
    selected: usize,
    /// List state for rendering.
    list_state: ListState,
    /// Theme colors.
    theme: Theme,
    /// Active sort mode.
    sort_mode: SessionSortMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SessionSortMode {
    Newest,
    Oldest,
    Title,
    Messages,
}

impl SessionPicker {
    pub fn new(sessions: Vec<SessionInfo>, theme: Theme) -> Self {
        let mut sessions = sessions;
        sessions.sort_by_key(|s| std::cmp::Reverse(s.created_at));
        let mut list_state = ListState::default();
        if !sessions.is_empty() {
            list_state.select(Some(0));
        }
        Self {
            sessions,
            selected: 0,
            list_state,
            theme,
            sort_mode: SessionSortMode::Newest,
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

    pub fn cycle_sort_mode(&mut self) {
        let selected_id = self.selected_session().map(|s| s.id.clone());
        self.sort_mode = match self.sort_mode {
            SessionSortMode::Newest => SessionSortMode::Oldest,
            SessionSortMode::Oldest => SessionSortMode::Title,
            SessionSortMode::Title => SessionSortMode::Messages,
            SessionSortMode::Messages => SessionSortMode::Newest,
        };
        self.apply_sort();
        if let Some(id) = selected_id
            && let Some(idx) = self.sessions.iter().position(|s| s.id == id)
        {
            self.selected = idx;
        }
        self.list_state.select(if self.sessions.is_empty() {
            None
        } else {
            Some(self.selected)
        });
    }

    pub fn sort_mode_label(&self) -> &'static str {
        match self.sort_mode {
            SessionSortMode::Newest => "newest",
            SessionSortMode::Oldest => "oldest",
            SessionSortMode::Title => "title",
            SessionSortMode::Messages => "messages",
        }
    }

    fn apply_sort(&mut self) {
        match self.sort_mode {
            SessionSortMode::Newest => {
                self.sessions
                    .sort_by_key(|s| std::cmp::Reverse(s.created_at));
            }
            SessionSortMode::Oldest => {
                self.sessions.sort_by_key(|s| s.created_at);
            }
            SessionSortMode::Title => {
                self.sessions.sort_by(|a, b| {
                    a.title
                        .to_lowercase()
                        .cmp(&b.title.to_lowercase())
                        .then_with(|| b.created_at.cmp(&a.created_at))
                });
            }
            SessionSortMode::Messages => {
                self.sessions.sort_by(|a, b| {
                    b.message_count
                        .cmp(&a.message_count)
                        .then_with(|| b.created_at.cmp(&a.created_at))
                });
            }
        }
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
                "Sessions",
                Style::default().fg(self.theme.accent),
            )),
            Line::from(""),
            Line::from(Span::styled(
                "j/k move  •  Enter resume  •  s sort  •  n new  •  Esc close",
                Style::default().fg(self.theme.dim),
            )),
        ]));
        frame.render_widget(header, chunks[0]);

        // Compute max title and when widths for aligned columns.
        // Truncate long titles to 50 chars so a single outlier doesn't
        // waste horizontal space.
        const TITLE_TRUNCATE: usize = 50;
        let mut truncated_titles: Vec<String> = Vec::with_capacity(self.sessions.len());
        let mut when_strings: Vec<String> = Vec::with_capacity(self.sessions.len());

        for s in &self.sessions {
            let title_chars: Vec<char> = s.title.chars().collect();
            let title = if title_chars.len() > TITLE_TRUNCATE {
                let truncated: String = title_chars[..TITLE_TRUNCATE].iter().collect();
                format!("{}…", truncated)
            } else {
                s.title.clone()
            };
            truncated_titles.push(title);
            when_strings.push(format_relative_time(s.created_at));
        }

        let max_title = truncated_titles
            .iter()
            .map(|t| t.chars().count())
            .max()
            .unwrap_or(0);
        let max_when = when_strings
            .iter()
            .map(|w| w.chars().count())
            .max()
            .unwrap_or(0);

        // Session list.
        let items: Vec<ListItem> = (0..self.sessions.len())
            .map(|i| {
                ListItem::new(Span::raw(session_row_label(
                    &self.sessions[i],
                    &truncated_titles[i],
                    &when_strings[i],
                    max_title,
                    max_when,
                )))
            })
            .collect();

        let list = List::new(items)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(self.theme.dim))
                    .title(format!("sort: {}", self.sort_mode_label())),
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

pub fn session_row_label(
    session: &SessionInfo,
    truncated_title: &str,
    when: &str,
    title_width: usize,
    when_width: usize,
) -> String {
    format!(
        "{:<title_width$}  │  {:<when_width$}  │  {} msgs",
        truncated_title, when, session.message_count
    )
}
