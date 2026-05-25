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

fn session_row_label(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn mk(id: &str, title: &str, created_at: u64, messages: usize) -> SessionInfo {
        SessionInfo {
            id: id.to_string(),
            title: title.to_string(),
            model: None,
            branch: None,
            token_count: 0,
            created_at,
            message_count: messages,
        }
    }

    #[test]
    fn cycle_sort_mode_reorders_and_preserves_selection() {
        let sessions = vec![
            mk("a", "zeta", 3000, 2),
            mk("b", "alpha", 1000, 10),
            mk("c", "beta", 2000, 5),
        ];
        let mut picker = SessionPicker::new(sessions, Theme::default());
        assert_eq!(picker.selected_session().map(|s| s.id.as_str()), Some("a"));

        picker.select_down();
        let selected = picker.selected_session().map(|s| s.id.clone());
        picker.cycle_sort_mode();
        assert_eq!(picker.sort_mode_label(), "oldest");
        assert_eq!(picker.selected_session().map(|s| s.id.clone()), selected);

        picker.cycle_sort_mode();
        assert_eq!(picker.sort_mode_label(), "title");
        assert_eq!(picker.sessions[0].title, "alpha");

        picker.cycle_sort_mode();
        assert_eq!(picker.sort_mode_label(), "messages");
        assert_eq!(picker.sessions[0].message_count, 10);
    }

    #[test]
    fn session_row_label_aligns_both_separators() {
        let session = SessionInfo {
            id: "s1".to_string(),
            title: "conversation".to_string(),
            model: Some("gpt-5.5".to_string()),
            branch: Some("feature/ui".to_string()),
            token_count: 3200,
            created_at: 1_000_000_000_000,
            message_count: 18,
        };
        // Simulate: max_title=21, max_when=8 ("just now")
        let max_w = 21usize;
        let max_when = 8usize;

        let short = "conversation".to_string();
        let row_short = session_row_label(&session, &short, "18h ago", max_w, max_when);
        let row_justnow = session_row_label(&session, &short, "just now", max_w, max_when);
        let long = "quite long title here".to_string();
        let row_long = session_row_label(&session, &long, "5m ago", max_w, max_when);

        // Find byte positions of both separators in each row
        let seps: Vec<(usize, usize)> = [&row_short, &row_justnow, &row_long]
            .iter()
            .map(|r| {
                let first = r.find('│').unwrap();
                let second = r
                    .char_indices()
                    .filter(|(_, c)| *c == '│')
                    .nth(1)
                    .map(|(i, _)| i)
                    .unwrap();
                (first, second)
            })
            .collect();

        // All first separators at same byte position
        assert_eq!(seps[0].0, seps[1].0, "first │ should align across rows");
        assert_eq!(seps[0].0, seps[2].0, "first │ should align across rows");

        // All second separators at same byte position
        assert_eq!(seps[0].1, seps[1].1, "second │ should align across rows");
        assert_eq!(seps[0].1, seps[2].1, "second │ should align across rows");

        // Verify expected positions:
        // title(21) + "  "(2) = 23 for first │
        // then │(3 bytes) + "  "(2) + when(8) + "  "(2) before second │
        // = 23 + 3 + 2 + 8 + 2 = 38 for second │
        assert_eq!(seps[0].0, 23, "first │ at byte 23");
        assert_eq!(seps[0].1, 38, "second │ at byte 38");
    }

    #[test]
    fn truncation_handles_multi_byte_chars_safely() {
        // Title with multi-byte UTF-8 chars — Vec<char> slicing
        // must not panic.
        let title = "áéíóú — accented chars"; // 27 chars, 31 bytes
        let title_chars: Vec<char> = title.chars().collect();
        assert!(title_chars.len() > 5);
        // Truncate to 5 chars
        let truncated: String = title_chars[..5].iter().collect();
        assert_eq!(truncated.chars().count(), 5);
        assert_eq!(truncated, "áéíóú");
    }
}
