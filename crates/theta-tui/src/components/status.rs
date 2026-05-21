//! Status bar component.

use crossterm::event::Event;
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

use crate::components::{Action, Component};
use crate::theme::Theme;

pub struct StatusBar {
    pub model: String,
    pub session_id: String,
    pub thinking: String,
    pub agent_state: String,
    pub tool_progress: String,
    theme: Theme,
}

impl StatusBar {
    pub fn new(theme: Theme) -> Self {
        Self {
            model: String::new(),
            session_id: String::new(),
            thinking: String::new(),
            agent_state: String::new(),
            tool_progress: String::new(),
            theme,
        }
    }

    pub fn set_agent_state(&mut self, state: &str) {
        self.agent_state = state.to_string();
    }

    pub fn set_tool_progress(&mut self, progress: &str) {
        self.tool_progress = progress.to_string();
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }
}

impl Component for StatusBar {
    fn render(&mut self, area: Rect, frame: &mut Frame) {
        let total_width = area.width as usize;
        let session_id = short_middle(&self.session_id, 14);
        let model_str = short_middle(&self.model, 24);
        let thinking_str = format!(" | thinking: {}", self.thinking);
        let session_str = format!(" | session: {session_id}");

        let left = vec![
            Span::styled(model_str, Style::default().fg(self.theme.accent)),
            Span::styled(thinking_str, Style::default().fg(self.theme.dim)),
            Span::styled(session_str, Style::default().fg(self.theme.dim)),
        ];

        let state_color = if self.agent_state.starts_with("error")
            || self.agent_state.starts_with("tool error")
        {
            self.theme.error
        } else if self.agent_state.starts_with("streaming")
            || self.agent_state.starts_with("thinking")
            || self.agent_state.starts_with("tool")
            || self.agent_state.starts_with("compacting")
            || self.agent_state.starts_with("retrying")
        {
            self.theme.warning
        } else {
            self.theme.success
        };

        let mut right_text = if self.tool_progress.is_empty() {
            format!("[{}]", self.agent_state)
        } else {
            format!("[{}] {}", self.agent_state, self.tool_progress)
        };
        right_text = truncate_chars(&right_text, total_width.saturating_div(2).max(12));

        let right = vec![Span::styled(right_text, Style::default().fg(state_color))];

        // Pad to fill width.
        let left_str: String = left.iter().map(|s| s.content.as_ref()).collect();
        let right_str: String = right.iter().map(|s| s.content.as_ref()).collect();

        let pad = if left_str.len() + right_str.len() < total_width {
            " ".repeat(total_width - left_str.len() - right_str.len())
        } else {
            " ".to_string()
        };

        let mut spans = left;
        spans.push(Span::raw(pad));
        spans.extend(right);

        let para =
            Paragraph::new(Line::from(spans)).style(Style::default().bg(Color::Rgb(30, 30, 30)));
        frame.render_widget(para, area);
    }

    fn handle_event(&mut self, _event: &Event) -> Option<Action> {
        None
    }

    fn is_focused(&self) -> bool {
        false
    }

    fn focus(&mut self, _focused: bool) {}
}

fn short_middle(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars || max_chars < 5 {
        return text.to_string();
    }
    let head_len = (max_chars - 3) / 2;
    let tail_len = max_chars - 3 - head_len;
    let head: String = text.chars().take(head_len).collect();
    let tail: String = text
        .chars()
        .rev()
        .take(tail_len)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head}...{tail}")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}
