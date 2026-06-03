use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};

use crate::components::session_picker::SessionInfo;
use crate::theme::Theme;

#[derive(Debug, Clone, Copy)]
pub enum TreeFilter {
    Default,
    NoTools,
    UserOnly,
    LabeledOnly,
    All,
}

impl TreeFilter {
    pub fn parse(s: &str) -> Self {
        match s {
            "no-tools" => Self::NoTools,
            "user-only" => Self::UserOnly,
            "labeled-only" => Self::LabeledOnly,
            "all" => Self::All,
            _ => Self::Default,
        }
    }
}

pub struct TreeSelector {
    pub visible: bool,
    sessions: Vec<SessionInfo>,
    selected: usize,
    state: ListState,
    theme: Theme,
    pub filter: TreeFilter,
}

impl TreeSelector {
    pub fn new(theme: Theme) -> Self {
        let mut state = ListState::default();
        state.select(Some(0));
        Self {
            visible: false,
            sessions: Vec::new(),
            selected: 0,
            state,
            theme,
            filter: TreeFilter::Default,
        }
    }

    pub fn set_sessions(&mut self, sessions: Vec<SessionInfo>, filter: TreeFilter) {
        self.sessions = sessions;
        self.filter = filter;
        self.selected = 0;
        self.state.select(Some(0));
        self.visible = true;
    }

    pub fn select_up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
        self.state.select(Some(self.selected));
    }
    pub fn select_down(&mut self) {
        if !self.sessions.is_empty() {
            self.selected = (self.selected + 1).min(self.sessions.len() - 1);
            self.state.select(Some(self.selected));
        }
    }
    pub fn selected(&self) -> Option<&SessionInfo> {
        self.sessions.get(self.selected)
    }

    pub fn render(&mut self, area: Rect, frame: &mut Frame) {
        if !self.visible {
            return;
        }
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(area);
        let rows: Vec<ListItem> = self
            .sessions
            .iter()
            .map(|s| ListItem::new(tree_row_label(s)))
            .collect();
        let header = Paragraph::new(Line::from(Span::styled(
            format!("Tree view ({})", filter_label(self.filter)),
            Style::default().fg(self.theme.accent),
        )));
        frame.render_widget(header, chunks[0]);
        let list = List::new(rows)
            .block(
                Block::default()
                    .title("Sessions")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(self.theme.border)),
            )
            .highlight_style(Style::default().fg(self.theme.accent).bg(Color::DarkGray))
            .highlight_symbol("> ");
        frame.render_stateful_widget(list, chunks[1], &mut self.state);
        let footer = Paragraph::new(Span::styled(
            "Up/Down move | Enter resume | Esc close",
            Style::default().fg(self.theme.dim),
        ));
        frame.render_widget(footer, chunks[2]);
    }
}

pub fn tree_row_label(session: &SessionInfo) -> String {
    format!(
        "{} | {} | {} msgs | {}",
        session.branch.clone().unwrap_or_else(|| "-".into()),
        session.model.clone().unwrap_or_else(|| "unknown".into()),
        session.message_count,
        session.title
    )
}

fn filter_label(filter: TreeFilter) -> &'static str {
    match filter {
        TreeFilter::Default => "default",
        TreeFilter::NoTools => "no-tools",
        TreeFilter::UserOnly => "user-only",
        TreeFilter::LabeledOnly => "labeled-only",
        TreeFilter::All => "all",
    }
}
