//! MiMo cluster selector — modal overlay shown when a MiMo model is selected
//! and no cluster has been chosen yet. Measures latency and lets the user pick.

use crossterm::event::{Event, KeyCode};
use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    prelude::Modifier,
    style::Style,
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Wrap},
};

use crate::theme::Theme;

/// A MiMo cluster entry with measured latency.
#[derive(Debug, Clone)]
pub struct MimoClusterEntry {
    pub label: String,
    pub url: String,
    pub latency_ms: Option<u64>,
}

/// Cluster selector overlay.
pub struct MimoClusterSelector {
    /// Clusters to display (with measured latencies).
    pub clusters: Vec<MimoClusterEntry>,
    /// Whether the overlay is visible.
    pub visible: bool,
    /// Whether latency measurement is in progress.
    pub measuring: bool,
    /// Currently selected index.
    selected: usize,
    /// List state for ratatui's built-in selection highlight.
    list_state: ListState,
}

impl MimoClusterSelector {
    pub fn new() -> Self {
        let mut s = Self {
            clusters: Vec::new(),
            visible: false,
            measuring: false,
            selected: 0,
            list_state: ListState::default(),
        };
        s.list_state.select(Some(0));
        s
    }

    /// Open the selector and display the given clusters.
    /// If `current_url` matches a cluster, it is pre-selected.
    pub fn open(&mut self, clusters: Vec<MimoClusterEntry>, current_url: Option<&str>) {
        self.clusters = clusters;
        self.visible = true;
        self.measuring = false;
        if let Some(url) = current_url {
            if let Some(pos) = self.clusters.iter().position(|c| c.url == url) {
                self.selected = pos;
            } else {
                self.selected = 0;
            }
        } else {
            self.selected = 0;
        }
        self.list_state.select(Some(self.selected));
    }

    /// Show the selector in "measuring" state while latencies are computed.
    pub fn start_measuring(&mut self) {
        self.visible = true;
        self.measuring = true;
        self.clusters.clear();
        self.selected = 0;
        self.list_state.select(Some(0));
    }

    /// Returns the currently selected cluster URL, if any.
    pub fn selected_url(&self) -> Option<&str> {
        self.clusters.get(self.selected).map(|c| c.url.as_str())
    }

    pub fn close(&mut self) {
        self.visible = false;
    }

    pub fn handle_event(&mut self, event: &Event) -> bool {
        if !self.visible {
            return false;
        }
        match event {
            Event::Key(key) => match key.code {
                KeyCode::Up | KeyCode::Char('k') => {
                    if self.selected > 0 {
                        self.selected -= 1;
                        self.list_state.select(Some(self.selected));
                    }
                    true
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if self.selected + 1 < self.clusters.len() {
                        self.selected += 1;
                        self.list_state.select(Some(self.selected));
                    }
                    true
                }
                _ => false,
            },
            _ => false,
        }
    }

    pub fn render(&mut self, area: Rect, frame: &mut Frame, theme: &Theme) {
        if !self.visible {
            return;
        }

        let popup_area = centered_rect(60, 50, area);

        let block = Block::default()
            .title("MiMo Cluster Selector")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(theme.highlight));

        if self.measuring {
            let text = "Measuring latency to each cluster...";
            let p = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
            frame.render_widget(p, popup_area);
            return;
        }

        if self.clusters.is_empty() {
            let text = "No clusters available.\n\nPress Esc to close.";
            let p = Paragraph::new(text).block(block).wrap(Wrap { trim: false });
            frame.render_widget(p, popup_area);
            return;
        }

        // Build list items with latency display.
        let items: Vec<ListItem> = self
            .clusters
            .iter()
            .map(|c| {
                let latency_str = match c.latency_ms {
                    Some(ms) => format!("  {}ms", ms),
                    None => "  unreachable".to_string(),
                };
                let line = Line::from(vec![
                    Span::raw(format!("  {}  ", c.label)),
                    Span::styled(latency_str, Style::default().add_modifier(Modifier::DIM)),
                ]);
                ListItem::new(line)
            })
            .collect();

        let list = List::new(items)
            .block(block)
            .highlight_style(Style::default().fg(theme.bg).bg(theme.highlight));

        // Clone the list state to work around ratatui's mutable borrow.
        let mut state = self.list_state.clone();
        frame.render_stateful_widget(list, popup_area, &mut state);
        // Sync back the selected index (ratatui may have updated it).
        if let Some(i) = state.selected() {
            self.selected = i;
        }
    }
}

/// Return a rectangle centered in `area` with the given width/height percentages.
fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

impl Default for MimoClusterSelector {
    fn default() -> Self {
        Self::new()
    }
}
