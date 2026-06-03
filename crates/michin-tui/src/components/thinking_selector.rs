//! Thinking level selector overlay — shown on `/thinking` (no args) to pick
//! a valid level for the current model.

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::Span,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};

use crate::theme::Theme;

/// A thinking level entry in the selector.
#[derive(Debug, Clone)]
pub struct ThinkingLevelEntry {
    pub id: String,
    pub label: String,
}

/// Thinking level selector overlay component.
pub struct ThinkingSelector {
    /// Available thinking levels for the current model.
    levels: Vec<ThinkingLevelEntry>,
    /// Currently selected index.
    pub selected: usize,
    /// List state.
    pub list_state: ListState,
    /// Theme.
    theme: Theme,
    /// Whether to show the selector.
    pub visible: bool,
    /// Whether user confirmed (Enter) or cancelled (Esc).
    pub confirmed: bool,
}

impl ThinkingSelector {
    pub fn new(theme: Theme) -> Self {
        let mut list_state = ListState::default();
        list_state.select(Some(0));
        Self {
            levels: Vec::new(),
            selected: 0,
            list_state,
            theme,
            visible: false,
            confirmed: false,
        }
    }

    /// Show the selector with the given levels.
    pub fn show(&mut self, levels: Vec<ThinkingLevelEntry>, current: Option<&str>) {
        self.levels = levels;
        self.visible = true;
        self.confirmed = false;
        // Select the current level if it's in the list.
        self.selected = current
            .and_then(|c| self.levels.iter().position(|l| l.id == c))
            .unwrap_or(0);
        self.list_state.select(Some(self.selected));
    }

    /// Hide the selector.
    pub fn hide(&mut self) {
        self.visible = false;
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    /// Whether levels have been loaded.
    pub fn has_levels(&self) -> bool {
        !self.levels.is_empty()
    }

    /// Get the selected level ID, if any.
    pub fn selected_level(&self) -> Option<&str> {
        self.levels.get(self.selected).map(|e| e.id.as_str())
    }

    /// Move selection up.
    pub fn select_up(&mut self) {
        if self.levels.is_empty() {
            return;
        }
        self.selected = self.selected.saturating_sub(1);
        self.list_state.select(Some(self.selected));
    }

    /// Move selection down.
    pub fn select_down(&mut self) {
        if self.levels.is_empty() {
            return;
        }
        self.selected = (self.selected + 1).min(self.levels.len().saturating_sub(1));
        self.list_state.select(Some(self.selected));
    }

    /// Render the selector overlay.
    pub fn render(&mut self, area: Rect, frame: &mut Frame) {
        if !self.visible || self.levels.is_empty() {
            return;
        }

        // Center the overlay on screen.
        let overlay_width = area.width.min(40);
        let overlay_height = area.height.min(12);
        let overlay_x = (area.width.saturating_sub(overlay_width)) / 2;
        let overlay_y = (area.height.saturating_sub(overlay_height)) / 2;

        let overlay = Rect {
            x: area.x + overlay_x,
            y: area.y + overlay_y,
            width: overlay_width,
            height: overlay_height,
        };

        let block = Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(self.theme.accent))
            .title(" Thinking Level ");

        let inner = block.inner(overlay);
        frame.render_widget(block, overlay);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(1)])
            .split(inner);

        // Level list.
        let items: Vec<ListItem> = self
            .levels
            .iter()
            .map(|entry| {
                let label = format!("  {:8}  {}", entry.id, entry.label);
                ListItem::new(Span::raw(label))
            })
            .collect();

        let list = List::new(items)
            .highlight_style(Style::default().fg(self.theme.accent).bg(Color::DarkGray))
            .highlight_symbol("> ");

        frame.render_stateful_widget(list, chunks[0], &mut self.list_state);

        // Footer help.
        let help = Paragraph::new(Span::styled(
            "Up/Down move | Enter select | Esc close",
            Style::default().fg(self.theme.dim),
        ));
        frame.render_widget(help, chunks[1]);
    }
}
