//! Model selector overlay — shown on Ctrl+P to switch models mid-conversation.

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::Span,
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};

use crate::theme::Theme;

/// A model entry in the selector.
#[derive(Debug, Clone)]
pub struct ModelEntry {
    pub id: String,
    pub name: String,
    pub provider: String,
    pub context_window: u32,
}

/// Model selector overlay component.
pub struct ModelSelector {
    /// All available models.
    all_models: Vec<ModelEntry>,
    /// Currently displayed models (filtered).
    pub filtered: Vec<usize>,
    /// Currently selected index in filtered.
    selected: usize,
    /// List state.
    list_state: ListState,
    /// Search query text.
    query: String,
    /// Theme.
    theme: Theme,
    /// Whether to show the selector.
    pub visible: bool,
    /// Whether user confirmed (Enter) or cancelled (Esc).
    pub confirmed: bool,
}

impl ModelSelector {
    pub fn new(models: Vec<ModelEntry>, theme: Theme) -> Self {
        let indices: Vec<usize> = (0..models.len()).collect();
        let mut list_state = ListState::default();
        if !indices.is_empty() {
            list_state.select(Some(0));
        }
        Self {
            all_models: models,
            filtered: indices,
            selected: 0,
            list_state,
            query: String::new(),
            theme,
            visible: false,
            confirmed: false,
        }
    }

    /// Show the selector.
    pub fn show(&mut self) {
        self.visible = true;
        self.confirmed = false;
        self.query.clear();
        self.filter_models();
    }

    /// Hide the selector.
    pub fn hide(&mut self) {
        self.visible = false;
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    pub fn set_models(&mut self, models: Vec<ModelEntry>) {
        self.all_models = models;
        self.filter_models();
    }

    /// Get the selected model entry, if any.
    pub fn selected_model(&self) -> Option<&ModelEntry> {
        self.filtered
            .get(self.selected)
            .and_then(|&idx| self.all_models.get(idx))
    }

    /// Move selection up.
    pub fn select_up(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        self.selected = self.selected.saturating_sub(1);
        self.list_state.select(Some(self.selected));
    }

    /// Move selection down.
    pub fn select_down(&mut self) {
        if self.filtered.is_empty() {
            return;
        }
        self.selected = (self.selected + 1).min(self.filtered.len().saturating_sub(1));
        self.list_state.select(Some(self.selected));
    }

    /// Add a character to the search query.
    pub fn push_query(&mut self, c: char) {
        self.query.push(c);
        self.filter_models();
    }

    /// Remove last character from search query.
    pub fn pop_query(&mut self) {
        self.query.pop();
        self.filter_models();
    }

    /// Filter models by query (substring match on id, name, and provider).
    fn filter_models(&mut self) {
        let q = self.query.to_lowercase();
        if q.is_empty() {
            self.filtered = (0..self.all_models.len()).collect();
        } else {
            self.filtered = self
                .all_models
                .iter()
                .enumerate()
                .filter(|(_, m)| {
                    m.id.to_lowercase().contains(&q)
                        || m.name.to_lowercase().contains(&q)
                        || m.provider.to_lowercase().contains(&q)
                })
                .map(|(i, _)| i)
                .collect();
        }
        self.selected = 0;
        if self.filtered.is_empty() {
            self.list_state.select(None);
        } else {
            self.list_state.select(Some(0));
        }
    }

    /// Render the selector overlay.
    pub fn render(&mut self, area: Rect, frame: &mut Frame) {
        if !self.visible {
            return;
        }

        // Center the overlay on screen.
        let overlay_width = area.width.min(60);
        let overlay_height = area.height.min(16);
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
            .title(" Models ");

        let inner = block.inner(overlay);
        frame.render_widget(block, overlay);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(2),
                Constraint::Min(0),
                Constraint::Length(1),
            ])
            .split(inner);

        // Search bar.
        let cursor = if self.query.len() == self.query.chars().count() {
            "█"
        } else {
            ""
        };
        let search_text = format!("filter> {}{}", self.query, cursor);
        let search = Paragraph::new(Span::styled(
            search_text,
            Style::default().fg(self.theme.accent),
        ));
        frame.render_widget(search, chunks[0]);

        // Model list.
        let items: Vec<ListItem> = self
            .filtered
            .iter()
            .map(|&idx| {
                let m = &self.all_models[idx];
                ListItem::new(Span::raw(format_model_row(m)))
            })
            .collect();

        let list = List::new(items)
            .highlight_style(Style::default().fg(self.theme.accent).bg(Color::DarkGray))
            .highlight_symbol("> ");

        frame.render_stateful_widget(list, chunks[1], &mut self.list_state);

        // Footer help.
        let help = Paragraph::new(Span::styled(
            "Type filter | Up/Down move | Enter select | Esc close",
            Style::default().fg(self.theme.dim),
        ));
        frame.render_widget(help, chunks[2]);
    }
}

pub fn format_model_row(model: &ModelEntry) -> String {
    let ctx = format_context_window(model.context_window);
    format!(
        "{:18}  {:24}  {:13}  {ctx:>6}",
        model.id, model.name, model.provider
    )
}

fn format_context_window(context_window: u32) -> String {
    if context_window >= 1_000_000 {
        format!("{}M", context_window / 1_000_000)
    } else {
        format!("{}K", context_window / 1000)
    }
}
