//! Model selector overlay — shown on Ctrl+P to switch models mid-conversation.
//!
//! Displays favorites in a pinned upper section, remaining models below.
//! Press `f` to toggle the selected model as a favorite.

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
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
    /// Full unfiltered model list — preserved for restore after filter.
    saved_models: Vec<ModelEntry>,
    /// Indices of favorites into `all_models`.
    favorite_indices: Vec<usize>,
    /// Indices of non-favorite models into `all_models`.
    other_indices: Vec<usize>,
    /// Display order: favorites first, then others. Each entry is an index into `all_models`.
    display_order: Vec<usize>,
    /// Number of favorites in `display_order`.
    favorite_count: usize,
    /// Currently selected index in `display_order`.
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
    /// Indices into `all_models` of all models that match the current query.
    filtered_indices: Vec<usize>,
}

impl ModelSelector {
    pub fn new(models: Vec<ModelEntry>, favorites: Vec<String>, theme: Theme) -> Self {
        let favorite_indices = Self::resolve_favorites(&models, &favorites);
        let other_indices: Vec<usize> = (0..models.len())
            .filter(|i| !favorite_indices.contains(i))
            .collect();
        let display_order = [favorite_indices.clone(), other_indices.clone()].concat();
        let favorite_count = favorite_indices.len();
        let mut list_state = ListState::default();
        if !display_order.is_empty() {
            list_state.select(Some(0));
        }
        Self {
            all_models: models.clone(),
            saved_models: models,
            favorite_indices,
            other_indices,
            display_order,
            favorite_count,
            selected: 0,
            list_state,
            query: String::new(),
            theme,
            visible: false,
            confirmed: false,
            filtered_indices: Vec::new(),
        }
    }

    /// Show the selector.
    pub fn show(&mut self) {
        self.visible = true;
        self.confirmed = false;
        self.query.clear();
        self.rebuild_display();
    }

    /// Hide the selector.
    pub fn hide(&mut self) {
        self.visible = false;
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    pub fn set_models(&mut self, models: Vec<ModelEntry>) {
        self.all_models = models.clone();
        self.saved_models = models;
        // Re-resolve favorites against the new model list — indices from the
        // old list may be out of bounds if the new list is shorter.
        let favorites: Vec<String> = self
            .favorite_indices
            .iter()
            .filter_map(|&i| self.all_models.get(i).map(|m| m.id.clone()))
            .collect();
        self.favorite_indices = Self::resolve_favorites(&self.all_models, &favorites);
        self.other_indices = (0..self.all_models.len())
            .filter(|i| !self.favorite_indices.contains(i))
            .collect();
        self.rebuild_display();
    }

    /// Update the favorites list and rebuild the display.
    pub fn set_favorites(&mut self, favorites: Vec<String>) {
        self.favorite_indices = Self::resolve_favorites(&self.all_models, &favorites);
        self.other_indices = (0..self.all_models.len())
            .filter(|i| !self.favorite_indices.contains(i))
            .collect();
        self.rebuild_display();
    }

    /// Filter the picker to only show models from a specific provider.
    /// Preserves the full model list in saved_models for restore.
    pub fn show_filtered_for_provider(&mut self, provider: &str) {
        let filtered: Vec<ModelEntry> = self
            .saved_models
            .iter()
            .filter(|m| m.provider == provider)
            .cloned()
            .collect();
        self.all_models = filtered;
        self.rebuild_display();
        self.show();
    }

    /// Restore the full model list after filtering.
    pub fn restore_all_models(&mut self) {
        self.set_models(self.saved_models.clone());
    }

    /// Get the selected model entry, if any.
    pub fn selected_model(&self) -> Option<&ModelEntry> {
        self.display_order
            .get(self.selected)
            .and_then(|&idx| self.all_models.get(idx))
    }

    /// Move selection up.
    pub fn select_up(&mut self) {
        if self.display_order.is_empty() {
            return;
        }
        self.selected = self.selected.saturating_sub(1);
        self.list_state.select(Some(self.selected));
    }

    /// Move selection down.
    pub fn select_down(&mut self) {
        if self.display_order.is_empty() {
            return;
        }
        self.selected = (self.selected + 1).min(self.display_order.len().saturating_sub(1));
        self.list_state.select(Some(self.selected));
    }

    /// Add a character to the search query.
    pub fn push_query(&mut self, c: char) {
        self.query.push(c);
        self.rebuild_display();
    }

    /// Remove last character from search query.
    pub fn pop_query(&mut self) {
        self.query.pop();
        self.rebuild_display();
    }

    /// Check if the currently selected model is a favorite.
    pub fn selected_is_favorite(&self) -> bool {
        self.display_order
            .get(self.selected)
            .is_some_and(|&idx| self.favorite_indices.contains(&idx))
    }

    /// Number of favorites in the current display.
    pub fn favorite_count(&self) -> usize {
        self.favorite_count
    }

    /// The display order: indices into the model list, favorites first.
    pub fn display_order(&self) -> &[usize] {
        &self.display_order
    }

    /// All available models.
    pub fn all_models(&self) -> &[ModelEntry] {
        &self.all_models
    }

    /// Resolve favorite model IDs to indices into the model list.
    fn resolve_favorites(models: &[ModelEntry], favorites: &[String]) -> Vec<usize> {
        favorites
            .iter()
            .filter_map(|fav| models.iter().position(|m| m.id == *fav))
            .collect()
    }

    /// Rebuild `display_order` from favorites + filtered others.
    fn rebuild_display(&mut self) {
        let q = self.query.to_lowercase();
        let matches_query = |m: &ModelEntry| -> bool {
            q.is_empty()
                || m.id.to_lowercase().contains(&q)
                || m.name.to_lowercase().contains(&q)
                || m.provider.to_lowercase().contains(&q)
        };

        let fav_matching: Vec<usize> = self
            .favorite_indices
            .iter()
            .copied()
            .filter(|&i| matches_query(&self.all_models[i]))
            .collect();

        let other_matching: Vec<usize> = self
            .other_indices
            .iter()
            .copied()
            .filter(|&i| matches_query(&self.all_models[i]))
            .collect();

        self.favorite_count = fav_matching.len();
        self.filtered_indices = [fav_matching.clone(), other_matching.clone()].concat();
        self.display_order = self.filtered_indices.clone();
        self.selected = 0;
        if self.display_order.is_empty() {
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

        // Dynamic width: 85% of terminal, clamped between 80 and 120.
        let term_width = area.width as usize;
        let desired = (term_width * 85) / 100;
        let overlay_width = (desired.clamp(80, 120).min(term_width)) as u16;
        let overlay_height = area.height.min(20);
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
            .style(Style::default().bg(self.theme.bg))
            .border_style(Style::default().fg(self.theme.accent))
            .title(" Models ");

        let inner = block.inner(overlay);
        frame.render_widget(block, overlay);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // search bar
                Constraint::Min(0),    // model list
                Constraint::Length(1), // footer
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

        // Compute dynamic column widths from the models being displayed.
        let inner_width = (overlay_width as usize).saturating_sub(4); // borders + padding
        let ctx_width: usize = 12; // e.g. "272K " or "1M    " — fits "Ctx Window"
        let gap: usize = 2; // gap between columns
        let provider_width: usize = self
            .display_order
            .iter()
            .map(|&i| self.all_models[i].provider.len())
            .max()
            .unwrap_or(8)
            .max(8);
        // Remaining space for id + name after provider and ctx.
        let reserved = provider_width + gap + ctx_width + gap;
        let available = inner_width.saturating_sub(reserved);
        let id_width = self
            .display_order
            .iter()
            .map(|&i| self.all_models[i].id.len())
            .max()
            .unwrap_or(10)
            .min(available / 2)
            .max(8);
        let name_width = available.saturating_sub(id_width + gap).max(8);

        // Build the list items with section headers.
        let mut items: Vec<ListItem> = Vec::new();
        let mut row_to_display: Vec<usize> = Vec::new(); // maps list row -> display_order index

        if self.favorite_count > 0 {
            // Favorites section header.
            items.push(ListItem::new(Line::from(Span::styled(
                "  ★ Favorites",
                Style::default()
                    .fg(self.theme.accent)
                    .add_modifier(ratatui::style::Modifier::BOLD),
            ))));
            row_to_display.push(usize::MAX);
            // Column labels.
            items.push(ListItem::new(Line::from(Span::styled(
                format!(
                    "  {:id_width$}  {:name_width$}  {:provider_width$}  {:>ctx_width$}",
                    "ID", "Name", "Provider", "Ctx Window"
                ),
                Style::default().fg(self.theme.dim),
            ))));
            row_to_display.push(usize::MAX);

            for i in 0..self.favorite_count {
                let model_idx = self.display_order[i];
                let m = &self.all_models[model_idx];
                let row_text = format!(
                    "  {}",
                    format_model_row_sized(m, id_width, name_width, provider_width)
                );
                items.push(ListItem::new(Line::from(Span::styled(
                    row_text,
                    Style::default().fg(self.theme.accent),
                ))));
                row_to_display.push(i);
            }
        }

        // "All models" section header (only if there are non-favorites).
        let other_start = self.favorite_count;
        let other_count = self.display_order.len() - other_start;
        if other_count > 0 {
            let header_label = if self.favorite_count > 0 {
                "  All"
            } else {
                "  Models"
            };
            items.push(ListItem::new(Line::from(Span::styled(
                header_label,
                Style::default().fg(self.theme.dim),
            ))));
            row_to_display.push(usize::MAX);
            // Column labels.
            items.push(ListItem::new(Line::from(Span::styled(
                format!(
                    "  {:id_width$}  {:name_width$}  {:provider_width$}  {:>ctx_width$}",
                    "ID", "Name", "Provider", "Ctx Window"
                ),
                Style::default().fg(self.theme.dim),
            ))));
            row_to_display.push(usize::MAX);

            for i in other_start..self.display_order.len() {
                let model_idx = self.display_order[i];
                let m = &self.all_models[model_idx];
                let row_text = format!(
                    "  {}",
                    format_model_row_sized(m, id_width, name_width, provider_width)
                );
                items.push(ListItem::new(Span::raw(row_text)));
                row_to_display.push(i);
            }
        }

        if items.is_empty() {
            items.push(ListItem::new(Span::styled(
                "  No matching models",
                Style::default().fg(self.theme.dim),
            )));
        }

        // Map `selected` (display_order index) to list row index.
        let mapped_selected = if self.display_order.is_empty() {
            None
        } else {
            row_to_display.iter().position(|&r| r == self.selected)
        };

        let list = List::new(items)
            .highlight_style(Style::default().fg(self.theme.accent).bg(Color::DarkGray))
            .highlight_symbol("> ");

        let mut list_state = self.list_state;
        list_state.select(mapped_selected);
        frame.render_stateful_widget(list, chunks[1], &mut list_state);

        // Footer help.
        let fav_hint = "Ctrl+F=toggle favorite";
        let help = Paragraph::new(Span::styled(
            format!("↑↓ move | Enter select | {fav_hint} | Esc close"),
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

/// Format a model row with explicit column widths, truncating with ellipsis when needed.
fn format_model_row_sized(
    model: &ModelEntry,
    id_width: usize,
    name_width: usize,
    provider_width: usize,
) -> String {
    let ctx = format_context_window(model.context_window);
    let id = truncate_str(&model.id, id_width);
    let name = truncate_str(&model.name, name_width);
    let provider = truncate_str(&model.provider, provider_width);
    format!("{id:<id_width$}  {name:<name_width$}  {provider:<provider_width$}  {ctx:>6}")
}

/// Truncate a string to `max_chars`, appending `…` if truncated.
fn truncate_str(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        s.to_string()
    } else if max_chars <= 1 {
        s.chars().take(max_chars).collect()
    } else {
        let truncated: String = s.chars().take(max_chars - 1).collect();
        format!("{truncated}…")
    }
}

fn format_context_window(context_window: u32) -> String {
    if context_window >= 1_000_000 {
        let m = context_window / 1_000_000;
        format!("{m}M")
    } else {
        let k = context_window / 1000;
        format!("{k}K")
    }
}
