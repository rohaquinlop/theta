//! Theme selector overlay — shown on `/themes` command or `Ctrl+T` long-press.
//!
//! Displays all available themes (built-in + user) with live color previews.
//! Selecting a theme applies it immediately and persists to `config.toml`.

use std::collections::HashMap;

use ratatui::{
    Frame,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Line, Span},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph},
};

use crate::theme::Theme;

/// Theme selector overlay component.
pub struct ThemeSelector {
    /// All available theme names (built-in + user).
    names: Vec<String>,
    /// User themes for rendering previews.
    user_themes: HashMap<String, Theme>,
    /// Currently selected index.
    selected: usize,
    /// List state.
    list_state: ListState,
    /// Whether to show the selector.
    pub visible: bool,
    /// Whether user confirmed (Enter) or cancelled (Esc).
    pub confirmed: bool,
}

impl ThemeSelector {
    pub fn new(names: Vec<String>, user_themes: HashMap<String, Theme>) -> Self {
        let mut list_state = ListState::default();
        if !names.is_empty() {
            list_state.select(Some(0));
        }
        Self {
            names,
            user_themes,
            selected: 0,
            list_state,
            visible: false,
            confirmed: false,
        }
    }

    /// Show the selector, pre-selecting the current theme.
    pub fn show(&mut self, current_theme_name: &str) {
        self.visible = true;
        self.confirmed = false;
        self.selected = self
            .names
            .iter()
            .position(|n| n == current_theme_name)
            .unwrap_or(0);
        self.list_state.select(Some(self.selected));
    }

    /// Hide the selector.
    pub fn hide(&mut self) {
        self.visible = false;
    }

    /// Update the theme list (e.g. after reloading user themes).
    pub fn set_themes(&mut self, names: Vec<String>, user_themes: HashMap<String, Theme>) {
        self.names = names;
        self.user_themes = user_themes;
    }

    /// Get the selected theme name, if any.
    pub fn selected_theme(&self) -> Option<&str> {
        self.names.get(self.selected).map(|s| s.as_str())
    }

    /// Move selection up.
    pub fn select_up(&mut self) {
        if self.names.is_empty() {
            return;
        }
        self.selected = self.selected.saturating_sub(1);
        self.list_state.select(Some(self.selected));
    }

    /// Move selection down.
    pub fn select_down(&mut self) {
        if self.names.is_empty() {
            return;
        }
        self.selected = (self.selected + 1).min(self.names.len().saturating_sub(1));
        self.list_state.select(Some(self.selected));
    }

    /// Resolve a theme by the selected name.
    pub fn resolve_selected(&self) -> Option<Theme> {
        let name = self.selected_theme()?;
        Some(Theme::named_with_users(name, &self.user_themes))
    }

    /// Render the selector overlay.
    pub fn render(&mut self, area: Rect, frame: &mut Frame, _current_theme: &Theme) {
        if !self.visible || self.names.is_empty() {
            return;
        }

        // Overlay: 70% width, 60% height, centered.
        let overlay_width = (area.width * 70 / 100).clamp(50, 80);
        let overlay_height = (area.height * 60 / 100).clamp(10, 24);
        let overlay_x = (area.width.saturating_sub(overlay_width)) / 2;
        let overlay_y = (area.height.saturating_sub(overlay_height)) / 2;

        let overlay = Rect {
            x: area.x + overlay_x,
            y: area.y + overlay_y,
            width: overlay_width,
            height: overlay_height,
        };

        // Resolve the selected theme for preview.
        let preview_theme = self.resolve_selected().unwrap_or_default();

        let block = Block::default()
            .borders(Borders::ALL)
            .style(Style::default().bg(preview_theme.bg))
            .border_style(Style::default().fg(preview_theme.accent))
            .title(format!(" Themes — {} ", self.names[self.selected]));

        let inner = block.inner(overlay);
        frame.render_widget(block, overlay);

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(6), // color preview swatches
                Constraint::Length(1), // separator
                Constraint::Min(0),    // theme list
                Constraint::Length(1), // footer
            ])
            .split(inner);

        // Color preview: show key theme colors as labeled blocks.
        self.render_preview(chunks[0], frame, &preview_theme);

        // Separator.
        let sep = "─".repeat(chunks[1].width as usize);
        frame.render_widget(
            Paragraph::new(Span::styled(sep, Style::default().fg(preview_theme.dim))),
            chunks[1],
        );

        // Theme list.
        let items: Vec<ListItem> = self
            .names
            .iter()
            .map(|name| {
                let theme = Theme::named_with_users(name, &self.user_themes);
                ListItem::new(Line::from(vec![
                    Span::styled(
                        format!("  {:16}  ", name),
                        Style::default().fg(theme.fg).bg(theme.bg),
                    ),
                    Span::raw(color_block(theme.accent) + " "),
                    Span::raw(color_block(theme.success) + " "),
                    Span::raw(color_block(theme.error) + " "),
                    Span::raw(color_block(theme.warning) + " "),
                    Span::raw(color_block(theme.highlight)),
                ]))
            })
            .collect();

        let list = List::new(items).highlight_style(
            Style::default()
                .fg(preview_theme.accent)
                .bg(preview_theme.highlight),
        );

        frame.render_stateful_widget(list, chunks[2], &mut self.list_state);

        // Footer.
        let help = Paragraph::new(Span::styled(
            "↑↓ move | Enter select | Esc close",
            Style::default().fg(preview_theme.dim),
        ));
        frame.render_widget(help, chunks[3]);
    }

    /// Render color swatch rows for the preview theme.
    fn render_preview(&self, area: Rect, frame: &mut Frame, theme: &Theme) {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
                Constraint::Length(1),
            ])
            .split(area);

        // Row 1: accent, bg, fg, dim
        let line1 = Line::from(vec![
            label_span("accent ", theme.dim),
            Span::styled("  ", Style::default().bg(theme.accent)),
            Span::raw("  "),
            label_span("bg ", theme.dim),
            Span::styled("  ", Style::default().bg(theme.bg)),
            Span::raw("  "),
            label_span("fg ", theme.dim),
            Span::styled("  ", Style::default().fg(theme.fg)),
            Span::raw("  "),
            label_span("dim ", theme.dim),
            Span::styled("  ", Style::default().fg(theme.dim)),
        ]);
        frame.render_widget(Paragraph::new(line1), rows[0]);

        // Row 2: success, error, warning
        let line2 = Line::from(vec![
            label_span("success ", theme.dim),
            Span::styled("  ", Style::default().fg(theme.success)),
            Span::raw("  "),
            label_span("error ", theme.dim),
            Span::styled("  ", Style::default().fg(theme.error)),
            Span::raw("  "),
            label_span("warning ", theme.dim),
            Span::styled("  ", Style::default().fg(theme.warning)),
        ]);
        frame.render_widget(Paragraph::new(line2), rows[1]);

        // Row 3: border, highlight, user_bubble
        let line3 = Line::from(vec![
            label_span("border ", theme.dim),
            Span::styled("  ", Style::default().fg(theme.border)),
            Span::raw("  "),
            label_span("highlight ", theme.dim),
            Span::styled("  ", Style::default().bg(theme.highlight)),
            Span::raw("  "),
            label_span("bubble ", theme.dim),
            Span::styled("  ", Style::default().bg(theme.user_bubble)),
        ]);
        frame.render_widget(Paragraph::new(line3), rows[2]);

        // Row 4: markdown colors
        let line4 = Line::from(vec![
            label_span("md_h1 ", theme.dim),
            Span::styled("  ", Style::default().fg(theme.md_heading_1)),
            Span::raw("  "),
            label_span("md_h2 ", theme.dim),
            Span::styled("  ", Style::default().fg(theme.md_heading_2)),
            Span::raw("  "),
            label_span("md_link ", theme.dim),
            Span::styled("  ", Style::default().fg(theme.md_link)),
            Span::raw("  "),
            label_span("code ", theme.dim),
            Span::styled(
                "  ",
                Style::default()
                    .fg(theme.code_fg.unwrap_or(theme.accent))
                    .bg(theme.code_bg),
            ),
        ]);
        frame.render_widget(Paragraph::new(line4), rows[3]);
    }
}

fn label_span<'a>(text: &'a str, color: Color) -> Span<'a> {
    Span::styled(text, Style::default().fg(color))
}

/// Render a small color block using the block characters.
fn color_block(color: Color) -> String {
    match color {
        Color::Reset => "░░".to_string(),
        _ => "██".to_string(),
    }
}
