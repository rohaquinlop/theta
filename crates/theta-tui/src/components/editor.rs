//! Input editor — multiline text input with visual-line cursor navigation,
//! inline fuzzy autocomplete for @ files and / commands, clipboard, and
//! proper terminal cursor positioning via `frame.set_cursor()`.

use crossterm::event::{Event, KeyCode, KeyModifiers, MouseButton, MouseEventKind};
use ratatui::{
    Frame,
    layout::Rect,
    style::Style,
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, Paragraph},
};
use std::path::PathBuf;

use crate::components::fuzzy::fuzzy_filter;
use crate::components::{Action, Component};
use crate::keybinding::{EnterBehavior, is_enter_send, is_follow_up_key, is_newline_key};
use crate::theme::Theme;

/// Inline autocomplete state.
#[derive(Debug, Clone)]
struct AutocompleteState {
    items: Vec<String>,
    selected: usize,
    /// Byte position where @ or / was typed.
    prefix_start: usize,
    /// Filter query between trigger and cursor.
    query: String,
    /// '@' or '/'.
    trigger: char,
}

/// Multiline text editor with visual-line cursor navigation.
///
/// ## Cursor model
///
/// Canonical cursor position is a byte offset into `text`.
/// `vis_lines` (built by `rebuild_visual_lines`) maps each visual line
/// to `Vec<usize>` of byte offsets per character, from which
/// `(vis_line, vis_col)` is derived on demand.
///
/// `desired_col` tracks horizontal aim during vertical navigation,
/// preserving column position across lines of varying length.
pub struct Editor {
    pub text: String,
    /// Byte offset into `text`.
    pub cursor: usize,
    /// `[byte_offset, …]` per character on each visual line.
    vis_lines: Vec<Vec<usize>>,
    /// Horizontal aim during vertical navigation.
    desired_col: usize,
    /// Inner width used for vis_lines cache.
    pub cached_width: usize,
    /// Cache invalidation flag.
    pub cache_dirty: bool,

    focused: bool,
    theme: Theme,
    history: Vec<String>,
    /// Position in history ring.
    history_idx: usize,
    /// Stash for history restore.
    saved_text: String,
    scroll: usize,
    autocomplete: Option<AutocompleteState>,
    working_dir: PathBuf,
    slash_commands: Vec<String>,
    enter_behavior: EnterBehavior,
    /// For hit-testing + cursor placement.
    pub last_inner_area: Option<Rect>,
}

impl Editor {
    pub fn new(
        theme: Theme,
        working_dir: PathBuf,
        slash_commands: Vec<String>,
        enter_behavior: String,
    ) -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            vis_lines: Vec::new(),
            desired_col: 0,
            cached_width: 0,
            cache_dirty: true,
            focused: false,
            theme,
            history: Vec::new(),
            history_idx: 0,
            saved_text: String::new(),
            scroll: 0,
            autocomplete: None,
            working_dir,
            slash_commands,
            enter_behavior: EnterBehavior::parse(&enter_behavior),
            last_inner_area: None,
        }
    }

    pub fn set_text(&mut self, text: &str) {
        self.text = text.to_string();
        self.cursor = self.text.len();
        self.desired_col = 0;
        self.scroll = 0;
        self.cache_dirty = true;
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn desired_height(&mut self, width: usize, max_height: u16) -> u16 {
        let inner_width = width.saturating_sub(2).max(1);
        self.rebuild_visual_lines(inner_width);
        let lines = self.vis_lines.len() as u16;
        lines.saturating_add(2).clamp(3, max_height.max(3))
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    /// Insert at cursor. Used by path picker.
    pub fn insert_at_cursor(&mut self, s: &str) {
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
        self.after_mutate();
    }

    /// Delete the last character.
    pub fn delete_last_char(&mut self) {
        if let Some(c) = self.text.chars().last() {
            let len = c.len_utf8();
            self.text.truncate(self.text.len() - len);
            if self.cursor > self.text.len() {
                self.cursor = self.text.len();
            }
            self.after_mutate();
        }
    }

    // ------------------------------------------------------------------
    // Visual line cache
    // ------------------------------------------------------------------

    /// Rebuild visual line cache if dirty or width changed.
    pub fn rebuild_visual_lines(&mut self, width: usize) {
        if !self.cache_dirty && self.cached_width == width {
            return;
        }
        self.vis_lines = build_vis_lines(&self.text, width);
        self.cached_width = width;
        self.cache_dirty = false;
    }

    /// Visual column at byte offset, rebuilding cache as needed.
    fn byte_to_vis_col(&mut self, byte: usize) -> usize {
        let width = if self.cached_width > 0 {
            self.cached_width
        } else {
            80
        };
        self.rebuild_visual_lines(width);
        let (_, col) = byte_to_vis(&self.vis_lines, &self.text, byte);
        col
    }

    fn nav_width(&self) -> usize {
        if self.cached_width > 0 {
            self.cached_width
        } else {
            80
        }
    }

    /// Post-mutation: rebuild cache, clamp cursor, maintain scroll.
    pub fn after_mutate(&mut self) {
        self.cache_dirty = true;
        self.rebuild_visual_lines(self.nav_width());
        // Clamp cursor to valid range.
        if !self.text.is_empty() && self.cursor > self.text.len() {
            self.cursor = self.text.len();
        }
        let (_vl, vc) = byte_to_vis(&self.vis_lines, &self.text, self.cursor);
        self.desired_col = vc;
        self.ensure_cursor_visible();
    }

    pub fn clamp_scroll(&mut self) {
        if self.vis_lines.is_empty() {
            self.scroll = 0;
            return;
        }
        let height = self
            .last_inner_area
            .map(|a| a.height as usize)
            .unwrap_or(3)
            .max(1);
        let max_scroll = self.vis_lines.len().saturating_sub(height);
        self.scroll = self.scroll.min(max_scroll);
    }

    // ------------------------------------------------------------------
    // Text editing operations
    // ------------------------------------------------------------------

    pub fn insert_char(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
        self.after_mutate();
    }

    fn delete_before(&mut self) {
        if self.cursor > 0
            && let Some(prev) = self.text[..self.cursor].chars().last()
        {
            let len = prev.len_utf8();
            self.text.replace_range(self.cursor - len..self.cursor, "");
            self.cursor -= len;
            self.after_mutate();
        }
    }

    fn delete_after(&mut self) {
        if self.cursor < self.text.len()
            && let Some(next) = self.text[self.cursor..].chars().next()
        {
            self.text
                .replace_range(self.cursor..self.cursor + next.len_utf8(), "");
            self.after_mutate();
        }
    }

    // ------------------------------------------------------------------
    // Cursor navigation (visual-line based)
    // ------------------------------------------------------------------

    /// Scroll to keep cursor line visible.
    fn ensure_cursor_visible(&mut self) {
        if self.vis_lines.is_empty() {
            return;
        }
        let (vl, _) = byte_to_vis(&self.vis_lines, &self.text, self.cursor);
        let height = self
            .last_inner_area
            .map(|a| a.height as usize)
            .unwrap_or(3)
            .max(1);
        if vl < self.scroll {
            self.scroll = vl;
        } else if vl >= self.scroll + height {
            self.scroll = vl.saturating_add(1).saturating_sub(height);
        }
        self.clamp_scroll();
    }

    /// Move cursor up. Returns true if at first line (caller handles history).
    pub fn move_up(&mut self) -> bool {
        self.rebuild_visual_lines(self.nav_width());
        if self.vis_lines.is_empty() {
            return false;
        }
        let (vl, _) = byte_to_vis(&self.vis_lines, &self.text, self.cursor);
        if vl == 0 {
            // At first visual line — signal caller to handle history.
            return true;
        }
        let target_line = vl.saturating_sub(1);
        let clamped_col = self
            .desired_col
            .min(self.vis_lines[target_line].len().saturating_sub(1));
        if self.vis_lines[target_line].is_empty() {
            // Empty line: position cursor at the start of this line.
            self.cursor = line_start_byte_for_width(&self.text, self.nav_width(), target_line);
        } else {
            self.cursor = self.vis_lines[target_line]
                .get(clamped_col)
                .copied()
                .unwrap_or(0);
            // If past end of line, go to last byte on that line.
            if self.cursor > self.text.len() {
                self.cursor = self.vis_lines[target_line].last().copied().unwrap_or(0);
                // Move past that char to the end.
                self.cursor += self.text[self.cursor..]
                    .chars()
                    .next()
                    .map(|c| c.len_utf8())
                    .unwrap_or(0);
            }
        }
        self.clamp_scroll();
        // Update desired_col to the new position.
        self.desired_col = self.byte_to_vis_col(self.cursor);
        self.ensure_cursor_visible();
        false
    }

    pub fn move_down(&mut self) -> bool {
        self.rebuild_visual_lines(self.nav_width());
        if self.vis_lines.is_empty() {
            return false;
        }
        let (vl, _) = byte_to_vis(&self.vis_lines, &self.text, self.cursor);
        if vl >= self.vis_lines.len().saturating_sub(1) {
            // Already at last visual line — signal caller to handle history.
            return true;
        }
        let target_line = vl + 1;
        let clamped_col = self
            .desired_col
            .min(self.vis_lines[target_line].len().saturating_sub(1));
        if self.vis_lines[target_line].is_empty() {
            // Empty line: position cursor at the start of this line.
            self.cursor = line_start_byte_for_width(&self.text, self.nav_width(), target_line);
        } else {
            self.cursor = self.vis_lines[target_line]
                .get(clamped_col)
                .copied()
                .unwrap_or(0);
            if self.cursor > self.text.len() {
                self.cursor = self.vis_lines[target_line].last().copied().unwrap_or(0);
                self.cursor += self.text[self.cursor..]
                    .chars()
                    .next()
                    .map(|c| c.len_utf8())
                    .unwrap_or(0);
            }
        }
        self.clamp_scroll();
        // Update desired_col to the new position.
        self.desired_col = self.byte_to_vis_col(self.cursor);
        self.ensure_cursor_visible();
        false
    }

    pub fn move_left(&mut self) {
        self.rebuild_visual_lines(self.nav_width());
        if self.cursor == 0 {
            return;
        }
        // If cursor is at start of a visual line (excl. the very first),
        // wrap to end of previous visual line.
        let (vl, vc) = byte_to_vis(&self.vis_lines, &self.text, self.cursor);
        if vc == 0 && vl > 0 {
            // Go to the last byte offset of the previous visual line.
            let prev = &self.vis_lines[vl - 1];
            if let Some(&last_byte) = prev.last() {
                let orig = self.cursor;
                self.cursor = last_byte;
                // Advance past that character so we're AFTER it.
                if let Some(ch) = self.text[last_byte..].chars().next() {
                    self.cursor = last_byte + ch.len_utf8();
                }
                // Wrapped visual line with no gap (e.g. "Hello"+"World"
                // at width=5): end of prev line and start of current line
                // share the same byte offset. Move onto the last char.
                if self.cursor == orig {
                    self.cursor = last_byte;
                }
            }
        } else if let Some(prev) = self.text[..self.cursor].chars().last() {
            self.cursor -= prev.len_utf8();
        }
        self.desired_col = self.byte_to_vis_col(self.cursor);
    }

    pub fn move_right(&mut self) {
        self.rebuild_visual_lines(self.nav_width());
        if self.cursor >= self.text.len() {
            return;
        }
        let (vl, vc) = byte_to_vis(&self.vis_lines, &self.text, self.cursor);
        if vc >= self.vis_lines[vl].len().saturating_sub(1) && vl + 1 < self.vis_lines.len() {
            // At end of visual line (not the last) — wrap to start of next.
            if let Some(&next_byte) = self.vis_lines[vl + 1].first() {
                self.cursor = next_byte;
            }
        } else if let Some(next) = self.text[self.cursor..].chars().next() {
            self.cursor += next.len_utf8();
        }
        self.desired_col = self.byte_to_vis_col(self.cursor);
    }

    pub fn move_word_left(&mut self) {
        // Skip delimiters leftwards, then skip word chars leftwards.
        while self.cursor > 0 {
            if let Some(prev) = self.text[..self.cursor].chars().last() {
                if is_word_char(prev) {
                    break;
                }
                self.cursor -= prev.len_utf8();
            }
        }
        while self.cursor > 0 {
            if let Some(prev) = self.text[..self.cursor].chars().last() {
                if !is_word_char(prev) {
                    break;
                }
                self.cursor -= prev.len_utf8();
            }
        }
        self.desired_col = self.byte_to_vis_col(self.cursor);
    }

    pub fn move_word_right(&mut self) {
        // Skip delimiters rightwards, then skip word chars rightwards.
        while self.cursor < self.text.len() {
            if let Some(next) = self.text[self.cursor..].chars().next() {
                if is_word_char(next) {
                    break;
                }
                self.cursor += next.len_utf8();
            }
        }
        while self.cursor < self.text.len() {
            if let Some(next) = self.text[self.cursor..].chars().next() {
                if !is_word_char(next) {
                    break;
                }
                self.cursor += next.len_utf8();
            }
        }
        self.desired_col = self.byte_to_vis_col(self.cursor);
    }

    pub fn move_line_start(&mut self) {
        // Move to the start of the current visual line.
        self.rebuild_visual_lines(self.nav_width());
        let (vl, _) = byte_to_vis(&self.vis_lines, &self.text, self.cursor);
        if vl < self.vis_lines.len() {
            if self.vis_lines[vl].is_empty() {
                self.cursor = line_start_byte_for_width(&self.text, self.nav_width(), vl);
            } else {
                self.cursor = *self.vis_lines[vl].first().unwrap_or(&0);
            }
        }
        self.desired_col = 0;
    }

    pub fn move_line_end(&mut self) {
        // Move to the end of the current visual line.
        self.rebuild_visual_lines(self.nav_width());
        let (vl, _) = byte_to_vis(&self.vis_lines, &self.text, self.cursor);
        if vl < self.vis_lines.len() {
            if self.vis_lines[vl].is_empty() {
                // Empty line: start and end are the same position.
                self.cursor = line_start_byte_for_width(&self.text, self.nav_width(), vl);
                self.desired_col = 0;
            } else if let Some(&last_byte) = self.vis_lines[vl].last() {
                // After the last character on the line.
                if let Some(ch) = self.text[last_byte..].chars().next() {
                    self.cursor = last_byte + ch.len_utf8();
                } else {
                    self.cursor = last_byte;
                }
                self.desired_col = self.vis_lines[vl].len();
            }
        }
    }

    /// Move cursor up one page. Returns `true` if at first visual line.
    pub fn move_page_up(&mut self) -> bool {
        self.rebuild_visual_lines(self.nav_width());
        let height = self
            .last_inner_area
            .map(|a| a.height as usize)
            .unwrap_or(10)
            .max(1);
        for _ in 0..height {
            if self.move_up() {
                return true;
            }
        }
        false
    }

    pub fn move_page_down(&mut self) -> bool {
        self.rebuild_visual_lines(self.nav_width());
        let height = self
            .last_inner_area
            .map(|a| a.height as usize)
            .unwrap_or(10)
            .max(1);
        for _ in 0..height {
            if self.move_down() {
                return true;
            }
        }
        false
    }

    fn move_text_start(&mut self) {
        self.cursor = 0;
        self.desired_col = 0;
        self.scroll = 0;
    }

    fn move_text_end(&mut self) {
        self.cursor = self.text.len();
        self.desired_col = self.byte_to_vis_col(self.cursor);
        self.clamp_scroll();
    }

    // ------------------------------------------------------------------
    // Submit & History
    // ------------------------------------------------------------------

    fn submit(&mut self) -> Option<String> {
        let text = self.text.trim().to_string();
        self.text.clear();
        self.cursor = 0;
        self.desired_col = 0;
        self.scroll = 0;
        self.autocomplete = None;
        self.cache_dirty = true;
        if text.is_empty() {
            return None;
        }
        self.history.push(text.clone());
        self.history_idx = self.history.len();
        Some(text)
    }

    fn history_up(&mut self) {
        if self.history.is_empty() {
            return;
        }
        if self.history_idx == self.history.len() {
            self.saved_text = self.text.clone();
        }
        if self.history_idx > 0 {
            self.history_idx -= 1;
            self.text = self.history[self.history_idx].clone();
            self.cursor = self.text.len();
            self.desired_col = 0;
            self.scroll = 0;
            self.cache_dirty = true;
        }
    }

    fn history_down(&mut self) {
        if self.history.is_empty() {
            return;
        }
        if self.history_idx < self.history.len() - 1 {
            self.history_idx += 1;
            self.text = self.history[self.history_idx].clone();
            self.cursor = self.text.len();
            self.desired_col = 0;
            self.scroll = 0;
            self.cache_dirty = true;
        } else if self.history_idx == self.history.len() - 1 {
            self.history_idx += 1;
            self.text = self.saved_text.clone();
            self.cursor = self.text.len();
            self.desired_col = 0;
            self.scroll = 0;
            self.cache_dirty = true;
        }
    }

    // ------------------------------------------------------------------
    // Autocomplete
    // ------------------------------------------------------------------

    fn start_autocomplete(&mut self, trigger: char) {
        let prefix_start = self.cursor;
        self.autocomplete = Some(AutocompleteState {
            items: Vec::new(),
            selected: 0,
            prefix_start,
            query: String::new(),
            trigger,
        });
        self.update_autocomplete_items();
    }

    fn update_autocomplete_items(&mut self) {
        let Some(ref mut ac) = self.autocomplete else {
            return;
        };

        // Query = text between prefix_start and cursor.
        if self.cursor >= ac.prefix_start {
            ac.query = self.text[ac.prefix_start..self.cursor].to_string();
        } else {
            ac.query.clear();
        }

        ac.items = match ac.trigger {
            '@' => file_mention_matches(&self.working_dir, &ac.query),
            '/' => fuzzy_command_matches(&self.slash_commands, &ac.query),
            _ => Vec::new(),
        };

        ac.selected = 0;
    }

    fn accept_autocomplete(&mut self) {
        let _trigger = self
            .autocomplete
            .as_ref()
            .map(|ac| ac.trigger)
            .unwrap_or('/');
        let start = self
            .autocomplete
            .as_ref()
            .map(|ac| ac.prefix_start)
            .unwrap_or(self.cursor);

        let Some(ref ac) = self.autocomplete else {
            return;
        };
        let Some(item) = ac.items.get(ac.selected).cloned() else {
            return;
        };
        let is_dir = item.ends_with('/');

        // Replace query text with the selected item.
        let end = self.cursor;
        self.text.replace_range(start..end, &item);
        self.cursor = start + item.len();
        self.cache_dirty = true;

        if is_dir {
            // Keep autocomplete open so user can keep navigating.
            self.autocomplete.as_mut().unwrap().prefix_start = start;
            self.autocomplete.as_mut().unwrap().query.clear();
            self.update_autocomplete_items();
        } else {
            // Insert space after file, dismiss autocomplete.
            self.text.insert(self.cursor, ' ');
            self.cursor += 1;
            self.cache_dirty = true;
            self.autocomplete = None;
        }
    }

    pub fn autocomplete_items(&self) -> Vec<String> {
        self.autocomplete
            .as_ref()
            .map(|ac| ac.items.clone())
            .unwrap_or_default()
    }

    pub fn autocomplete_selected(&self) -> usize {
        self.autocomplete
            .as_ref()
            .map(|ac| ac.selected)
            .unwrap_or(0)
    }

    pub fn autocomplete_active(&self) -> bool {
        self.autocomplete.is_some()
    }

    fn dismiss_autocomplete(&mut self) {
        self.autocomplete = None;
    }

    fn select_next(&mut self) {
        if let Some(ref mut ac) = self.autocomplete
            && !ac.items.is_empty()
        {
            ac.selected = (ac.selected + 1) % ac.items.len();
        }
    }

    fn select_prev(&mut self) {
        if let Some(ref mut ac) = self.autocomplete {
            ac.selected = ac.selected.saturating_sub(1);
        }
    }

    fn refresh_slash_autocomplete(&mut self) {
        let at_start = self.text.starts_with('/');
        if !at_start || self.cursor == 0 {
            self.dismiss_autocomplete();
            return;
        }

        let upto_cursor = &self.text[..self.cursor];
        if !upto_cursor.starts_with('/') {
            self.dismiss_autocomplete();
            return;
        }

        let in_first_token = !upto_cursor.contains(' ') && !upto_cursor.contains('\n');
        if !in_first_token {
            self.dismiss_autocomplete();
            return;
        }

        let prefix_start = 1;
        if self.autocomplete.is_none() {
            self.autocomplete = Some(AutocompleteState {
                items: Vec::new(),
                selected: 0,
                prefix_start,
                query: String::new(),
                trigger: '/',
            });
        }

        if let Some(ref mut ac) = self.autocomplete {
            ac.prefix_start = prefix_start;
            ac.trigger = '/';
        }
        self.update_autocomplete_items();
    }
}

impl Component for Editor {
    fn render(&mut self, area: Rect, frame: &mut Frame) {
        let input_border = if self.focused {
            self.theme.accent
        } else {
            self.theme.warning
        };
        let block = Block::default()
            .borders(Borders::TOP | Borders::BOTTOM)
            .border_style(Style::default().fg(input_border));

        let inner = block.inner(area);
        if inner.width == 0 || inner.height == 0 {
            let para = Paragraph::new("").block(block);
            frame.render_widget(para, area);
            return;
        }
        let width = inner.width as usize;
        let height = inner.height as usize;
        if width == 0 || height == 0 {
            let para = Paragraph::new("").block(block);
            frame.render_widget(para, area);
            return;
        }

        self.rebuild_visual_lines(width);
        let total_lines = self.vis_lines.len();

        // Find cursor visual line and auto-scroll.
        let (cursor_vl, _) = byte_to_vis(&self.vis_lines, &self.text, self.cursor);
        if cursor_vl < self.scroll {
            self.scroll = cursor_vl;
        } else if cursor_vl >= self.scroll + height {
            self.scroll = cursor_vl.saturating_add(1).saturating_sub(height);
        }
        self.clamp_scroll();

        // Build visible text lines.
        let end = (self.scroll + height).min(total_lines);
        let mut visible_lines: Vec<Line> = Vec::with_capacity(end.saturating_sub(self.scroll));

        for line_idx in self.scroll..end {
            let chars_in_line = &self.vis_lines[line_idx];
            let mut spans: Vec<Span> = Vec::with_capacity(chars_in_line.len().max(1));
            for &byte_offset in chars_in_line {
                let c = self.text[byte_offset..].chars().next().unwrap_or(' ');
                let style = if self.focused && byte_offset == self.cursor {
                    Style::default().fg(self.theme.bg).bg(self.theme.accent)
                } else {
                    Style::default()
                };
                spans.push(Span::styled(c.to_string(), style));
            }
            // Block cursor at end of current line.
            if self.focused && cursor_vl == line_idx {
                let (_, cursor_col) = byte_to_vis(&self.vis_lines, &self.text, self.cursor);
                if cursor_col == spans.len() {
                    spans.push(Span::styled(
                        " ",
                        Style::default().fg(self.theme.bg).bg(self.theme.accent),
                    ));
                }
            }
            // Block cursor on empty line.
            if spans.is_empty() && self.focused && cursor_vl == line_idx {
                spans.push(Span::styled(
                    " ",
                    Style::default().fg(self.theme.bg).bg(self.theme.accent),
                ));
            }
            visible_lines.push(Line::from(spans));
        }

        frame.render_widget(Clear, area);
        frame.render_widget(block, area);
        frame.render_widget(Paragraph::new(Text::from(visible_lines)), inner);

        if self.focused {
            let (cursor_line, cursor_col) = byte_to_vis(&self.vis_lines, &self.text, self.cursor);
            if cursor_line >= self.scroll && cursor_line < self.scroll + height {
                let x = inner.x.saturating_add(cursor_col as u16);
                let y = inner
                    .y
                    .saturating_add(cursor_line.saturating_sub(self.scroll) as u16);
                frame.set_cursor_position((x, y));
            }
        }

        self.last_inner_area = Some(inner);
    }

    fn handle_event(&mut self, event: &Event) -> Option<Action> {
        if !self.focused {
            return None;
        }
        if let Event::Paste(pasted) = event {
            self.text.insert_str(self.cursor, pasted);
            self.cursor += pasted.len();
            self.after_mutate();
            self.refresh_slash_autocomplete();
            return None;
        }
        if let Event::Mouse(mouse) = event {
            match mouse.kind {
                MouseEventKind::Down(MouseButton::Left) => {
                    // Click-to-place cursor at the mouse position.
                    if let Some((line, col)) = self.mouse_to_cell(mouse.column, mouse.row) {
                        let vis_line = self.scroll + line;
                        if vis_line < self.vis_lines.len() {
                            let chars_on_line = &self.vis_lines[vis_line];
                            let clamped_col = col.min(chars_on_line.len().saturating_sub(1));
                            if let Some(&byte_offset) = chars_on_line.get(clamped_col) {
                                self.cursor = byte_offset;
                            } else {
                                // Clicked past end of line → go to end.
                                if let Some(&last_byte) = chars_on_line.last()
                                    && let Some(ch) = self.text[last_byte..].chars().next()
                                {
                                    self.cursor = last_byte + ch.len_utf8();
                                }
                            }
                            self.desired_col = self.byte_to_vis_col(self.cursor);
                        }
                    }
                }
                MouseEventKind::ScrollUp => {
                    self.scroll = self.scroll.saturating_sub(3);
                }
                MouseEventKind::ScrollDown => {
                    self.scroll = self.scroll.saturating_add(3);
                    self.clamp_scroll();
                }
                _ => {}
            }
            return None;
        }
        let Event::Key(key) = event else {
            return None;
        };

        // If autocomplete is active, handle its keys first.
        if self.autocomplete.is_some() {
            if is_newline_key(key, self.enter_behavior) {
                self.insert_char('\n');
                return None;
            }
            match key {
                crossterm::event::KeyEvent {
                    code: KeyCode::Tab, ..
                } => {
                    self.accept_autocomplete();
                    return None;
                }
                crossterm::event::KeyEvent {
                    code: KeyCode::Esc, ..
                } => {
                    self.dismiss_autocomplete();
                    return None;
                }
                crossterm::event::KeyEvent {
                    code: KeyCode::Up, ..
                } => {
                    self.select_prev();
                    return None;
                }
                crossterm::event::KeyEvent {
                    code: KeyCode::Down,
                    ..
                } => {
                    self.select_next();
                    return None;
                }
                crossterm::event::KeyEvent {
                    code: KeyCode::Enter,
                    ..
                } => {
                    if self
                        .autocomplete
                        .as_ref()
                        .is_some_and(|ac| !ac.items.is_empty())
                    {
                        self.accept_autocomplete();
                        return None;
                    }
                    // No autocomplete items — dismiss and fall through to send.
                    self.dismiss_autocomplete();
                }
                crossterm::event::KeyEvent {
                    code: KeyCode::Char(c),
                    ..
                } => {
                    self.insert_char(*c);
                    let slash_mode = self
                        .autocomplete
                        .as_ref()
                        .map(|ac| ac.trigger == '/')
                        .unwrap_or(false);
                    if slash_mode {
                        self.refresh_slash_autocomplete();
                    } else {
                        self.update_autocomplete_items();
                    }
                    return None;
                }
                crossterm::event::KeyEvent {
                    code: KeyCode::Backspace,
                    ..
                } => {
                    if let Some(ref ac) = self.autocomplete
                        && self.cursor <= ac.prefix_start
                    {
                        self.delete_before();
                        self.dismiss_autocomplete();
                        return None;
                    }
                    self.delete_before();
                    self.update_autocomplete_items();
                    return None;
                }
                _ => {}
            }
        }

        if is_enter_send(key, self.enter_behavior) {
            if let Some(text) = self.submit() {
                return Some(Action::SendMessage(text));
            }
            return None;
        }
        if is_follow_up_key(key) {
            if let Some(text) = self.submit() {
                return Some(Action::FollowUpMessage(text));
            }
            return None;
        }
        if is_newline_key(key, self.enter_behavior) {
            self.insert_char('\n');
            return None;
        }

        match key {
            // ── Autocomplete triggers ──
            crossterm::event::KeyEvent {
                code: KeyCode::Char('@'),
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                self.insert_char('@');
                self.start_autocomplete('@');
                return None;
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Char('/'),
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                self.insert_char('/');
                if self.text.trim().is_empty()
                    || self.text.ends_with(' ')
                    || self.text.ends_with('\n')
                    || self.text == "/"
                {
                    self.start_autocomplete('/');
                }
                return None;
            }
            // ── Regular character insertion ──
            crossterm::event::KeyEvent {
                code: KeyCode::Char(c),
                modifiers,
                ..
            } if modifiers.is_empty() || *modifiers == KeyModifiers::SHIFT => {
                self.insert_char(*c);
                self.refresh_slash_autocomplete();
            }
            // ── Tab → 2 spaces ──
            crossterm::event::KeyEvent {
                code: KeyCode::Tab, ..
            } => {
                self.text.insert_str(self.cursor, "  ");
                self.cursor += 2;
                self.after_mutate();
            }
            // ── Backspace / Delete ──
            crossterm::event::KeyEvent {
                code: KeyCode::Backspace,
                ..
            } => {
                self.delete_before();
                self.refresh_slash_autocomplete();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Delete,
                ..
            } => {
                self.delete_after();
                self.refresh_slash_autocomplete();
            }
            // ── Cursor navigation ──
            crossterm::event::KeyEvent {
                code: KeyCode::Left,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SUPER) => {
                self.move_text_start();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Right,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SUPER) => {
                self.move_text_end();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Left,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                self.move_word_left();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                self.move_word_right();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Left,
                ..
            } => {
                self.move_left();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Right,
                ..
            } => {
                self.move_right();
            }
            // ── Vertical navigation: Up/Down moves cursor ──
            crossterm::event::KeyEvent {
                code: KeyCode::Up,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SUPER) => {
                self.move_text_start();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Down,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SUPER) => {
                self.move_text_end();
            }
            // ── History browsing: Alt+Up / Alt+Down (before bare Up/Down) ──
            crossterm::event::KeyEvent {
                code: KeyCode::Up,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                self.history_up();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                self.history_down();
            }
            // ── Vertical navigation: Up/Down moves cursor ──
            #[allow(clippy::collapsible_match)]
            crossterm::event::KeyEvent {
                code: KeyCode::Up, ..
            } => {
                if self.move_up() {
                    // At first visual line → browse history.
                    self.history_up();
                }
            }
            // ── Down: moves cursor down, history at last line ──
            #[allow(clippy::collapsible_match)]
            crossterm::event::KeyEvent {
                code: KeyCode::Down,
                ..
            } => {
                if self.move_down() {
                    // At last visual line → browse history forward.
                    self.history_down();
                }
            }
            // ── Page Up / Page Down ──
            #[allow(clippy::collapsible_match)]
            crossterm::event::KeyEvent {
                code: KeyCode::PageUp,
                ..
            } => {
                if self.move_page_up() {
                    self.history_up();
                }
            }
            #[allow(clippy::collapsible_match)]
            crossterm::event::KeyEvent {
                code: KeyCode::PageDown,
                ..
            } => {
                if self.move_page_down() {
                    self.history_down();
                }
            }
            // ── Line start/end ──
            crossterm::event::KeyEvent {
                code: KeyCode::Home,
                ..
            } => {
                self.move_line_start();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::End, ..
            } => {
                self.move_line_end();
            }
            _ => {}
        }
        None
    }

    fn is_focused(&self) -> bool {
        self.focused
    }

    fn focus(&mut self, focused: bool) {
        self.focused = focused;
    }
}

impl Editor {
    pub fn mouse_to_cell(&self, col: u16, row: u16) -> Option<(usize, usize)> {
        let area = self.last_inner_area?;
        if col < area.x || row < area.y || col >= area.x + area.width || row >= area.y + area.height
        {
            return None;
        }
        let line = (row - area.y) as usize;
        let col = (col - area.x) as usize;
        Some((line, col))
    }
}

// ---------------------------------------------------------------------------
// Visual-line helpers
// ---------------------------------------------------------------------------

/// Build visual line layout: for each visual line, a `Vec<usize>` of byte
/// offsets of each character in that line.
pub fn build_vis_lines(text: &str, width: usize) -> Vec<Vec<usize>> {
    if width == 0 {
        return vec![vec![]];
    }
    let mut lines: Vec<Vec<usize>> = Vec::new();
    let mut current: Vec<usize> = Vec::new();
    let mut col = 0usize;

    for (byte_idx, ch) in text.char_indices() {
        if ch == '\n' {
            lines.push(std::mem::take(&mut current));
            col = 0;
            continue;
        }
        if col >= width {
            lines.push(std::mem::take(&mut current));
            col = 0;
        }
        current.push(byte_idx);
        col += 1;
    }
    // Always push the last (possibly empty) line.
    lines.push(current);
    lines
}

/// Convert byte offset to (vis_line, vis_col) using the cached layout.
pub fn byte_to_vis(vis_lines: &[Vec<usize>], text: &str, byte: usize) -> (usize, usize) {
    if vis_lines.is_empty() {
        return (0, 0);
    }

    let starts = build_vis_line_starts_from_layout(vis_lines, text);
    let text_len = text.len();
    if byte >= text_len {
        let last_idx = vis_lines.len() - 1;
        return (last_idx, vis_lines[last_idx].len());
    }

    let line_idx = match starts.binary_search(&byte) {
        Ok(idx) => idx,
        Err(ins) => ins.saturating_sub(1),
    }
    .min(vis_lines.len().saturating_sub(1));

    let line = &vis_lines[line_idx];
    for (col_idx, &b) in line.iter().enumerate() {
        if b == byte {
            return (line_idx, col_idx);
        }
    }

    (line_idx, line.len())
}

fn build_vis_line_starts_from_layout(vis_lines: &[Vec<usize>], text: &str) -> Vec<usize> {
    if vis_lines.is_empty() {
        return Vec::new();
    }

    let mut starts = Vec::with_capacity(vis_lines.len());
    let mut pos = 0usize;

    for (i, line) in vis_lines.iter().enumerate() {
        starts.push(pos.min(text.len()));

        // Advance by this visual line's character count.
        for _ in 0..line.len() {
            if pos >= text.len() {
                break;
            }
            let mut iter = text[pos..].chars();
            if let Some(ch) = iter.next() {
                pos += ch.len_utf8();
            } else {
                break;
            }
        }

        // Between visual lines: consume newline boundary if present.
        if i + 1 < vis_lines.len() && pos < text.len() && text.as_bytes()[pos] == b'\n' {
            pos += 1;
        }
    }

    starts
}

/// Returns true for code-like word characters.
fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Compute the byte offset for the start of a visual line at a given width.
///
/// This is robust for consecutive empty lines because it follows the same
/// wrapping/newline rules as `build_vis_lines` while tracking explicit starts.
fn line_start_byte_for_width(text: &str, width: usize, target_line: usize) -> usize {
    if target_line == 0 {
        return 0;
    }
    if width == 0 {
        return text.len();
    }

    let mut current_line = 0usize;
    let mut col = 0usize;

    for (byte_idx, ch) in text.char_indices() {
        if col == 0 && current_line == target_line {
            return byte_idx;
        }

        if ch == '\n' {
            current_line += 1;
            col = 0;
            continue;
        }

        if col >= width {
            current_line += 1;
            col = 0;
            if current_line == target_line {
                return byte_idx;
            }
        }

        col += 1;
    }

    // After consuming text, there is always one trailing visual line.
    if current_line + 1 == target_line {
        return text.len();
    }

    text.len()
}

/// Convert (vis_line, vis_col) to byte offset.
pub fn vis_to_byte(vis_lines: &[Vec<usize>], text_len: usize, line: usize, col: usize) -> usize {
    let Some(chars) = vis_lines.get(line) else {
        return text_len;
    };
    if col >= chars.len() {
        return text_len;
    }
    chars[col]
}

// ---------------------------------------------------------------------------
// Fuzzy file matching
// ---------------------------------------------------------------------------

/// Return file mention matches: recursive, relative paths, fuzzy-ranked.
/// Respects .gitignore, but expands into gitignored directories when the
/// user types an exact directory path (e.g. `docs/`).
pub fn file_mention_matches(base_dir: &std::path::Path, query: &str) -> Vec<String> {
    let mut entries = git_tracked_and_untracked_files(base_dir)
        .unwrap_or_else(|| recursive_file_paths(base_dir, query.starts_with('.')));
    entries.sort();
    entries.dedup();

    let trimmed = query.trim();
    if trimmed.is_empty() {
        return entries.into_iter().take(50).collect();
    }

    let mut filtered = fuzzy_filter(&entries, trimmed, |s| s)
        .into_iter()
        .cloned()
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        filtered = entries
            .into_iter()
            .filter(|path| path.contains(trimmed))
            .collect();
    }

    // If fuzzy search found nothing and the query resolves to a real
    // directory on disk, list that directory's contents directly.
    // This lets users navigate into gitignored directories.
    if filtered.is_empty() {
        let dir_candidate = trimmed.trim_end_matches('/');
        let abs_dir = base_dir.join(dir_candidate);
        if abs_dir.is_dir() {
            let include_hidden = trimmed.starts_with('.');
            let mut dir_entries = Vec::new();
            collect_file_paths(base_dir, &abs_dir, include_hidden, &mut dir_entries);
            dir_entries.sort();
            dir_entries.dedup();
            return dir_entries.into_iter().take(50).collect();
        }
    }

    filtered.into_iter().take(50).collect()
}

fn git_tracked_and_untracked_files(base_dir: &std::path::Path) -> Option<Vec<String>> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(base_dir)
        .arg("ls-files")
        .arg("--cached")
        .arg("--others")
        .arg("--exclude-standard")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let files = String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| line.replace('\\', "/"))
        .collect::<Vec<_>>();
    if files.is_empty() { None } else { Some(files) }
}

fn recursive_file_paths(base_dir: &std::path::Path, include_hidden: bool) -> Vec<String> {
    let mut out = Vec::new();
    collect_file_paths(base_dir, base_dir, include_hidden, &mut out);
    out
}

fn collect_file_paths(
    base_dir: &std::path::Path,
    current_dir: &std::path::Path,
    include_hidden: bool,
    out: &mut Vec<String>,
) {
    let Ok(read_dir) = std::fs::read_dir(current_dir) else {
        return;
    };

    for entry in read_dir.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if !include_hidden && name.starts_with('.') {
            continue;
        }
        if matches!(
            name.as_str(),
            "target" | "node_modules" | ".git" | ".theta" | "dist" | "build"
        ) {
            continue;
        }

        let path = entry.path();
        let Ok(file_type) = entry.file_type() else {
            continue;
        };

        if file_type.is_dir() {
            collect_file_paths(base_dir, &path, include_hidden, out);
        } else if file_type.is_file()
            && let Ok(relative) = path.strip_prefix(base_dir)
        {
            out.push(relative.to_string_lossy().replace('\\', "/"));
        }
    }
}

fn fuzzy_command_matches(commands: &[String], query: &str) -> Vec<String> {
    if let Some(skill_query) = query.strip_prefix("skill:") {
        let skill_commands: Vec<&String> = commands
            .iter()
            .filter(|c| c.starts_with("skill:"))
            .collect();
        let filtered = fuzzy_filter(&skill_commands, skill_query, |s| &s[6..]);
        return filtered.into_iter().take(10).cloned().cloned().collect();
    }

    let mut out: Vec<String> = Vec::new();

    // Built-in commands first.
    let builtins: Vec<&String> = commands
        .iter()
        .filter(|c| !c.starts_with("skill:"))
        .collect();
    for cmd in fuzzy_filter(&builtins, query, |s| s) {
        out.push(cmd.to_string());
        if out.len() >= 10 {
            return out;
        }
    }

    // Then skill matches (proactive: match /git-... → /skill:git-...).
    let skill_commands: Vec<&String> = commands
        .iter()
        .filter(|c| c.starts_with("skill:"))
        .collect();
    for cmd in fuzzy_filter(&skill_commands, query, |s| &s[6..]) {
        out.push(cmd.to_string());
        if out.len() >= 10 {
            break;
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
