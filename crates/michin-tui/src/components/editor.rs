//! Input editor — multiline text input backed by `tui-textarea` with
//! inline fuzzy autocomplete for @ files and / commands, paste handling.

use crossterm::event::{Event, KeyCode, KeyModifiers, MouseButton, MouseEventKind};
use fff_search::shared::SharedFilePicker;
use fff_search::{FileSearchConfig, FuzzySearchOptions, PaginationArgs, QueryParser};
use ratatui::{
    Frame,
    layout::Rect,
    style::Style,
    widgets::{Block, Borders},
};
use ratatui_textarea::{CursorMove, Input, Key, Scrolling, TextArea, WrapMode};
use std::path::{Path, PathBuf};

use crate::components::fuzzy::fuzzy_filter;
use crate::components::{Action, Component};
use crate::keybinding::{EnterBehavior, is_enter_send, is_follow_up_key, is_newline_key};
use crate::theme::Theme;

/// Inline autocomplete state.
#[derive(Debug, Clone)]
struct AutocompleteState {
    items: Vec<String>,
    selected: usize,
    /// Visual (row, col) where @ or / was typed.
    prefix_row: usize,
    prefix_col: usize,
    /// Filter query between trigger and cursor.
    query: String,
    /// '@' or '/'.
    trigger: char,
}

/// Lazy file index cache. Populated once on first `@` trigger,
/// then filtered in-memory per keystroke. Rebuilds when project
/// files changed — detected via `.git/index` mtime (git repos)
/// or root dir mtime (non-git fallback).
///
/// When `fff_picker` is set, this cache is unused — FFF handles
/// file discovery.
struct FileIndex {
    entries: Vec<String>,
    cache_key: Option<FileIndexKey>,
}

/// Tracks the on-disk state used to decide if the index is stale.
struct FileIndexKey {
    /// Mtime of `.git/index` or working dir root (non-git fallback).
    stamp: std::time::SystemTime,
}

impl FileIndex {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            cache_key: None,
        }
    }

    fn ensure_built(&mut self, base_dir: &Path) {
        let current = index_stamp(base_dir);
        let fresh = self
            .cache_key
            .as_ref()
            .is_some_and(|key| key.stamp == current);
        if fresh && !self.entries.is_empty() {
            return;
        }
        self.entries = build_file_index(base_dir);
        self.cache_key = Some(FileIndexKey { stamp: current });
    }

    #[allow(dead_code)]
    fn invalidate(&mut self) {
        self.cache_key = None;
        self.entries.clear();
    }
}

/// Return a SystemTime that changes when the project's file listing changes.
fn index_stamp(base_dir: &Path) -> std::time::SystemTime {
    // Git repos: `.git/index` mtime updates on any index change.
    let git_index = base_dir.join(".git").join("index");
    if git_index.exists() {
        return std::fs::metadata(&git_index)
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
    }
    // Non-git fallback: root dir mtime (platform-dependent).
    std::fs::metadata(base_dir)
        .and_then(|m| m.modified())
        .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
}

/// Build file list from `git ls-files` or recursive `read_dir` fallback.
fn build_file_index(base_dir: &Path) -> Vec<String> {
    let mut entries = git_tracked_and_untracked_files(base_dir)
        .unwrap_or_else(|| recursive_file_paths(base_dir, false));
    entries.sort();
    entries.dedup();
    entries
}

/// Multiline text editor with visual-line cursor navigation.
///
/// Backed by `tui-textarea` for all text editing, cursor movement,
/// undo/redo, kill/yank, and line navigation. The Editor layer adds
/// MichiN-specific features: autocomplete, history, enter behavior,
/// and app-global keybindings.
pub struct Editor {
    textarea: TextArea<'static>,
    focused: bool,
    theme: Theme,
    history: Vec<String>,
    /// Position in history ring.
    history_idx: usize,
    /// Stash for history restore.
    saved_text: String,
    autocomplete: Option<AutocompleteState>,
    file_index: FileIndex,
    /// Optional FFF shared picker for frecency-aware autocomplete.
    /// When set, @-mention uses FFF fuzzy_search instead of
    /// git ls-files + nucleo fuzzy filter.
    fff_picker: Option<SharedFilePicker>,
    working_dir: PathBuf,
    slash_commands: Vec<String>,
    enter_behavior: EnterBehavior,
    /// For popup positioning + mouse hit-testing.
    pub last_inner_area: Option<Rect>,
}

impl Editor {
    pub fn new(
        theme: Theme,
        working_dir: PathBuf,
        slash_commands: Vec<String>,
        enter_behavior: String,
        fff_picker: Option<SharedFilePicker>,
    ) -> Self {
        let mut textarea = TextArea::default();
        textarea.set_tab_length(2);
        textarea.set_max_histories(100);
        // tui-textarea's built-in visual cursor (REVERSED block) is
        // the only cursor — no terminal cursor set via frame.
        // Remove the default underline on cursor line.
        textarea.set_cursor_line_style(Style::default());
        textarea.set_wrap_mode(WrapMode::WordOrGlyph);

        Self {
            textarea,
            focused: false,
            theme,
            history: Vec::new(),
            history_idx: 0,
            saved_text: String::new(),
            autocomplete: None,
            file_index: FileIndex::new(),
            fff_picker,
            working_dir,
            slash_commands,
            enter_behavior: EnterBehavior::parse(&enter_behavior),
            last_inner_area: None,
        }
    }

    pub fn set_text(&mut self, text: &str) {
        let lines: Vec<String> = text.lines().map(|s| s.to_string()).collect();
        let lines = if lines.is_empty() {
            vec![String::new()]
        } else {
            lines
        };
        self.textarea = TextArea::new(lines);
        self.textarea.set_tab_length(2);
        self.textarea.set_max_histories(100);
        self.textarea.set_wrap_mode(WrapMode::WordOrGlyph);
        self.apply_theme_styles();
        // Place cursor at end.
        self.textarea.move_cursor(CursorMove::Bottom);
        self.textarea.move_cursor(CursorMove::End);
    }

    pub fn text(&self) -> String {
        self.textarea.lines().join("\n")
    }

    pub fn desired_height(&self, width: usize, max_height: u16) -> u16 {
        let available = width.max(1);
        let display_lines: usize = self
            .textarea
            .lines()
            .iter()
            .map(|line| {
                let line_width: usize = line.chars().map(unicode_width).sum();
                if line_width == 0 {
                    1
                } else {
                    // Ceiling division for word-wrapped lines.
                    line_width.div_ceil(available)
                }
            })
            .sum();
        // Borders add 2 lines, minimum 3 rows.
        (display_lines as u16)
            .saturating_add(2)
            .clamp(3, max_height.max(3))
    }

    /// Return cursor position as (row, col). For tests.
    pub fn cursor_position(&self) -> (usize, usize) {
        let c = self.textarea.cursor();
        (c.0, c.1)
    }

    /// Return number of lines. For tests.
    pub fn line_count(&self) -> usize {
        self.textarea.lines().len()
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
        self.apply_theme_styles();
    }

    fn apply_theme_styles(&mut self) {
        let text_style = Style::default().fg(self.theme.fg);
        // Reverse-video block cursor using theme accent.
        let cursor_style = Style::default().fg(self.theme.bg).bg(self.theme.accent);
        self.textarea.set_style(text_style);
        self.textarea.set_cursor_style(cursor_style);
        // No underline on cursor line.
        self.textarea.set_cursor_line_style(Style::default());
    }

    /// Insert at cursor. Used by path picker.
    pub fn insert_at_cursor(&mut self, s: &str) {
        for c in s.chars() {
            self.textarea.insert_char(c);
        }
    }

    /// Delete the last character.
    pub fn delete_last_char(&mut self) {
        self.textarea.delete_char();
    }

    // ------------------------------------------------------------------
    // Submit & History
    // ------------------------------------------------------------------

    fn submit(&mut self) -> Option<String> {
        let text = self.textarea.lines().join("\n");
        let text = text.trim().to_string();
        // Reset textarea.
        self.textarea = TextArea::default();
        self.textarea.set_tab_length(2);
        self.textarea.set_max_histories(100);
        self.textarea.set_wrap_mode(WrapMode::WordOrGlyph);
        self.apply_theme_styles();
        self.autocomplete = None;
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
            self.saved_text = self.textarea.lines().join("\n");
        }
        if self.history_idx > 0 {
            self.history_idx -= 1;
            let text = self.history[self.history_idx].clone();
            self.set_text(&text);
        }
    }

    fn history_down(&mut self) {
        if self.history.is_empty() {
            return;
        }
        if self.history_idx < self.history.len() - 1 {
            self.history_idx += 1;
            let text = self.history[self.history_idx].clone();
            self.set_text(&text);
        } else if self.history_idx == self.history.len() - 1 {
            self.history_idx += 1;
            let text = self.saved_text.clone();
            self.set_text(&text);
        }
    }

    // ------------------------------------------------------------------
    // Cursor ↔ byte helpers (for autocomplete)
    // ------------------------------------------------------------------

    /// Convert visual (row, col) from tui-textarea cursor to byte offset.
    fn visual_to_byte(&self, row: usize, col: usize) -> usize {
        let lines = self.textarea.lines();
        let mut byte = 0;
        for (i, line) in lines.iter().enumerate() {
            if i == row {
                let chars_in_col: usize = line.chars().take(col).map(|c| c.len_utf8()).sum();
                return byte + chars_in_col;
            }
            byte += line.len() + 1; // +1 for the newline
        }
        byte
    }

    /// Get the cursor position as (row, col) from textarea.
    fn cursor_pos(&self) -> (usize, usize) {
        let c = self.textarea.cursor();
        (c.0, c.1)
    }

    /// Full text from textarea.
    fn full_text(&self) -> String {
        self.textarea.lines().join("\n")
    }

    // ------------------------------------------------------------------
    // Autocomplete
    // ------------------------------------------------------------------

    fn start_autocomplete(&mut self, trigger: char) {
        let (row, col) = self.cursor_pos();
        self.autocomplete = Some(AutocompleteState {
            items: Vec::new(),
            selected: 0,
            prefix_row: row,
            prefix_col: col,
            query: String::new(),
            trigger,
        });
        self.update_autocomplete_items();
    }

    fn update_autocomplete_items(&mut self) {
        // Extract autocomplete state to avoid dual mutable+immutable borrow.
        let (prefix_row, prefix_col, trigger) = match self.autocomplete.as_ref() {
            Some(ac) => (ac.prefix_row, ac.prefix_col, ac.trigger),
            None => return,
        };

        let full = self.full_text();
        let (cursor_row, cursor_col) = self.cursor_pos();
        let start_byte = self.visual_to_byte(prefix_row, prefix_col);
        let end_byte = self.visual_to_byte(cursor_row, cursor_col);

        let Some(ref mut ac) = self.autocomplete else {
            return;
        };

        // Query = text from prefix_start to cursor.
        if end_byte >= start_byte && start_byte <= full.len() {
            let end = end_byte.min(full.len());
            ac.query = full[start_byte..end].to_string();
        } else {
            ac.query.clear();
            ac.items.clear();
            return;
        }

        let items = match trigger {
            '@' => {
                if let Some(ref picker) = self.fff_picker {
                    fff_file_matches(picker, &ac.query)
                } else {
                    self.file_index.ensure_built(&self.working_dir);
                    file_mention_matches_from_cache(
                        &self.file_index.entries,
                        &self.working_dir,
                        &ac.query,
                    )
                }
            }
            '/' => fuzzy_command_matches(&self.slash_commands, &ac.query),
            _ => Vec::new(),
        };

        ac.items = items;
        ac.selected = 0;
    }

    fn accept_autocomplete(&mut self) {
        let Some(ref ac) = self.autocomplete else {
            return;
        };
        let Some(item) = ac.items.get(ac.selected).cloned() else {
            return;
        };
        let is_dir = item.ends_with('/');
        let prefix_row = ac.prefix_row;
        let prefix_col = ac.prefix_col;

        let full = self.full_text();
        let (cursor_row, cursor_col) = self.cursor_pos();
        let start_byte = self.visual_to_byte(prefix_row, prefix_col);
        let end_byte = self.visual_to_byte(cursor_row, cursor_col);

        // Build replacement text.
        let mut new_text = String::with_capacity(full.len() + item.len());
        new_text.push_str(&full[..start_byte.min(full.len())]);
        new_text.push_str(&item);
        if end_byte < full.len() {
            new_text.push_str(&full[end_byte..]);
        }

        if !is_dir {
            // Insert trailing space.
            let trail_byte = start_byte + item.len();
            if trail_byte <= new_text.len() {
                new_text.insert(trail_byte, ' ');
            }
        }

        let new_cursor_byte = start_byte + item.len() + if is_dir { 0 } else { 1 };
        self.set_text(&new_text);

        // Position cursor at the end of inserted text.
        if let Some((row, col)) = self.byte_to_visual(new_cursor_byte) {
            self.textarea
                .move_cursor(CursorMove::Jump(row as u16, col as u16));
        }

        if is_dir {
            // Keep autocomplete open so user can keep navigating.
            if let Some(ref mut ac) = self.autocomplete {
                ac.prefix_row = prefix_row;
                ac.prefix_col = prefix_col;
                ac.query.clear();
            }
            self.update_autocomplete_items();
        } else {
            self.autocomplete = None;
        }
    }
    fn byte_to_visual(&self, target: usize) -> Option<(usize, usize)> {
        let lines = self.textarea.lines();
        let mut byte = 0;
        for (row, line) in lines.iter().enumerate() {
            let line_end = byte + line.len();
            if target <= line_end {
                let offset = target - byte;
                let col: usize = line[..offset.min(line.len())].chars().count();
                return Some((row, col));
            }
            byte = line_end + 1; // +1 for newline
        }
        // Target past end — place at end of last line.
        let last = lines.len().saturating_sub(1);
        let col = lines.last().map(|l| l.chars().count()).unwrap_or(0);
        Some((last, col))
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
            if ac.items.is_empty() {
                return;
            }
            ac.selected = if ac.selected == 0 {
                ac.items.len() - 1
            } else {
                ac.selected - 1
            };
        }
    }

    fn refresh_slash_autocomplete(&mut self) {
        let full = self.full_text();
        let at_start = full.starts_with('/');
        if !at_start {
            self.dismiss_autocomplete();
            return;
        }
        let (_, cursor_col) = self.cursor_pos();
        if cursor_col == 0 {
            self.dismiss_autocomplete();
            return;
        }

        let in_first_token = !full[..self.visual_to_byte(0, cursor_col).min(full.len())]
            .contains(' ')
            && !full.contains('\n');
        if !in_first_token {
            self.dismiss_autocomplete();
            return;
        }

        let prefix_row = 0;
        let prefix_col = 1; // skip the '/'
        if self.autocomplete.is_none() {
            self.autocomplete = Some(AutocompleteState {
                items: Vec::new(),
                selected: 0,
                prefix_row,
                prefix_col,
                query: String::new(),
                trigger: '/',
            });
        }

        if let Some(ref mut ac) = self.autocomplete {
            ac.prefix_row = prefix_row;
            ac.prefix_col = prefix_col;
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
            frame.render_widget(block, area);
            return;
        }

        // When the editor area changes (lines added/removed, resize),
        // ratatui-textarea's viewport retains a stale top_row that can
        // exceed the new screen line count — leaving blank space below
        // the content. Fix: pre-render to update the screen map, then
        // reset the viewport to top so the widget recalculates scroll
        // from a clean state. The cursor is moved to (0,0) before the
        // scroll so InViewport is a no-op, then restored via Jump.
        if Some(inner) != self.last_inner_area {
            // Pass 1: update screen map for new dimensions.
            frame.render_widget(&self.textarea, inner);
            let saved = self.textarea.cursor();
            // Move cursor to (0,0) so InViewport inside scroll() is a
            // no-op — prevents the scroll from clamping the cursor to
            // the old viewport bounds.
            self.textarea.move_cursor(CursorMove::Top);
            self.textarea.move_cursor(CursorMove::Head);
            // Reset viewport to top. Widget recalculates from
            // prev_top_row=0 using the actual cursor position.
            self.textarea.scroll(Scrolling::Delta {
                rows: -i16::MAX,
                cols: 0,
            });
            // Restore cursor to its pre-reset position.
            self.textarea
                .move_cursor(CursorMove::Jump(saved.0 as u16, saved.1 as u16));
        }

        frame.render_widget(block, area);
        frame.render_widget(&self.textarea, inner);

        self.last_inner_area = Some(inner);
    }

    fn handle_event(&mut self, event: &Event) -> Option<Action> {
        if !self.focused {
            return None;
        }

        // ── Paste ──
        if let Event::Paste(pasted) = event {
            let lines: Vec<&str> = pasted.lines().collect();
            let len = lines.len();
            for (i, line) in lines.iter().enumerate() {
                if !line.is_empty() {
                    let trimmed = line.trim_end_matches('\r');
                    for c in trimmed.chars() {
                        self.textarea.insert_char(c);
                    }
                }
                // Insert newline between lines, but not after the last one
                // to avoid a trailing empty line.
                if i + 1 < len {
                    self.textarea.insert_newline();
                }
            }
            self.refresh_slash_autocomplete();
            return None;
        }

        // ── Mouse ──
        if let Event::Mouse(mouse) = event {
            if let Some(ref area) = self.last_inner_area {
                match mouse.kind {
                    MouseEventKind::Down(MouseButton::Left) => {
                        let col = mouse.column.saturating_sub(area.x) as usize;
                        let row = mouse.row.saturating_sub(area.y) as usize;
                        self.textarea
                            .move_cursor(CursorMove::Jump(row as u16, col as u16));
                        self.textarea.move_cursor(CursorMove::InViewport);
                    }
                    MouseEventKind::ScrollUp => {
                        self.textarea.scroll(Scrolling::PageUp);
                    }
                    MouseEventKind::ScrollDown => {
                        self.textarea.scroll(Scrolling::PageDown);
                    }
                    _ => {}
                }
            }
            return None;
        }

        let Event::Key(key) = event else {
            return None;
        };

        // ── Autocomplete-active key intercepts ──
        if self.autocomplete.is_some() {
            if is_newline_key(key, self.enter_behavior) {
                let input = key_to_tui_input(key);
                self.textarea.input(input);
                self.refresh_slash_autocomplete();
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
                        .is_some_and(|ac| ac.trigger == '/' && !ac.items.is_empty())
                    {
                        self.accept_autocomplete();
                        return None;
                    }
                    self.dismiss_autocomplete();
                }
                crossterm::event::KeyEvent {
                    code: KeyCode::Char(' '),
                    ..
                } => {
                    self.textarea.insert_char(' ');
                    if self
                        .autocomplete
                        .as_ref()
                        .is_some_and(|ac| ac.trigger == '@')
                    {
                        self.dismiss_autocomplete();
                    } else if self.autocomplete.is_some() {
                        self.refresh_slash_autocomplete();
                    }
                    return None;
                }
                crossterm::event::KeyEvent {
                    code: KeyCode::Char(c),
                    ..
                } => {
                    let c = if key.modifiers.contains(KeyModifiers::SHIFT) {
                        c.to_uppercase().next().unwrap_or(*c)
                    } else {
                        *c
                    };
                    self.textarea.insert_char(c);
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
                // Alt+Backspace/Delete in autocomplete: delete whitespace only.
                crossterm::event::KeyEvent {
                    code: KeyCode::Backspace,
                    modifiers: KeyModifiers::ALT,
                    ..
                } => {
                    delete_whitespace_backward(&mut self.textarea);
                    self.update_autocomplete_items();
                    return None;
                }
                crossterm::event::KeyEvent {
                    code: KeyCode::Delete,
                    modifiers: KeyModifiers::ALT,
                    ..
                } => {
                    delete_whitespace_forward(&mut self.textarea);
                    self.update_autocomplete_items();
                    return None;
                }
                crossterm::event::KeyEvent {
                    code: KeyCode::Backspace,
                    ..
                } => {
                    if let Some(ref ac) = self.autocomplete {
                        let (cur_row, cur_col) = self.cursor_pos();
                        if cur_row == ac.prefix_row && cur_col <= ac.prefix_col {
                            self.textarea.delete_char();
                            self.dismiss_autocomplete();
                            return None;
                        }
                    }
                    self.textarea.delete_char();
                    self.update_autocomplete_items();
                    return None;
                }
                _ => {}
            }
        }

        // ── Submit keys (check before character handling) ──
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

        // ── Custom key intercepts (BEFORE fallthrough to tui-textarea) ──

        match key {
            // ── Autocomplete triggers ──
            crossterm::event::KeyEvent {
                code: KeyCode::Char('@'),
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                self.textarea.insert_char('@');
                self.start_autocomplete('@');
                return None;
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Char('/'),
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                self.textarea.insert_char('/');
                let full = self.full_text();
                let trimmed = full.trim();
                if trimmed.is_empty() || full.ends_with(' ') || full.ends_with('\n') || full == "/"
                {
                    self.start_autocomplete('/');
                }
                return None;
            }

            // ── macOS: Cmd+Delete → kill current line ──
            crossterm::event::KeyEvent {
                code: KeyCode::Delete,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SUPER) => {
                delete_current_line(&mut self.textarea);
                return None;
            }
            // ── macOS: Cmd+Backspace → kill from start to cursor ──
            crossterm::event::KeyEvent {
                code: KeyCode::Backspace,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SUPER) => {
                delete_current_line(&mut self.textarea);
                return None;
            }

            // ── Cmd+Left/Right → text start/end ──
            crossterm::event::KeyEvent {
                code: KeyCode::Left,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SUPER) => {
                self.textarea.move_cursor(CursorMove::Top);
                self.textarea.move_cursor(CursorMove::Head);
                return None;
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Right,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SUPER) => {
                self.textarea.move_cursor(CursorMove::Bottom);
                self.textarea.move_cursor(CursorMove::End);
                return None;
            }
            // Cmd+Up → top
            crossterm::event::KeyEvent {
                code: KeyCode::Up,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SUPER) => {
                self.textarea.move_cursor(CursorMove::Top);
                self.textarea.move_cursor(CursorMove::Head);
                return None;
            }
            // Cmd+Down → bottom
            crossterm::event::KeyEvent {
                code: KeyCode::Down,
                modifiers,
                ..
            } if modifiers.contains(KeyModifiers::SUPER) => {
                self.textarea.move_cursor(CursorMove::Bottom);
                self.textarea.move_cursor(CursorMove::End);
                return None;
            }

            // ── Alt+Left / Alt+Right → word navigation ──
            crossterm::event::KeyEvent {
                code: KeyCode::Left,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                self.textarea.move_cursor(CursorMove::WordBack);
                return None;
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Right,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                self.textarea.move_cursor(CursorMove::WordForward);
                return None;
            }
            // ── Option+Backspace / Option+Delete → delete whitespace only ──
            crossterm::event::KeyEvent {
                code: KeyCode::Backspace,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                delete_whitespace_backward(&mut self.textarea);
                return None;
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Delete,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                delete_whitespace_forward(&mut self.textarea);
                return None;
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Up,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                self.history_up();
                return None;
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                self.history_down();
                return None;
            }

            // ── Bare Up/Down: move cursor; at boundary → history (unless selecting) ──
            crossterm::event::KeyEvent {
                code: KeyCode::Up,
                modifiers,
                ..
            } => {
                let (row, _col) = self.cursor_pos();
                if row == 0 && !modifiers.contains(KeyModifiers::SHIFT) {
                    self.history_up();
                } else {
                    let input = key_to_tui_input(key);
                    self.textarea.input(input);
                    self.refresh_slash_autocomplete();
                }
                return None;
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Down,
                modifiers,
                ..
            } => {
                let line_count = self.textarea.lines().len();
                let (row, _col) = self.cursor_pos();
                if row >= line_count.saturating_sub(1) && !modifiers.contains(KeyModifiers::SHIFT) {
                    self.history_down();
                } else {
                    let input = key_to_tui_input(key);
                    self.textarea.input(input);
                    self.refresh_slash_autocomplete();
                }
                return None;
            }

            // ── PageUp/PageDown: at boundary → history ──
            crossterm::event::KeyEvent {
                code: KeyCode::PageUp,
                ..
            } => {
                let (row, _col) = self.cursor_pos();
                let input = key_to_tui_input(key);
                self.textarea.input(input);
                if row == 0 {
                    self.history_up();
                }
                self.refresh_slash_autocomplete();
                return None;
            }
            crossterm::event::KeyEvent {
                code: KeyCode::PageDown,
                ..
            } => {
                let line_count = self.textarea.lines().len();
                let (row, _col) = self.cursor_pos();
                let input = key_to_tui_input(key);
                self.textarea.input(input);
                if row >= line_count.saturating_sub(1) {
                    self.history_down();
                }
                self.refresh_slash_autocomplete();
                return None;
            }

            // ── Tab → 2 spaces (tui-textarea default with tab_length=2 already does this) ──
            // Fall through to textarea.input() which handles Tab → spaces.

            // ── Catch all: regular character input + refresh slash autocomplete ──
            _ => {
                let input = key_to_tui_input(key);
                // Don't pass default (null) input — it would be a no-op anyway.
                if input == Input::default() {
                    return None;
                }
                self.textarea.input(input);
                // Only refresh slash autocomplete on non-navigation keys.
                if is_text_mutation_key(key) {
                    self.refresh_slash_autocomplete();
                }
            }
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

// ---------------------------------------------------------------------------
// crossterm KeyEvent → tui_textarea::Input conversion
// ---------------------------------------------------------------------------

fn unicode_width(c: char) -> usize {
    unicode_width::UnicodeWidthChar::width(c).unwrap_or(0)
}

fn key_to_tui_input(key: &crossterm::event::KeyEvent) -> Input {
    let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
    let alt = key.modifiers.contains(KeyModifiers::ALT);
    let shift = key.modifiers.contains(KeyModifiers::SHIFT);

    let tui_key = match key.code {
        KeyCode::Char(c) => Key::Char(c),
        KeyCode::Backspace => Key::Backspace,
        KeyCode::Delete => Key::Delete,
        KeyCode::Enter => Key::Enter,
        KeyCode::Tab => Key::Tab,
        KeyCode::Left => Key::Left,
        KeyCode::Right => Key::Right,
        KeyCode::Up => Key::Up,
        KeyCode::Down => Key::Down,
        KeyCode::Home => Key::Home,
        KeyCode::End => Key::End,
        KeyCode::PageUp => Key::PageUp,
        KeyCode::PageDown => Key::PageDown,
        KeyCode::Esc => Key::Esc,
        KeyCode::F(n) => Key::F(n),
        _ => return Input::default(),
    };

    Input {
        key: tui_key,
        ctrl,
        alt,
        shift,
    }
}

/// Returns true if a key event mutates text (insert, delete, etc.).
fn is_text_mutation_key(key: &crossterm::event::KeyEvent) -> bool {
    matches!(
        key.code,
        KeyCode::Char(_) | KeyCode::Backspace | KeyCode::Delete | KeyCode::Tab | KeyCode::Enter
    )
}

/// Delete entire current line (Cmd+Delete/Cmd+Backspace).
/// Uses selection-based deletion that handles all cursor positions
/// and empty lines correctly.
fn delete_current_line(textarea: &mut TextArea) {
    // Single line can't be removed, just clear it.
    if textarea.lines().len() == 1 {
        textarea.select_all();
        textarea.cut();
        return;
    }
    let is_last = textarea.cursor().0 + 1 >= textarea.lines().len();
    // Select current line content and cut.
    textarea.move_cursor(CursorMove::Head);
    textarea.start_selection();
    textarea.move_cursor(CursorMove::End);
    textarea.cut();
    if is_last {
        textarea.delete_newline();
    } else {
        // Cursor on now-empty line. delete_next_char jumps to next line
        // and deletes the newline, collapsing the empty line.
        textarea.delete_next_char();
    }
}

/// Delete consecutive whitespace left of cursor.
fn delete_whitespace_backward(textarea: &mut TextArea) {
    loop {
        let c = textarea.cursor();
        let (row, col) = (c.0, c.1);
        if col == 0 {
            break;
        }
        let line = &textarea.lines()[row];
        let prev = line[..col].chars().last();
        if prev.is_some_and(|c| c.is_whitespace()) {
            textarea.move_cursor(CursorMove::Back);
            textarea.delete_next_char();
        } else {
            break;
        }
    }
}

/// Delete consecutive whitespace right of cursor.
fn delete_whitespace_forward(textarea: &mut TextArea) {
    loop {
        let c = textarea.cursor();
        let (row, col) = (c.0, c.1);
        let line = &textarea.lines()[row];
        let next = line[col..].chars().next();
        if next.is_some_and(|c| c.is_whitespace()) {
            textarea.delete_next_char();
        } else {
            break;
        }
    }
}

// Fuzzy file matching
// ---------------------------------------------------------------------------

/// Backward-compat: builds index every call (slow). Used by tests.
pub fn file_mention_matches(base_dir: &Path, query: &str) -> Vec<String> {
    let entries = build_file_index(base_dir);
    file_mention_matches_from_cache(&entries, base_dir, query)
}

/// Return file mention matches from a pre-built cache: fuzzy-ranked.
pub fn file_mention_matches_from_cache(
    entries: &[String],
    base_dir: &Path,
    query: &str,
) -> Vec<String> {
    let mut entries = entries.to_vec();
    let trimmed = query.trim();

    // Git excludes gitignored directories (e.g. docs/). When query has a
    // slash and the directory prefix exists on disk, supplement entries
    // with filesystem listing from that dir so partial paths like
    // `docs/risk` still match.
    if let Some(slash_pos) = trimmed.rfind('/') {
        let dir_prefix = &trimmed[..slash_pos + 1];
        let abs_dir = base_dir.join(dir_prefix.trim_end_matches('/'));
        if abs_dir.is_dir() {
            let include_hidden = trimmed.starts_with('.');
            let mut dir_entries = Vec::new();
            collect_file_paths(base_dir, &abs_dir, include_hidden, &mut dir_entries);
            dir_entries.sort();
            dir_entries.dedup();
            for entry in dir_entries {
                if !entries.contains(&entry) {
                    entries.push(entry);
                }
            }
            entries.sort();
        }
    }

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

fn git_tracked_and_untracked_files(base_dir: &Path) -> Option<Vec<String>> {
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
            "target" | "node_modules" | ".git" | ".michin" | "dist" | "build"
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

/// FFF-powered file matching for @-mention autocomplete.
/// Uses frecency-ranked fuzzy search from the FFF index.
fn fff_file_matches(picker: &SharedFilePicker, query: &str) -> Vec<String> {
    let guard = match picker.read() {
        Ok(g) => g,
        Err(_) => return Vec::new(),
    };
    let picker = match guard.as_ref() {
        Some(p) => p,
        None => return Vec::new(),
    };

    let parser = QueryParser::new(FileSearchConfig);
    let fff_query = parser.parse(query);

    let results = picker.fuzzy_search(
        &fff_query,
        None,
        FuzzySearchOptions {
            max_threads: 0,
            current_file: None,
            pagination: PaginationArgs {
                offset: 0,
                limit: 50,
            },
            ..Default::default()
        },
    );

    results
        .items
        .iter()
        .map(|item| {
            let path = item.relative_path(picker);
            path.replace('\\', "/")
        })
        .collect()
}
