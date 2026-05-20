//! Input editor — multiline text input with inline fuzzy autocomplete for @ files and / commands.

use crossterm::event::{Event, KeyCode, KeyModifiers};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Clear, Paragraph},
};
use std::path::PathBuf;

use crate::components::fuzzy::fuzzy_filter;
use crate::components::{Action, Component};
use crate::theme::Theme;

/// State for inline autocomplete (file paths or slash commands).
#[derive(Debug, Clone)]
struct AutocompleteState {
    /// Available items (file names or command names).
    items: Vec<String>,
    /// Currently selected index.
    selected: usize,
    /// Byte position in text where @ or / was typed.
    prefix_start: usize,
    /// The filter query (text between @ or / and cursor).
    query: String,
}

/// Multiline text editor for user input.
pub struct Editor {
    /// The text buffer.
    text: String,
    /// Cursor position (byte offset).
    cursor: usize,
    /// Whether focused.
    focused: bool,
    /// Theme.
    theme: Theme,
    /// History of submitted messages.
    history: Vec<String>,
    /// Current history index (for up/down browsing).
    history_idx: usize,
    /// Temporary save for history browsing.
    saved_text: String,
    /// Scroll offset in visual lines.
    scroll: usize,
    /// Inline autocomplete state (None = not active).
    autocomplete: Option<AutocompleteState>,
    /// Working directory for file autocomplete.
    working_dir: PathBuf,
    /// Known slash commands for command autocomplete.
    slash_commands: Vec<String>,
}

impl Editor {
    pub fn new(theme: Theme, working_dir: PathBuf, slash_commands: Vec<String>) -> Self {
        Self {
            text: String::new(),
            cursor: 0,
            focused: false,
            theme,
            history: Vec::new(),
            history_idx: 0,
            saved_text: String::new(),
            scroll: 0,
            autocomplete: None,
            working_dir,
            slash_commands,
        }
    }

    pub fn set_text(&mut self, text: &str) {
        self.text = text.to_string();
        self.cursor = self.text.len();
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    pub fn desired_height(&self, width: usize, max_height: u16) -> u16 {
        let inner_width = width.saturating_sub(2).max(1);
        let lines = wrap_text(&self.text, inner_width).len() as u16;
        lines.saturating_add(2).clamp(3, max_height.max(3))
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    /// Insert text at the current cursor position (used by path picker).
    pub fn insert_at_cursor(&mut self, s: &str) {
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    /// Delete the last character.
    pub fn delete_last_char(&mut self) {
        if let Some(c) = self.text.chars().last() {
            let len = c.len_utf8();
            self.text.truncate(self.text.len() - len);
            if self.cursor > self.text.len() {
                self.cursor = self.text.len();
            }
        }
    }

    // ------------------------------------------------------------------
    // Text editing operations
    // ------------------------------------------------------------------

    fn insert_char(&mut self, c: char) {
        self.text.insert(self.cursor, c);
        self.cursor += c.len_utf8();
    }

    fn delete_before(&mut self) {
        if self.cursor > 0
            && let Some(prev) = self.text[..self.cursor].chars().last()
        {
            let len = prev.len_utf8();
            self.text.replace_range(self.cursor - len..self.cursor, "");
            self.cursor -= len;
        }
    }

    fn delete_after(&mut self) {
        if self.cursor < self.text.len()
            && let Some(next) = self.text[self.cursor..].chars().next()
        {
            self.text
                .replace_range(self.cursor..self.cursor + next.len_utf8(), "");
        }
    }

    fn move_left(&mut self) {
        if self.cursor > 0
            && let Some(prev) = self.text[..self.cursor].chars().last()
        {
            self.cursor -= prev.len_utf8();
        }
    }

    fn move_right(&mut self) {
        if self.cursor < self.text.len()
            && let Some(next) = self.text[self.cursor..].chars().next()
        {
            self.cursor += next.len_utf8();
        }
    }

    fn move_word_left(&mut self) {
        while self.cursor > 0 {
            if let Some(prev) = self.text[..self.cursor].chars().last() {
                if prev.is_whitespace() {
                    self.move_left();
                } else {
                    break;
                }
            }
        }
        while self.cursor > 0 {
            if let Some(prev) = self.text[..self.cursor].chars().last() {
                if !prev.is_whitespace() {
                    self.move_left();
                } else {
                    break;
                }
            }
        }
    }

    #[allow(dead_code)]
    fn move_word_right(&mut self) {
        while self.cursor < self.text.len() {
            if let Some(next) = self.text[self.cursor..].chars().next() {
                if !next.is_whitespace() {
                    self.move_right();
                } else {
                    break;
                }
            }
        }
        while self.cursor < self.text.len() {
            if let Some(next) = self.text[self.cursor..].chars().next() {
                if next.is_whitespace() {
                    self.move_right();
                } else {
                    break;
                }
            }
        }
    }

    fn move_start(&mut self) {
        self.cursor = 0;
    }

    fn move_end(&mut self) {
        self.cursor = self.text.len();
    }

    fn submit(&mut self) -> Option<String> {
        let text = self.text.trim().to_string();
        self.text.clear();
        self.cursor = 0;
        self.scroll = 0;
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
            self.saved_text = self.text.clone();
        }
        if self.history_idx > 0 {
            self.history_idx -= 1;
            self.text = self.history[self.history_idx].clone();
            self.cursor = self.text.len();
            self.scroll = 0;
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
            self.scroll = 0;
        } else if self.history_idx == self.history.len() - 1 {
            self.history_idx += 1;
            self.text = self.saved_text.clone();
            self.cursor = self.text.len();
            self.scroll = 0;
        }
    }

    // ------------------------------------------------------------------
    // Autocomplete
    // ------------------------------------------------------------------

    /// Start autocomplete after typing @ or /.
    fn start_autocomplete(&mut self, trigger: char) {
        let prefix_start = self.cursor; // right after @ or /
        self.autocomplete = Some(AutocompleteState {
            items: Vec::new(),
            selected: 0,
            prefix_start,
            query: String::new(),
        });
        self.update_autocomplete(trigger);
    }

    /// Update autocomplete items based on current query.
    fn update_autocomplete(&mut self, trigger: char) {
        let Some(ref mut ac) = self.autocomplete else {
            return;
        };

        // Extract query: text between prefix_start and cursor.
        if self.cursor >= ac.prefix_start {
            ac.query = self.text[ac.prefix_start..self.cursor].to_string();
        } else {
            ac.query.clear();
        }

        ac.items = match trigger {
            '@' => file_mention_matches(&self.working_dir, &ac.query),
            '/' => fuzzy_command_matches(&self.slash_commands, &ac.query),
            _ => Vec::new(),
        };

        ac.selected = 0;
    }

    /// Apply the selected autocomplete item.
    fn accept_autocomplete(&mut self, trigger: char) {
        let Some(ref ac) = self.autocomplete else {
            return;
        };
        let Some(item) = ac.items.get(ac.selected) else {
            return;
        };
        let is_dir = item.ends_with('/');

        // Replace query text with the selected item.
        let start = ac.prefix_start;
        let end = self.cursor;
        self.text.replace_range(start..end, item);

        if is_dir {
            // Keep autocomplete open so user can keep navigating.
            self.cursor = start + item.len();
            self.autocomplete.as_mut().unwrap().prefix_start = start;
            self.autocomplete.as_mut().unwrap().query.clear();
            self.update_autocomplete(trigger);
        } else {
            // Insert space after file, dismiss autocomplete.
            self.cursor = start + item.len();
            self.text.insert(self.cursor, ' ');
            self.cursor += 1;
            self.autocomplete = None;
        }
    }

    /// Return autocomplete items for external rendering.
    pub fn autocomplete_items(&self) -> Vec<String> {
        self.autocomplete
            .as_ref()
            .map(|ac| ac.items.clone())
            .unwrap_or_default()
    }

    /// Selected index in autocomplete items.
    pub fn autocomplete_selected(&self) -> usize {
        self.autocomplete
            .as_ref()
            .map(|ac| ac.selected)
            .unwrap_or(0)
    }

    /// Whether autocomplete is currently active.
    pub fn autocomplete_active(&self) -> bool {
        self.autocomplete.is_some()
    }

    /// Dismiss autocomplete without applying.
    fn dismiss_autocomplete(&mut self) {
        self.autocomplete = None;
    }

    fn select_next(&mut self) {
        if let Some(ref mut ac) = self.autocomplete
            && !ac.items.is_empty()
        {
            ac.selected = (ac.selected + 1).min(ac.items.len().saturating_sub(1));
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
            return;
        }

        let upto_cursor = &self.text[..self.cursor];
        if !upto_cursor.starts_with('/') {
            return;
        }

        let in_first_token = !upto_cursor.contains(' ') && !upto_cursor.contains('\n');
        if !in_first_token {
            return;
        }

        let prefix_start = 1;
        if self.autocomplete.is_none() {
            self.autocomplete = Some(AutocompleteState {
                items: Vec::new(),
                selected: 0,
                prefix_start,
                query: String::new(),
            });
        }

        if let Some(ref mut ac) = self.autocomplete {
            ac.prefix_start = prefix_start;
        }
        self.update_autocomplete('/');
    }
}

impl Component for Editor {
    fn render(&mut self, area: Rect, frame: &mut Frame) {
        let cursor_style = if self.focused {
            Style::default().fg(self.theme.accent).bg(Color::DarkGray)
        } else {
            Style::default().fg(self.theme.dim)
        };

        let input_border = if self.focused {
            self.theme.accent
        } else {
            self.theme.warning
        };
        let block = Block::default()
            .borders(Borders::TOP | Borders::BOTTOM)
            .border_style(Style::default().fg(input_border));

        let inner = block.inner(area);
        let width = inner.width as usize;
        let height = inner.height as usize;
        if width == 0 || height == 0 {
            let para = Paragraph::new("").block(block);
            frame.render_widget(para, area);
            return;
        }

        // Wrap text into visual lines.
        let visual_lines = wrap_text(&self.text, width);
        let total_lines = visual_lines.len();

        // Find cursor visual line.
        let cursor_line = cursor_visual_line(&self.text, self.cursor, width);

        // Auto-scroll.
        if cursor_line < self.scroll {
            self.scroll = cursor_line;
        } else if cursor_line >= self.scroll + height {
            self.scroll = cursor_line.saturating_sub(height.saturating_sub(1));
        }
        self.scroll = self.scroll.min(total_lines.saturating_sub(height));

        // Build visible text lines.
        let end = (self.scroll + height).min(total_lines);
        let visible_lines: Vec<Line> = visual_lines[self.scroll..end]
            .iter()
            .enumerate()
            .map(|(line_idx, line)| {
                let abs_line = self.scroll + line_idx;
                let spans: Vec<Span> = line
                    .iter()
                    .map(|&char_idx| {
                        let c = self.text[char_idx..].chars().next().unwrap_or(' ');
                        let at_cursor = self.focused && char_idx == self.cursor;
                        Span::styled(
                            c.to_string(),
                            if at_cursor {
                                cursor_style
                            } else {
                                Style::default()
                            },
                        )
                    })
                    .collect();
                let mut spans = spans;
                if self.focused && self.cursor >= self.text.len() && abs_line == cursor_line {
                    spans.push(Span::styled(" ", cursor_style));
                }
                Line::from(spans)
            })
            .collect();

        frame.render_widget(Clear, area);
        frame.render_widget(Paragraph::new(Text::from(visible_lines)).block(block), area);
    }

    fn handle_event(&mut self, event: &Event) -> Option<Action> {
        if !self.focused {
            return None;
        }
        let Event::Key(key) = event else {
            return None;
        };

        // If autocomplete is active, handle its keys first.
        if self.autocomplete.is_some() {
            match key {
                crossterm::event::KeyEvent {
                    code: KeyCode::Tab, ..
                } => {
                    let trigger = if self.text.as_bytes().get(
                        self.autocomplete
                            .as_ref()
                            .unwrap()
                            .prefix_start
                            .wrapping_sub(1),
                    ) == Some(&b'@')
                    {
                        '@'
                    } else {
                        '/'
                    };
                    self.accept_autocomplete(trigger);
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
                    let trigger = if self.text.as_bytes().get(
                        self.autocomplete
                            .as_ref()
                            .unwrap()
                            .prefix_start
                            .wrapping_sub(1),
                    ) == Some(&b'@')
                    {
                        '@'
                    } else {
                        '/'
                    };
                    self.accept_autocomplete(trigger);
                    return None;
                }
                crossterm::event::KeyEvent {
                    code: KeyCode::Char(c),
                    ..
                } => {
                    self.insert_char(*c);
                    let trigger = if self.text.as_bytes().get(
                        self.autocomplete
                            .as_ref()
                            .unwrap()
                            .prefix_start
                            .wrapping_sub(1),
                    ) == Some(&b'@')
                    {
                        '@'
                    } else {
                        '/'
                    };
                    self.update_autocomplete(trigger);
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
                    let trigger = if self.text.as_bytes().get(
                        self.autocomplete
                            .as_ref()
                            .unwrap()
                            .prefix_start
                            .wrapping_sub(1),
                    ) == Some(&b'@')
                    {
                        '@'
                    } else {
                        '/'
                    };
                    self.update_autocomplete(trigger);
                    return None;
                }
                _ => {}
            }
        }

        match key {
            crossterm::event::KeyEvent {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                if let Some(text) = self.submit() {
                    return Some(Action::SendMessage(text));
                }
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Enter,
                modifiers: KeyModifiers::ALT,
                ..
            } => {
                if let Some(text) = self.submit() {
                    return Some(Action::FollowUpMessage(text));
                }
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Char('j'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.insert_char('\n');
            }
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
            crossterm::event::KeyEvent {
                code: KeyCode::Char(c),
                modifiers: KeyModifiers::NONE | KeyModifiers::SHIFT,
                ..
            } => {
                self.insert_char(*c);
                self.refresh_slash_autocomplete();
            }
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
            crossterm::event::KeyEvent {
                code: KeyCode::Up,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                self.history_up();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Down,
                modifiers: KeyModifiers::NONE,
                ..
            } => {
                self.history_down();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Home,
                ..
            } => {
                self.move_start();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::End, ..
            } => {
                self.move_end();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Char('a'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.move_start();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Char('e'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                self.move_end();
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Char('w'),
                modifiers: KeyModifiers::CONTROL,
                ..
            } => {
                let old = self.cursor;
                self.move_word_left();
                let new = self.cursor;
                self.text.replace_range(new..old, "");
            }
            crossterm::event::KeyEvent {
                code: KeyCode::Tab, ..
            } => {
                self.text.insert_str(self.cursor, "  ");
                self.cursor += 2;
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

// ---------------------------------------------------------------------------
// Fuzzy file matching
// ---------------------------------------------------------------------------

/// Return Codex-style file mention matches:
/// gitignore-aware, recursive, relative paths, fuzzy-ranked.
fn file_mention_matches(base_dir: &std::path::Path, query: &str) -> Vec<String> {
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

    // If user types "/git-...", proactively suggest "/skill:git-..." matches.
    let skill_commands: Vec<&String> = commands
        .iter()
        .filter(|c| c.starts_with("skill:"))
        .collect();
    for cmd in fuzzy_filter(&skill_commands, query, |s| &s[6..]) {
        out.push(cmd.to_string());
        if out.len() >= 10 {
            return out;
        }
    }

    // Normal command matching.
    let cmds: Vec<&String> = commands.iter().collect();
    for cmd in fuzzy_filter(&cmds, query, |s| s) {
        if !out.iter().any(|existing| existing == cmd.as_str()) {
            out.push(cmd.to_string());
            if out.len() >= 10 {
                break;
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Text wrapping helpers
// ---------------------------------------------------------------------------

/// Split text into visual lines of at most `width` chars.
fn wrap_text(text: &str, width: usize) -> Vec<Vec<usize>> {
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
    if !current.is_empty() || lines.is_empty() {
        lines.push(current);
    }
    lines
}

/// Find which visual line the given byte cursor is on.
fn cursor_visual_line(text: &str, cursor: usize, width: usize) -> usize {
    let mut line = 0usize;
    let mut col = 0usize;

    for (byte_idx, ch) in text.char_indices() {
        if byte_idx >= cursor {
            return line;
        }
        if ch == '\n' {
            line += 1;
            col = 0;
            continue;
        }
        if col >= width {
            line += 1;
            col = 0;
        }
        col += 1;
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(name: &str) -> std::path::PathBuf {
        let root =
            std::env::temp_dir().join(format!("theta-tui-editor-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        root
    }

    #[test]
    fn file_mentions_recurse_and_fuzzy_match_relative_paths() {
        let root = temp_root("fuzzy-path");
        std::fs::create_dir_all(root.join("src/components")).unwrap();
        std::fs::write(root.join("src/main.rs"), "").unwrap();
        std::fs::write(root.join("src/components/chat.rs"), "").unwrap();
        std::fs::write(root.join("README.md"), "").unwrap();

        let matches = file_mention_matches(&root, "src chat");

        assert_eq!(matches, vec!["src/components/chat.rs".to_string()]);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn file_mentions_empty_query_returns_visible_files() {
        let root = temp_root("empty");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/lib.rs"), "").unwrap();
        std::fs::write(root.join("Cargo.toml"), "").unwrap();

        let matches = file_mention_matches(&root, "");

        assert_eq!(
            matches,
            vec!["Cargo.toml".to_string(), "src/lib.rs".to_string()]
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn file_mentions_use_git_exclude_standard_when_available() {
        let root = temp_root("git-aware");
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/visible.rs"), "").unwrap();
        std::fs::write(root.join("ignored.log"), "").unwrap();
        std::fs::write(root.join(".gitignore"), "*.log\n").unwrap();
        let _ = std::process::Command::new("git")
            .arg("-C")
            .arg(&root)
            .arg("init")
            .output();

        let matches = file_mention_matches(&root, "");

        assert!(matches.contains(&"src/visible.rs".to_string()));
        assert!(!matches.contains(&"ignored.log".to_string()));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn file_mentions_skips_hidden_paths_by_default() {
        let root = temp_root("hidden");
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::write(root.join(".git/config"), "").unwrap();
        std::fs::write(root.join("visible.rs"), "").unwrap();

        let matches = file_mention_matches(&root, "");

        assert_eq!(matches, vec!["visible.rs".to_string()]);
        let _ = std::fs::remove_dir_all(root);
    }
}
