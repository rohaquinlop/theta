//! Chat message display — scrollable conversation view with markdown styling.

use crossterm::event::{Event, KeyCode, MouseButton, MouseEventKind};
use pulldown_cmark::{CodeBlockKind, Event as MdEvent, Options, Parser, Tag, TagEnd};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Modifier, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Padding, Paragraph},
};
use std::collections::HashMap;
use std::sync::OnceLock;
use syntect::{
    easy::HighlightLines,
    highlighting::{Style as SyntectStyle, Theme as SyntectTheme, ThemeSet},
    parsing::SyntaxSet,
    util::LinesWithEndings,
};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

use crate::components::{Action, Component};
use crate::theme::Theme;

/// A single chat message to display.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub text: String,
    pub tool_name: Option<String>,
    pub is_streaming: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatRole {
    User,
    Assistant,
    Thinking,
    Tool,
    System,
}

/// Scrollable chat message list.
pub struct Chat {
    pub messages: Vec<ChatMessage>,
    scroll_top: usize,
    auto_follow_tail: bool,
    theme: Theme,
    focused: bool,
    last_visible_lines: Vec<VisibleLine>,
    last_inner_area: Option<Rect>,
    select_anchor: Option<(usize, usize)>,
    select_head: Option<(usize, usize)>,
    selecting: bool,
    active_tool_message_idx: HashMap<String, usize>,
    cached_inner_width: Option<usize>,
    cached_wrapped_lines: Vec<Line<'static>>,
    cached_visible_line_texts: Vec<String>,
    /// For each message in self.messages, the (start, end) range into
    /// cached_wrapped_lines / cached_visible_line_texts.
    cached_msg_ranges: Vec<(usize, usize)>,
    cached_message_count: usize,
    cache_dirty: bool,
}

#[derive(Debug, Clone)]
struct VisibleLine {
    text: String,
    url_ranges: Vec<(usize, usize, String)>,
}

impl Chat {
    /// Benchmark helper: simulate old no-cache path by formatting and wrapping
    /// the whole transcript every call. Returns wrapped line count.
    pub fn benchmark_full_rebuild_no_cache(&self, inner_width: usize) -> usize {
        let mut lines: Vec<Line> = Vec::new();
        for msg in &self.messages {
            lines.extend(self.format_message(msg, inner_width));
        }
        wrap_styled_lines(&lines, inner_width).len()
    }

    /// Benchmark helper: rebuild internal cache if needed and return wrapped line count.
    pub fn benchmark_cached_rebuild(&mut self, inner_width: usize) -> usize {
        self.rebuild_render_cache(inner_width);
        self.cached_wrapped_lines.len()
    }

    pub fn invalidate_render_cache(&mut self) {
        self.cache_dirty = true;
    }

    pub fn cache_dirty(&self) -> bool {
        self.cache_dirty
    }

    pub fn clear_messages(&mut self) {
        self.messages.clear();
        self.active_tool_message_idx.clear();
        self.cached_msg_ranges.clear();
        self.cached_message_count = 0;
        self.cache_dirty = true;
        self.select_anchor = None;
        self.select_head = None;
        self.selecting = false;
    }

    pub fn new(theme: Theme) -> Self {
        Self {
            messages: Vec::new(),
            scroll_top: 0,
            auto_follow_tail: true,
            theme,
            focused: false,
            last_visible_lines: Vec::new(),
            last_inner_area: None,
            select_anchor: None,
            select_head: None,
            selecting: false,
            active_tool_message_idx: HashMap::new(),
            cached_inner_width: None,
            cached_wrapped_lines: Vec::new(),
            cached_visible_line_texts: Vec::new(),
            cached_msg_ranges: Vec::new(),
            cached_message_count: 0,
            cache_dirty: true,
        }
    }

    pub fn add_message(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
        if let Some(tool_name) = self.messages.last().and_then(|m| {
            if m.role == ChatRole::Tool && m.is_streaming {
                m.tool_name.clone()
            } else {
                None
            }
        }) {
            self.active_tool_message_idx
                .insert(tool_name, self.messages.len() - 1);
        }
        // Append to render cache immediately so the cache stays in sync.
        self.append_last_to_cache();
    }

    /// Add or update a "preparing" tool message. If a preparing message for
    /// this tool already exists (from ToolCallPrepared), update it in-place.
    /// Otherwise push a new message. Returns the message index.
    pub fn upsert_tool_message(&mut self, name: &str, text: &str, is_streaming: bool) -> usize {
        if let Some(&idx) = self.active_tool_message_idx.get(name)
            && let Some(msg) = self.messages.get_mut(idx)
            && msg.role == ChatRole::Tool
            && msg.tool_name.as_deref() == Some(name)
        {
            msg.text = text.to_string();
            msg.is_streaming = is_streaming;
            self.update_msg_in_cache(idx);
            return idx;
        }
        let idx = self.messages.len();
        self.messages.push(ChatMessage {
            role: ChatRole::Tool,
            text: text.to_string(),
            tool_name: Some(name.to_string()),
            is_streaming,
        });
        if is_streaming {
            self.active_tool_message_idx.insert(name.to_string(), idx);
        }
        self.append_last_to_cache();
        idx
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
        self.cache_dirty = true;
    }

    pub fn update_last(&mut self, text: &str, role: ChatRole, is_streaming: bool) {
        if let Some(last) = self.messages.last_mut()
            && last.role == role
            && last.is_streaming
        {
            last.text.push_str(text);
            last.is_streaming = is_streaming;
            self.update_last_in_cache();
            return;
        }
        self.messages.push(ChatMessage {
            role,
            text: text.to_string(),
            tool_name: None,
            is_streaming,
        });
        // Append to render cache immediately so subsequent
        // TextDelta events hit the fast incremental path.
        self.append_last_to_cache();
    }

    pub fn update_tool(&mut self, name: &str, text: &str, is_streaming: bool) {
        if let Some(&idx) = self.active_tool_message_idx.get(name)
            && let Some(msg) = self.messages.get_mut(idx)
            && msg.role == ChatRole::Tool
            && msg.tool_name.as_deref() == Some(name)
            && msg.is_streaming
        {
            msg.text.push_str(text);
            msg.is_streaming = is_streaming;
            if !is_streaming {
                self.active_tool_message_idx.remove(name);
            }
            // Incremental: re-format this message in-place in the cache.
            self.update_msg_in_cache(idx);
            return;
        }

        let idx = self.messages.len();
        self.messages.push(ChatMessage {
            role: ChatRole::Tool,
            text: text.trim_start_matches('\n').to_string(),
            tool_name: Some(name.to_string()),
            is_streaming,
        });
        if is_streaming {
            self.active_tool_message_idx.insert(name.to_string(), idx);
        }
        self.append_last_to_cache();
    }

    pub fn complete_tool_compact(&mut self, name: &str, text: &str) {
        if let Some(&idx) = self.active_tool_message_idx.get(name)
            && let Some(msg) = self.messages.get_mut(idx)
            && msg.role == ChatRole::Tool
            && msg.tool_name.as_deref() == Some(name)
        {
            msg.text = text.to_string();
            msg.is_streaming = false;
            self.active_tool_message_idx.remove(name);
            self.update_msg_in_cache(idx);
            return;
        }

        self.messages.push(ChatMessage {
            role: ChatRole::Tool,
            text: text.to_string(),
            tool_name: Some(name.to_string()),
            is_streaming: false,
        });
        self.append_last_to_cache();
    }

    pub fn finish_last(&mut self, role: ChatRole) {
        if let Some(last) = self.messages.last_mut()
            && last.role == role
        {
            last.is_streaming = false;
            if role == ChatRole::Tool
                && let Some(name) = last.tool_name.as_deref()
            {
                self.active_tool_message_idx.remove(name);
            }
            self.update_last_in_cache();
        }
    }

    /// Format a message into styled lines with markdown parsing.
    fn format_message(&self, msg: &ChatMessage, content_width: usize) -> Vec<Line<'static>> {
        const USER_BUBBLE_OUTER_MARGIN: usize = 0;
        const USER_BUBBLE_INNER_PAD: usize = 1;

        let is_skill_invocation = msg.role == ChatRole::User && msg.text.starts_with("/skill:");
        let (prefix, role_style): (&str, Style) = match msg.role {
            ChatRole::User if is_skill_invocation => (
                "◈ ",
                Style::default()
                    .fg(self.theme.accent)
                    .add_modifier(ratatui::style::Modifier::BOLD),
            ),
            ChatRole::User => (
                "",
                Style::default()
                    .fg(self.theme.accent)
                    .add_modifier(ratatui::style::Modifier::BOLD),
            ),
            ChatRole::Assistant => ("", Style::default().fg(self.theme.fg)),
            ChatRole::Thinking => ("[thinking] ", Style::default().fg(self.theme.dim)),
            ChatRole::Tool => ("[tool] ", Style::default().fg(self.theme.warning)),
            ChatRole::System => ("[system] ", Style::default().fg(self.theme.dim)),
        };

        let text = if msg.role == ChatRole::Tool {
            let body = truncate_output(&msg.text, 500);
            if let Some(name) = msg.tool_name.as_deref() {
                format!("{name}: {body}")
            } else {
                body
            }
        } else {
            msg.text.clone()
        };

        let cursor = if msg.is_streaming {
            Some(Span::styled(
                "\u{258c}",
                Style::default().fg(self.theme.accent),
            ))
        } else {
            None
        };

        let markdown_width = if msg.role == ChatRole::User {
            content_width
                .saturating_sub(USER_BUBBLE_OUTER_MARGIN * 2)
                .saturating_sub(USER_BUBBLE_INNER_PAD * 2)
        } else {
            content_width
        };

        let mut lines = compact_blank_lines(format_markdown(
            &text,
            role_style,
            &self.theme,
            prefix,
            markdown_width,
        ));
        if msg.role == ChatRole::User {
            lines = wrap_user_bubble(
                lines,
                self.theme.user_bubble,
                content_width,
                USER_BUBBLE_OUTER_MARGIN,
                USER_BUBBLE_INNER_PAD,
            );
        }

        if let Some(ref c) = cursor {
            if lines.is_empty() {
                lines.push(Line::from(vec![
                    Span::styled(prefix.to_string(), role_style),
                    c.clone(),
                ]));
            } else if let Some(last) = lines.last_mut() {
                last.spans.push(c.clone());
            }
        }

        lines
    }
}

fn wrap_user_bubble(
    mut lines: Vec<Line<'static>>,
    bg: Color,
    content_width: usize,
    outer_margin: usize,
    inner_pad: usize,
) -> Vec<Line<'static>> {
    if lines.is_empty() || content_width == 0 {
        return lines;
    }
    let usable_width = content_width.saturating_sub(outer_margin * 2);
    if usable_width == 0 {
        return lines;
    }

    let max_line_width = lines
        .iter()
        .map(|line| {
            let text = line
                .spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>();
            UnicodeWidthStr::width(text.as_str())
        })
        .max()
        .unwrap_or(0);

    let min_inner_width = inner_pad * 2 + 1;
    let bubble_inner_width = (max_line_width + inner_pad * 2)
        .min(usable_width)
        .max(min_inner_width);
    let bubble_style = Style::default().bg(bg);
    let margin = " ".repeat(outer_margin);

    let mut out = Vec::with_capacity(lines.len());

    for line in &mut lines {
        for span in &mut line.spans {
            span.style = span.style.bg(bg);
        }
        let line_text = line
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<String>();
        let line_width = UnicodeWidthStr::width(line_text.as_str());
        let fill_width = bubble_inner_width.saturating_sub(inner_pad * 2 + line_width);

        let mut spans = Vec::with_capacity(line.spans.len() + 4);
        spans.push(Span::raw(margin.clone()));
        spans.push(Span::styled(" ".repeat(inner_pad), bubble_style));
        spans.extend(line.spans.clone());
        spans.push(Span::styled(" ".repeat(fill_width), bubble_style));
        spans.push(Span::styled(" ".repeat(inner_pad), bubble_style));
        out.push(Line::from(spans));
    }

    out
}

impl Component for Chat {
    fn render(&mut self, area: Rect, frame: &mut Frame) {
        let render_start = std::time::Instant::now();
        let block = Block::default()
            .borders(Borders::NONE)
            .padding(Padding::horizontal(1));
        let inner = block.inner(area);
        let inner_width = area.width.saturating_sub(2) as usize;
        self.rebuild_render_cache(inner_width);

        let viewport_height = area.height as usize;
        let total_visual_rows = self.cached_wrapped_lines.len();
        let max_scroll = total_visual_rows.saturating_sub(viewport_height);
        if self.auto_follow_tail {
            self.scroll_top = max_scroll;
        }
        self.clamp_scroll_to_bounds(max_scroll);
        let scroll_top = self.scroll_top;

        let mut visible = self
            .cached_wrapped_lines
            .iter()
            .skip(scroll_top)
            .take(inner.height as usize)
            .cloned()
            .collect::<Vec<_>>();

        if self.selecting && let (Some(anchor), Some(head)) = (self.select_anchor, self.select_head) {
            let (start, end) = ordered_selection(anchor, head);
            let visible_last = visible.len().saturating_sub(1);
            let from_line = start.0.min(visible_last);
            let to_line = end.0.min(visible_last);
            for visible_line_idx in from_line..=to_line {
                if let Some(line) = visible.get_mut(visible_line_idx) {
                    let from = if visible_line_idx == start.0 {
                        start.1
                    } else {
                        0
                    };
                    let to = if visible_line_idx == end.0 {
                        end.1
                    } else {
                        usize::MAX
                    };
                    highlight_line_range(line, from, to, self.theme.highlight);
                }
            }
        }

        let para = Paragraph::new(Text::from(visible.clone()))
            .block(block)
            .scroll((0, 0));

        frame.render_widget(para, area);

        let start = scroll_top.min(self.cached_visible_line_texts.len());
        let end = (start + inner.height as usize).min(self.cached_visible_line_texts.len());
        self.last_visible_lines = self.cached_visible_line_texts[start..end]
            .iter()
            .map(|text| VisibleLine {
                text: text.clone(),
                url_ranges: extract_url_ranges(text),
            })
            .collect();
        self.last_inner_area = Some(inner);
        tracing::debug!(
            elapsed_ms = render_start.elapsed().as_millis(),
            visible_lines = self.last_visible_lines.len(),
            "chat render"
        );
    }

    fn handle_event(&mut self, event: &Event) -> Option<Action> {
        match event {
            Event::Key(key) if self.focused => match key.code {
                KeyCode::Up => {
                    self.scroll_top = self.scroll_top.saturating_sub(1);
                    self.auto_follow_tail = false;
                }
                KeyCode::Down => {
                    self.scroll_top = self.scroll_top.saturating_add(1);
                }
                KeyCode::PageUp => {
                    self.scroll_top = self.scroll_top.saturating_sub(10);
                    self.auto_follow_tail = false;
                }
                KeyCode::PageDown => {
                    self.scroll_top = self.scroll_top.saturating_add(10);
                }
                KeyCode::Home => {
                    self.scroll_top = 0;
                    self.auto_follow_tail = false;
                }
                KeyCode::End => {
                    self.auto_follow_tail = true;
                }
                _ => {}
            },
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollUp => {
                    self.scroll_top = self.scroll_top.saturating_sub(3);
                    self.auto_follow_tail = false;
                }
                MouseEventKind::ScrollDown => {
                    self.scroll_top = self.scroll_top.saturating_add(3);
                }
                MouseEventKind::Down(MouseButton::Left) => {
                    if let Some(pos) = self.mouse_to_cell(mouse.column, mouse.row) {
                        if mouse
                            .modifiers
                            .contains(crossterm::event::KeyModifiers::CONTROL)
                            && let Some(url) = self.url_at(pos)
                        {
                            self.selecting = false;
                            return Some(Action::OpenUrl(url));
                        }
                        self.select_anchor = Some(pos);
                        self.select_head = Some(pos);
                        self.selecting = true;
                        if let Some(text) = self.selection_text(pos) {
                            return Some(Action::CopySelection(text));
                        }
                    }
                }
                MouseEventKind::Drag(MouseButton::Left) => {
                    if self.selecting
                        && let Some(pos) = self.mouse_to_cell(mouse.column, mouse.row)
                    {
                        self.select_head = Some(pos);
                    }
                    if self.selecting
                        && let Some(pos) = self.mouse_to_cell(mouse.column, mouse.row)
                        && let Some(text) = self.selection_text(pos)
                    {
                        return Some(Action::CopySelection(text));
                    }
                }
                MouseEventKind::Up(MouseButton::Left) => {
                    if self.selecting
                        && let Some(pos) = self.mouse_to_cell(mouse.column, mouse.row)
                    {
                        self.select_head = Some(pos);
                    }
                    if self.selecting
                        && let Some(pos) = self.mouse_to_cell(mouse.column, mouse.row)
                        && let Some(text) = self.selection_text(pos)
                    {
                        self.selecting = false;
                        self.select_anchor = None;
                        self.select_head = None;
                        return Some(Action::CopySelection(text));
                    }
                    self.selecting = false;
                    self.select_anchor = None;
                    self.select_head = None;
                }
                _ => {}
            },
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

impl Chat {
    fn rebuild_render_cache(&mut self, inner_width: usize) {
        if !self.cache_dirty && self.cached_inner_width == Some(inner_width) {
            return;
        }
        let rebuild_start = std::time::Instant::now();
        if self.cached_inner_width == Some(inner_width)
            && self.messages.len() == self.cached_message_count + 1
            && inner_width > 0
            && let Some(msg) = self.messages.last()
        {
            if self.cached_message_count > 0
                && let Some(prev) = self.messages.get(self.cached_message_count - 1)
                && should_insert_gap(prev.role.clone(), msg.role.clone())
            {
                self.cached_visible_line_texts.push(String::new());
                self.cached_wrapped_lines.push(Line::raw(""));
            }
            let start_line = self.cached_wrapped_lines.len();
            let lines = self.format_message(msg, inner_width);
            for line in wrap_styled_lines(&lines, inner_width) {
                self.cached_visible_line_texts.push(line_text(&line));
                self.cached_wrapped_lines.push(line);
            }
            let end_line = self.cached_wrapped_lines.len();
            self.cached_msg_ranges.push((start_line, end_line));
            self.cached_message_count = self.messages.len();
            self.cache_dirty = false;
            tracing::debug!(
                elapsed_ms = rebuild_start.elapsed().as_millis(),
                wrapped_lines = self.cached_wrapped_lines.len(),
                "chat cache incremental append"
            );
            return;
        }
        self.cached_inner_width = Some(inner_width);
        self.cached_wrapped_lines.clear();
        self.cached_visible_line_texts.clear();
        self.cached_msg_ranges.clear();
        if inner_width == 0 {
            self.cache_dirty = false;
            return;
        }

        let mut prev_role: Option<ChatRole> = None;
        for msg in &self.messages {
            if let Some(prev) = prev_role.clone()
                && should_insert_gap(prev, msg.role.clone())
            {
                let _line_idx = self.cached_wrapped_lines.len();
                self.cached_visible_line_texts.push(String::new());
                self.cached_wrapped_lines.push(Line::raw(""));
            }
            let start_line = self.cached_wrapped_lines.len();
            let lines = self.format_message(msg, inner_width);
            for line in wrap_styled_lines(&lines, inner_width) {
                self.cached_visible_line_texts.push(line_text(&line));
                self.cached_wrapped_lines.push(line);
            }
            let end_line = self.cached_wrapped_lines.len();
            self.cached_msg_ranges.push((start_line, end_line));
            prev_role = Some(msg.role.clone());
        }
        self.cached_message_count = self.messages.len();
        self.cache_dirty = false;
        tracing::debug!(
            elapsed_ms = rebuild_start.elapsed().as_millis(),
            wrapped_lines = self.cached_wrapped_lines.len(),
            "chat cache rebuild"
        );
    }

    /// Re-format and re-wrap a specific message in the cached lines,
    /// replacing its previous range in-place. Avoids full rebuild on every
    /// streaming token delta.
    fn replace_msg_in_cache(&mut self, msg_idx: usize, inner_width: usize) {
        let Some((start, end)) = self.cached_msg_ranges.get(msg_idx).copied() else {
            self.cache_dirty = true;
            return;
        };
        let len = end.saturating_sub(start);
        if len > 0 && start < self.cached_wrapped_lines.len() {
            self.cached_wrapped_lines
                .drain(start..end.min(self.cached_wrapped_lines.len()));
            self.cached_visible_line_texts
                .drain(start..end.min(self.cached_visible_line_texts.len()));
        }
        let msg = &self.messages[msg_idx];
        let lines = self.format_message(msg, inner_width);
        let new_lines: Vec<Line<'static>> = wrap_styled_lines(&lines, inner_width);
        let new_texts: Vec<String> = new_lines.iter().map(line_text).collect();
        let new_count = new_lines.len();
        // Splice new lines at the same position
        let insert_pos = start.min(self.cached_wrapped_lines.len());
        for (i, line) in new_lines.into_iter().enumerate() {
            let pos = insert_pos + i;
            if pos < self.cached_wrapped_lines.len() {
                self.cached_wrapped_lines[pos] = line;
            } else {
                self.cached_wrapped_lines.push(line);
            }
        }
        for (i, text) in new_texts.into_iter().enumerate() {
            let pos = insert_pos + i;
            if pos < self.cached_visible_line_texts.len() {
                self.cached_visible_line_texts[pos] = text;
            } else {
                self.cached_visible_line_texts.push(text);
            }
        }
        // Update the range for this message
        self.cached_msg_ranges[msg_idx] = (insert_pos, insert_pos + new_count);
        // Shift all subsequent message ranges by the delta
        let delta = new_count as isize - len as isize;
        if delta != 0 {
            for range in self.cached_msg_ranges.iter_mut().skip(msg_idx + 1) {
                range.0 = (range.0 as isize + delta) as usize;
                range.1 = (range.1 as isize + delta) as usize;
            }
        }
    }

    /// Append the just-pushed last message to the render cache immediately,
    /// including a gap line when needed. This keeps the cache in sync so
    /// subsequent streaming deltas hit the fast replace_msg_in_cache path
    /// instead of waiting for the next rebuild_render_cache call.
    fn append_last_to_cache(&mut self) {
        let Some(inner_width) = self.cached_inner_width else {
            self.cache_dirty = true;
            return;
        };
        if inner_width == 0 {
            return;
        }
        let msg_idx = self.messages.len() - 1;
        let msg = &self.messages[msg_idx];
        let insert_gap = if self.cached_message_count > 0
            && let Some(prev) = self.messages.get(self.cached_message_count - 1)
        {
            should_insert_gap(prev.role.clone(), msg.role.clone())
        } else {
            false
        };
        if insert_gap {
            self.cached_visible_line_texts.push(String::new());
            self.cached_wrapped_lines.push(Line::raw(""));
        }
        let start_line = self.cached_wrapped_lines.len();
        let lines = self.format_message(msg, inner_width);
        for line in wrap_styled_lines(&lines, inner_width) {
            self.cached_visible_line_texts.push(line_text(&line));
            self.cached_wrapped_lines.push(line);
        }
        let end_line = self.cached_wrapped_lines.len();
        self.cached_msg_ranges.push((start_line, end_line));
        self.cached_message_count = self.messages.len();
        self.cache_dirty = false;
    }

    /// Update the last message in the render cache incrementally.
    /// Falls back to cache_dirty if the cache is not ready.
    fn update_last_in_cache(&mut self) {
        let Some(inner_width) = self.cached_inner_width else {
            self.cache_dirty = true;
            return;
        };
        if inner_width == 0 || self.cached_msg_ranges.len() != self.messages.len() {
            self.cache_dirty = true;
            return;
        }
        self.replace_msg_in_cache(self.messages.len() - 1, inner_width);
    }

    /// Update a specific message in the render cache incrementally.
    /// Falls back to cache_dirty if the cache is not ready.
    pub fn update_msg_in_cache(&mut self, msg_idx: usize) {
        let Some(inner_width) = self.cached_inner_width else {
            self.cache_dirty = true;
            return;
        };
        if inner_width == 0 || self.cached_msg_ranges.len() != self.messages.len() {
            self.cache_dirty = true;
            return;
        }
        self.replace_msg_in_cache(msg_idx, inner_width);
    }

    fn mouse_to_cell(&self, col: u16, row: u16) -> Option<(usize, usize)> {
        let area = self.last_inner_area?;
        if col < area.x || row < area.y || col >= area.x + area.width || row >= area.y + area.height
        {
            return None;
        }
        let line = (row - area.y) as usize;
        let col = (col - area.x) as usize;
        Some((line, col))
    }

    fn selection_text(&self, head: (usize, usize)) -> Option<String> {
        let anchor = self.select_anchor?;
        let (start, end) = if anchor <= head {
            (anchor, head)
        } else {
            (head, anchor)
        };
        if self.last_visible_lines.is_empty() {
            return None;
        }
        let mut out = String::new();
        for line_idx in start.0..=end.0 {
            let line = self
                .last_visible_lines
                .get(line_idx)
                .map(|l| l.text.as_str())
                .unwrap_or("");
            let chars: Vec<char> = line.chars().collect();
            let from = if line_idx == start.0 {
                start.1.min(chars.len())
            } else {
                0
            };
            let to = if line_idx == end.0 {
                end.1.min(chars.len())
            } else {
                chars.len()
            };
            if from < to {
                out.push_str(&chars[from..to].iter().collect::<String>());
            }
            if line_idx != end.0 {
                out.push('\n');
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }

    fn url_at(&self, pos: (usize, usize)) -> Option<String> {
        let line = self.last_visible_lines.get(pos.0)?;
        let chars: Vec<char> = line.text.chars().collect();
        if chars.is_empty() {
            return None;
        }
        let idx = pos.1.min(chars.len());
        for (start, end, url) in &line.url_ranges {
            if idx >= *start && idx < *end {
                return Some(url.clone());
            }
        }
        None
    }
}

impl Chat {
    fn clamp_scroll_to_bounds(&mut self, max_scroll: usize) {
        self.scroll_top = self.scroll_top.min(max_scroll);
        if self.scroll_top >= max_scroll {
            self.auto_follow_tail = true;
            self.scroll_top = max_scroll;
        }
    }
}

fn ordered_selection(a: (usize, usize), b: (usize, usize)) -> ((usize, usize), (usize, usize)) {
    if a <= b { (a, b) } else { (b, a) }
}

fn highlight_line_range(line: &mut Line<'static>, start_col: usize, end_col: usize, bg: Color) {
    let mut col = 0usize;
    let mut new_spans = Vec::new();
    for span in &line.spans {
        let chars: Vec<char> = span.content.chars().collect();
        if chars.is_empty() {
            continue;
        }
        let span_start = col;
        let span_end = col + chars.len();
        col = span_end;
        let sel_start = start_col.max(span_start);
        let sel_end = end_col.min(span_end);
        if sel_start >= sel_end {
            new_spans.push(span.clone());
            continue;
        }
        let rel_a = sel_start - span_start;
        let rel_b = sel_end - span_start;
        if rel_a > 0 {
            new_spans.push(Span::styled(
                chars[..rel_a].iter().collect::<String>(),
                span.style,
            ));
        }
        new_spans.push(Span::styled(
            chars[rel_a..rel_b].iter().collect::<String>(),
            span.style.bg(bg),
        ));
        if rel_b < chars.len() {
            new_spans.push(Span::styled(
                chars[rel_b..].iter().collect::<String>(),
                span.style,
            ));
        }
    }
    line.spans = new_spans;
}

fn truncate_output(text: &str, max_len: usize) -> String {
    if text.chars().count() <= max_len {
        text.to_string()
    } else {
        let truncated: String = text.chars().take(max_len).collect();
        format!("{}... ({} chars total)", truncated, text.chars().count())
    }
}

fn should_insert_gap(prev: ChatRole, curr: ChatRole) -> bool {
    role_group(prev) != role_group(curr)
}

fn role_group(role: ChatRole) -> u8 {
    match role {
        ChatRole::User => 1,
        ChatRole::Assistant | ChatRole::Thinking => 2,
        ChatRole::Tool => 3,
        ChatRole::System => 4,
    }
}

fn compact_blank_lines(lines: Vec<Line<'static>>) -> Vec<Line<'static>> {
    let mut out = Vec::with_capacity(lines.len());
    let mut prev_blank = false;
    for line in lines {
        let is_blank = line
            .spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<String>()
            .trim()
            .is_empty();
        if is_blank && prev_blank {
            continue;
        }
        prev_blank = is_blank;
        out.push(line);
    }
    out
}

fn wrap_styled_lines(lines: &[Line<'static>], width: usize) -> Vec<Line<'static>> {
    if width == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    for line in lines {
        if line.spans.is_empty() {
            out.push(Line::raw(""));
            continue;
        }
        let mut current_spans: Vec<Span<'static>> = Vec::new();
        let mut current_width = 0usize;
        for span in &line.spans {
            let mut buf = String::new();
            for ch in span.content.chars() {
                let ch_w = UnicodeWidthChar::width(ch).unwrap_or(0).max(1);
                if current_width + ch_w > width && (current_width > 0 || !buf.is_empty()) {
                    if !buf.is_empty() {
                        current_spans.push(Span::styled(std::mem::take(&mut buf), span.style));
                    }
                    out.push(Line::from(std::mem::take(&mut current_spans)));
                    current_width = 0;
                }
                buf.push(ch);
                current_width += ch_w;
            }
            if !buf.is_empty() {
                current_spans.push(Span::styled(buf, span.style));
            }
        }
        out.push(Line::from(current_spans));
    }
    out
}

fn line_text(line: &Line<'static>) -> String {
    line.spans
        .iter()
        .map(|s| s.content.as_ref())
        .collect::<String>()
}

fn extract_url_ranges(text: &str) -> Vec<(usize, usize, String)> {
    let chars: Vec<char> = text.chars().collect();
    let mut ranges = Vec::new();
    let mut i = 0usize;
    while i < chars.len() {
        let rest: String = chars[i..].iter().collect();
        let offset = if rest.starts_with("http://") || rest.starts_with("https://") {
            Some(0usize)
        } else if let Some(pos) = rest.find(" http://") {
            Some(pos + 1)
        } else {
            rest.find(" https://").map(|pos| pos + 1)
        };
        let Some(off) = offset else {
            break;
        };
        i += off;
        let start = i;
        while i < chars.len() && !chars[i].is_whitespace() {
            i += 1;
        }
        let mut url: String = chars[start..i].iter().collect();
        while url.chars().last().is_some_and(|c| ",.;:)]}\"'".contains(c)) {
            url.pop();
            i = i.saturating_sub(1);
        }
        if url.starts_with("http://") || url.starts_with("https://") {
            ranges.push((start, i, url));
        }
    }
    ranges
}

// ---------------------------------------------------------------------------
// Markdown formatting
// ---------------------------------------------------------------------------

/// Parse text line-by-line and produce styled Lines.
fn format_markdown(
    text: &str,
    base_style: Style,
    theme: &Theme,
    prefix: &str,
    content_width: usize,
) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    let mut line = if prefix.is_empty() {
        Vec::new()
    } else {
        vec![Span::styled(prefix.to_string(), base_style)]
    };
    let mut style_stack = vec![base_style];
    let mut block_stack: Vec<BlockState> = Vec::new();
    let mut in_item = false;
    let mut current_item_continuation_prefix = String::new();
    let mut code_block_lang: Option<String> = None;
    let mut in_code_block = false;
    let mut link_targets: Vec<String> = Vec::new();
    let mut pending_task_marker: Option<bool> = None;
    let mut in_table = false;
    let mut in_table_cell = false;
    let mut table_header_rows = 0usize;
    let mut table_rows: Vec<Vec<String>> = Vec::new();
    let mut current_table_row: Vec<String> = Vec::new();
    let mut current_table_cell = String::new();

    let mut opts = Options::empty();
    opts.insert(Options::ENABLE_STRIKETHROUGH);
    opts.insert(Options::ENABLE_TABLES);
    opts.insert(Options::ENABLE_TASKLISTS);
    opts.insert(Options::ENABLE_FOOTNOTES);

    for event in Parser::new_ext(text, opts) {
        match event {
            MdEvent::Start(tag) => match tag {
                Tag::Heading { level, .. } => {
                    flush_line(&mut out, &mut line);
                    let heading_style = match level as u8 {
                        1 => Style::default()
                            .fg(theme.md_heading_1)
                            .add_modifier(Modifier::BOLD),
                        2 => Style::default()
                            .fg(theme.md_heading_2)
                            .add_modifier(Modifier::BOLD),
                        _ => Style::default()
                            .fg(theme.md_heading_2)
                            .add_modifier(Modifier::BOLD),
                    };
                    style_stack.push(base_style.patch(heading_style));
                }
                Tag::Strong => style_stack.push(
                    style_stack
                        .last()
                        .copied()
                        .unwrap_or(base_style)
                        .add_modifier(Modifier::BOLD),
                ),
                Tag::Emphasis => style_stack.push(
                    style_stack
                        .last()
                        .copied()
                        .unwrap_or(base_style)
                        .add_modifier(Modifier::ITALIC),
                ),
                Tag::Strikethrough => style_stack.push(
                    style_stack
                        .last()
                        .copied()
                        .unwrap_or(base_style)
                        .add_modifier(Modifier::CROSSED_OUT),
                ),
                Tag::Link { dest_url, .. } => {
                    style_stack.push(
                        style_stack
                            .last()
                            .copied()
                            .unwrap_or(base_style)
                            .fg(theme.md_link)
                            .add_modifier(Modifier::UNDERLINED),
                    );
                    link_targets.push(dest_url.to_string());
                }
                Tag::BlockQuote(_) => block_stack.push(BlockState::Quote),
                Tag::List(start) => block_stack.push(BlockState::List {
                    ordered: start.is_some(),
                    next: start.unwrap_or(1),
                }),
                Tag::Item => {
                    flush_line(&mut out, &mut line);
                    let indent = list_indent(&block_stack);
                    let marker = list_marker(&mut block_stack);
                    line.push(Span::styled(
                        format!("{indent}{marker}"),
                        Style::default().fg(theme.md_list_marker),
                    ));
                    current_item_continuation_prefix =
                        format!("{indent}{}", " ".repeat(marker.chars().count()));
                    if let Some(checked) = pending_task_marker.take() {
                        let marker = if checked { "☑ " } else { "☐ " };
                        line.push(Span::styled(
                            marker.to_string(),
                            Style::default().fg(theme.md_task_marker),
                        ));
                        current_item_continuation_prefix
                            .push_str(&" ".repeat(marker.chars().count()));
                    }
                    in_item = true;
                }
                Tag::CodeBlock(kind) => {
                    flush_line(&mut out, &mut line);
                    in_code_block = true;
                    code_block_lang = match kind {
                        CodeBlockKind::Fenced(lang) => {
                            let l = lang.trim().to_lowercase();
                            if l.is_empty() { None } else { Some(l) }
                        }
                        CodeBlockKind::Indented => None,
                    };
                }
                Tag::Table(_) => {
                    flush_line(&mut out, &mut line);
                    in_table = true;
                    table_header_rows = 0;
                    table_rows.clear();
                    current_table_row.clear();
                    current_table_cell.clear();
                }
                Tag::TableHead => {}
                Tag::TableRow => {
                    current_table_row.clear();
                }
                Tag::TableCell => {
                    in_table_cell = true;
                    current_table_cell.clear();
                }
                Tag::Paragraph if !line.is_empty() && !only_prefix(&line, prefix) => {
                    flush_line(&mut out, &mut line);
                }
                Tag::Paragraph => {}
                _ => {}
            },
            MdEvent::End(tag) => match tag {
                TagEnd::Heading(_) => {
                    if style_stack.len() > 1 {
                        style_stack.pop();
                    }
                    flush_line(&mut out, &mut line);
                }
                TagEnd::Strong | TagEnd::Emphasis | TagEnd::Strikethrough
                    if style_stack.len() > 1 =>
                {
                    style_stack.pop();
                }
                TagEnd::Strong | TagEnd::Emphasis | TagEnd::Strikethrough => {}
                TagEnd::Link => {
                    if style_stack.len() > 1 {
                        style_stack.pop();
                    }
                    if let Some(url) = link_targets.pop() {
                        line.push(Span::styled(
                            format!(" ({url})"),
                            Style::default().fg(theme.md_quote),
                        ));
                    }
                }
                TagEnd::BlockQuote(_) => {
                    pop_last_matching(&mut block_stack, |b| matches!(b, BlockState::Quote));
                    flush_line(&mut out, &mut line);
                }
                TagEnd::List(_) => {
                    pop_last_matching(&mut block_stack, |b| matches!(b, BlockState::List { .. }));
                    flush_line(&mut out, &mut line);
                }
                TagEnd::Item => {
                    flush_line(&mut out, &mut line);
                    in_item = false;
                    current_item_continuation_prefix.clear();
                }
                TagEnd::CodeBlock => {
                    in_code_block = false;
                    code_block_lang = None;
                    flush_line(&mut out, &mut line);
                }
                TagEnd::TableCell => {
                    in_table_cell = false;
                    current_table_row.push(current_table_cell.trim().to_string());
                    current_table_cell.clear();
                }
                TagEnd::TableRow if !current_table_row.is_empty() => {
                    table_rows.push(std::mem::take(&mut current_table_row));
                }
                TagEnd::TableRow => {}
                TagEnd::TableHead => {
                    table_header_rows = table_rows.len();
                }
                TagEnd::Table => {
                    in_table = false;
                    flush_line(&mut out, &mut line);
                    out.extend(render_table(
                        &table_rows,
                        table_header_rows,
                        base_style,
                        theme,
                        prefix,
                        content_width,
                    ));
                    table_rows.clear();
                }
                TagEnd::Paragraph => {
                    flush_line(&mut out, &mut line);
                }
                _ => {}
            },
            MdEvent::Text(t) => {
                if in_table && in_table_cell {
                    current_table_cell.push_str(&t);
                    continue;
                }
                if in_code_block {
                    for raw in t.lines() {
                        let mut spans = vec![Span::styled(
                            format!("{prefix}\u{2503} "),
                            Style::default().fg(theme.md_rule_border),
                        )];
                        spans.extend(highlight_code_line(raw, code_block_lang.as_deref(), theme));
                        out.push(Line::from(spans));
                    }
                    if t.ends_with('\n') {
                        out.push(Line::from(vec![Span::styled(
                            format!("{prefix}\u{2503}"),
                            Style::default().fg(theme.md_rule_border),
                        )]));
                    }
                } else {
                    for (idx, chunk) in t.split('\n').enumerate() {
                        if idx > 0 {
                            flush_line(&mut out, &mut line);
                            if in_item {
                                line.push(Span::styled(
                                    current_item_continuation_prefix.clone(),
                                    Style::default(),
                                ));
                            }
                        }
                        if chunk.is_empty() {
                            continue;
                        }
                        let mut style = style_stack.last().copied().unwrap_or(base_style);
                        if block_stack.iter().any(|b| matches!(b, BlockState::Quote)) {
                            style = style.fg(theme.md_quote).add_modifier(Modifier::ITALIC);
                        }
                        line.push(Span::styled(chunk.to_string(), style));
                    }
                }
            }
            MdEvent::Code(t) => {
                if in_table && in_table_cell {
                    current_table_cell.push_str(&t);
                    continue;
                }
                let code_style = Style::default()
                    .fg(theme.md_inline_code)
                    .add_modifier(Modifier::ITALIC);
                line.push(Span::styled(t.to_string(), code_style));
            }
            MdEvent::SoftBreak | MdEvent::HardBreak => {
                flush_line(&mut out, &mut line);
                if in_item {
                    line.push(Span::styled(
                        current_item_continuation_prefix.clone(),
                        Style::default(),
                    ));
                }
            }
            MdEvent::Rule => {
                flush_line(&mut out, &mut line);
                out.push(Line::from(vec![
                    Span::styled(prefix.to_string(), base_style),
                    Span::styled(
                        "────────────────",
                        Style::default().fg(theme.md_rule_border),
                    ),
                ]));
            }
            MdEvent::TaskListMarker(checked) => {
                if in_item {
                    let marker = if checked { "☑ " } else { "☐ " };
                    line.push(Span::styled(
                        marker.to_string(),
                        Style::default().fg(theme.md_task_marker),
                    ));
                    current_item_continuation_prefix.push_str(&" ".repeat(marker.chars().count()));
                } else {
                    // Fallback for parser ordering differences.
                    pending_task_marker = Some(checked);
                }
            }
            _ => {}
        }
    }

    flush_line(&mut out, &mut line);
    out
}

fn highlight_code_line(line: &str, lang: Option<&str>, theme: &Theme) -> Vec<Span<'static>> {
    let ps = syntax_set();
    let syntax = lang
        .and_then(map_lang_token)
        .and_then(|token| {
            ps.find_syntax_by_token(token)
                .or_else(|| ps.find_syntax_by_extension(token))
        })
        .unwrap_or_else(|| ps.find_syntax_plain_text());

    let mut highlighter = HighlightLines::new(syntax, syntect_theme());
    let base = Style::default().fg(theme.code_fg.unwrap_or(Color::Cyan));
    let mut spans = Vec::new();
    for snippet in LinesWithEndings::from(line) {
        if let Ok(ranges) = highlighter.highlight_line(snippet, ps) {
            spans.extend(ranges.into_iter().map(|(style, segment)| {
                Span::styled(segment.to_string(), ratatui_style_from_syntect(style, base))
            }));
        } else {
            spans.push(Span::styled(snippet.to_string(), base));
        }
    }
    spans
}

#[derive(Clone, Copy)]
enum BlockState {
    Quote,
    List { ordered: bool, next: u64 },
}

fn flush_line(out: &mut Vec<Line<'static>>, line: &mut Vec<Span<'static>>) {
    if line.is_empty() {
        return;
    }
    out.push(Line::from(std::mem::take(line)));
}

fn only_prefix(line: &[Span<'static>], prefix: &str) -> bool {
    line.len() == 1 && line[0].content.as_ref() == prefix
}

fn list_indent(block_stack: &[BlockState]) -> String {
    let depth = block_stack
        .iter()
        .filter(|b| matches!(b, BlockState::List { .. }))
        .count();
    "  ".repeat(depth.saturating_sub(1))
}

fn list_marker(block_stack: &mut [BlockState]) -> String {
    for block in block_stack.iter_mut().rev() {
        if let BlockState::List { ordered, next } = block {
            if *ordered {
                let marker = format!("{next}. ");
                *next += 1;
                return marker;
            }
            return "• ".to_string();
        }
    }
    "• ".to_string()
}

fn render_table(
    rows: &[Vec<String>],
    header_rows: usize,
    base_style: Style,
    theme: &Theme,
    prefix: &str,
    content_width: usize,
) -> Vec<Line<'static>> {
    if rows.is_empty() {
        return Vec::new();
    }
    let cols = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if cols == 0 {
        return Vec::new();
    }

    let prefix_width = UnicodeWidthStr::width(prefix);
    let available = content_width
        .saturating_sub(prefix_width)
        .max(cols.saturating_mul(2) + 1);
    let separator_width = cols.saturating_mul(3) + 1;
    let cell_budget = available.saturating_sub(separator_width);
    let min_col_width = 4usize;
    let mut col_widths = vec![min_col_width; cols];
    for (c, w) in col_widths.iter_mut().enumerate() {
        *w = rows
            .iter()
            .filter_map(|r| r.get(c))
            .map(|s| UnicodeWidthStr::width(s.as_str()))
            .max()
            .unwrap_or(min_col_width)
            .max(min_col_width);
    }
    let total: usize = col_widths.iter().sum();
    if total > cell_budget && cell_budget > 0 {
        let mut scaled: Vec<usize> = col_widths
            .iter()
            .map(|w| ((*w as f64 / total as f64) * cell_budget as f64).floor() as usize)
            .map(|w| w.max(2))
            .collect();
        let mut remain = cell_budget.saturating_sub(scaled.iter().sum::<usize>());
        let mut idx = 0usize;
        while remain > 0 {
            scaled[idx % cols] += 1;
            remain -= 1;
            idx += 1;
        }
        col_widths = scaled;
    }

    let mut out = Vec::new();
    for (r_idx, row) in rows.iter().enumerate() {
        let wrapped: Vec<Vec<String>> = (0..cols)
            .map(|c| {
                let cell = row.get(c).map(String::as_str).unwrap_or("");
                wrap_to_width(cell, col_widths[c])
            })
            .collect();
        let row_height = wrapped.iter().map(Vec::len).max().unwrap_or(1);
        for h in 0..row_height {
            let mut spans = vec![Span::styled(prefix.to_string(), base_style)];
            spans.push(Span::styled("|", Style::default().fg(theme.md_rule_border)));
            for (c, width) in col_widths.iter().enumerate().take(cols) {
                spans.push(Span::raw(" "));
                let piece = wrapped
                    .get(c)
                    .and_then(|lines| lines.get(h))
                    .cloned()
                    .unwrap_or_default();
                let style = if r_idx < header_rows {
                    Style::default()
                        .fg(theme.md_table_header)
                        .add_modifier(Modifier::BOLD)
                } else {
                    Style::default().fg(theme.fg)
                };
                spans.push(Span::styled(pad_to_width(&piece, *width), style));
                spans.push(Span::raw(" "));
                spans.push(Span::styled("|", Style::default().fg(theme.md_rule_border)));
            }
            out.push(Line::from(spans));
        }
        if r_idx + 1 == header_rows {
            out.push(table_separator_line(prefix, &col_widths, theme, base_style));
        }
    }
    out
}

fn table_separator_line(
    prefix: &str,
    widths: &[usize],
    theme: &Theme,
    base_style: Style,
) -> Line<'static> {
    let mut spans = vec![Span::styled(prefix.to_string(), base_style)];
    spans.push(Span::styled("|", Style::default().fg(theme.md_rule_border)));
    for width in widths {
        spans.push(Span::styled(
            format!("{}|", "-".repeat(width + 2)),
            Style::default().fg(theme.md_rule_border),
        ));
    }
    Line::from(spans)
}

fn wrap_to_width(text: &str, width: usize) -> Vec<String> {
    if width == 0 {
        return vec![String::new()];
    }
    let mut out = Vec::new();
    for paragraph in text.split('\n') {
        let mut line = String::new();
        for word in paragraph.split_whitespace() {
            if line.is_empty() {
                if UnicodeWidthStr::width(word) <= width {
                    line.push_str(word);
                } else {
                    for chunk in hard_wrap(word, width) {
                        out.push(chunk);
                    }
                }
                continue;
            }
            let candidate = format!("{line} {word}");
            if UnicodeWidthStr::width(candidate.as_str()) <= width {
                line = candidate;
            } else {
                out.push(std::mem::take(&mut line));
                if UnicodeWidthStr::width(word) <= width {
                    line.push_str(word);
                } else {
                    for chunk in hard_wrap(word, width) {
                        out.push(chunk);
                    }
                }
            }
        }
        if !line.is_empty() {
            out.push(line);
        } else if paragraph.is_empty() {
            out.push(String::new());
        }
    }
    if out.is_empty() {
        vec![String::new()]
    } else {
        out
    }
}

fn hard_wrap(text: &str, width: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        current.push(ch);
        if UnicodeWidthStr::width(current.as_str()) >= width {
            out.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    if out.is_empty() {
        vec![String::new()]
    } else {
        out
    }
}

fn pad_to_width(text: &str, width: usize) -> String {
    let current = UnicodeWidthStr::width(text);
    if current >= width {
        text.to_string()
    } else {
        format!("{text}{}", " ".repeat(width - current))
    }
}

fn pop_last_matching<F>(stack: &mut Vec<BlockState>, mut predicate: F)
where
    F: FnMut(&BlockState) -> bool,
{
    if let Some(pos) = stack.iter().rposition(&mut predicate) {
        stack.remove(pos);
    }
}

fn syntax_set() -> &'static SyntaxSet {
    static SYNTAX_SET: OnceLock<SyntaxSet> = OnceLock::new();
    SYNTAX_SET.get_or_init(SyntaxSet::load_defaults_newlines)
}

fn syntect_theme() -> &'static SyntectTheme {
    static SELECTED_THEME: OnceLock<SyntectTheme> = OnceLock::new();
    SELECTED_THEME.get_or_init(|| {
        let set = ThemeSet::load_defaults();
        set.themes
            .get("base16-ocean.dark")
            .or_else(|| set.themes.get("InspiredGitHub"))
            .or_else(|| set.themes.values().next())
            .cloned()
            .unwrap_or_default()
    })
}

fn ratatui_style_from_syntect(s: SyntectStyle, base: Style) -> Style {
    let mut style = base.fg(Color::Rgb(s.foreground.r, s.foreground.g, s.foreground.b));
    if s.font_style
        .contains(syntect::highlighting::FontStyle::BOLD)
    {
        style = style.add_modifier(Modifier::BOLD);
    }
    if s.font_style
        .contains(syntect::highlighting::FontStyle::ITALIC)
    {
        style = style.add_modifier(Modifier::ITALIC);
    }
    if s.font_style
        .contains(syntect::highlighting::FontStyle::UNDERLINE)
    {
        style = style.add_modifier(Modifier::UNDERLINED);
    }
    style
}

fn map_lang_token(lang: &str) -> Option<&str> {
    match lang {
        "rs" => Some("rust"),
        "js" => Some("javascript"),
        "ts" => Some("typescript"),
        "py" => Some("python"),
        "sh" => Some("bash"),
        "yml" => Some("yaml"),
        "shell" => Some("bash"),
        "" => None,
        other => Some(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rendered_text(lines: &[Line<'static>]) -> String {
        lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn normalized_rendered_text(lines: &[Line<'static>]) -> String {
        let raw = rendered_text(lines);
        let mut rows: Vec<&str> = raw.lines().collect();
        while rows.first().is_some_and(|r| r.is_empty()) {
            rows.remove(0);
        }
        while rows.last().is_some_and(|r| r.is_empty()) {
            rows.pop();
        }
        rows.join("\n")
    }

    #[test]
    fn test_markdown_headers() {
        let theme = Theme::default();
        let style = Style::default();
        let lines = format_markdown("# Top\n## Mid\n### Low\ntext", style, &theme, "", 80);
        let rendered: Vec<String> = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        assert!(rendered.iter().any(|l| l.contains("Top")));
        assert!(rendered.iter().any(|l| l.contains("Mid")));
        assert!(rendered.iter().any(|l| l.contains("Low")));
        assert!(rendered.iter().any(|l| l.contains("text")));
    }

    #[test]
    fn test_code_block() {
        let theme = Theme::default();
        let style = Style::default();
        let lines = format_markdown(
            "before\n```rust\nlet x = 1;\n```\nafter",
            style,
            &theme,
            "",
            80,
        );
        // before, code header, code line, after
        assert!(lines.len() >= 3);
    }

    #[test]
    fn test_task_list_markers() {
        let theme = Theme::default();
        let style = Style::default();
        let lines = format_markdown("- [ ] todo\n- [x] done", style, &theme, "", 80);
        let rendered: Vec<String> = lines
            .iter()
            .map(|l| {
                l.spans
                    .iter()
                    .map(|s| s.content.as_ref())
                    .collect::<String>()
            })
            .collect();
        assert!(rendered.iter().any(|l| l.contains("☐")));
        assert!(rendered.iter().any(|l| l.contains("☑")));
    }

    #[test]
    fn test_skill_invocation_prefix_rendered() {
        let chat = Chat::new(Theme::default());
        let lines = chat.format_message(
            &ChatMessage {
                role: ChatRole::User,
                text: "/skill:git-commit".into(),
                tool_name: None,
                is_streaming: false,
            },
            80,
        );
        let has_marker = lines.iter().any(|l| {
            let txt = l
                .spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>();
            txt.contains("◈ ")
        });
        assert!(has_marker);
    }

    #[test]
    fn test_user_message_uses_bubble_background() {
        let chat = Chat::new(Theme::default());
        let lines = chat.format_message(
            &ChatMessage {
                role: ChatRole::User,
                text: "hello".into(),
                tool_name: None,
                is_streaming: false,
            },
            80,
        );
        let has_bubble_bg = lines
            .iter()
            .flat_map(|l| l.spans.iter())
            .any(|s| s.style.bg == Some(chat.theme.user_bubble));
        assert!(has_bubble_bg);
    }

    #[test]
    fn test_add_message_does_not_force_scroll_to_bottom() {
        let mut chat = Chat::new(Theme::default());
        chat.scroll_top = 5;
        chat.auto_follow_tail = false;
        chat.add_message(ChatMessage {
            role: ChatRole::Assistant,
            text: "new content".into(),
            tool_name: None,
            is_streaming: false,
        });
        assert_eq!(chat.scroll_top, 5);
        assert!(!chat.auto_follow_tail);
    }

    #[test]
    fn test_markdown_golden_nested_lists_and_links() {
        let theme = Theme::default();
        let style = Style::default();
        let md = "# Title\n- parent\n  - child with [link](https://example.com)\n> quote";
        let lines = format_markdown(md, style, &theme, "", 80);
        let got = normalized_rendered_text(&lines);
        let expected = "Title\n• parent\n  • child with link (https://example.com)\nquote";
        assert_eq!(got, expected);
    }

    #[test]
    fn test_markdown_golden_table_wraps_inside_width() {
        let theme = Theme::default();
        let style = Style::default();
        let md = "| Name | Description |\n| --- | --- |\n| A | verylongtoken_without_spaces_that_must_wrap |\n| B | short |";
        let lines = format_markdown(md, style, &theme, "", 34);
        let got = normalized_rendered_text(&lines);
        let expected = "\
| A   | verylongtoken_without_sp |
|     | aces_that_must_wrap      |
| B   | short                    |";
        assert_eq!(got, expected);
    }

    #[test]
    fn compact_tool_completion_updates_started_row() {
        let mut chat = Chat::new(Theme::default());
        chat.add_message(ChatMessage {
            role: ChatRole::Tool,
            text: "running".to_string(),
            tool_name: Some("read".to_string()),
            is_streaming: true,
        });
        chat.complete_tool_compact("read", "done: src/main.rs");
        assert_eq!(chat.messages.len(), 1);
        assert_eq!(chat.messages[0].text, "done: src/main.rs");
        assert!(!chat.messages[0].is_streaming);
    }

    #[test]
    fn inserts_gap_between_role_groups() {
        assert!(should_insert_gap(ChatRole::User, ChatRole::Assistant));
        assert!(!should_insert_gap(ChatRole::Assistant, ChatRole::Thinking));
        assert!(should_insert_gap(ChatRole::Assistant, ChatRole::Tool));
    }

    #[test]
    #[ignore = "perf characterization; run manually"]
    fn perf_large_history_render_cache() {
        let mut chat = Chat::new(Theme::default());
        for i in 0..2500 {
            chat.add_message(ChatMessage {
                role: if i % 2 == 0 {
                    ChatRole::User
                } else {
                    ChatRole::Assistant
                },
                text: format!("message {i} {}", "x".repeat(120)),
                tool_name: None,
                is_streaming: false,
            });
        }
        let start = std::time::Instant::now();
        chat.rebuild_render_cache(120);
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs() < 10);
    }

    #[test]
    fn test_clear_messages() {
        let mut chat = Chat::new(Theme::default());
        chat.add_message(ChatMessage {
            role: ChatRole::User,
            text: "hello".into(),
            tool_name: None,
            is_streaming: false,
        });
        chat.add_message(ChatMessage {
            role: ChatRole::Assistant,
            text: "hi".into(),
            tool_name: None,
            is_streaming: false,
        });
        assert_eq!(chat.messages.len(), 2);

        chat.clear_messages();

        assert!(chat.messages.is_empty());
        assert!(chat.active_tool_message_idx.is_empty());
        assert_eq!(chat.cached_message_count, 0);
        assert!(chat.cache_dirty);
    }
}
