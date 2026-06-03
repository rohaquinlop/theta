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

/// A chat message to display.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub text: String,
    pub tool_call_id: Option<String>,
    pub is_streaming: bool,
    pub is_error: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChatRole {
    User,
    Assistant,
    Thinking,
    Tool,
    Skill,
    System,
}

/// Scrollable chat message list.
pub struct Chat {
    pub messages: Vec<ChatMessage>,
    pub scroll_top: usize,
    pub auto_follow_tail: bool,
    pub theme: Theme,
    focused: bool,
    last_visible_lines: Vec<VisibleLine>,
    last_inner_area: Option<Rect>,
    select_anchor: Option<(usize, usize)>,
    select_head: Option<(usize, usize)>,
    selecting: bool,
    pub active_tool_message_idx: HashMap<String, usize>,
    cached_inner_width: Option<usize>,
    cached_wrapped_lines: Vec<Line<'static>>,
    cached_visible_line_texts: Vec<String>,
    pub cached_msg_ranges: Vec<(usize, usize)>,
    pub cached_message_count: usize,
    pub cache_dirty: bool,
}

#[derive(Debug, Clone)]
struct VisibleLine {
    text: String,
    url_ranges: Vec<(usize, usize, String)>,
}

impl Chat {
    #[cfg(debug_assertions)]
    /// Benchmark helper: simulate old no-cache path by formatting and wrapping
    /// the whole transcript every call. Returns wrapped line count.
    pub fn benchmark_full_rebuild_no_cache(&self, inner_width: usize) -> usize {
        let mut lines: Vec<Line> = Vec::new();
        for msg in &self.messages {
            lines.extend(self.format_message(msg, inner_width));
        }
        wrap_styled_lines(&lines, inner_width).len()
    }

    #[cfg(debug_assertions)]
    /// Benchmark helper: rebuild internal cache if needed and return wrapped line count.
    pub fn benchmark_cached_rebuild(&mut self, inner_width: usize) -> usize {
        self.rebuild_render_cache(inner_width);
        self.cached_wrapped_lines.len()
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
        if let Some(call_id) = self.messages.last().and_then(|m| {
            if m.role == ChatRole::Tool && m.is_streaming {
                m.tool_call_id.clone()
            } else {
                None
            }
        }) {
            self.active_tool_message_idx
                .insert(call_id, self.messages.len() - 1);
        }
        // Append to render cache immediately so the cache stays in sync.
        self.append_last_to_cache();
    }

    /// Add or update a tool message, keyed by tool_call_id.
    /// If a message with this call_id already exists, update it in-place.
    /// Otherwise push a new message. Returns the message index.
    pub fn upsert_tool_message(
        &mut self,
        call_id: &str,
        text: &str,
        is_streaming: bool,
        is_error: bool,
    ) -> usize {
        if let Some(&idx) = self.active_tool_message_idx.get(call_id)
            && let Some(msg) = self.messages.get_mut(idx)
            && msg.role == ChatRole::Tool
            && msg.tool_call_id.as_deref() == Some(call_id)
        {
            msg.text = text.to_string();
            msg.is_streaming = is_streaming;
            msg.is_error = is_error;
            self.update_msg_in_cache(idx);
            return idx;
        }
        let idx = self.messages.len();
        self.messages.push(ChatMessage {
            role: ChatRole::Tool,
            text: text.to_string(),
            tool_call_id: Some(call_id.to_string()),
            is_streaming,
            is_error,
        });
        if is_streaming {
            self.active_tool_message_idx
                .insert(call_id.to_string(), idx);
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
            tool_call_id: None,
            is_streaming,
            is_error: false,
        });
        // Append to render cache immediately so subsequent
        // TextDelta events hit the fast incremental path.
        self.append_last_to_cache();
    }

    pub fn complete_tool_compact(&mut self, call_id: &str, text: &str, is_error: bool) {
        if let Some(&idx) = self.active_tool_message_idx.get(call_id)
            && let Some(msg) = self.messages.get_mut(idx)
            && msg.role == ChatRole::Tool
            && msg.tool_call_id.as_deref() == Some(call_id)
        {
            msg.text = text.to_string();
            msg.is_streaming = false;
            msg.is_error = is_error;
            self.active_tool_message_idx.remove(call_id);
            self.update_msg_in_cache(idx);
            return;
        }

        // Fallback: search all tool messages by call_id in case the index
        // entry was lost (e.g. broadcast channel lag dropped ToolCallPrepared).
        if let Some(idx) = self
            .messages
            .iter()
            .rposition(|m| m.role == ChatRole::Tool && m.tool_call_id.as_deref() == Some(call_id))
        {
            self.messages[idx].text = text.to_string();
            self.messages[idx].is_streaming = false;
            self.messages[idx].is_error = is_error;
            self.active_tool_message_idx.remove(call_id);
            self.update_msg_in_cache(idx);
            return;
        }

        self.messages.push(ChatMessage {
            role: ChatRole::Tool,
            text: text.to_string(),
            tool_call_id: Some(call_id.to_string()),
            is_streaming: false,
            is_error,
        });
        self.append_last_to_cache();
    }

    pub fn finish_last(&mut self, role: ChatRole) {
        // Search backwards — the target message may not be the very last
        // (e.g. tool/skill messages can follow the assistant message).
        if let Some(msg) = self.messages.iter_mut().rev().find(|m| m.role == role) {
            msg.is_streaming = false;
            if role == ChatRole::Tool
                && let Some(call_id) = msg.tool_call_id.as_deref()
            {
                self.active_tool_message_idx.remove(call_id);
            }
            // Find the index for cache update.
            if let Some(idx) = self.messages.iter().rposition(|m| m.role == role) {
                self.update_msg_in_cache(idx);
            }
        }
    }

    /// Format a message into styled lines with markdown parsing.
    pub fn format_message(&self, msg: &ChatMessage, content_width: usize) -> Vec<Line<'static>> {
        const USER_BUBBLE_OUTER_MARGIN: usize = 0;
        const USER_BUBBLE_INNER_PAD: usize = 0;

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
            ChatRole::Tool => {
                let style = if msg.is_error {
                    Style::default().fg(self.theme.error)
                } else {
                    Style::default().fg(self.theme.warning)
                };
                ("[tool] ", style)
            }
            ChatRole::System => ("[system] ", Style::default().fg(self.theme.dim)),
            ChatRole::Skill => (
                "▸ skill ",
                Style::default()
                    .fg(self.theme.success)
                    .add_modifier(Modifier::BOLD),
            ),
        };

        // Tool messages already include the full display text
        // (e.g. "bash: ls -la (done)"), so render body as-is.
        let text = if msg.role == ChatRole::Tool {
            truncate_output(&msg.text, 500)
        } else {
            msg.text.clone()
        };

        // read tool: style range "{offset}-{end}" in dim, rest in warning.
        if msg.role == ChatRole::Tool
            && !msg.is_error
            && let Some((label, range)) = split_read_tool_text(&text)
        {
            let label_style = Style::default().fg(self.theme.warning);
            let range_style = Style::default().fg(self.theme.dim);
            let tool_style = Style::default().fg(self.theme.warning);
            return vec![Line::from(vec![
                Span::styled(prefix, tool_style),
                Span::styled(label.to_string(), label_style),
                Span::styled(range.to_string(), range_style),
            ])];
        }

        // edit tool: style [+N/-M] with green +N and red -M.
        if msg.role == ChatRole::Tool
            && !msg.is_error
            && let Some(parts) = split_edit_tool_text(&text)
        {
            let tool_style = Style::default().fg(self.theme.warning);
            let added_style = Style::default().fg(self.theme.success);
            let removed_style = Style::default().fg(self.theme.error);
            return vec![Line::from(vec![
                Span::styled(prefix, tool_style),
                Span::styled(parts.prefix.to_string(), tool_style),
                Span::styled(parts.added.to_string(), added_style),
                Span::styled(parts.slash_minus.to_string(), tool_style),
                Span::styled(parts.removed.to_string(), removed_style),
                Span::styled(parts.suffix.to_string(), tool_style),
            ])];
        }

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

        if self.selecting
            && let (Some(anchor), Some(head)) = (self.select_anchor, self.select_head)
        {
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
    pub fn rebuild_render_cache(&mut self, inner_width: usize) {
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
        // Incremental replacement only works for the last message.
        // Interior messages have inter-message gap lines that the
        // drain-splice cycle doesn't preserve, causing cached_msg_ranges
        // to drift out of sync with the actual cached line positions.
        if msg_idx + 1 < self.messages.len() {
            self.cache_dirty = true;
            return;
        }
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
        // If the new message is shorter (fewer lines), remove stale
        // lines left over from the old message that the overwrite
        // loop didn't consume.
        if new_count < len {
            let overflow = len - new_count;
            let drain_start = insert_pos + new_count;
            let drain_end = (drain_start + overflow)
                .min(self.cached_wrapped_lines.len())
                .min(self.cached_visible_line_texts.len());
            self.cached_wrapped_lines.drain(drain_start..drain_end);
            self.cached_visible_line_texts.drain(drain_start..drain_end);
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

/// Split a read tool summary like "read /tmp/a.rs:11-30" into ("read /tmp/a.rs:", "11-30").
/// Returns None if the text doesn't match the expected read format.
fn split_read_tool_text(text: &str) -> Option<(&str, &str)> {
    if !text.starts_with("read ") {
        return None;
    }
    let rest = text.strip_prefix("read ")?;
    let last_colon = rest.rfind(':')?;
    let range = &rest[last_colon + 1..];
    // Validate range looks like "N-M"
    if range.contains('-') && range.chars().all(|c| c.is_ascii_digit() || c == '-') {
        let split = text.len() - range.len();
        Some((&text[..split], range))
    } else {
        None
    }
}

/// Parts of an edit tool summary with diff, e.g. "edit /tmp/a.rs  [+2/-1]".
struct EditToolParts<'a> {
    prefix: &'a str,      // "edit /tmp/a.rs  [+\"
    added: &'a str,       // "2"
    slash_minus: &'a str, // "/-"
    removed: &'a str,     // "1"
    suffix: &'a str,      // "]"
}

/// Split an edit tool summary like "edit /tmp/a.rs  [+2/-1]" into styled parts.
fn split_edit_tool_text(text: &str) -> Option<EditToolParts<'_>> {
    if !text.starts_with("edit ") {
        return None;
    }
    // Look for the diff stats block.
    let bracket = text.find(" [+")?;
    let after_bracket = &text[bracket + 3..]; // "N/-M]"
    let slash = after_bracket.find('/')?;
    let dash_pos = slash + 1;
    if dash_pos >= after_bracket.len() || after_bracket.as_bytes()[dash_pos] != b'-' {
        return None;
    }
    if !after_bracket.ends_with(']') {
        return None;
    }
    let added_num = &after_bracket[..slash];
    let removed_num = &after_bracket[dash_pos + 1..after_bracket.len() - 1];
    // Validate both are digits
    if !added_num.chars().all(|c| c.is_ascii_digit())
        || !removed_num.chars().all(|c| c.is_ascii_digit())
    {
        return None;
    }
    // Include the +/- signs in the colored portions.
    Some(EditToolParts {
        prefix: &text[..bracket + 2], // up to and including " ["
        added: &text[bracket + 2..bracket + 3 + slash], // "+N"
        slash_minus: "/",
        removed: &text[bracket + 3 + dash_pos..bracket + 3 + after_bracket.len() - 1], // "-M"
        suffix: "]",
    })
}

pub fn should_insert_gap(prev: ChatRole, curr: ChatRole) -> bool {
    role_group(prev) != role_group(curr)
}

fn role_group(role: ChatRole) -> u8 {
    match role {
        ChatRole::User => 1,
        ChatRole::Assistant | ChatRole::Thinking => 2,
        ChatRole::Tool => 3,
        ChatRole::System => 4,
        ChatRole::Skill => 5,
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
pub fn format_markdown(
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
    let mut item_content_first = false;
    let mut in_code_block = false;
    let mut code_highlighter: Option<HighlightLines> = None;
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
                    ensure_block_spacing(&mut out);
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
                Tag::BlockQuote(_) => {
                    flush_line(&mut out, &mut line);
                    ensure_block_spacing(&mut out);
                    block_stack.push(BlockState::Quote);
                }
                Tag::List(start) => {
                    flush_line(&mut out, &mut line);
                    // Only add spacing for top-level lists, not nested ones.
                    if !block_stack
                        .iter()
                        .any(|b| matches!(b, BlockState::List { .. }))
                    {
                        ensure_block_spacing(&mut out);
                    }
                    block_stack.push(BlockState::List {
                        ordered: start.is_some(),
                        next: start.unwrap_or(1),
                    });
                }
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
                    item_content_first = false;
                }
                Tag::CodeBlock(kind) => {
                    flush_line(&mut out, &mut line);
                    ensure_block_spacing(&mut out);
                    in_code_block = true;
                    let lang = match kind {
                        CodeBlockKind::Fenced(lang) => {
                            let l = lang.trim().to_lowercase();
                            if l.is_empty() { None } else { Some(l) }
                        }
                        CodeBlockKind::Indented => None,
                    };
                    // Create highlighter once per code block so multi-line
                    // constructs (block comments, strings) carry state across lines.
                    let ps = syntax_set();
                    let syntax = lang
                        .as_deref()
                        .and_then(map_lang_token)
                        .and_then(|token| {
                            ps.find_syntax_by_token(token)
                                .or_else(|| ps.find_syntax_by_extension(token))
                        })
                        .unwrap_or_else(|| ps.find_syntax_plain_text());
                    code_highlighter = Some(HighlightLines::new(syntax, syntect_theme()));
                }
                Tag::Table(_) => {
                    flush_line(&mut out, &mut line);
                    ensure_block_spacing(&mut out);
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
                Tag::Paragraph
                    if !line.is_empty()
                        && !only_prefix(&line, prefix)
                        // Don't flush on Paragraph start when inside a list item
                        // and the line only has the item marker (no content yet).
                        // pulldown-cmark wraps list items in Paragraph when there
                        // are blank lines between items; flushing here would split
                        // the marker onto its own line.
                        && (!in_item || item_content_first) =>
                {
                    flush_line(&mut out, &mut line);
                }
                // Paragraphs are block-level: separate from preceding blocks
                // with a blank line when they start after content.
                // (Heading, List, CodeBlock, Table, BlockQuote spacing is
                // handled at their own Start events.)
                Tag::Paragraph if !in_item => {
                    ensure_block_spacing(&mut out);
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
                    code_highlighter = None;
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
                    // pulldown-cmark 0.13 emits header cells as direct
                    // children of TableHead, not wrapped in TableRow.
                    // Push accumulated cells before recording header count.
                    if !current_table_row.is_empty() {
                        table_rows.push(std::mem::take(&mut current_table_row));
                    }
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
                    let ps = syntax_set();
                    let base = Style::default().fg(theme.code_fg.unwrap_or(Color::Cyan));
                    for raw in t.lines() {
                        let mut spans = vec![Span::styled(
                            format!("{prefix}\u{2503} "),
                            Style::default().fg(theme.md_rule_border),
                        )];
                        if let Some(ref mut highlighter) = code_highlighter {
                            for snippet in LinesWithEndings::from(raw) {
                                if let Ok(ranges) = highlighter.highlight_line(snippet, ps) {
                                    spans.extend(ranges.into_iter().map(|(style, segment)| {
                                        Span::styled(
                                            segment.to_string(),
                                            ratatui_style_from_syntect(style, base),
                                        )
                                    }));
                                } else {
                                    spans.push(Span::styled(snippet.to_string(), base));
                                }
                            }
                        } else {
                            spans.push(Span::styled(raw.to_string(), base));
                        }
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
                                item_content_first = false;
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
                        if in_item {
                            item_content_first = true;
                        }
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
                if in_item {
                    item_content_first = true;
                }
            }
            MdEvent::SoftBreak | MdEvent::HardBreak => {
                if in_table && in_table_cell {
                    current_table_cell.push(' ');
                    continue;
                }
                flush_line(&mut out, &mut line);
                if in_item {
                    item_content_first = false;
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

/// Push a blank line to `out` if the last line is not already blank and
/// the output is not empty. Used to separate block-level elements.
fn ensure_block_spacing(out: &mut Vec<Line<'static>>) {
    if out.is_empty() {
        return;
    }
    let last_is_blank = out.last().is_some_and(|l| {
        l.spans
            .iter()
            .map(|s| s.content.as_ref())
            .collect::<String>()
            .trim()
            .is_empty()
    });
    if !last_is_blank {
        out.push(Line::raw(""));
    }
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
        "jsx" => Some("jsx"),
        "ts" => Some("typescript"),
        "tsx" => Some("tsx"),
        "py" => Some("python"),
        "rb" => Some("ruby"),
        "sh" | "shell" | "bash" | "zsh" => Some("bash"),
        "yml" => Some("yaml"),
        "md" | "markdown" => Some("markdown"),
        "toml" => Some("toml"),
        "json" | "jsonc" => Some("json"),
        "css" | "scss" | "sass" => Some("css"),
        "html" | "htm" => Some("html"),
        "go" => Some("go"),
        "java" => Some("java"),
        "c" | "h" => Some("c"),
        "cpp" | "cc" | "cxx" | "hpp" => Some("cpp"),
        "csharp" | "cs" => Some("csharp"),
        "swift" => Some("swift"),
        "kt" | "kotlin" => Some("kotlin"),
        "zig" => Some("zig"),
        "nix" => Some("nix"),
        "lua" => Some("lua"),
        "php" => Some("php"),
        "perl" | "pl" => Some("perl"),
        "r" => Some("r"),
        "sql" => Some("sql"),
        "elixir" | "ex" => Some("elixir"),
        "scala" => Some("scala"),
        "clojure" | "clj" => Some("clojure"),
        "haskell" | "hs" => Some("haskell"),
        "protobuf" | "proto" => Some("protobuf"),
        "graphql" | "gql" => Some("graphql"),
        "dockerfile" => Some("dockerfile"),
        "makefile" | "make" => Some("makefile"),
        "cmake" => Some("cmake"),
        "vim" => Some("vim"),
        "latex" | "tex" => Some("tex"),
        "terraform" | "tf" => Some("terraform"),
        "" => None,
        other => Some(other),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: collect styled lines into plain strings.
    fn lines_text(lines: &[Line]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn table_with_headers_renders_header_row() {
        let theme = Theme::default();
        let style = Style::default();
        let md = "| Name | Value |\n|------|-------|\n| foo  | 42    |\n| bar  | 99    |\n";
        let lines = format_markdown(md, style, &theme, "", 80);
        let texts = lines_text(&lines);
        // First line should contain the header "Name" (not empty).
        assert!(
            texts.iter().any(|t| t.contains("Name")),
            "table header 'Name' should appear: {:?}",
            texts
        );
        // Both header cells should be present.
        assert!(
            texts.iter().any(|t| t.contains("Value")),
            "table header 'Value' should appear: {:?}",
            texts
        );
        // Data cells should also be present.
        assert!(
            texts.iter().any(|t| t.contains("foo")),
            "data cell 'foo' should appear: {:?}",
            texts
        );
    }

    #[test]
    fn list_items_with_blank_lines_keep_marker_and_text_together() {
        let theme = Theme::default();
        let style = Style::default();
        // Blank lines between items trigger Paragraph wrapping.
        let md = "1. First item\n\n2. Second item\n";
        let lines = format_markdown(md, style, &theme, "", 80);
        let texts = lines_text(&lines);
        // The first non-empty line should contain both "1." and "First item".
        let first = texts.iter().find(|t| !t.trim().is_empty()).unwrap();
        assert!(
            first.contains("1.") && first.contains("First"),
            "marker and text should be on same line, got: '{}'",
            first
        );
    }

    #[test]
    fn simple_list_no_blank_lines_works() {
        let theme = Theme::default();
        let style = Style::default();
        let md = "1. First\n2. Second\n3. Third\n";
        let lines = format_markdown(md, style, &theme, "", 80);
        let texts = lines_text(&lines);
        let first = texts.iter().find(|t| t.contains("1.")).unwrap();
        assert!(
            first.contains("First"),
            "simple list: marker and text on same line, got: '{}'",
            first
        );
    }

    #[test]
    fn sublist_rendering() {
        let theme = Theme::default();
        let style = Style::default();
        let md = "1. Parent\n   * Child\n   * Child 2\n2. Parent 2\n";
        let lines = format_markdown(md, style, &theme, "", 80);
        let texts = lines_text(&lines);
        assert!(
            texts
                .iter()
                .any(|t| t.contains("Parent") && t.contains("1.")),
            "parent item: {:?}",
            texts
        );
        assert!(
            texts.iter().any(|t| t.contains("Child") && t.contains("•")),
            "child item: {:?}",
            texts
        );
    }

    #[test]
    fn multi_paragraph_list_item() {
        let theme = Theme::default();
        let style = Style::default();
        let md = "1. First paragraph\n\n   Second paragraph\n2. Next\n";
        let lines = format_markdown(md, style, &theme, "", 80);
        let texts = lines_text(&lines);
        // First paragraph should be on same line as marker.
        assert!(
            texts
                .iter()
                .any(|t| t.contains("1.") && t.contains("First paragraph")),
            "first paragraph on marker line: {:?}",
            texts
        );
        // Second paragraph on continuation line.
        assert!(
            texts
                .iter()
                .any(|t| t.contains("Second paragraph") && !t.contains("1.")),
            "second paragraph on continuation line: {:?}",
            texts
        );
    }

    #[test]
    fn code_block_has_syntax_highlighting() {
        let theme = Theme::default();
        let style = Style::default();
        let md = "```rust\nfn main() {}\n```";
        let lines = format_markdown(md, style, &theme, "", 80);
        // The code line should contain multiple spans (syntax-highlighted),
        // not just a single plain-text span.
        let code_line = lines
            .iter()
            .find(|l| l.spans.iter().any(|s| s.content.as_ref().contains("fn")))
            .expect("should have a line with 'fn'");
        assert!(
            code_line.spans.len() > 2,
            "syntax highlighting should produce multiple spans, got {}: {:?}",
            code_line.spans.len(),
            code_line.spans
        );
    }

    #[test]
    fn code_block_plain_text_for_unknown_lang() {
        let theme = Theme::default();
        let style = Style::default();
        // Unknown language should still render content, just without highlighting.
        let md = "```notalang\nsome text\n```";
        let lines = format_markdown(md, style, &theme, "", 80);
        let texts = lines_text(&lines);
        assert!(
            texts.iter().any(|t| t.contains("some text")),
            "code block content should render: {:?}",
            texts
        );
    }

    #[test]
    fn map_lang_token_handles_common_aliases() {
        assert_eq!(map_lang_token("rs"), Some("rust"));
        assert_eq!(map_lang_token("js"), Some("javascript"));
        assert_eq!(map_lang_token("ts"), Some("typescript"));
        assert_eq!(map_lang_token("py"), Some("python"));
        assert_eq!(map_lang_token("rb"), Some("ruby"));
        assert_eq!(map_lang_token("go"), Some("go"));
        assert_eq!(map_lang_token("java"), Some("java"));
        assert_eq!(map_lang_token("c"), Some("c"));
        assert_eq!(map_lang_token("cpp"), Some("cpp"));
        assert_eq!(map_lang_token("sql"), Some("sql"));
        assert_eq!(map_lang_token("toml"), Some("toml"));
        assert_eq!(map_lang_token("json"), Some("json"));
        assert_eq!(map_lang_token("css"), Some("css"));
        assert_eq!(map_lang_token("html"), Some("html"));
        assert_eq!(map_lang_token("shell"), Some("bash"));
        assert_eq!(map_lang_token("sh"), Some("bash"));
        assert_eq!(map_lang_token("zsh"), Some("bash"));
        assert_eq!(map_lang_token(""), None);
        // Unknown tokens pass through for syntect lookup.
        assert_eq!(map_lang_token("notalang"), Some("notalang"));
    }

    #[test]
    fn blockquote_rendering() {
        let theme = Theme::default();
        let style = Style::default();
        let md = "> quoted text\n\nnormal text";
        let lines = format_markdown(md, style, &theme, "", 80);
        let texts = lines_text(&lines);
        assert!(
            texts.iter().any(|t| t.contains("quoted text")),
            "blockquote content should render: {:?}",
            texts
        );
        assert!(
            texts.iter().any(|t| t.contains("normal text")),
            "text after blockquote should render: {:?}",
            texts
        );
    }

    #[test]
    fn table_then_paragraph_has_spacing() {
        let theme = Theme::default();
        let style = Style::default();
        let md = "| a | b |\n|---|---|\n| 1 | 2 |\n\nafter";
        let lines = format_markdown(md, style, &theme, "", 80);
        let texts = lines_text(&lines);
        let last_data_idx = texts
            .iter()
            .rposition(|t| t.contains('1') && t.contains('2'))
            .unwrap();
        // There should be a blank line between the table and "after".
        assert!(
            texts
                .get(last_data_idx + 1)
                .is_some_and(|l| l.trim().is_empty()),
            "blank line expected between table and paragraph: {:?}",
            texts
        );
    }
}
