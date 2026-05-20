//! Chat message display — scrollable conversation view with markdown styling.

use crossterm::event::{Event, KeyCode, MouseEventKind};
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Padding, Paragraph, Wrap},
};
use unicode_width::UnicodeWidthStr;

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
    Tool,
    System,
}

/// Scrollable chat message list.
pub struct Chat {
    pub messages: Vec<ChatMessage>,
    scroll_from_bottom: usize,
    theme: Theme,
    focused: bool,
}

impl Chat {
    pub fn new(theme: Theme) -> Self {
        Self {
            messages: Vec::new(),
            scroll_from_bottom: 0,
            theme,
            focused: false,
        }
    }

    pub fn add_message(&mut self, msg: ChatMessage) {
        self.messages.push(msg);
        self.scroll_from_bottom = 0;
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    pub fn update_last(&mut self, text: &str, role: ChatRole, is_streaming: bool) {
        if let Some(last) = self.messages.last_mut()
            && last.role == role
            && last.is_streaming
        {
            last.text.push_str(text);
            last.is_streaming = is_streaming;
            return;
        }
        self.messages.push(ChatMessage {
            role,
            text: text.to_string(),
            tool_name: None,
            is_streaming,
        });
        self.scroll_from_bottom = 0;
    }

    pub fn update_tool(&mut self, name: &str, text: &str, is_streaming: bool) {
        if let Some(msg) = self.messages.iter_mut().rev().find(|msg| {
            msg.role == ChatRole::Tool && msg.tool_name.as_deref() == Some(name) && msg.is_streaming
        }) {
            msg.text.push_str(text);
            msg.is_streaming = is_streaming;
            return;
        }

        self.messages.push(ChatMessage {
            role: ChatRole::Tool,
            text: text.trim_start_matches('\n').to_string(),
            tool_name: Some(name.to_string()),
            is_streaming,
        });
        self.scroll_from_bottom = 0;
    }

    pub fn finish_last(&mut self, role: ChatRole) {
        if let Some(last) = self.messages.last_mut()
            && last.role == role
        {
            last.is_streaming = false;
        }
    }

    /// Format a message into styled lines with markdown parsing.
    fn format_message(&self, msg: &ChatMessage) -> Vec<Line<'static>> {
        let (prefix, role_style): (&str, Style) = match msg.role {
            ChatRole::User => (" ", Style::default().fg(self.theme.fg)),
            ChatRole::Assistant => ("", Style::default().fg(self.theme.fg)),
            ChatRole::Tool => ("  ", Style::default().fg(self.theme.warning)),
            ChatRole::System => ("  ", Style::default().fg(self.theme.dim)),
        };

        let text = if msg.role == ChatRole::Tool {
            let body = truncate_output(&msg.text, 500);
            if let Some(name) = msg.tool_name.as_deref() {
                format!("[tool:{name}] {body}")
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

        let mut lines = format_markdown(&text, role_style, &self.theme, prefix);

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

impl Component for Chat {
    fn render(&mut self, area: Rect, frame: &mut Frame) {
        let block = Block::default()
            .borders(Borders::NONE)
            .padding(Padding::horizontal(1));
        let mut lines: Vec<Line> = Vec::new();
        for (idx, msg) in self.messages.iter().enumerate() {
            if idx > 0 {
                lines.push(Line::raw(""));
            }
            lines.extend(self.format_message(msg));
        }

        let inner_width = area.width.saturating_sub(2) as usize;
        let viewport_height = area.height as usize;
        let total_visual_rows = visual_row_count(&lines, inner_width);
        let max_scroll = total_visual_rows.saturating_sub(viewport_height);
        self.scroll_from_bottom = self.scroll_from_bottom.min(max_scroll);
        let scroll_top = max_scroll.saturating_sub(self.scroll_from_bottom);

        let para = Paragraph::new(Text::from(lines))
            .wrap(Wrap { trim: false })
            .block(block)
            .scroll((scroll_top.min(u16::MAX as usize) as u16, 0));

        frame.render_widget(para, area);
    }

    fn handle_event(&mut self, event: &Event) -> Option<Action> {
        match event {
            Event::Key(key) if self.focused => match key.code {
                KeyCode::Up => self.scroll_from_bottom = self.scroll_from_bottom.saturating_add(1),
                KeyCode::Down => {
                    self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(1)
                }
                KeyCode::PageUp => {
                    self.scroll_from_bottom = self.scroll_from_bottom.saturating_add(10)
                }
                KeyCode::PageDown => {
                    self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(10)
                }
                KeyCode::Home => self.scroll_from_bottom = usize::MAX,
                KeyCode::End => self.scroll_from_bottom = 0,
                _ => {}
            },
            Event::Mouse(mouse) => match mouse.kind {
                MouseEventKind::ScrollUp => {
                    self.scroll_from_bottom = self.scroll_from_bottom.saturating_add(3)
                }
                MouseEventKind::ScrollDown => {
                    self.scroll_from_bottom = self.scroll_from_bottom.saturating_sub(3)
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

fn truncate_output(text: &str, max_len: usize) -> String {
    if text.chars().count() <= max_len {
        text.to_string()
    } else {
        let truncated: String = text.chars().take(max_len).collect();
        format!("{}... ({} chars total)", truncated, text.chars().count())
    }
}

fn visual_row_count(lines: &[Line<'_>], width: usize) -> usize {
    if width == 0 {
        return 0;
    }

    lines
        .iter()
        .map(|line| {
            let text = line
                .spans
                .iter()
                .map(|s| s.content.as_ref())
                .collect::<String>();
            let w = UnicodeWidthStr::width(text.as_str());
            w.max(1).div_ceil(width)
        })
        .sum()
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
) -> Vec<Line<'static>> {
    let mut lines = Vec::new();
    let mut in_code_block = false;
    let mut code_lang: Option<String> = None;

    for raw in text.lines() {
        let trimmed = raw.trim();

        // Fenced code block toggle.
        if let Some(lang) = trimmed.strip_prefix("```") {
            if in_code_block {
                in_code_block = false;
                code_lang = None;
            } else {
                in_code_block = true;
                let normalized = lang.trim().to_lowercase();
                code_lang = if normalized.is_empty() {
                    None
                } else {
                    Some(normalized)
                };
                if !lang.is_empty() {
                    lines.push(Line::from(vec![Span::styled(
                        format!("{prefix}\u{2503} {lang}"),
                        Style::default()
                            .fg(theme.dim)
                            .add_modifier(ratatui::style::Modifier::ITALIC),
                    )]));
                } else {
                    lines.push(Line::from(vec![Span::styled(
                        format!("{prefix}\u{2503}"),
                        Style::default().fg(theme.dim),
                    )]));
                }
            }
            continue;
        }

        if in_code_block {
            let mut spans = vec![Span::styled(
                format!("{prefix}\u{2503} "),
                Style::default().fg(theme.border),
            )];
            spans.extend(highlight_code_line(raw, code_lang.as_deref(), theme));
            lines.push(Line::from(spans));
            continue;
        }

        // Heading detection.
        if let Some(heading) = trimmed.strip_prefix("### ") {
            let h_style = Style::default()
                .fg(theme.accent)
                .add_modifier(ratatui::style::Modifier::BOLD);
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), base_style),
                Span::styled(format!("\u{2592} {heading}"), h_style),
            ]));
            continue;
        }
        if let Some(heading) = trimmed.strip_prefix("## ") {
            let h_style = Style::default()
                .fg(theme.accent)
                .add_modifier(ratatui::style::Modifier::BOLD);
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), base_style),
                Span::styled(format!("\u{2593} {heading}"), h_style),
            ]));
            continue;
        }
        if let Some(heading) = trimmed.strip_prefix("# ") {
            let h_style = Style::default()
                .fg(theme.success)
                .add_modifier(ratatui::style::Modifier::BOLD);
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), base_style),
                Span::styled(format!("\u{2588} {heading}"), h_style),
            ]));
            continue;
        }

        // Blockquote.
        if let Some(quoted) = trimmed.strip_prefix("> ") {
            let q_style = Style::default()
                .fg(theme.dim)
                .add_modifier(ratatui::style::Modifier::ITALIC);
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), base_style),
                Span::styled(format!("  {quoted}"), q_style),
            ]));
            continue;
        }

        // Bullet / list item.
        let is_bullet = trimmed
            .strip_prefix("- ")
            .or_else(|| trimmed.strip_prefix("* "));
        if let Some(item) = is_bullet {
            let bullet_style = Style::default().fg(theme.warning);
            let mut spans = vec![
                Span::styled(prefix.to_string(), base_style),
                Span::styled("  \u{2022} ", bullet_style),
            ];
            spans.extend(inline_format(item.to_string(), base_style));
            lines.push(Line::from(spans));
            continue;
        }

        // Numbered list.
        if trimmed.chars().take_while(|c| c.is_ascii_digit()).count() > 0
            && trimmed
                .chars()
                .skip_while(|c| c.is_ascii_digit())
                .take(2)
                .collect::<String>()
                .starts_with(". ")
        {
            let num_style = Style::default().fg(theme.warning);
            lines.push(Line::from(vec![
                Span::styled(prefix.to_string(), base_style),
                Span::styled(format!("  {trimmed}"), num_style),
            ]));
            continue;
        }

        // Empty line.
        if trimmed.is_empty() {
            lines.push(Line::raw(""));
            continue;
        }

        // Regular text with inline formatting.
        let mut spans = vec![Span::styled(prefix.to_string(), base_style)];
        spans.extend(inline_format(raw.to_string(), base_style));
        lines.push(Line::from(spans));
    }

    lines
}

fn highlight_code_line(line: &str, lang: Option<&str>, theme: &Theme) -> Vec<Span<'static>> {
    let base = Style::default().fg(theme.code_fg.unwrap_or(Color::Cyan));
    if line.trim_start().starts_with("//") {
        return vec![Span::styled(
            line.to_string(),
            Style::default().fg(theme.dim),
        )];
    }

    if !matches!(lang, Some("rust") | Some("rs")) {
        return vec![Span::styled(line.to_string(), base)];
    }

    let keywords = [
        "fn", "pub", "struct", "impl", "trait", "enum", "let", "mut", "use", "mod", "match", "if",
        "else", "for", "while", "loop", "return", "async", "await",
    ];

    line.split_inclusive(char::is_whitespace)
        .map(|chunk| {
            let token = chunk.trim();
            if token.starts_with("\"") && token.ends_with("\"") && token.len() >= 2 {
                Span::styled(chunk.to_string(), Style::default().fg(theme.success))
            } else if keywords.contains(&token) {
                Span::styled(chunk.to_string(), Style::default().fg(theme.warning))
            } else {
                Span::styled(chunk.to_string(), base)
            }
        })
        .collect()
}

/// Apply inline formatting: `code` (dim/italic), **bold**, *italic*, ~~strike~~.
fn inline_format(text: String, base: Style) -> Vec<Span<'static>> {
    let mut spans: Vec<Span<'static>> = Vec::new();
    let mut remaining = text.as_str();

    while !remaining.is_empty() {
        // Inline code: `text`
        if let Some(start) = remaining.find('`') {
            // Text before the backtick.
            if start > 0 {
                spans.push(Span::styled(remaining[..start].to_string(), base));
            }
            remaining = &remaining[start + 1..];
            if let Some(end) = remaining.find('`') {
                let code_style = Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(ratatui::style::Modifier::ITALIC);
                spans.push(Span::styled(remaining[..end].to_string(), code_style));
                remaining = &remaining[end + 1..];
            } else {
                // Unclosed backtick — treat as literal.
                spans.push(Span::raw(format!("`{remaining}")));
                remaining = "";
            }
        } else {
            spans.push(Span::styled(remaining.to_string(), base));
            remaining = "";
        }
    }

    if spans.is_empty() {
        vec![Span::raw(text)]
    } else {
        spans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_markdown_headers() {
        let theme = Theme::default();
        let style = Style::default();
        let lines = format_markdown("# Top\n## Mid\n### Low\ntext", style, &theme, "");
        assert_eq!(lines.len(), 4);
    }

    #[test]
    fn test_code_block() {
        let theme = Theme::default();
        let style = Style::default();
        let lines = format_markdown("before\n```rust\nlet x = 1;\n```\nafter", style, &theme, "");
        // before, code header, code line, after
        assert!(lines.len() >= 3);
    }
}
