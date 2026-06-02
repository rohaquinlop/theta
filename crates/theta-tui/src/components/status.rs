//! Status bar component.

use crossterm::event::Event;
use ratatui::{
    Frame,
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::Paragraph,
};

use crate::components::{Action, Component};
use crate::theme::Theme;

pub struct StatusBar {
    pub model: String,
    pub session_id: String,
    pub thinking: String,
    pub agent_state: String,
    pub detail: String,
    pub last_end_reason: String,
    pub last_turn_decision: String,
    pub turn_index: u32,
    pub show_diagnostics: bool,
    /// Context token percentage (0-100) from last API call.
    pub ctx_pct: u32,
    /// Context window size (set by TUI on model switch).
    pub context_window: u32,
    /// Reserve tokens for the model's response.
    pub reserve_tokens: u32,
    /// Context tokens from last API call.
    pub context_tokens: u32,
    /// Extension status rows: rows[0] is primary bottom row.
    pub extension_rows: Vec<StatusRow>,
    /// Number of rows that need their own visual row (from tui.row() callbacks),
    /// excluding status lines merged into primary row.
    extension_row_count: usize,
    spinner_idx: usize,
    last_dot_tick: std::time::Instant,
    theme: Theme,
}

/// A single row of extension status data.
#[derive(Debug, Clone, Default)]
pub struct StatusRow {
    pub left: Vec<String>,
    pub center: Vec<String>,
    pub right: Vec<String>,
}

impl StatusBar {
    pub fn new(theme: Theme) -> Self {
        Self {
            model: String::new(),
            session_id: String::new(),
            thinking: String::new(),
            agent_state: String::new(),
            detail: String::new(),
            last_end_reason: String::new(),
            last_turn_decision: String::new(),
            turn_index: 0,
            show_diagnostics: false,
            ctx_pct: 0,
            context_window: 0,
            reserve_tokens: 4096,
            context_tokens: 0,
            extension_rows: Vec::new(),
            extension_row_count: 0,
            spinner_idx: 0,
            last_dot_tick: std::time::Instant::now(),
            theme,
        }
    }

    pub fn set_agent_state(&mut self, state: &str) {
        self.agent_state = state.to_string();
    }

    pub fn set_detail(&mut self, detail: &str) {
        self.detail = detail.to_string();
    }

    pub fn set_theme(&mut self, theme: Theme) {
        self.theme = theme;
    }

    pub fn set_turn_index(&mut self, turn_index: u32) {
        self.turn_index = turn_index;
    }

    pub fn set_show_diagnostics(&mut self, show: bool) {
        self.show_diagnostics = show;
    }

    pub fn set_extension_rows(&mut self, rows: Vec<StatusRow>) {
        self.extension_rows = rows;
    }

    pub fn set_extension_row_count(&mut self, count: usize) {
        self.extension_row_count = count;
    }

    /// Total rows needed: 1 primary + extension-only rows.
    pub fn desired_height(&self) -> u16 {
        1 + self.extension_row_count as u16
    }
}

impl Component for StatusBar {
    fn render(&mut self, area: Rect, frame: &mut Frame) {
        let total_width = area.width as usize;

        // Build the agent state badge (right-aligned).
        let state_color = if self.agent_state.starts_with("error")
            || self.agent_state.starts_with("tool error")
            || self.agent_state == "Failed"
        {
            self.theme.error
        } else if self.agent_state.starts_with("streaming")
            || self.agent_state.starts_with("thinking")
            || self.agent_state.starts_with("tool")
            || self.agent_state.starts_with("compacting")
            || self.agent_state.starts_with("retrying")
            || self.agent_state == "ModelCall"
            || self.agent_state == "ToolExec"
            || self.agent_state == "Retrying"
            || self.agent_state == "Blocked"
        {
            self.theme.warning
        } else {
            self.theme.success
        };

        let mode = mode_from_state(&self.agent_state);
        let right_badge = if self.show_diagnostics {
            format!("[{mode}] turn:{}", self.turn_index)
        } else {
            format!("[{mode}]")
        };
        // Dots animation for thinking/streaming states (shown left of the badge).
        // Throttled to 500ms per frame so it doesn't flash at render speed.
        const DOT_INTERVAL: std::time::Duration = std::time::Duration::from_millis(500);
        let thinking_dots = if mode == "thinking" || mode == "stream" {
            const FRAMES: [&str; 3] = [".", "..", "..."];
            if self.last_dot_tick.elapsed() >= DOT_INTERVAL {
                self.spinner_idx = self.spinner_idx.wrapping_add(1);
                self.last_dot_tick = std::time::Instant::now();
            }
            FRAMES[self.spinner_idx % FRAMES.len()]
        } else {
            self.spinner_idx = 0;
            ""
        };

        // Determine total rows: always primary row (0) + any extension-defined rows.
        // Works for any extension row index because extension_rows is a dense vec
        // sized to max_row_idx + 1 by the scripting engine.
        let num_rows = 1 + self.extension_rows.len();
        // Render as many rows as fit in available height.
        let renderable_rows = (area.height as usize).min(num_rows);

        for row_idx in 0..renderable_rows {
            let y = area.y + row_idx as u16;
            let row_area = Rect {
                x: area.x,
                y,
                width: area.width,
                height: 1,
            };

            let ext_row = self.extension_rows.get(row_idx);

            if row_idx == 0 {
                // Primary row: model + ext-left + detail + ext-right + agent-state
                self.render_primary_row(
                    frame,
                    row_area,
                    total_width,
                    ext_row,
                    &right_badge,
                    thinking_dots,
                    state_color,
                );
            } else {
                // Extension-only rows: ext-left + ext-center + ext-right
                self.render_extension_row(frame, row_area, total_width, ext_row);
            }
        }
    }

    fn handle_event(&mut self, _event: &Event) -> Option<Action> {
        None
    }

    fn is_focused(&self) -> bool {
        false
    }

    fn focus(&mut self, _focused: bool) {}
}

impl StatusBar {
    // ── private render helpers ───────────────────────────────────

    /// Render the primary (bottom) row: model + ext-left + detail + ext-right + agent-state.
    #[allow(clippy::too_many_arguments)]
    fn render_primary_row(
        &self,
        frame: &mut Frame,
        area: Rect,
        total_width: usize,
        ext_row: Option<&StatusRow>,
        right_badge: &str,
        thinking_dots: &str,
        state_color: Color,
    ) {
        let model_max_chars = if self.ctx_pct > 0 { 16 } else { 28 };
        let model_str = short_middle(&self.model, model_max_chars);
        let thinking_str = short_middle(&self.thinking, 10);

        let ctx_color = if self.ctx_pct >= 90 {
            self.theme.error
        } else if self.ctx_pct >= 70 {
            self.theme.warning
        } else {
            self.theme.success
        };

        let mut model_spans = vec![
            Span::styled("[".to_string(), Style::default().fg(self.theme.border)),
            Span::styled(model_str, Style::default().fg(self.theme.dim)),
            Span::styled(":".to_string(), Style::default().fg(self.theme.border)),
            Span::styled(thinking_str, Style::default().fg(self.theme.dim)),
        ];
        if self.ctx_pct > 0 {
            model_spans.push(Span::styled(
                " ".to_string(),
                Style::default().fg(self.theme.border),
            ));
            model_spans.push(Span::styled(
                "ctx:".to_string(),
                Style::default().fg(self.theme.dim),
            ));
            model_spans.push(Span::styled(
                format!("{}%", self.ctx_pct),
                Style::default().fg(ctx_color),
            ));
        }
        model_spans.push(Span::styled(
            "]".to_string(),
            Style::default().fg(self.theme.border),
        ));

        let ext_left = ext_row.map(|r| join_parts(&r.left)).unwrap_or_default();
        let ext_right = ext_row.map(|r| join_parts(&r.right)).unwrap_or_default();

        let detail_str = if self.detail.trim().is_empty() {
            String::new()
        } else {
            format!(" {}", truncate_chars(self.detail.trim(), 48))
        };

        let state_span = Span::styled(right_badge.to_string(), Style::default().fg(state_color));

        // Layout: [model:effort] [ext-left]  <detail>  [ext-right] [idle/status]
        let model_text: String = model_spans.iter().map(|s| s.content.as_ref()).collect();
        let fixed = model_text.len()
            + ext_left.len()
            + detail_str.len()
            + ext_right.len()
            + right_badge.len();
        let pad = if fixed < total_width {
            " ".repeat(total_width - fixed)
        } else {
            String::new()
        };

        let mut spans = model_spans;
        if !ext_left.is_empty() {
            spans.push(Span::styled(
                format!(" {ext_left}"),
                Style::default().fg(self.theme.accent),
            ));
        }
        spans.push(Span::raw(pad));
        if !detail_str.is_empty() {
            spans.push(Span::styled(
                detail_str,
                Style::default().fg(self.theme.dim),
            ));
        }
        if !ext_right.is_empty() {
            spans.push(Span::styled(
                format!(" {ext_right}"),
                Style::default().fg(self.theme.accent),
            ));
        }

        // Right-anchor the state badge so it never gets clipped.
        let dots_str = if thinking_dots.is_empty() {
            String::new()
        } else {
            format!("{thinking_dots} ")
        };
        let badge_space = " ";
        let badge_len = badge_space.len() + dots_str.len() + right_badge.len();
        let left_max = total_width.saturating_sub(badge_len);
        let left_line = Line::from(truncate_line_chars(spans, left_max));
        // Pad to fill the gap before the badge.
        let left_visible: String = left_line.iter().flat_map(|s| s.content.chars()).collect();
        let gap = if left_visible.len() < left_max {
            " ".repeat(left_max - left_visible.len())
        } else {
            String::new()
        };

        let mut all_spans: Vec<Span> = left_line.spans;
        if !gap.is_empty() {
            all_spans.push(Span::raw(gap));
        }
        if !dots_str.is_empty() {
            all_spans.push(Span::styled(
                dots_str,
                Style::default().fg(self.theme.warning),
            ));
        }
        all_spans.push(Span::raw(badge_space));
        all_spans.push(state_span);

        let line = Line::from(all_spans);
        let para = Paragraph::new(line).style(Style::default().bg(Color::Reset));
        frame.render_widget(para, area);
    }

    /// Render an extension row: ext-left + ext-center + ext-right.
    fn render_extension_row(
        &self,
        frame: &mut Frame,
        area: Rect,
        total_width: usize,
        ext_row: Option<&StatusRow>,
    ) {
        let Some(row) = ext_row else {
            let para = Paragraph::new("").style(Style::default().bg(Color::Reset));
            frame.render_widget(para, area);
            return;
        };

        let left_text = join_parts(&row.left);
        let center_text = join_parts(&row.center);
        let right_text = join_parts(&row.right);

        let fixed = left_text.len() + center_text.len() + right_text.len();
        let remaining = total_width.saturating_sub(fixed);
        let left_pad = remaining / 2;
        let right_pad = remaining - left_pad;

        let mut spans: Vec<Span> = Vec::new();
        if !left_text.is_empty() {
            spans.push(Span::styled(
                left_text,
                Style::default().fg(self.theme.accent),
            ));
        }
        spans.push(Span::raw(" ".repeat(left_pad)));
        if !center_text.is_empty() {
            spans.push(Span::styled(
                center_text,
                Style::default().fg(self.theme.accent),
            ));
        }
        spans.push(Span::raw(" ".repeat(right_pad)));
        if !right_text.is_empty() {
            spans.push(Span::styled(
                right_text,
                Style::default().fg(self.theme.accent),
            ));
        }

        let line = Line::from(truncate_line_chars(spans, total_width));
        let para = Paragraph::new(line).style(Style::default().bg(Color::Reset));
        frame.render_widget(para, area);
    }
}

/// Truncate spans by character count, preserving styles.
/// Drops spans from the right until the total char count fits within max_chars.
fn truncate_line_chars(spans: Vec<Span>, max_chars: usize) -> Vec<Span> {
    let mut result: Vec<Span> = Vec::new();
    let mut remaining = max_chars;
    for span in spans {
        if remaining == 0 {
            break;
        }
        let char_count = span.content.chars().count();
        if char_count <= remaining {
            result.push(span);
            remaining -= char_count;
        } else {
            let truncated: String = span.content.chars().take(remaining).collect();
            if !truncated.is_empty() {
                result.push(Span::styled(truncated, span.style));
            }
            break;
        }
    }
    result
}

/// Join slot parts with spaces between each, stripping any that are empty.
fn join_parts(parts: &[String]) -> String {
    parts
        .iter()
        .filter(|s| !s.is_empty())
        .cloned()
        .collect::<Vec<_>>()
        .join(" ")
}

fn mode_from_state(state: &str) -> &str {
    if state == "Retrying" || state.starts_with("retrying") {
        "retry"
    } else if state == "ToolExec" || state.starts_with("tool") {
        "tool"
    } else if state.starts_with("thinking") {
        "thinking"
    } else if state == "ModelCall" || state.starts_with("streaming") {
        "stream"
    } else if state == "Blocked" {
        "blocked"
    } else if state == "Failed" {
        "failed"
    } else if state == "Cancelled" || state.starts_with("cancel") {
        "cancel"
    } else if state.starts_with("compacting") {
        "compact"
    } else if state.starts_with("error") {
        "error"
    } else {
        "idle"
    }
}

fn short_middle(text: &str, max_chars: usize) -> String {
    let count = text.chars().count();
    if count <= max_chars || max_chars < 5 {
        return text.to_string();
    }
    let head_len = (max_chars - 3) / 2;
    let tail_len = max_chars - 3 - head_len;
    let head: String = text.chars().take(head_len).collect();
    let tail: String = text
        .chars()
        .rev()
        .take(tail_len)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect();
    format!("{head}...{tail}")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    if max_chars <= 3 {
        return ".".repeat(max_chars);
    }
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}...")
    } else {
        truncated
    }
}
