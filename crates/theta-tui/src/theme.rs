//! Color theme definitions.

use ratatui::style::Color;

/// A named color theme.
#[derive(Debug, Clone)]
pub struct Theme {
    /// Primary accent color.
    pub accent: Color,
    /// Background color.
    pub bg: Color,
    /// Text color.
    pub fg: Color,
    /// Dimmed/secondary text.
    pub dim: Color,
    /// Success color (tool results, confirmations).
    pub success: Color,
    /// Error color.
    pub error: Color,
    /// Warning color.
    pub warning: Color,
    /// Border color.
    pub border: Color,
    /// Highlight color for selected items.
    pub highlight: Color,
    /// User message bubble.
    pub user_bubble: Color,
    /// Assistant message bubble.
    pub assistant_bubble: Color,
    /// Code block foreground (optional, falls back to Cyan).
    pub code_fg: Option<Color>,
    /// Code block background.
    pub code_bg: Color,
    /// Markdown heading level 1 color.
    pub md_heading_1: Color,
    /// Markdown heading level 2/3 color.
    pub md_heading_2: Color,
    /// Markdown list marker color.
    pub md_list_marker: Color,
    /// Markdown quote text color.
    pub md_quote: Color,
    /// Markdown link color.
    pub md_link: Color,
    /// Markdown inline code color.
    pub md_inline_code: Color,
    /// Markdown rule and table border color.
    pub md_rule_border: Color,
    /// Markdown table header color.
    pub md_table_header: Color,
    /// Markdown task marker color.
    pub md_task_marker: Color,
}

impl Default for Theme {
    fn default() -> Self {
        Self {
            accent: Color::Cyan,
            bg: Color::Reset,
            fg: Color::Reset,
            dim: Color::DarkGray,
            success: Color::Green,
            error: Color::Red,
            warning: Color::Yellow,
            border: Color::Rgb(60, 60, 60),
            highlight: Color::Rgb(70, 70, 140),
            user_bubble: Color::Rgb(52, 66, 92),
            assistant_bubble: Color::Reset,
            code_fg: None, // falls back to accent
            code_bg: Color::Rgb(30, 30, 40),
            md_heading_1: Color::Green,
            md_heading_2: Color::Cyan,
            md_list_marker: Color::Yellow,
            md_quote: Color::DarkGray,
            md_link: Color::Cyan,
            md_inline_code: Color::Cyan,
            md_rule_border: Color::Rgb(60, 60, 60),
            md_table_header: Color::Cyan,
            md_task_marker: Color::Cyan,
        }
    }
}

/// Monokai-inspired theme.
impl Theme {
    pub fn named(name: &str) -> Self {
        match name {
            "monokai" => Self::monokai(),
            _ => Self::default(),
        }
    }

    pub fn names() -> &'static [&'static str] {
        &["default", "monokai"]
    }

    pub fn monokai() -> Self {
        Self {
            accent: Color::Rgb(166, 226, 46),   // green
            bg: Color::Rgb(39, 40, 34),         // dark gray
            fg: Color::Rgb(248, 248, 242),      // off-white
            dim: Color::Rgb(117, 113, 94),      // comment gray
            success: Color::Rgb(166, 226, 46),  // green
            error: Color::Rgb(249, 38, 114),    // red/pink
            warning: Color::Rgb(230, 219, 116), // yellow
            border: Color::Rgb(58, 58, 48),
            highlight: Color::Rgb(73, 72, 62),
            user_bubble: Color::Rgb(70, 70, 58),
            assistant_bubble: Color::Rgb(39, 40, 34),
            code_fg: Some(Color::Rgb(230, 219, 116)), // yellow for monokai
            code_bg: Color::Rgb(58, 58, 48),
            md_heading_1: Color::Rgb(166, 226, 46),
            md_heading_2: Color::Rgb(102, 217, 239),
            md_list_marker: Color::Rgb(230, 219, 116),
            md_quote: Color::Rgb(117, 113, 94),
            md_link: Color::Rgb(102, 217, 239),
            md_inline_code: Color::Rgb(230, 219, 116),
            md_rule_border: Color::Rgb(58, 58, 48),
            md_table_header: Color::Rgb(249, 38, 114),
            md_task_marker: Color::Rgb(166, 226, 46),
        }
    }
}
