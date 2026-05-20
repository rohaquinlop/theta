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
            user_bubble: Color::Rgb(40, 50, 70),
            assistant_bubble: Color::Reset,
            code_fg: None, // falls back to accent
            code_bg: Color::Rgb(30, 30, 40),
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
            user_bubble: Color::Rgb(58, 58, 48),
            assistant_bubble: Color::Rgb(39, 40, 34),
            code_fg: Some(Color::Rgb(230, 219, 116)), // yellow for monokai
            code_bg: Color::Rgb(58, 58, 48),
        }
    }
}
