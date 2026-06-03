//! Color theme definitions.
//!
//! Built-in themes (`default`, `monokai`) plus user TOML themes from
//! `~/.theta/themes/*.toml`. User themes may `inherits` from any built-in
//! theme and override individual color keys.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

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

/// Serialized representation of a TOML theme file.
#[derive(Debug, Clone, serde::Deserialize)]
struct ThemeToml {
    /// Optional base built-in theme to inherit from.
    #[serde(default)]
    inherits: Option<String>,
    #[serde(default)]
    accent: Option<String>,
    #[serde(default)]
    bg: Option<String>,
    #[serde(default)]
    fg: Option<String>,
    #[serde(default)]
    dim: Option<String>,
    #[serde(default)]
    success: Option<String>,
    #[serde(default)]
    error: Option<String>,
    #[serde(default)]
    warning: Option<String>,
    #[serde(default)]
    border: Option<String>,
    #[serde(default)]
    highlight: Option<String>,
    #[serde(default)]
    user_bubble: Option<String>,
    #[serde(default)]
    assistant_bubble: Option<String>,
    #[serde(default)]
    code_fg: Option<String>,
    #[serde(default)]
    code_bg: Option<String>,
    #[serde(default)]
    md_heading_1: Option<String>,
    #[serde(default)]
    md_heading_2: Option<String>,
    #[serde(default)]
    md_list_marker: Option<String>,
    #[serde(default)]
    md_quote: Option<String>,
    #[serde(default)]
    md_link: Option<String>,
    #[serde(default)]
    md_inline_code: Option<String>,
    #[serde(default)]
    md_rule_border: Option<String>,
    #[serde(default)]
    md_table_header: Option<String>,
    #[serde(default)]
    md_task_marker: Option<String>,
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
    /// Resolve a theme by name from built-in themes.
    pub fn named(name: &str) -> Self {
        match name {
            "monokai" => Self::monokai(),
            _ => Self::default(),
        }
    }

    /// Resolve a theme by name, checking user themes first, then built-ins.
    pub fn named_with_users(name: &str, user_themes: &HashMap<String, Theme>) -> Self {
        if let Some(t) = user_themes.get(name) {
            return t.clone();
        }
        Self::named(name)
    }

    /// Built-in theme names.
    pub fn names() -> &'static [&'static str] {
        &["default", "monokai"]
    }

    /// All available theme names (built-in + user).
    pub fn all_names(user_themes: &HashMap<String, Theme>) -> Vec<String> {
        let mut names: Vec<String> = Self::names().iter().map(|s| s.to_string()).collect();
        for key in user_themes.keys() {
            if !names.contains(key) {
                names.push(key.clone());
            }
        }
        names
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

// ── User theme loading ──

/// Load all user TOML themes from `~/.theta/themes/*.toml`.
///
/// Each file's stem (without `.toml`) becomes the theme name.
/// If a theme has `inherits`, its fields override the parent built-in theme.
pub fn load_user_themes() -> HashMap<String, Theme> {
    let dir = user_theme_dir();
    load_user_themes_from(&dir)
}

fn user_theme_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".theta")
        .join("themes")
}

fn load_user_themes_from(dir: &Path) -> HashMap<String, Theme> {
    let mut themes = HashMap::new();
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return themes,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "toml") {
            let name = match path.file_stem().and_then(|s| s.to_str()) {
                Some(n) => n.to_string(),
                None => continue,
            };
            match load_theme_file(&path) {
                Ok(theme) => {
                    themes.insert(name, theme);
                }
                Err(e) => {
                    tracing::warn!("Failed to load theme {}: {e}", path.display());
                }
            }
        }
    }
    themes
}

fn load_theme_file(path: &Path) -> Result<Theme, String> {
    let contents = std::fs::read_to_string(path).map_err(|e| format!("read: {e}"))?;
    let toml: ThemeToml = toml::from_str(&contents).map_err(|e| format!("parse: {e}"))?;

    let mut theme = match toml.inherits.as_deref() {
        Some("monokai") => Theme::monokai(),
        Some("default") | None => Theme::default(),
        Some(other) => {
            tracing::warn!(
                "Unknown inherits '{other}' in {}, using default",
                path.display()
            );
            Theme::default()
        }
    };

    if let Some(v) = toml.accent.as_deref() {
        theme.accent = parse_color(v)?;
    }
    if let Some(v) = toml.bg.as_deref() {
        theme.bg = parse_color(v)?;
    }
    if let Some(v) = toml.fg.as_deref() {
        theme.fg = parse_color(v)?;
    }
    if let Some(v) = toml.dim.as_deref() {
        theme.dim = parse_color(v)?;
    }
    if let Some(v) = toml.success.as_deref() {
        theme.success = parse_color(v)?;
    }
    if let Some(v) = toml.error.as_deref() {
        theme.error = parse_color(v)?;
    }
    if let Some(v) = toml.warning.as_deref() {
        theme.warning = parse_color(v)?;
    }
    if let Some(v) = toml.border.as_deref() {
        theme.border = parse_color(v)?;
    }
    if let Some(v) = toml.highlight.as_deref() {
        theme.highlight = parse_color(v)?;
    }
    if let Some(v) = toml.user_bubble.as_deref() {
        theme.user_bubble = parse_color(v)?;
    }
    if let Some(v) = toml.assistant_bubble.as_deref() {
        theme.assistant_bubble = parse_color(v)?;
    }
    if let Some(v) = toml.code_fg.as_deref() {
        theme.code_fg = Some(parse_color(v)?);
    }
    if let Some(v) = toml.code_bg.as_deref() {
        theme.code_bg = parse_color(v)?;
    }
    if let Some(v) = toml.md_heading_1.as_deref() {
        theme.md_heading_1 = parse_color(v)?;
    }
    if let Some(v) = toml.md_heading_2.as_deref() {
        theme.md_heading_2 = parse_color(v)?;
    }
    if let Some(v) = toml.md_list_marker.as_deref() {
        theme.md_list_marker = parse_color(v)?;
    }
    if let Some(v) = toml.md_quote.as_deref() {
        theme.md_quote = parse_color(v)?;
    }
    if let Some(v) = toml.md_link.as_deref() {
        theme.md_link = parse_color(v)?;
    }
    if let Some(v) = toml.md_inline_code.as_deref() {
        theme.md_inline_code = parse_color(v)?;
    }
    if let Some(v) = toml.md_rule_border.as_deref() {
        theme.md_rule_border = parse_color(v)?;
    }
    if let Some(v) = toml.md_table_header.as_deref() {
        theme.md_table_header = parse_color(v)?;
    }
    if let Some(v) = toml.md_task_marker.as_deref() {
        theme.md_task_marker = parse_color(v)?;
    }

    Ok(theme)
}

/// Parse a color string. Supports:
/// - Named colors: `"red"`, `"green"`, `"blue"`, `"cyan"`, `"yellow"`, `"magenta"`,
///   `"white"`, `"black"`, `"dark_gray"`, `"gray"`, `"reset"`, `"none"`
/// - Hex: `"#ff8800"` or `"ff8800"`
/// - RGB tuple: `"rgb(255, 136, 0)"`
fn parse_color(s: &str) -> Result<Color, String> {
    let s = s.trim();

    // Hex
    let hex = s.strip_prefix('#').unwrap_or(s);
    if hex.len() == 6 && hex.chars().all(|c| c.is_ascii_hexdigit()) {
        let r = u8::from_str_radix(&hex[0..2], 16).unwrap();
        let g = u8::from_str_radix(&hex[2..4], 16).unwrap();
        let b = u8::from_str_radix(&hex[4..6], 16).unwrap();
        return Ok(Color::Rgb(r, g, b));
    }

    // RGB tuple
    if let Some(inner) = s.strip_prefix("rgb(").and_then(|s| s.strip_suffix(')')) {
        let parts: Vec<&str> = inner.split(',').map(str::trim).collect();
        if parts.len() == 3 {
            let r: u8 = parts[0].parse().map_err(|_| format!("bad rgb: {s}"))?;
            let g: u8 = parts[1].parse().map_err(|_| format!("bad rgb: {s}"))?;
            let b: u8 = parts[2].parse().map_err(|_| format!("bad rgb: {s}"))?;
            return Ok(Color::Rgb(r, g, b));
        }
    }

    // Named colors
    match s.to_lowercase().as_str() {
        "reset" | "none" => Ok(Color::Reset),
        "black" => Ok(Color::Black),
        "red" => Ok(Color::Red),
        "green" => Ok(Color::Green),
        "yellow" => Ok(Color::Yellow),
        "blue" => Ok(Color::Blue),
        "magenta" => Ok(Color::Magenta),
        "cyan" => Ok(Color::Cyan),
        "gray" | "grey" | "light_gray" | "light_grey" => Ok(Color::Gray),
        "dark_gray" | "dark_grey" => Ok(Color::DarkGray),
        "white" => Ok(Color::White),
        _ => Err(format!("unknown color: {s}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hex_color() {
        assert!(matches!(
            parse_color("#ff8800"),
            Ok(Color::Rgb(255, 136, 0))
        ));
        assert!(matches!(parse_color("ff8800"), Ok(Color::Rgb(255, 136, 0))));
    }

    #[test]
    fn parse_rgb_tuple() {
        assert!(matches!(
            parse_color("rgb(10, 20, 30)"),
            Ok(Color::Rgb(10, 20, 30))
        ));
    }

    #[test]
    fn parse_named_colors() {
        assert!(matches!(parse_color("red"), Ok(Color::Red)));
        assert!(matches!(parse_color("cyan"), Ok(Color::Cyan)));
        assert!(matches!(parse_color("dark_gray"), Ok(Color::DarkGray)));
        assert!(matches!(parse_color("reset"), Ok(Color::Reset)));
    }

    #[test]
    fn parse_invalid_color() {
        assert!(parse_color("notacolor").is_err());
        assert!(parse_color("#xyz").is_err());
    }

    #[test]
    fn load_theme_from_toml_string() {
        let toml_str = r##"
            inherits = "monokai"
            accent = "#a6e22e"
            error = "red"
        "##;
        let parsed: ThemeToml = toml::from_str(toml_str).unwrap();
        assert_eq!(parsed.inherits.as_deref(), Some("monokai"));
        assert_eq!(parsed.accent.as_deref(), Some("#a6e22e"));
        assert_eq!(parsed.error.as_deref(), Some("red"));
    }

    #[test]
    fn load_user_themes_from_disk() {
        let themes = load_user_themes();
        // Should find catppuccin_mocha and catppuccin_latte if placed in ~/.theta/themes/.
        // This test is a no-op if the files aren't present (e.g. CI).
        for name in ["catppuccin_mocha", "catppuccin_latte"] {
            if let Some(theme) = themes.get(name) {
                // Mocha is dark, latte is light — both should parse with non-reset fg.
                assert_ne!(theme.fg, Color::Reset, "{name} fg should not be reset");
            }
        }
    }
}
