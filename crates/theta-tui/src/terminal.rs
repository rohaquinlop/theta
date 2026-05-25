//! Terminal setup and teardown.

use std::io::{self, Stdout};

use crossterm::{
    ExecutableCommand,
    event::{
        DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
        KeyboardEnhancementFlags, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
    },
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};

/// Setup raw mode and alternate screen.
pub fn setup() -> io::Result<()> {
    enable_raw_mode()?;
    io::stdout().execute(EnterAlternateScreen)?;
    io::stdout().execute(EnableMouseCapture)?;
    io::stdout().execute(EnableBracketedPaste)?;
    // Enable keyboard protocol so modified keys (Shift+Enter, etc.) are
    // reported distinctly. Supported by Kitty, iTerm2, WezTerm, etc.
    // Terminals that don't support it simply ignore this request.
    let _ = io::stdout().execute(PushKeyboardEnhancementFlags(
        KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES,
    ));
    Ok(())
}

/// Setup raw mode and alternate screen, and set terminal window title.
pub fn setup_with_title(title: &str) -> io::Result<()> {
    enable_raw_mode()?;
    set_window_title(title)?;
    io::stdout().execute(EnterAlternateScreen)?;
    io::stdout().execute(EnableMouseCapture)?;
    io::stdout().execute(EnableBracketedPaste)?;
    let _ = io::stdout().execute(PushKeyboardEnhancementFlags(
        KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES,
    ));
    Ok(())
}

/// Set the terminal window title via ANSI escape sequence.
/// Uses OSC 0 (icon + window title) and OSC 2 (window title only)
/// for broad terminal compatibility.
pub fn set_window_title(title: &str) -> io::Result<()> {
    use std::io::Write;
    // OSC 0 ; title ST — sets both icon and window title
    write!(io::stdout(), "\x1b]0;{title}\x07")?;
    // OSC 2 ; title ST — sets window title explicitly
    write!(io::stdout(), "\x1b]2;{title}\x07")?;
    io::stdout().flush()?;
    Ok(())
}

/// Restore terminal to normal mode.
pub fn restore() -> io::Result<()> {
    let _ = io::stdout().execute(PopKeyboardEnhancementFlags);
    io::stdout().execute(DisableBracketedPaste)?;
    io::stdout().execute(DisableMouseCapture)?;
    io::stdout().execute(LeaveAlternateScreen)?;
    disable_raw_mode()?;
    Ok(())
}

/// Get the terminal size as (cols, rows).
pub fn size() -> io::Result<(u16, u16)> {
    crossterm::terminal::size()
}

/// Create a ratatui Terminal with Crossterm backend on stdout.
pub fn create_terminal() -> io::Result<ratatui::Terminal<ratatui::backend::CrosstermBackend<Stdout>>>
{
    let backend = ratatui::backend::CrosstermBackend::new(io::stdout());
    ratatui::Terminal::new(backend)
}
