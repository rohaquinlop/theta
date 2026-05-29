//! TUI components.

use crossterm::event::Event;
use ratatui::{Frame, layout::Rect};

pub mod chat;
pub mod editor;
pub mod fuzzy;
pub mod login_flow;
pub mod mimo_cluster;
pub mod model_selector;
pub mod session_picker;
pub mod status;
pub mod thinking_selector;
pub mod tree_selector;

pub use login_flow::{LoginFlow, ProviderEntry, known_providers};
pub use mimo_cluster::{MimoClusterEntry, MimoClusterSelector};
pub use model_selector::{ModelEntry, ModelSelector};
pub use session_picker::{SessionInfo, SessionPicker};
pub use thinking_selector::ThinkingSelector;

/// A command or skill entry for autocomplete.
#[derive(Debug, Clone)]
pub struct CommandEntry {
    /// The text to insert (e.g., "help", "model gpt-5.5").
    pub name: String,
    /// Short description.
    pub description: String,
}

/// Actions that components can request from the App.
#[derive(Debug, Clone)]
pub enum Action {
    SendMessage(String),
    SteerMessage(String),
    FollowUpMessage(String),
    /// Cancel current agent turn if streaming, or initiate quit confirmation.
    Cancel,
    Quit,
    SwitchModel(String),
    SetThinking(String),
    ClearChat,
    SessionInfo,
    ForkSession,
    ShowHelp,
    ShowModelSelector,
    ShowThinkingSelector,
    CycleTheme,
    ShowTree,
    CopySelection(String),
    OpenUrl(String),
    None,
}

/// A renderable TUI component.
pub trait Component: Send {
    fn render(&mut self, area: Rect, frame: &mut Frame);
    fn handle_event(&mut self, event: &Event) -> Option<Action>;
    fn is_focused(&self) -> bool;
    fn focus(&mut self, focused: bool);
}
