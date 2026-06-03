//! Keybinding manager.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use crate::components::Action;

/// Maps key events to actions.
#[derive(Debug, Clone)]
pub struct Keybinding {
    pub key: KeyEvent,
    pub action: Action,
    pub description: &'static str,
}

/// Resolve a key event to an action.
pub fn resolve_event(event: &crossterm::event::Event, bindings: &[Keybinding]) -> Option<Action> {
    let crossterm::event::Event::Key(key_event) = event else {
        return None;
    };
    for binding in bindings {
        if key_matches(key_event, &binding.key) {
            return Some(binding.action.clone());
        }
    }
    None
}

fn key_matches(pressed: &KeyEvent, binding: &KeyEvent) -> bool {
    if pressed.code != binding.code {
        return false;
    }
    // If binding has no modifiers, ignore pressed modifiers (accept Ctrl+C, Alt+Enter as the key).
    // If binding has modifiers, pressed must match exactly.
    if binding.modifiers.is_empty() {
        return true;
    }
    pressed.modifiers == binding.modifiers
}

/// Editor Enter behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnterBehavior {
    Send,
    Newline,
}

impl EnterBehavior {
    pub fn parse(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "newline" => Self::Newline,
            _ => Self::Send,
        }
    }
}

/// Returns true if key is plain Enter.
pub fn is_enter_send(key: &KeyEvent, behavior: EnterBehavior) -> bool {
    key.code == KeyCode::Enter
        && key.modifiers == KeyModifiers::NONE
        && matches!(behavior, EnterBehavior::Send)
}

/// Returns true if key should insert newline.
pub fn is_newline_key(key: &KeyEvent, behavior: EnterBehavior) -> bool {
    (key.code == KeyCode::Enter
        && (key.modifiers == KeyModifiers::SHIFT
            || key.modifiers == KeyModifiers::ALT
            || (key.modifiers == KeyModifiers::NONE && matches!(behavior, EnterBehavior::Newline))))
        || (key.code == KeyCode::Char('j') && key.modifiers == KeyModifiers::CONTROL)
}

/// Returns true if key is follow-up submit.
pub fn is_follow_up_key(key: &KeyEvent) -> bool {
    key.code == KeyCode::Enter && key.modifiers == KeyModifiers::CONTROL
}

/// Default keybindings for the chat app.
pub fn default_bindings() -> Vec<Keybinding> {
    vec![
        Keybinding {
            key: KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
            action: Action::Cancel,
            description: "Cancel / Quit confirm",
        },
        Keybinding {
            key: KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
            action: Action::Cancel,
            description: "Cancel / Quit confirm",
        },
        Keybinding {
            key: KeyEvent::new(KeyCode::Char('p'), KeyModifiers::CONTROL),
            action: Action::ShowModelSelector,
            description: "Switch model",
        },
        Keybinding {
            key: KeyEvent::new(KeyCode::Char('t'), KeyModifiers::CONTROL),
            action: Action::CycleTheme,
            description: "Cycle theme",
        },
        Keybinding {
            key: KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL),
            action: Action::EditQueuedMessage,
            description: "Focus queued messages (navigate with arrows, Enter to edit)",
        },
    ]
}
