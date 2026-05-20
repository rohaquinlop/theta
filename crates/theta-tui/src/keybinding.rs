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

/// Default keybindings for the chat app.
pub fn default_bindings() -> Vec<Keybinding> {
    vec![
        Keybinding {
            key: KeyEvent::new(KeyCode::Char('c'), KeyModifiers::CONTROL),
            action: Action::Quit,
            description: "Quit",
        },
        Keybinding {
            key: KeyEvent::new(KeyCode::Esc, KeyModifiers::NONE),
            action: Action::Quit,
            description: "Quit",
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
    ]
}
