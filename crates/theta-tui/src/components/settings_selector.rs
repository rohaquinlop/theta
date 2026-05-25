use crossterm::event::{Event, KeyCode};
use ratatui::{
    Frame,
    layout::Rect,
    style::Style,
    text::Line,
    widgets::{Block, Borders, List, ListItem, ListState},
};

use crate::theme::Theme;

#[derive(Debug, Clone)]
pub struct SettingsView {
    pub steering_mode: String,
    pub follow_up_mode: String,
    pub transport_preference: String,
    pub show_thinking: bool,
}

pub struct SettingsSelector {
    pub visible: bool,
    selected: usize,
    state: ListState,
    pub view: SettingsView,
    theme: Theme,
}

impl SettingsSelector {
    pub fn new(theme: Theme, view: SettingsView) -> Self {
        let mut state = ListState::default();
        state.select(Some(0));
        Self {
            visible: false,
            selected: 0,
            state,
            view,
            theme,
        }
    }

    pub fn show(&mut self) {
        self.visible = true;
    }
    pub fn hide(&mut self) {
        self.visible = false;
    }

    pub fn render(&mut self, area: Rect, frame: &mut Frame) {
        if !self.visible {
            return;
        }
        let list_items = vec![
            ListItem::new(vec![
                Line::from(format!("steeringMode: {}", self.view.steering_mode)),
                Line::from("  Alt+Enter while streaming: steer or queue follow-up"),
            ]),
            ListItem::new(vec![
                Line::from(format!("followUpMode: {}", self.view.follow_up_mode)),
                Line::from("  Ctrl+Enter while streaming: queue follow-up or steer"),
            ]),
            ListItem::new(vec![
                Line::from(format!(
                    "transportPreference: {}",
                    self.view.transport_preference
                )),
                Line::from("  Transport hint for provider requests (auto/http/sse)"),
            ]),
            ListItem::new(vec![
                Line::from(format!(
                    "showThinking: {}",
                    if self.view.show_thinking { "on" } else { "off" }
                )),
                Line::from("  Show model thinking text in UI by default"),
            ]),
        ];
        let list = List::new(list_items)
            .block(
                Block::default()
                    .title("Settings (Enter/Space toggle, Esc save+close)")
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(self.theme.border)),
            )
            .highlight_style(Style::default().fg(self.theme.accent));
        frame.render_stateful_widget(list, area, &mut self.state);
    }

    pub fn handle_event(&mut self, event: &Event) -> bool {
        if !self.visible {
            return false;
        }
        let Event::Key(key) = event else {
            return true;
        };
        match key.code {
            KeyCode::Esc => self.hide(),
            KeyCode::Up => {
                self.selected = self.selected.saturating_sub(1);
                self.state.select(Some(self.selected));
            }
            KeyCode::Down => {
                self.selected = (self.selected + 1).min(3);
                self.state.select(Some(self.selected));
            }
            KeyCode::Enter | KeyCode::Char(' ') => self.toggle_selected(),
            _ => {}
        }
        true
    }

    fn toggle_selected(&mut self) {
        match self.selected {
            0 => {
                self.view.steering_mode = if self.view.steering_mode == "steer" {
                    "follow-up".into()
                } else {
                    "steer".into()
                }
            }
            1 => {
                self.view.follow_up_mode = if self.view.follow_up_mode == "follow-up" {
                    "steer".into()
                } else {
                    "follow-up".into()
                }
            }
            2 => {
                self.view.transport_preference = match self.view.transport_preference.as_str() {
                    "auto" => "http".into(),
                    "http" => "sse".into(),
                    _ => "auto".into(),
                }
            }
            3 => self.view.show_thinking = !self.view.show_thinking,
            _ => {}
        }
    }

    pub fn current_view(&self) -> SettingsView {
        self.view.clone()
    }
}
