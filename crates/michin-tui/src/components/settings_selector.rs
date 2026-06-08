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
    pub show_tool_diffs: bool,
    pub tool_progress_hz: u64,
    pub enter_behavior: String,
    pub max_context_window: Option<u32>,
    pub auto_escalate: bool,
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

    const ITEM_COUNT: usize = 9;

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
            ListItem::new(vec![
                Line::from(format!(
                    "showToolDiffs: {}",
                    if self.view.show_tool_diffs {
                        "on"
                    } else {
                        "off"
                    }
                )),
                Line::from("  Show edit diffs in tool output (off by default)"),
            ]),
            ListItem::new(vec![
                Line::from(format!("toolProgressHz: {}", self.view.tool_progress_hz)),
                Line::from("  Tool progress update frequency (1-60)"),
            ]),
            ListItem::new(vec![
                Line::from(format!("enterBehavior: {}", self.view.enter_behavior)),
                Line::from("  Enter key in editor: send or insert newline"),
            ]),
            ListItem::new(vec![
                Line::from(format!(
                    "maxContextWindow: {}",
                    match self.view.max_context_window {
                        Some(n) => format_number(n),
                        None => "off (model default)".to_string(),
                    }
                )),
                Line::from("  Max context token cap (off=model limit)"),
            ]),
            ListItem::new(vec![
                Line::from(format!(
                    "autoEscalate: {}",
                    if self.view.auto_escalate { "on" } else { "off" }
                )),
                Line::from("  Allow flash model to self-escalate to pro within turn"),
            ]),
        ];
        let list = List::new(list_items)
            .block(
                Block::default()
                    .title("Settings (Enter/Space toggle, Esc save+close)")
                    .borders(Borders::ALL)
                    .style(Style::default().bg(self.theme.bg))
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
                self.selected = (self.selected + 1).min(Self::ITEM_COUNT - 1);
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
            4 => self.view.show_tool_diffs = !self.view.show_tool_diffs,
            5 => {
                self.view.tool_progress_hz = match self.view.tool_progress_hz {
                    1 => 5,
                    5 => 10,
                    10 => 20,
                    20 => 30,
                    30 => 60,
                    _ => 1,
                };
            }
            6 => {
                self.view.enter_behavior = if self.view.enter_behavior == "send" {
                    "newline".into()
                } else {
                    "send".into()
                };
            }
            7 => {
                self.view.max_context_window = match self.view.max_context_window {
                    None => Some(50_000),
                    Some(50_000) => Some(100_000),
                    Some(100_000) => Some(150_000),
                    Some(150_000) => Some(200_000),
                    Some(200_000) => Some(250_000),
                    Some(250_000) => Some(300_000),
                    Some(300_000) => None,
                    Some(_) => Some(250_000),
                }
            }
            8 => {
                self.view.auto_escalate = !self.view.auto_escalate;
            }
            _ => {}
        }
    }

    pub fn current_view(&self) -> SettingsView {
        self.view.clone()
    }
}

fn format_number(n: u32) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push('_');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
