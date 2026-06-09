use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use michin_ai::{ContentBlock, Message, Model, Provider, ThinkingLevel, Tool};

use crate::cache_shape::CacheShape;
use crate::types::AgentTool;
use crate::types::{RunReport, RunReportEvent, TurnEndReason};

#[derive(Clone)]
pub struct AgentState {
    pub system_prompt: Vec<ContentBlock>,
    pub model: Model,
    pub tools: Vec<Arc<dyn AgentTool>>,
    /// Skills + extensions. Appended to the system prompt before each LLM call.
    pub resource_context: Option<Vec<ContentBlock>>,
    pub messages: Vec<Message>,
    pub is_streaming: bool,
    pub thinking_level: ThinkingLevel,
    pub available_models: Vec<Model>,
    pub last_turn_end_reason: Option<TurnEndReason>,
    pub current_run_report: Option<RunReport>,
    pub last_run_report: Option<RunReport>,
    pub current_run_id: Option<String>,
    pub current_turn_id: Option<String>,
    pub executed_tool_call_ids_in_turn: HashSet<String>,
    pub(crate) system_prompt_tokens: u32,
    pub(crate) resource_context_tokens: u32,
    pub(crate) michin_ai_tools: Vec<Tool>,
    /// Per-model-id circuit breaker state. Scoped to this agent instance
    /// so concurrent agents (tests, multi-session) don't share breakers.
    pub(crate) circuit_breakers: HashMap<String, BreakerState>,
    pub(crate) consecutive_compacts: u32,
    pub(crate) compaction_paused: bool,
    /// Prefix-cache shape from the previous turn (for diff diagnostics).
    pub(crate) prev_cache_shape: Option<CacheShape>,
    /// Whether plan mode is active (read-only exploration, no code mutation).
    pub plan_mode: bool,
    /// Cumulative API cache metrics, keyed by provider.
    pub cache_stats: HashMap<Provider, CacheStats>,
    /// Caveman communication mode: None = off, Some("full") = active.
    /// Persisted in settings.json.
    pub caveman_mode: Option<String>,
    /// Model to switch to when assistant requests escalation.
    /// None = escalation disabled.
    pub escalation_model: Option<Model>,
    /// Set true when escalation fires in the current turn. Cleared each turn.
    /// Separates loop-initiated escalation from user-initiated model switches.
    pub escalation_fired: bool,
    /// Content blocks appended to the system context at request time.
    /// EXCLUDED from cache-shape hash.
    pub volatile_overlays: Vec<ContentBlock>,
}

/// Cumulative API cache metrics per provider.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct CacheStats {
    pub total_cache_read_tokens: u64,
    pub total_cache_write_tokens: u64,
    pub total_input_tokens: u64,
}

impl CacheStats {
    /// Cache read ratio within cached tokens only: `read / (read + write)`.
    /// Does NOT include non-cached input tokens in the denominator.
    /// Use `effective_hit_ratio()` for the ratio against total input.
    pub fn hit_ratio(&self) -> f64 {
        let total = self.total_cache_read_tokens + self.total_cache_write_tokens;
        if total == 0 {
            0.0
        } else {
            self.total_cache_read_tokens as f64 / total as f64
        }
    }

    /// Effective cache hit ratio against total input: `read / total_input`.
    /// Returns 0.0 when no input tokens were sent.
    pub fn effective_hit_ratio(&self) -> f64 {
        if self.total_input_tokens == 0 {
            0.0
        } else {
            self.total_cache_read_tokens as f64 / self.total_input_tokens as f64
        }
    }
}

/// Circuit breaker per model key.
#[derive(Debug, Clone)]
pub struct BreakerState {
    pub consecutive_failures: u32,
    pub opened_at: Option<Instant>,
}

impl BreakerState {
    pub fn new() -> Self {
        Self {
            consecutive_failures: 0,
            opened_at: None,
        }
    }
}

impl Default for BreakerState {
    fn default() -> Self {
        Self::new()
    }
}

impl AgentState {
    pub fn new(model: Model, available_models: Vec<Model>) -> Self {
        Self {
            system_prompt: Vec::new(),
            model,
            tools: Vec::new(),
            resource_context: None,
            messages: Vec::new(),
            is_streaming: false,
            thinking_level: ThinkingLevel::Off,
            available_models,
            last_turn_end_reason: None,
            current_run_report: None,
            last_run_report: None,
            current_run_id: None,
            current_turn_id: None,
            executed_tool_call_ids_in_turn: HashSet::new(),
            system_prompt_tokens: 0,
            resource_context_tokens: 0,
            michin_ai_tools: Vec::new(),
            circuit_breakers: HashMap::new(),
            consecutive_compacts: 0,
            compaction_paused: false,
            prev_cache_shape: None,
            plan_mode: false,
            cache_stats: HashMap::new(),
            caveman_mode: None,
            escalation_model: None,
            escalation_fired: false,
            volatile_overlays: Vec::new(),
        }
    }

    pub fn push_run_event(
        &mut self,
        kind: &str,
        fields: impl IntoIterator<Item = (String, String)>,
    ) {
        if let Some(report) = self.current_run_report.as_mut() {
            let mut map = std::collections::BTreeMap::new();
            if let Some(run_id) = &self.current_run_id {
                map.insert("run_id".to_string(), run_id.clone());
            }
            if let Some(turn_id) = &self.current_turn_id {
                map.insert("turn_id".to_string(), turn_id.clone());
            }
            map.insert("model".to_string(), self.model.id.clone());
            map.insert("provider".to_string(), format!("{:?}", self.model.provider));
            for (k, v) in fields {
                map.insert(k.clone(), redact_field(&k, &v));
            }
            report.events.push(RunReportEvent {
                ts_ms: now_ms(),
                kind: kind.to_string(),
                fields: map,
            });
        }
    }

    pub fn add_user_message(&mut self, content: Vec<ContentBlock>, timestamp: u64) {
        self.messages.push(Message::User { content, timestamp });
    }

    pub fn add_assistant_message(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    pub fn add_tool_result(&mut self, msg: Message) {
        self.messages.push(msg);
    }

    /// Get only the messages that should be sent to the LLM.
    /// Does NOT include resource_context — callers must prepend it separately.
    pub fn llm_messages(&self) -> Vec<Message> {
        self.messages
            .iter()
            .filter(|m| {
                matches!(
                    m,
                    Message::User { .. } | Message::Assistant { .. } | Message::ToolResult { .. }
                )
            })
            .cloned()
            .collect()
    }

    pub fn resource_context_tokens(&self) -> u32 {
        self.resource_context_tokens
    }

    pub fn update_cached_tokens(&mut self) {
        self.system_prompt_tokens = approximate_tokens_for_blocks(&self.system_prompt);
        self.resource_context_tokens = self
            .resource_context
            .as_deref()
            .map(approximate_tokens_for_blocks)
            .unwrap_or(0);
    }

    pub fn rebuild_michin_ai_tools(&mut self) {
        let mut tools: Vec<Tool> = self
            .tools
            .iter()
            .map(|t| Tool {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters(),
            })
            .collect();
        // Sort by name for byte-stable prefix serialization — prevents
        // spurious cache busts when tool registration order changes.
        tools.sort_by(|a, b| a.name.cmp(&b.name));
        self.michin_ai_tools = tools;
    }

    /// Update cumulative cache stats from API usage.
    pub fn update_cache_stats(&mut self, provider: Provider, usage: &michin_ai::Usage) {
        if usage.cache_read_tokens == 0 && usage.cache_write_tokens == 0 {
            return;
        }
        let entry = self.cache_stats.entry(provider).or_default();
        entry.total_cache_read_tokens += usage.cache_read_tokens as u64;
        entry.total_cache_write_tokens += usage.cache_write_tokens as u64;
        entry.total_input_tokens += usage.input_tokens as u64;
    }

    pub fn load_messages(&mut self, messages: Vec<Message>) {
        self.messages = messages;
    }

    pub fn last_model_id(&self) -> Option<&str> {
        self.messages.iter().rev().find_map(|m| match m {
            Message::Assistant { model, .. } => model.as_deref(),
            _ => None,
        })
    }

    pub fn token_count(&self) -> u32 {
        let msg_tokens: u32 = self.messages.iter().map(|m| m.token_count()).sum();
        msg_tokens + self.system_prompt_tokens + self.resource_context_tokens
    }

    /// The last API-reported input token count (real, from the most recent
    /// assistant message's usage). This is the actual prompt token count as
    /// counted by the provider.
    pub fn last_real_input_tokens(&self) -> Option<u32> {
        self.messages.iter().rev().find_map(|m| match m {
            Message::Assistant { usage, .. } => usage.as_ref().map(|u| u.input_tokens),
            _ => None,
        })
    }

    /// Best-effort context consumption: API-reported input tokens if available,
    /// otherwise the approximate token count.
    pub fn context_tokens(&self) -> u32 {
        self.last_real_input_tokens()
            .unwrap_or_else(|| self.token_count())
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Approximate token count for a slice of content blocks by serializing to JSON.
fn approximate_tokens_for_blocks(blocks: &[ContentBlock]) -> u32 {
    blocks
        .iter()
        .map(|b| michin_ai::approximate_token_count(&serde_json::to_string(b).unwrap_or_default()))
        .sum()
}

fn redact_field(key: &str, value: &str) -> String {
    let key_lower = key.to_ascii_lowercase();
    let looks_sensitive_key = [
        "token",
        "secret",
        "password",
        "authorization",
        "cookie",
        "api_key",
        "apikey",
        "access_key",
        "refresh",
    ]
    .iter()
    .any(|p| key_lower.contains(p));
    let value_lower = value.to_ascii_lowercase();
    let looks_sensitive_value = value.starts_with("sk-")
        || value_lower.contains("bearer ")
        || value_lower.contains("authorization:")
        || value_lower.contains("api_key=")
        || value_lower.contains("token=");

    if looks_sensitive_key || looks_sensitive_value {
        "[REDACTED]".to_string()
    } else {
        value.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use michin_ai::{Api, Modality, ModelCompat, Provider};

    fn test_model() -> Model {
        Model {
            id: "test-model".into(),
            name: "Test Model".into(),
            api: Api::OpenAiCompletions,
            provider: Provider::OpenAI,
            base_url: "https://test.api".into(),
            reasoning: false,
            thinking_level_map: std::collections::HashMap::new(),
            input: vec![Modality::Text],
            context_window: 128_000,
            max_tokens: 16_384,
            compat: ModelCompat::for_openai(),
        }
    }

    #[test]
    fn cache_stats_hit_ratio() {
        let mut stats = CacheStats::default();
        assert_eq!(stats.hit_ratio(), 0.0);
        assert_eq!(stats.effective_hit_ratio(), 0.0);

        stats.total_cache_read_tokens = 900;
        stats.total_cache_write_tokens = 100;
        stats.total_input_tokens = 1000;
        assert!((stats.hit_ratio() - 0.9).abs() < f64::EPSILON);
        assert!((stats.effective_hit_ratio() - 0.9).abs() < f64::EPSILON);

        // hit_ratio and effective_hit_ratio diverge when non-cached tokens exist.
        let mut stats2 = CacheStats::default();
        stats2.total_cache_read_tokens = 500;
        stats2.total_cache_write_tokens = 0;
        stats2.total_input_tokens = 2000;
        assert!((stats2.hit_ratio() - 1.0).abs() < f64::EPSILON);
        assert!((stats2.effective_hit_ratio() - 0.25).abs() < f64::EPSILON);
    }

    #[test]
    fn update_cache_stats_skips_zero_usage() {
        let model = test_model();
        let mut state = AgentState::new(model.clone(), vec![model]);
        let usage = michin_ai::Usage {
            input_tokens: 1000,
            output_tokens: 100,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        };
        state.update_cache_stats(Provider::XiaomiMiMo, &usage);
        assert!(state.cache_stats.is_empty());
    }

    #[test]
    fn update_cache_stats_accumulates() {
        let model = test_model();
        let mut state = AgentState::new(model.clone(), vec![model]);

        let usage1 = michin_ai::Usage {
            input_tokens: 1000,
            output_tokens: 100,
            cache_read_tokens: 800,
            cache_write_tokens: 200,
        };
        state.update_cache_stats(Provider::DeepSeek, &usage1);

        let usage2 = michin_ai::Usage {
            input_tokens: 1200,
            output_tokens: 150,
            cache_read_tokens: 1000,
            cache_write_tokens: 200,
        };
        state.update_cache_stats(Provider::DeepSeek, &usage2);

        let stats = state.cache_stats.get(&Provider::DeepSeek).unwrap();
        assert_eq!(stats.total_cache_read_tokens, 1800);
        assert_eq!(stats.total_cache_write_tokens, 400);
        assert_eq!(stats.total_input_tokens, 2200);
    }

    #[test]
    fn update_cache_stats_per_provider_isolation() {
        let model = test_model();
        let mut state = AgentState::new(model.clone(), vec![model]);

        state.update_cache_stats(
            Provider::DeepSeek,
            &michin_ai::Usage {
                input_tokens: 1000,
                output_tokens: 100,
                cache_read_tokens: 900,
                cache_write_tokens: 100,
            },
        );
        state.update_cache_stats(
            Provider::XiaomiMiMo,
            &michin_ai::Usage {
                input_tokens: 500,
                output_tokens: 50,
                cache_read_tokens: 400,
                cache_write_tokens: 100,
            },
        );

        let ds = state.cache_stats.get(&Provider::DeepSeek).unwrap();
        let mimo = state.cache_stats.get(&Provider::XiaomiMiMo).unwrap();
        assert_eq!(ds.total_cache_read_tokens, 900);
        assert_eq!(mimo.total_cache_read_tokens, 400);
    }
}
