//! Prefix-cache shape diagnostics.
//!
//! Hashes the system prompt and normalized tool schemas each turn,
//! diffs against the previous turn, and reports why a cache miss
//! happened (system, tools, log_rewrite). This directly supports
//! the DeepSeek prefix-cache performance investigation.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use michin_ai::Tool;

/// Snapshot of the prefix shape at the start of a turn.
#[derive(Debug, Clone, Default)]
pub struct CacheShape {
    pub system_hash: u64,
    pub tools_hash: u64,
}

/// Result of diffing two cache shapes.
#[derive(Debug, Clone)]
pub struct CacheShapeDiff {
    pub first_turn: bool,
    pub system_changed: bool,
    pub tools_changed: bool,
    pub log_rewritten: bool,
    pub per_tool_tokens: Vec<(String, u32)>,
}

impl CacheShapeDiff {
    /// Whether anything changed that would bust the prefix cache.
    pub fn cache_busted(&self) -> bool {
        self.first_turn || self.system_changed || self.tools_changed || self.log_rewritten
    }

    /// Human-readable reason for the cache miss.
    pub fn reason(&self) -> String {
        if self.first_turn {
            return "first_turn".to_string();
        }
        let mut parts = Vec::new();
        if self.system_changed {
            parts.push("system");
        }
        if self.tools_changed {
            parts.push("tools");
        }
        if self.log_rewritten {
            parts.push("log_rewrite");
        }
        if parts.is_empty() {
            "hit".to_string()
        } else {
            parts.join(", ")
        }
    }
}

/// Compute the hash of the system prompt blocks.
pub fn hash_system_prompt(blocks: &[michin_ai::ContentBlock]) -> u64 {
    let mut hasher = DefaultHasher::new();
    for block in blocks {
        match serde_json::to_vec(block) {
            Ok(bytes) => bytes.hash(&mut hasher),
            Err(_) => {
                format!("{block:?}").hash(&mut hasher);
            }
        }
    }
    hasher.finish()
}

/// Compute the hash of tool schemas.
///
/// Sorts by name for byte-stable serialization. `rebuild_michin_ai_tools`
/// already sorts the stored `michin_ai_tools`, but this function operates
/// on an arbitrary `&[Tool]` slice and must be independently stable.
pub fn hash_tool_schemas(tools: &[Tool]) -> u64 {
    let mut hasher = DefaultHasher::new();
    let mut sorted: Vec<&Tool> = tools.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    for tool in &sorted {
        tool.name.hash(&mut hasher);
        tool.description.hash(&mut hasher);
        if let Ok(json) = serde_json::to_vec(&tool.parameters) {
            json.hash(&mut hasher);
        } else {
            format!("{:?}", tool.parameters).hash(&mut hasher);
        }
    }
    hasher.finish()
}

/// Estimate per-tool token cost by serializing to JSON and approximating.
pub fn estimate_tool_tokens(tool: &Tool) -> u32 {
    let json = serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
    });
    michin_ai::approximate_token_count(&serde_json::to_string(&json).unwrap_or_default())
}

/// Compute per-tool token estimates for a list of tools.
pub fn per_tool_tokens(tools: &[Tool]) -> Vec<(String, u32)> {
    tools
        .iter()
        .map(|t| (t.name.clone(), estimate_tool_tokens(t)))
        .collect()
}

/// Diff two cache shapes and compute a CacheShapeDiff.
pub fn diff_shapes(
    prev: Option<&CacheShape>,
    current: &CacheShape,
    tools: &[Tool],
    log_rewritten: bool,
) -> CacheShapeDiff {
    let (first_turn, system_changed, tools_changed) = match prev {
        Some(p) => (
            false,
            p.system_hash != current.system_hash,
            p.tools_hash != current.tools_hash,
        ),
        None => (true, false, false),
    };

    let per_tool = per_tool_tokens(tools);

    CacheShapeDiff {
        first_turn,
        system_changed,
        tools_changed,
        log_rewritten,
        per_tool_tokens: per_tool,
    }
}
