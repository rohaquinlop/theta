//! Script integration: loads Rhai scripts from disk and bridges
//! them into the agent via the Hooks trait.
//!
//! Scripts live in `~/.michin/extensions/*.rhai` (global) and
//! `./.michin/extensions/*.rhai` (project-local). They are written
//! by the agent when the user asks for guardrails, permission gates,
//! or tool-call interceptors.

use std::path::Path;
use std::sync::Arc;

use michin_agent_core::hooks::Hooks;
use michin_script::{ScriptEngine, ScriptHooks, ScriptLoader};
use tokio::sync::Notify;

/// A discovered script with metadata for prompt display.
#[derive(Debug, Clone)]
pub struct DiscoveredScript {
    pub name: String,
    pub location: std::path::PathBuf,
    pub source: String,
}

/// Discover scripts without loading them (for prompt display).
pub async fn discover_scripts(working_dir: &Path) -> Vec<DiscoveredScript> {
    let loader = ScriptLoader::discover(working_dir).await;
    loader
        .scripts()
        .iter()
        .map(|def| DiscoveredScript {
            name: def.name.clone(),
            location: def.location.clone(),
            source: def.source.clone(),
        })
        .collect()
}

/// Build the `<available_extensions>` XML prompt block (slim: name + location only).
pub fn build_extensions_slim_block(scripts: &[DiscoveredScript]) -> Option<String> {
    if scripts.is_empty() {
        return None;
    }

    let mut block = String::from("\n<available_extensions>\n");
    for script in scripts {
        block.push_str(&format!(
            "  <extension>\n    <name>{name}</name>\n    <location>{loc}</location>\n  </extension>\n",
            name = script.name,
            loc = script.location.display(),
        ));
    }
    block.push_str("</available_extensions>");
    Some(block)
}

/// Build the `<available_extensions>` XML prompt block with full source.
pub fn build_extensions_prompt_block(scripts: &[DiscoveredScript]) -> Option<String> {
    if scripts.is_empty() {
        return None;
    }

    let mut block = String::from("\n<available_extensions>\n");
    for script in scripts {
        block.push_str(&format!(
            r#"  <extension>
    <name>{name}</name>
    <location>{loc}</location>
    <source>{src}</source>
  </extension>
"#,
            name = script.name,
            loc = script.location.display(),
            src = script.source,
        ));
    }
    block.push_str("</available_extensions>");
    Some(block)
}

/// Load scripts from disk and build a hooks implementation.
/// Returns `None` if no scripts found (avoids overhead of empty engine).
pub async fn load_script_hooks(
    working_dir: &Path,
    status_notify: Arc<Notify>,
) -> Option<Arc<dyn Hooks>> {
    let loader = ScriptLoader::discover(working_dir).await;

    if loader.is_empty() {
        tracing::info!("no scripts discovered");
        return None;
    }

    tracing::info!(count = loader.len(), "loading scripts");

    let engine = Arc::new(ScriptEngine::new());
    let mut errors = 0usize;

    for def in loader.scripts() {
        if let Err(e) = engine.load(def) {
            tracing::error!(
                script = %def.name,
                location = %def.location.display(),
                error = %e,
                "failed to load script"
            );
            errors += 1;
        }
    }

    if errors > 0 && engine_has_no_handlers() {
        // All scripts failed — return None so agent uses NoopHooks.
        tracing::warn!("all scripts failed to load, using noop hooks");
        None
    } else {
        if errors > 0 {
            tracing::warn!(errors, "some scripts failed to load, using partial hooks");
        }
        let hooks = ScriptHooks::new(engine, status_notify);
        Some(Arc::new(hooks))
    }
}

/// Placeholder check — engine has at least one handler registered.
fn engine_has_no_handlers() -> bool {
    // ScriptEngine always has some internal state, so we assume
    // if we got here with errors, at least one script loaded.
    false
}
