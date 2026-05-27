//! Extension system: trait-based plugin architecture for custom tools,
//! commands, and hooks. Extensions are compiled into the binary — no dynamic
//! loading in MVP.

use std::sync::Arc;

use async_trait::async_trait;

use crate::tools::ToolContext;
use theta_agent_core::hooks::Hooks;
use theta_agent_core::types::AgentTool;

/// Context passed to extensions on startup.
#[derive(Debug, Clone)]
pub struct ExtensionContext {
    /// Project working directory.
    pub working_dir: std::path::PathBuf,
    /// Shared tool context for creating built-in-style tools.
    pub tool_context: ToolContext,
}

/// A slash command registered by an extension.
#[derive(Debug, Clone)]
pub struct ExtensionCommand {
    /// Command name (without the `/` prefix).
    pub name: String,
    /// Short description for `/help`.
    pub description: String,
    /// Usage example.
    pub usage: String,
}

/// An extension that can register tools, commands, and hooks.
///
/// Extensions are compiled into the binary. Users who want custom tools
/// fork Theta, implement this trait, and register their extension in main.rs.
#[async_trait]
pub trait Extension: Send + Sync {
    /// Unique extension name.
    fn name(&self) -> &str;

    /// Extension version string.
    fn version(&self) -> &str;

    /// Called when the extension is loaded (before any agent runs).
    async fn on_startup(&self, _ctx: &ExtensionContext) -> anyhow::Result<()> {
        Ok(())
    }

    /// Called when the extension is unloaded (on shutdown).
    async fn on_shutdown(&self, _ctx: &ExtensionContext) -> anyhow::Result<()> {
        Ok(())
    }

    /// Custom tools provided by this extension.
    fn tools(&self) -> Vec<Arc<dyn AgentTool>> {
        vec![]
    }

    /// Custom slash commands provided by this extension.
    fn commands(&self) -> Vec<ExtensionCommand> {
        vec![]
    }

    /// Custom lifecycle hooks. Return `None` to use defaults.
    fn hooks(&self) -> Option<Arc<dyn Hooks>> {
        None
    }
}

/// Registry of all loaded extensions.
pub struct ExtensionRegistry {
    extensions: Vec<Arc<dyn Extension>>,
}

impl ExtensionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            extensions: Vec::new(),
        }
    }

    /// Register an extension.
    pub fn register(&mut self, ext: Arc<dyn Extension>) {
        self.extensions.push(ext);
    }

    /// Initialize all extensions.
    pub async fn startup(&self, ctx: &ExtensionContext) -> anyhow::Result<()> {
        for ext in &self.extensions {
            ext.on_startup(ctx).await?;
        }
        Ok(())
    }

    /// Shut down all extensions.
    pub async fn shutdown(&self, ctx: &ExtensionContext) -> anyhow::Result<()> {
        for ext in &self.extensions {
            ext.on_shutdown(ctx).await?;
        }
        Ok(())
    }

    /// Collect all tools from all extensions.
    pub fn all_tools(&self) -> Vec<Arc<dyn AgentTool>> {
        self.extensions.iter().flat_map(|e| e.tools()).collect()
    }

    /// Collect all commands from all extensions.
    pub fn all_commands(&self) -> Vec<ExtensionCommand> {
        self.extensions.iter().flat_map(|e| e.commands()).collect()
    }

    /// Collect all hooks from extensions (first non-None wins).
    pub fn all_hooks(&self) -> Vec<Arc<dyn Hooks>> {
        self.extensions.iter().filter_map(|e| e.hooks()).collect()
    }

    /// Get an extension by name.
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Extension>> {
        self.extensions.iter().find(|e| e.name() == name)
    }

    /// Number of loaded extensions.
    pub fn len(&self) -> usize {
        self.extensions.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.extensions.is_empty()
    }
}

impl Default for ExtensionRegistry {
    fn default() -> Self {
        Self::new()
    }
}
