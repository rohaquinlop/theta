//! Theta Script: Rhai-powered scriptable hooks for Theta.
//!
//! Loads `.rhai` scripts from `~/.theta/extensions/` (global) and
//! `./.theta/extensions/` (project-local). Scripts can register
//! `before_tool` and `after_tool` hooks that intercept tool calls.
//!
//! # Script API
//!
//! ```rhai
//! // Block dangerous commands
//! tool.before("bash", |call| {
//!     if call.args.command.contains("rm -rf") {
//!         return #{ blocked: true, reason: "Blocked: rm -rf" };
//!     }
//! });
//!
//! // Notify on large writes
//! tool.after("write", |call, result| {
//!     if call.args.content.len() > 10000 {
//!         ctx.notify("Large file write completed");
//!     }
//! });
//! ```
//!
//! Throws `AgentError::ToolExecution` with `{ blocked: true, reason: "..." }`
//! to block a tool call.

mod engine;
mod hooks;
mod loader;

pub use engine::ScriptEngine;
pub use hooks::ScriptHooks;
pub use loader::ScriptLoader;
