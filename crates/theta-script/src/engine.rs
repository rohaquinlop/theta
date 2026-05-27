//! Rhai scripting engine for Theta tool hooks.
//!
//! Scripts use `tool.before(name, callback)` and `tool.after(name, callback)`.
//! These are Rhai functions that store FnPtr callbacks for later evaluation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rhai::{AST, Dynamic, Engine, FnPtr, Scope};

use theta_agent_core::types::ExtensionStatusRow;

use crate::loader::ScriptDef;

/// Outcome of a before-tool hook.
#[derive(Debug, Clone)]
pub enum BeforeHookResult {
    Allow,
    Block { reason: String },
}

/// A single registration call.
#[derive(Clone)]
struct ToolHandler {
    /// Keep the AST alive (closure references it).
    #[allow(dead_code)]
    ast: AST,
    /// Rhai callback for the hook.
    callable: FnPtr,
}

#[derive(Clone)]
struct RegistrationContext {
    script_name: String,
    ast: AST,
}

/// Script engine: loads scripts, evaluates hooks.
///
/// The Rhai `Engine` is wrapped in a `Mutex` because Rhai evaluation
/// uses internal `Cell`/`RefCell` and is not safe to call concurrently
/// from multiple threads. All evaluation methods acquire this lock.
pub struct ScriptEngine {
    engine: Mutex<Engine>,
    handlers: Arc<Mutex<HashMap<String, Vec<ToolHandler>>>>,
    registration_context: Arc<Mutex<Option<RegistrationContext>>>,
    /// TUI status line callbacks: key → Rhai callback that returns String.
    tui_status_handlers: Arc<Mutex<HashMap<String, ToolHandler>>>,
    /// TUI row layout callbacks: index → Rhai callback returning #{ left, center, right }.
    tui_row_handlers: Arc<Mutex<HashMap<usize, ToolHandler>>>,
}

impl ScriptEngine {
    pub fn new() -> Self {
        let mut engine = Engine::new();
        let handlers = Arc::new(Mutex::new(HashMap::new()));
        let registration_context = Arc::new(Mutex::new(None));
        let tui_status_handlers = Arc::new(Mutex::new(HashMap::new()));
        let tui_row_handlers = Arc::new(Mutex::new(HashMap::new()));
        let shared_state: Arc<Mutex<HashMap<String, String>>> =
            Arc::new(Mutex::new(HashMap::new()));

        // Register set_state(key, value) — accessible from all Rhai scripts.
        {
            let ss = Arc::clone(&shared_state);
            engine.register_fn("set_state", move |key: &str, value: &str| {
                if let Ok(mut guard) = ss.lock() {
                    guard.insert(key.to_string(), value.to_string());
                }
            });
        }

        // Register get_state(key) -> String — returns "" if not found.
        {
            let ss = Arc::clone(&shared_state);
            engine.register_fn("get_state", move |key: &str| -> String {
                ss.lock()
                    .map(|guard| guard.get(key).cloned().unwrap_or_default())
                    .unwrap_or_default()
            });
        }

        // Register cwd() -> String — returns current working directory path.
        engine.register_fn("cwd", || -> String {
            std::env::current_dir()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "unknown".into())
        });

        // Register home_dir() -> String — returns the user's home directory path.
        engine.register_fn("home_dir", || -> String {
            dirs::home_dir()
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_else(|| "~".into())
        });

        {
            let handlers: Arc<Mutex<HashMap<String, Vec<ToolHandler>>>> = Arc::clone(&handlers);
            let registration_context: Arc<Mutex<Option<RegistrationContext>>> =
                Arc::clone(&registration_context);
            engine.register_fn(
                "before",
                move |_tool: rhai::Map, tool: &str, callback: FnPtr| {
                    let Some(ctx) = registration_context.lock().unwrap().clone() else {
                        return Dynamic::UNIT;
                    };
                    let key = format!("before_{tool}");
                    let handler = ToolHandler {
                        ast: ctx.ast,
                        callable: callback,
                    };
                    handlers
                        .lock()
                        .unwrap()
                        .entry(key)
                        .or_default()
                        .push(handler);
                    tracing::info!(script = %ctx.script_name, tool, "registered before_tool");
                    Dynamic::UNIT
                },
            );
        }

        {
            let handlers: Arc<Mutex<HashMap<String, Vec<ToolHandler>>>> = Arc::clone(&handlers);
            let registration_context: Arc<Mutex<Option<RegistrationContext>>> =
                Arc::clone(&registration_context);
            engine.register_fn(
                "after",
                move |_tool: rhai::Map, tool: &str, callback: FnPtr| {
                    let Some(ctx) = registration_context.lock().unwrap().clone() else {
                        return Dynamic::UNIT;
                    };
                    let key = format!("after_{tool}");
                    let handler = ToolHandler {
                        ast: ctx.ast,
                        callable: callback,
                    };
                    handlers
                        .lock()
                        .unwrap()
                        .entry(key)
                        .or_default()
                        .push(handler);
                    tracing::info!(script = %ctx.script_name, tool, "registered after_tool");
                    Dynamic::UNIT
                },
            );
        }

        {
            let tui_handlers = Arc::clone(&tui_status_handlers);
            let registration_context: Arc<Mutex<Option<RegistrationContext>>> =
                Arc::clone(&registration_context);
            engine.register_fn(
                "status",
                move |_tui: rhai::Map, key: &str, callback: FnPtr| {
                    let Some(ctx) = registration_context.lock().unwrap().clone() else {
                        return Dynamic::UNIT;
                    };
                    let handler = ToolHandler {
                        ast: ctx.ast,
                        callable: callback,
                    };
                    tui_handlers
                        .lock()
                        .unwrap()
                        .insert(key.to_string(), handler);
                    tracing::info!(script = %ctx.script_name, key, "registered tui.status");
                    Dynamic::UNIT
                },
            );
        }

        // Register tui.row(row_idx, callback) — callback returns #{ left, center, right }.
        {
            let row_handlers = Arc::clone(&tui_row_handlers);
            let registration_context: Arc<Mutex<Option<RegistrationContext>>> =
                Arc::clone(&registration_context);
            engine.register_fn(
                "row",
                move |_tui: rhai::Map, row_idx: i64, callback: FnPtr| {
                    let Some(ctx) = registration_context.lock().unwrap().clone() else {
                        return Dynamic::UNIT;
                    };
                    let idx = row_idx.max(0) as usize;
                    let handler = ToolHandler {
                        ast: ctx.ast,
                        callable: callback,
                    };
                    row_handlers.lock().unwrap().insert(idx, handler);
                    tracing::info!(script = %ctx.script_name, row_idx = idx, "registered tui.row");
                    Dynamic::UNIT
                },
            );
        }

        Self {
            engine: Mutex::new(engine),
            handlers,
            registration_context,
            tui_status_handlers,
            tui_row_handlers,
        }
    }

    /// Load a script file. The script calls `tool.before(name, fn)` / `tool.after(name, fn)`.
    /// Scripts can use `set_state(key, value)` and `get_state(key)` to share state
    /// across hooks (e.g., track caveman level across tool calls and status display).
    pub fn load(&self, def: &ScriptDef) -> Result<(), String> {
        let ast = self
            .engine
            .lock()
            .unwrap()
            .compile(&def.source)
            .map_err(|e| format!("Syntax error in {}: {e}", def.name))?;

        let mut scope = Scope::new();

        scope.push("tool", Dynamic::from(rhai::Map::new()));
        scope.push("tui", Dynamic::from(rhai::Map::new()));

        // ctx.notify(msg)
        {
            let script_name = def.name.clone();
            let notify = Arc::new(move |msg: &str| {
                tracing::info!(script = %script_name, %msg, "script notify");
                Dynamic::UNIT
            }) as Arc<dyn Fn(&str) -> Dynamic + Send + Sync>;
            let mut ctx_map = rhai::Map::new();
            ctx_map.insert("notify".into(), Dynamic::from(notify));
            scope.push("ctx", Dynamic::from(ctx_map));
        }

        *self.registration_context.lock().unwrap() = Some(RegistrationContext {
            script_name: def.name.clone(),
            ast: ast.clone(),
        });

        let eval_result = self
            .engine
            .lock()
            .unwrap()
            .eval_ast_with_scope::<Dynamic>(&mut scope, &ast)
            .map_err(|e| format!("Runtime error in {}: {e}", def.name));

        *self.registration_context.lock().unwrap() = None;

        let _ = eval_result?;

        Ok(())
    }

    /// Evaluate all `before` handlers for `tool_name`.
    pub fn eval_before(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
    ) -> Result<BeforeHookResult, String> {
        let key = format!("before_{tool_name}");
        let handlers = {
            let guard = self.handlers.lock().unwrap();
            guard.get(&key).cloned()
        };
        let Some(handlers) = handlers else {
            return Ok(BeforeHookResult::Allow);
        };
        for handler in &handlers {
            let result = self.run_hook(handler, args, None)?;
            if let Some(reason) = self.parse_block_result(&result) {
                return Ok(BeforeHookResult::Block { reason });
            }
        }
        Ok(BeforeHookResult::Allow)
    }

    /// Evaluate all `after` handlers for `tool_name`.
    pub fn eval_after(
        &self,
        tool_name: &str,
        args: &serde_json::Value,
        result_content: &str,
    ) -> Result<(), String> {
        let key = format!("after_{tool_name}");
        let handlers = {
            let guard = self.handlers.lock().unwrap();
            guard.get(&key).cloned()
        };
        let Some(handlers) = handlers else {
            return Ok(());
        };
        for handler in &handlers {
            let _ = self.run_hook(handler, args, Some(result_content))?;
        }
        Ok(())
    }

    /// Evaluate all registered TUI status callbacks.
    /// Returns key → text pairs for display in the TUI status area.
    pub fn eval_tui_statuses(&self) -> Vec<(String, String)> {
        let engine = self.engine.lock().unwrap();
        let guard = self.tui_status_handlers.lock().unwrap();
        let mut out = Vec::with_capacity(guard.len());
        for (key, handler) in guard.iter() {
            // Call with empty context — status callbacks take no args.
            let call_dyn = Dynamic::from(rhai::Map::new());
            match handler
                .callable
                .call::<Dynamic>(&engine, &handler.ast, (call_dyn,))
            {
                Ok(dyn_val) => {
                    let text = dyn_val.try_cast::<String>().unwrap_or_default();
                    if !text.is_empty() {
                        out.push((key.clone(), text));
                    }
                }
                Err(e) => {
                    tracing::warn!(key, error = %e, "tui.status eval error");
                }
            }
        }
        // Sort by key for stable ordering.
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Evaluate all registered TUI row callbacks.
    /// Returns vec of ExtensionStatusRow with left/center/right text slots,
    /// ordered by row index.
    pub fn eval_tui_rows(&self) -> Vec<ExtensionStatusRow> {
        let engine = self.engine.lock().unwrap();
        let guard = self.tui_row_handlers.lock().unwrap();
        if guard.is_empty() {
            return vec![];
        }

        let mut indices: Vec<usize> = guard.keys().copied().collect();
        indices.sort();

        let mut rows: Vec<ExtensionStatusRow> = Vec::new();
        for idx in indices {
            if let Some(handler) = guard.get(&idx) {
                let call_dyn = Dynamic::from(rhai::Map::new());
                match handler
                    .callable
                    .call::<Dynamic>(&engine, &handler.ast, (call_dyn,))
                {
                    Ok(dyn_val) => {
                        let row = self.parse_row_result(&dyn_val);
                        // Ensure vec has enough entries.
                        while rows.len() <= idx {
                            rows.push(ExtensionStatusRow::default());
                        }
                        rows[idx] = row;
                    }
                    Err(e) => {
                        tracing::warn!(row_idx = idx, error = %e, "tui.row eval error");
                    }
                }
            }
        }
        rows
    }

    /// Parse a Rhai map like #{ left: "text", center: "text", right: "text" }
    /// into an ExtensionStatusRow.
    fn parse_row_result(&self, val: &Dynamic) -> ExtensionStatusRow {
        let default = ExtensionStatusRow::default();
        if !val.is_map() {
            return default;
        }
        let map = val.clone().cast::<rhai::Map>();
        let left = map
            .get("left")
            .and_then(|v| v.clone().try_cast::<String>())
            .filter(|s| !s.is_empty())
            .map(|s| vec![s])
            .unwrap_or_default();
        let center = map
            .get("center")
            .and_then(|v| v.clone().try_cast::<String>())
            .filter(|s| !s.is_empty())
            .map(|s| vec![s])
            .unwrap_or_default();
        let right = map
            .get("right")
            .and_then(|v| v.clone().try_cast::<String>())
            .filter(|s| !s.is_empty())
            .map(|s| vec![s])
            .unwrap_or_default();
        ExtensionStatusRow {
            left,
            center,
            right,
        }
    }

    fn run_hook(
        &self,
        handler: &ToolHandler,
        args: &serde_json::Value,
        result: Option<&str>,
    ) -> Result<Dynamic, String> {
        let engine = self.engine.lock().unwrap();
        let args_str = serde_json::to_string(args).unwrap_or_else(|_| "null".into());
        let args_dyn: Dynamic = engine
            .parse_json(&args_str, false)
            .map_err(|e| format!("arg parse: {e}"))?
            .into();

        let mut call_map = rhai::Map::new();
        call_map.insert("args".into(), args_dyn);
        let call_dyn = Dynamic::from(call_map);

        if let Some(r) = result {
            handler
                .callable
                .call::<Dynamic>(&engine, &handler.ast, (call_dyn, r.to_string()))
                .map_err(|e| format!("hook error: {e}"))
        } else {
            handler
                .callable
                .call::<Dynamic>(&engine, &handler.ast, (call_dyn,))
                .map_err(|e| format!("hook error: {e}"))
        }
    }

    fn parse_block_result(&self, val: &Dynamic) -> Option<String> {
        if val.is_map() {
            let map = val.clone().cast::<rhai::Map>();
            let blocked = map
                .get("blocked")
                .map(|v| v.as_bool().unwrap_or(false))
                .unwrap_or(false);
            if blocked {
                return map
                    .get("reason")
                    .and_then(|r| r.clone().try_cast::<String>());
            }
        }
        None
    }
}

impl Default for ScriptEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_allow_without_handler() {
        let engine = ScriptEngine::new();
        let args = serde_json::json!({"command": "echo hello"});
        let result = engine.eval_before("bash", &args).unwrap();
        assert!(matches!(result, BeforeHookResult::Allow));
    }

    #[test]
    fn test_block_rm_rf() {
        let engine = ScriptEngine::new();

        let script = ScriptDef {
            name: "test".into(),
            location: PathBuf::from("test.rhai"),
            source: r#"
                tool.before("bash", |ctx| {
                    if ctx.args.command.contains("rm -rf") {
                        return #{ blocked: true, reason: "Blocked: rm -rf" };
                    }
                });
            "#
            .into(),
        };

        engine.load(&script).unwrap();

        let args = serde_json::json!({"command": "rm -rf /tmp/test"});
        let result = engine.eval_before("bash", &args).unwrap();
        assert!(
            matches!(&result, BeforeHookResult::Block { reason } if reason.contains("rm -rf")),
            "expected block, got {result:?}"
        );

        let args = serde_json::json!({"command": "ls -la"});
        let result = engine.eval_before("bash", &args).unwrap();
        assert!(matches!(result, BeforeHookResult::Allow));
    }

    #[test]
    fn test_env_protection() {
        let engine = ScriptEngine::new();

        let script = ScriptDef {
            name: "guard".into(),
            location: PathBuf::from("guard.rhai"),
            source: r#"
                tool.before("write", |ctx| {
                    if ctx.args.path.ends_with(".env") {
                        return #{ blocked: true, reason: "no .env writes" };
                    }
                });
            "#
            .into(),
        };

        engine.load(&script).unwrap();

        let args = serde_json::json!({"path": ".env", "content": "SECRET=123"});
        let result = engine.eval_before("write", &args).unwrap();
        assert!(matches!(result, BeforeHookResult::Block { .. }));

        let args = serde_json::json!({"path": "src/main.rs"});
        let result = engine.eval_before("write", &args).unwrap();
        assert!(matches!(result, BeforeHookResult::Allow));
    }

    #[test]
    fn test_tui_status_registration() {
        let engine = ScriptEngine::new();

        let script = ScriptDef {
            name: "status-demo".into(),
            location: PathBuf::from("status.rhai"),
            source: r#"
                tui.status("skill:git-commit", |ctx| {
                    return "committing...";
                });
            "#
            .into(),
        };

        engine.load(&script).unwrap();

        let statuses = engine.eval_tui_statuses();
        assert_eq!(statuses.len(), 1);
        assert_eq!(statuses[0].0, "skill:git-commit");
        assert_eq!(statuses[0].1, "committing...");
    }

    #[test]
    fn test_shared_state_across_hooks() {
        let engine = ScriptEngine::new();

        let script = ScriptDef {
            name: "state-demo".into(),
            location: PathBuf::from("state.rhai"),
            source: r#"
                // Initialize default state
                let current = get_state("level");
                if current == "" {
                    set_state("level", "ultra");
                }

                // After reading a file, update state
                tool.after("read", |ctx, _result| {
                    let path = ctx.args.get("path");
                    if path != () && path.to_string().contains("caveman") {
                        set_state("level", "full");
                    }
                });

                // Display state in TUI
                tui.status("caveman:level", |ctx| {
                    let level = get_state("level");
                    return `[caveman:${level}]`;
                });
            "#
            .into(),
        };

        engine.load(&script).unwrap();

        // Initial state: ultra
        let statuses = engine.eval_tui_statuses();
        assert_eq!(statuses[0].1, "[caveman:ultra]");

        // Simulate caveman skill being read → should update state
        let args = serde_json::json!({"path": "/some/path/caveman/SKILL.md", "offset": 1});
        engine
            .eval_after("read", &args, "# caveman skill content...")
            .unwrap();

        // State should now be "full"
        let statuses = engine.eval_tui_statuses();
        assert_eq!(statuses[0].1, "[caveman:full]");
    }

    #[test]
    fn test_tui_status_multiple_keys() {
        let engine = ScriptEngine::new();

        let script = ScriptDef {
            name: "multi-status".into(),
            location: PathBuf::from("multi.rhai"),
            source: r#"
                tui.status("project:build", |ctx| {
                    return "building...";
                });
                tui.status("project:lint", |ctx| {
                    return "linting...";
                });
            "#
            .into(),
        };

        engine.load(&script).unwrap();

        let statuses = engine.eval_tui_statuses();
        assert_eq!(statuses.len(), 2);
        // Sorted by key.
        assert_eq!(statuses[0].0, "project:build");
        assert_eq!(statuses[1].0, "project:lint");
    }
}
