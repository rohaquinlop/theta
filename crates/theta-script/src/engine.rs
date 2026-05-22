//! Rhai scripting engine for Theta tool hooks.
//!
//! Scripts use `tool.before(name, callback)` and `tool.after(name, callback)`.
//! These are Rhai functions that store FnPtr callbacks for later evaluation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rhai::{AST, Dynamic, Engine, FnPtr, Scope};

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
pub struct ScriptEngine {
    engine: Engine,
    handlers: Arc<Mutex<HashMap<String, Vec<ToolHandler>>>>,
    registration_context: Arc<Mutex<Option<RegistrationContext>>>,
}

impl ScriptEngine {
    pub fn new() -> Self {
        let mut engine = Engine::new();
        let handlers = Arc::new(Mutex::new(HashMap::new()));
        let registration_context = Arc::new(Mutex::new(None));

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

        Self {
            engine,
            handlers,
            registration_context,
        }
    }

    /// Load a script file. The script calls `tool_before(name, fn)` / `tool_after(name, fn)`.
    pub fn load(&self, def: &ScriptDef) -> Result<(), String> {
        let ast = self
            .engine
            .compile(&def.source)
            .map_err(|e| format!("Syntax error in {}: {e}", def.name))?;

        let mut scope = Scope::new();

        scope.push("tool", Dynamic::from(rhai::Map::new()));

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

    fn run_hook(
        &self,
        handler: &ToolHandler,
        args: &serde_json::Value,
        result: Option<&str>,
    ) -> Result<Dynamic, String> {
        let args_str = serde_json::to_string(args).unwrap_or_else(|_| "null".into());
        let args_dyn: Dynamic = self
            .engine
            .parse_json(&args_str, false)
            .map_err(|e| format!("arg parse: {e}"))?
            .into();

        let mut call_map = rhai::Map::new();
        call_map.insert("args".into(), args_dyn);
        let call_dyn = Dynamic::from(call_map);

        if let Some(r) = result {
            handler
                .callable
                .call::<Dynamic>(&self.engine, &handler.ast, (call_dyn, r.to_string()))
                .map_err(|e| format!("hook error: {e}"))
        } else {
            handler
                .callable
                .call::<Dynamic>(&self.engine, &handler.ast, (call_dyn,))
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
}
