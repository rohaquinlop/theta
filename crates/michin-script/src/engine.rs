//! Rhai scripting engine for MichiN tool hooks.
//!
//! Scripts use `tool.before(name, callback)` and `tool.after(name, callback)`.
//! These are Rhai functions that store FnPtr callbacks for later evaluation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use rhai::{AST, Dynamic, Engine, EvalAltResult, FnPtr, Position, Scope};

use michin_agent_core::command_policy;
use michin_agent_core::types::{ExtensionStatusRow, ToolExecutionMode};

/// Maximum bytes to capture from exec() stdout/stderr before truncating.
const MAX_OUTPUT_BYTES: usize = 256 * 1024;

use crate::loader::ScriptDef;

/// Outcome of a before-tool hook.
#[derive(Debug, Clone)]
pub enum BeforeHookResult {
    Allow,
    Block { reason: String },
}

/// Metadata for a tool registered via `tool.register()` in a Rhai script.
#[derive(Debug, Clone)]
pub struct RegisteredToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
    pub execution_mode: ToolExecutionMode,
    /// The script AST that contains the `execute()` function.
    pub ast: AST,
    pub script_name: String,
}

/// Result returned by a custom tool's `execute()` function.
#[derive(Debug, Clone)]
pub struct ToolExecResult {
    pub content: String,
    pub is_error: bool,
}

/// A single registration call.
#[derive(Clone)]
struct ToolHandler {
    /// Keep the AST alive (closure references it).
    ast: AST,
    /// Rhai callback for the hook.
    callable: FnPtr,
    /// Script name for state namespace.
    script_name: String,
}

#[derive(Clone)]
struct RegistrationContext {
    script_name: String,
    ast: AST,
}

/// Script engine: loads scripts, evaluates hooks, and manages custom tools.
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
    /// Custom tools registered via `tool.register()`.
    registered_tools: Arc<Mutex<Vec<RegisteredToolDef>>>,
    /// When true, all script registrations (before/after/status/row/register) are
    /// suppressed. Prevents duplicate handlers when re-evaluating the AST during
    /// `eval_tool_execute`.
    suppress_registration: Arc<Mutex<bool>>,
    /// Namespace prefix for set_state/get_state keys during script load.
    state_namespace: Arc<Mutex<Option<String>>>,
}

impl ScriptEngine {
    pub fn new() -> Self {
        let mut engine = Engine::new();
        engine.set_max_operations(1_000_000);
        let handlers = Arc::new(Mutex::new(HashMap::new()));
        let registration_context = Arc::new(Mutex::new(None));
        let tui_status_handlers = Arc::new(Mutex::new(HashMap::new()));
        let tui_row_handlers = Arc::new(Mutex::new(HashMap::new()));
        let registered_tools = Arc::new(Mutex::new(Vec::new()));
        let suppress_registration = Arc::new(Mutex::new(false));
        let shared_state: Arc<Mutex<HashMap<String, String>>> =
            Arc::new(Mutex::new(HashMap::new()));
        let state_namespace: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        // Register set_state(key, value) — accessible from all Rhai scripts.
        {
            let ss = Arc::clone(&shared_state);
            let sn_state = Arc::clone(&state_namespace);
            engine.register_fn("set_state", move |key: &str, value: &str| {
                let prefix = sn_state.lock().unwrap().clone().unwrap_or_default();
                let full_key = format!("{prefix}:{key}");
                if let Ok(mut guard) = ss.lock() {
                    guard.insert(full_key, value.to_string());
                }
            });
        }

        // Register get_state(key) -> String — returns "" if not found.
        {
            let ss = Arc::clone(&shared_state);
            let sn_state = Arc::clone(&state_namespace);
            engine.register_fn("get_state", move |key: &str| -> String {
                let prefix = sn_state.lock().unwrap().clone().unwrap_or_default();
                let full_key = format!("{prefix}:{key}");
                ss.lock()
                    .map(|guard| guard.get(&full_key).cloned().unwrap_or_default())
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
            let suppress = Arc::clone(&suppress_registration);
            engine.register_fn(
                "before",
                move |_tool: rhai::Map, tool: &str, callback: FnPtr| {
                    if *suppress.lock().unwrap() {
                        return Dynamic::UNIT;
                    }
                    let Some(ctx) = registration_context.lock().unwrap().clone() else {
                        return Dynamic::UNIT;
                    };
                    let key = format!("before_{tool}");
                    let handler = ToolHandler {
                        ast: ctx.ast,
                        callable: callback,
                        script_name: ctx.script_name.clone(),
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
            let suppress = Arc::clone(&suppress_registration);
            engine.register_fn(
                "after",
                move |_tool: rhai::Map, tool: &str, callback: FnPtr| {
                    if *suppress.lock().unwrap() {
                        return Dynamic::UNIT;
                    }
                    let Some(ctx) = registration_context.lock().unwrap().clone() else {
                        return Dynamic::UNIT;
                    };
                    let key = format!("after_{tool}");
                    let handler = ToolHandler {
                        ast: ctx.ast,
                        callable: callback,
                        script_name: ctx.script_name.clone(),
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
            let suppress = Arc::clone(&suppress_registration);
            engine.register_fn(
                "status",
                move |_tui: rhai::Map, key: &str, callback: FnPtr| {
                    if *suppress.lock().unwrap() {
                        return Dynamic::UNIT;
                    }
                    let Some(ctx) = registration_context.lock().unwrap().clone() else {
                        return Dynamic::UNIT;
                    };
                    let handler = ToolHandler {
                        ast: ctx.ast,
                        callable: callback,
                        script_name: ctx.script_name.clone(),
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

        // Register exec(command, args_array) -> #{ stdout, stderr, exit_code }
        // Kills the subprocess after 30 seconds to prevent hanging.
        engine.register_fn("exec", |command: &str, args: rhai::Array| -> Dynamic {
            let arg_strings: Vec<String> = args.iter().map(|a| a.to_string()).collect();

            // Check command safety policy.
            let decision = command_policy::evaluate_exec_command(command, &arg_strings);
            if decision.decision == michin_agent_core::SafetyDecisionKind::Rejected {
                let mut map = rhai::Map::new();
                map.insert("stdout".into(), Dynamic::from(String::new()));
                map.insert("stderr".into(), Dynamic::from(decision.details));
                map.insert("exit_code".into(), Dynamic::from(-1 as rhai::INT));
                return Dynamic::from(map);
            }

            let output = std::process::Command::new(command)
                .args(&arg_strings)
                .output();
            match output {
                Ok(out) => {
                    let mut stdout = String::from_utf8_lossy(&out.stdout).to_string();
                    let mut stderr = String::from_utf8_lossy(&out.stderr).to_string();
                    let mut truncated = false;
                    if stdout.len() > MAX_OUTPUT_BYTES {
                        let cap = cap_bytes(&stdout, MAX_OUTPUT_BYTES);
                        truncated |= stdout.len() != cap.len();
                        stdout = cap;
                    }
                    if stderr.len() > MAX_OUTPUT_BYTES {
                        let cap = cap_bytes(&stderr, MAX_OUTPUT_BYTES);
                        truncated |= stderr.len() != cap.len();
                        stderr = cap;
                    }
                    if truncated {
                        stderr.push_str("\n[output truncated at 256KB]");
                    }
                    let exit_code = out.status.code().unwrap_or(-1) as rhai::INT;
                    let mut map = rhai::Map::new();
                    map.insert("stdout".into(), Dynamic::from(stdout));
                    map.insert("stderr".into(), Dynamic::from(stderr));
                    map.insert("exit_code".into(), Dynamic::from(exit_code));
                    Dynamic::from(map)
                }
                Err(e) => {
                    let mut map = rhai::Map::new();
                    map.insert("stdout".into(), Dynamic::from(String::new()));
                    map.insert("stderr".into(), Dynamic::from(e.to_string()));
                    map.insert("exit_code".into(), Dynamic::from(-1 as rhai::INT));
                    Dynamic::from(map)
                }
            }
        });

        // Register exec_with_timeout(command, args_array, timeout_secs)
        // Returns the same shape as exec() but kills the process after the timeout.
        // Falls back to a timeout error #{ stdout: "", stderr: "timeout", exit_code: -1 }.
        engine.register_fn(
            "exec_with_timeout",
            |command: &str, args: rhai::Array, timeout_secs: i64| -> Dynamic {
                let arg_strings: Vec<String> = args.iter().map(|a| a.to_string()).collect();

                // Check command safety policy.
                let decision = command_policy::evaluate_exec_command(command, &arg_strings);
                if decision.decision == michin_agent_core::SafetyDecisionKind::Rejected {
                    let mut map = rhai::Map::new();
                    map.insert("stdout".into(), Dynamic::from(String::new()));
                    map.insert("stderr".into(), Dynamic::from(decision.details));
                    map.insert("exit_code".into(), Dynamic::from(-1 as rhai::INT));
                    return Dynamic::from(map);
                }

                let mut child = match std::process::Command::new(command)
                    .args(&arg_strings)
                    .stdout(std::process::Stdio::piped())
                    .stderr(std::process::Stdio::piped())
                    .spawn()
                {
                    Ok(c) => c,
                    Err(e) => {
                        let mut map = rhai::Map::new();
                        map.insert("stdout".into(), Dynamic::from(String::new()));
                        map.insert("stderr".into(), Dynamic::from(e.to_string()));
                        map.insert("exit_code".into(), Dynamic::from(-1 as rhai::INT));
                        return Dynamic::from(map);
                    }
                };
                let deadline = std::time::Instant::now()
                    + std::time::Duration::from_secs(timeout_secs.max(1) as u64);
                loop {
                    match child.try_wait() {
                        Ok(Some(_status)) => break,
                        Ok(None) => {
                            if std::time::Instant::now() >= deadline {
                                let _ = child.kill();
                                let _ = child.wait();
                                let mut map = rhai::Map::new();
                                map.insert("stdout".into(), Dynamic::from(String::new()));
                                map.insert(
                                    "stderr".into(),
                                    Dynamic::from(format!(
                                        "process timed out after {timeout_secs}s"
                                    )),
                                );
                                map.insert("exit_code".into(), Dynamic::from(-1 as rhai::INT));
                                return Dynamic::from(map);
                            }
                            std::thread::sleep(std::time::Duration::from_millis(50));
                        }
                        Err(e) => {
                            let mut map = rhai::Map::new();
                            map.insert("stdout".into(), Dynamic::from(String::new()));
                            map.insert("stderr".into(), Dynamic::from(e.to_string()));
                            map.insert("exit_code".into(), Dynamic::from(-1 as rhai::INT));
                            return Dynamic::from(map);
                        }
                    }
                }
                let result = child.wait_with_output().unwrap();
                let mut stdout = String::from_utf8_lossy(&result.stdout).to_string();
                let mut stderr = String::from_utf8_lossy(&result.stderr).to_string();
                let mut truncated = false;
                if stdout.len() > MAX_OUTPUT_BYTES {
                    let cap = cap_bytes(&stdout, MAX_OUTPUT_BYTES);
                    truncated |= stdout.len() != cap.len();
                    stdout = cap;
                }
                if stderr.len() > MAX_OUTPUT_BYTES {
                    let cap = cap_bytes(&stderr, MAX_OUTPUT_BYTES);
                    truncated |= stderr.len() != cap.len();
                    stderr = cap;
                }
                if truncated {
                    stderr.push_str("\n[output truncated at 256KB]");
                }
                let exit_code = result.status.code().unwrap_or(-1) as rhai::INT;
                let mut map = rhai::Map::new();
                map.insert("stdout".into(), Dynamic::from(stdout));
                map.insert("stderr".into(), Dynamic::from(stderr));
                map.insert("exit_code".into(), Dynamic::from(exit_code));
                Dynamic::from(map)
            },
        );

        // Register read_file(path) -> String
        engine.register_fn(
            "read_file",
            |path: &str| -> Result<String, Box<EvalAltResult>> {
                std::fs::read_to_string(path).map_err(|e| {
                    EvalAltResult::ErrorRuntime(
                        format!("read_file failed: {e}").into(),
                        Position::NONE,
                    )
                    .into()
                })
            },
        );

        // Register write_file(path, content)
        engine.register_fn(
            "write_file",
            |path: &str, content: &str| -> Result<(), Box<EvalAltResult>> {
                std::fs::write(path, content).map_err(|e| {
                    EvalAltResult::ErrorRuntime(
                        format!("write_file failed: {e}").into(),
                        Position::NONE,
                    )
                    .into()
                })
            },
        );

        // Register str_trim(s) -> String — returns a new trimmed copy.
        // Rhai's built-in trim() mutates in place and returns (), so it
        // cannot be used in expressions.
        engine.register_fn("str_trim", |s: &str| -> String { s.trim().to_string() });

        // Register tui.row(row_idx, callback) — callback returns #{ left, center, right }.
        {
            let row_handlers = Arc::clone(&tui_row_handlers);
            let registration_context: Arc<Mutex<Option<RegistrationContext>>> =
                Arc::clone(&registration_context);
            let suppress = Arc::clone(&suppress_registration);
            engine.register_fn(
                "row",
                move |_tui: rhai::Map, row_idx: i64, callback: FnPtr| {
                    if *suppress.lock().unwrap() {
                        return Dynamic::UNIT;
                    }
                    let Some(ctx) = registration_context.lock().unwrap().clone() else {
                        return Dynamic::UNIT;
                    };
                    let idx = row_idx.max(0) as usize;
                    let handler = ToolHandler {
                        ast: ctx.ast,
                        callable: callback,
                        script_name: ctx.script_name.clone(),
                    };
                    row_handlers.lock().unwrap().insert(idx, handler);
                    tracing::info!(script = %ctx.script_name, row_idx = idx, "registered tui.row");
                    Dynamic::UNIT
                },
            );
        }

        // Register tool.register(name, schema_map) — registers a custom tool.
        {
            let reg_tools = Arc::clone(&registered_tools);
            let suppress = Arc::clone(&suppress_registration);
            let registration_context: Arc<Mutex<Option<RegistrationContext>>> =
                Arc::clone(&registration_context);
            engine.register_fn(
                "register",
                move |_tool: rhai::Map, name: &str, schema: rhai::Map| {
                    // Skip if registration is suppressed (re-eval during execute).
                    if *suppress.lock().unwrap() {
                        return Dynamic::UNIT;
                    }
                    // Reject built-in tool names.
                    const BUILT_IN_NAMES: &[&str] = &["read", "write", "edit", "bash"];
                    if BUILT_IN_NAMES.contains(&name) {
                        tracing::warn!(
                            tool_name = name,
                            "custom tool rejected: name shadows built-in"
                        );
                        return Dynamic::UNIT;
                    }
                    let Some(ctx) = registration_context.lock().unwrap().clone() else {
                        return Dynamic::UNIT;
                    };

                    let description = schema
                        .get("description")
                        .and_then(|v| v.clone().try_cast::<String>())
                        .unwrap_or_default();

                    // Extract the nested parameters map from the schema, or use the full schema as fallback.
                    let parameters = schema
                        .get("parameters")
                        .filter(|v| v.is_map())
                        .map(|v| rhai_map_to_json_value(&v.clone().cast::<rhai::Map>()))
                        .unwrap_or_else(|| rhai_map_to_json_value(&schema));

                    let execution_mode = schema
                        .get("execution_mode")
                        .and_then(|v| v.clone().try_cast::<String>())
                        .map(|s| match s.as_str() {
                            "sequential" => ToolExecutionMode::Sequential,
                            _ => ToolExecutionMode::Parallel,
                        })
                        .unwrap_or(ToolExecutionMode::Parallel);

                    let tool_def = RegisteredToolDef {
                        name: name.to_string(),
                        description,
                        parameters,
                        execution_mode,
                        ast: ctx.ast.clone(),
                        script_name: ctx.script_name.clone(),
                    };
                    tracing::info!(
                        script = %ctx.script_name,
                        tool_name = name,
                        "registered custom tool"
                    );
                    reg_tools.lock().unwrap().push(tool_def);
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
            registered_tools,
            suppress_registration,
            state_namespace,
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

        // Set shared state namespace to this script's name.
        *self.state_namespace.lock().unwrap() = Some(def.name.clone());

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
        *self.state_namespace.lock().unwrap() = None;

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

    /// Return all custom tools registered via `tool.register()`.
    pub fn registered_tools(&self) -> Vec<RegisteredToolDef> {
        self.registered_tools.lock().unwrap().clone()
    }

    /// Execute the `execute(args)` function in a custom tool's script AST.
    /// Returns the content string and error flag from the Rhai return value.
    pub fn eval_tool_execute(
        &self,
        tool_def: &RegisteredToolDef,
        args: &serde_json::Value,
    ) -> Result<ToolExecResult, String> {
        let engine = self.engine.lock().unwrap();
        let args_str = serde_json::to_string(args).unwrap_or_else(|_| "{}".into());
        let args_dyn: Dynamic = engine
            .parse_json(&args_str, false)
            .map_err(|e| format!("arg parse: {e}"))?
            .into();

        // Evaluate the full AST to define const values and fn definitions
        // in the scope. call_fn skips top-level statements, so without this
        // step any `const` bindings (paths, config) would be undefined.
        *self.suppress_registration.lock().unwrap() = true;
        let mut scope = Scope::new();
        scope.push("tool", Dynamic::from(rhai::Map::new()));
        scope.push("tui", Dynamic::from(rhai::Map::new()));
        scope.push("ctx", Dynamic::from(rhai::Map::new()));

        // Errors here are real (bad const, syntax, etc.) — surface them.
        if let Err(e) = engine.eval_ast_with_scope::<Dynamic>(&mut scope, &tool_def.ast) {
            *self.suppress_registration.lock().unwrap() = false;
            return Err(format!("init error in {}: {e}", tool_def.script_name));
        }
        *self.suppress_registration.lock().unwrap() = false;

        // Second pass: call execute() with the populated scope.
        let result = engine
            .call_fn::<Dynamic>(&mut scope, &tool_def.ast, "execute", (args_dyn,))
            .map_err(|e| format!("execute() error in {}: {e}", tool_def.script_name))?;

        parse_tool_exec_result(&result)
    }

    /// Evaluate all registered TUI status callbacks.
    /// Returns key → text pairs for display in the TUI status area.
    pub fn eval_tui_statuses(&self) -> Vec<(String, String)> {
        let guard = self.tui_status_handlers.lock().unwrap();
        let mut out = Vec::with_capacity(guard.len());
        for (key, handler) in guard.iter() {
            // Set namespace so set_state/get_state in callbacks use the correct prefix.
            *self.state_namespace.lock().unwrap() = Some(handler.script_name.clone());
            let engine = self.engine.lock().unwrap();
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
            drop(engine);
            *self.state_namespace.lock().unwrap() = None;
        }
        // Sort by key for stable ordering.
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    /// Evaluate all registered TUI row callbacks.
    /// Returns vec of ExtensionStatusRow with left/center/right text slots,
    /// ordered by row index.
    pub fn eval_tui_rows(&self) -> Vec<ExtensionStatusRow> {
        let guard = self.tui_row_handlers.lock().unwrap();
        if guard.is_empty() {
            return vec![];
        }

        let mut indices: Vec<usize> = guard.keys().copied().collect();
        indices.sort();

        let mut rows: Vec<ExtensionStatusRow> = Vec::new();
        for idx in indices {
            if let Some(handler) = guard.get(&idx) {
                // Set namespace so set_state/get_state in callbacks use the correct prefix.
                *self.state_namespace.lock().unwrap() = Some(handler.script_name.clone());
                let engine = self.engine.lock().unwrap();
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
                drop(engine);
                *self.state_namespace.lock().unwrap() = None;
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
        // Set namespace so set_state/get_state callbacks use the correct prefix.
        *self.state_namespace.lock().unwrap() = Some(handler.script_name.clone());

        let engine = self.engine.lock().unwrap();
        let args_str = serde_json::to_string(args).unwrap_or_else(|_| "null".into());
        let args_dyn: Dynamic = engine
            .parse_json(&args_str, false)
            .map_err(|e| format!("arg parse: {e}"))?
            .into();

        let mut call_map = rhai::Map::new();
        call_map.insert("args".into(), args_dyn);
        let call_dyn = Dynamic::from(call_map);

        let res = if let Some(r) = result {
            handler
                .callable
                .call::<Dynamic>(&engine, &handler.ast, (call_dyn, r.to_string()))
                .map_err(|e| format!("hook error: {e}"))
        } else {
            handler
                .callable
                .call::<Dynamic>(&engine, &handler.ast, (call_dyn,))
                .map_err(|e| format!("hook error: {e}"))
        };

        *self.state_namespace.lock().unwrap() = None;
        res
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

/// Cap a string at `limit` bytes, preserving UTF-8 validity.
/// Returns the original if within limit.
fn cap_bytes(s: &str, limit: usize) -> String {
    if s.len() <= limit {
        return s.to_string();
    }
    let mut capped = s.as_bytes()[..limit].to_vec();
    // Chop off trailing partial UTF-8 byte.
    while !capped.is_empty() && capped.last().copied().unwrap() & 0b1100_0000 == 0b1000_0000 {
        capped.pop();
    }
    String::from_utf8(capped).unwrap_or_else(|_| s[..limit].to_string())
}

/// Convert a Rhai map to a `serde_json::Value`.
/// Handles nested maps, strings, numbers, and booleans.
fn rhai_map_to_json_value(map: &rhai::Map) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    for (key, val) in map {
        obj.insert(key.to_string(), rhai_dynamic_to_json(val));
    }
    serde_json::Value::Object(obj)
}

fn rhai_dynamic_to_json(val: &Dynamic) -> serde_json::Value {
    if val.is_string() {
        serde_json::Value::String(val.to_string())
    } else if val.is::<i64>() {
        serde_json::Value::Number(serde_json::Number::from(val.as_int().unwrap_or(0)))
    } else if val.is::<i32>() {
        serde_json::Value::Number(serde_json::Number::from(val.clone().cast::<i32>() as i64))
    } else if val.is::<f64>() {
        let f = val.as_float().unwrap_or(0.0);
        serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
    } else if val.is::<f32>() {
        let f = val.clone().cast::<f32>() as f64;
        serde_json::Number::from_f64(f)
            .map(serde_json::Value::Number)
            .unwrap_or(serde_json::Value::Null)
    } else if val.is::<bool>() {
        serde_json::Value::Bool(val.as_bool().unwrap_or(false))
    } else if val.is_map() {
        rhai_map_to_json_value(&val.clone().cast::<rhai::Map>())
    } else if val.is_array() {
        let arr = val.clone().cast::<rhai::Array>();
        serde_json::Value::Array(arr.iter().map(rhai_dynamic_to_json).collect())
    } else if val.is::<()>() {
        serde_json::Value::Null
    } else {
        // Fallback: serialize the string representation.
        serde_json::Value::String(val.to_string())
    }
}

/// Parse the return value of a custom tool's `execute()` function.
/// Accepts a string (content, no error) or a map with `content` and `is_error`.
fn parse_tool_exec_result(val: &Dynamic) -> Result<ToolExecResult, String> {
    if val.is_string() {
        Ok(ToolExecResult {
            content: val.to_string(),
            is_error: false,
        })
    } else if val.is_map() {
        let map = val.clone().cast::<rhai::Map>();
        let content = map
            .get("content")
            .and_then(|v| v.clone().try_cast::<String>())
            .unwrap_or_default();
        // Coerce is_error from bool, integer, or any truthy value.
        let is_error = match map.get("is_error") {
            Some(v) => {
                if v.is::<bool>() {
                    v.as_bool().unwrap_or(false)
                } else if v.is::<rhai::INT>() {
                    v.as_int().unwrap_or(0) != 0
                } else if v.is::<String>() {
                    v.to_string() == "true"
                } else {
                    false
                }
            }
            None => false,
        };
        Ok(ToolExecResult { content, is_error })
    } else {
        Ok(ToolExecResult {
            content: val.to_string(),
            is_error: false,
        })
    }
}
