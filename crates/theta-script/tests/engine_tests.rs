use std::path::PathBuf;
use theta_script::{BeforeHookResult, ScriptDef, ScriptEngine};

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
    assert_eq!(statuses[0].1, "building...");
    assert_eq!(statuses[1].0, "project:lint");
    assert_eq!(statuses[1].1, "linting...");
}
