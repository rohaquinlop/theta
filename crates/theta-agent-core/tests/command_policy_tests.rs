use serde_json::json;
use theta_agent_core::command_policy::{
    AuthorizationClass, evaluate_tool_call, parse_command_segments, required_user_authorization,
};
use theta_agent_core::types::{SafetyDecisionKind, ToolCall};

fn bash_call(cmd: &str) -> ToolCall {
    ToolCall {
        id: "c1".to_string(),
        name: "bash".to_string(),
        arguments: json!({ "command": cmd }),
    }
}

#[test]
fn parse_segments_respects_quotes_and_chaining() {
    let segs = parse_command_segments("echo 'a;b' && rg foo src | wc -l; git status");
    assert_eq!(segs.len(), 4);
    assert_eq!(segs[0].argv[0], "echo");
    assert_eq!(segs[1].argv[0], "rg");
    assert_eq!(segs[2].argv[0], "wc");
    assert_eq!(segs[3].argv[0], "git");
}

#[test]
fn always_on_policy_allows_read_only_commands() {
    for command in [
        "cd /Users/rhafid/opensource-projects/theta && git status",
        "git diff crates/theta/src/interactive.rs 2>/dev/null",
        "cargo test",
        "cargo clippy -- -D warnings",
        "cargo fmt --check",
        "npm test",
        "make check",
    ] {
        let d = evaluate_tool_call(&bash_call(command), true);
        assert_eq!(
            d.decision,
            SafetyDecisionKind::Allowed,
            "{command} should be allowed"
        );
    }
}

#[test]
fn always_on_policy_allows_bash_commands_not_catastrophic() {
    for command in [
        "git commit -m test",
        "git push origin main",
        "sed -i 's/a/b/' f.txt",
        "cargo add serde",
        "npm install express",
        "echo hi > out.txt",
    ] {
        let d = evaluate_tool_call(&bash_call(command), true);
        assert_eq!(d.decision, SafetyDecisionKind::Allowed);
    }
}

#[test]
fn non_bash_tools_always_allowed() {
    for name in ["read", "ls", "find", "grep", "write", "edit", "mock"] {
        let tc = ToolCall {
            id: "w1".to_string(),
            name: name.to_string(),
            arguments: json!({"path":"a","content":"b"}),
        };
        let d = evaluate_tool_call(&tc, true);
        assert_eq!(d.decision, SafetyDecisionKind::Allowed);
    }
}

#[test]
fn strict_mode_rejects_catastrophic_commands() {
    for command in [
        "rm -rf /",
        "sudo rm -rf /",
        "env FOO=bar rm -rf ~",
        "mkfs /dev/disk9",
        "shutdown now",
    ] {
        let d = evaluate_tool_call(&bash_call(command), true);
        assert_eq!(
            d.decision,
            SafetyDecisionKind::Rejected,
            "{command} should be rejected"
        );
    }
}

#[test]
fn strict_mode_allows_non_catastrophic_recursive_delete() {
    let d = evaluate_tool_call(&bash_call("rm -rf /tmp/foo"), true);
    assert_eq!(d.decision, SafetyDecisionKind::Allowed);
}

#[test]
fn non_strict_allows_recursive_delete() {
    let d = evaluate_tool_call(&bash_call("rm -rf /tmp/foo"), false);
    assert_eq!(d.decision, SafetyDecisionKind::Allowed);
}

#[test]
fn required_authorization_classification_is_generic() {
    let commit = ToolCall {
        id: "1".to_string(),
        name: "bash".to_string(),
        arguments: json!({"command":"git commit -m test"}),
    };
    assert_eq!(
        required_user_authorization(&commit),
        Some(AuthorizationClass::Commit)
    );
    let dep = ToolCall {
        id: "2".to_string(),
        name: "bash".to_string(),
        arguments: json!({"command":"npm install"}),
    };
    assert_eq!(
        required_user_authorization(&dep),
        Some(AuthorizationClass::DependencyMutation)
    );
    let file = ToolCall {
        id: "3".to_string(),
        name: "write".to_string(),
        arguments: json!({"path":"a","content":"b"}),
    };
    assert_eq!(
        required_user_authorization(&file),
        Some(AuthorizationClass::FileMutation)
    );
    let inspect = ToolCall {
        id: "4".to_string(),
        name: "bash".to_string(),
        arguments: json!({"command":"git diff"}),
    };
    assert_eq!(required_user_authorization(&inspect), None);
}
