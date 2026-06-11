use michin::interactive::{expand_skill_message, format_tool_summary};
use michin::skills::Skill;
use michin_agent_core::types::ToolResult;
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn read_summary_is_compact() {
    let result = ToolResult {
        tool_call_id: "id".into(),
        tool_name: "read".into(),
        content: vec![],
        details: Some(serde_json::json!({
            "path": "/tmp/a.rs",
            "total_lines": 100,
            "offset": 11,
            "lines_read": 20
        })),
        is_error: false,
    };
    let s = format_tool_summary(&result, 200);
    assert!(s.contains("read: /tmp/a.rs"));
    assert!(s.contains("read: /tmp/a.rs:11-30"));
    assert!(!s.contains("fn "));
}

#[test]
fn edit_summary_is_compact() {
    let result = ToolResult {
        tool_call_id: "id".into(),
        tool_name: "edit".into(),
        content: vec![],
        details: Some(serde_json::json!({
            "path": "/tmp/a.rs",
            "changes": 3,
            "diff": "--- a/tmp/a.rs\n+++ b/tmp/a.rs\n@@ -1 +1 @@\n-old\n+new\n+added"
        })),
        is_error: false,
    };
    let s = format_tool_summary(&result, 200);
    assert!(s.contains("edit: /tmp/a.rs"));
    assert!(s.contains("[+2/-1]"));
    assert!(!s.contains("@@"));
}

#[test]
fn skill_command_without_args_executes_now() {
    let skill = Skill {
        name: "git-commit".into(),
        description: "Commit workflow".into(),
        location: PathBuf::from("/tmp/skills/git-commit/SKILL.md"),
        body: "Do commit workflow".into(),
        extra: HashMap::new(),
    };
    let s = expand_skill_message("/skill:git-commit", &[skill]);
    assert!(s.contains("<skill name=\"git-commit\""));
    assert!(s.contains("Execute this skill now"));
    assert!(s.contains("Do not only acknowledge loading the skill"));
}

#[test]
fn skill_command_with_args_preserves_args_only() {
    let skill = Skill {
        name: "git-commit".into(),
        description: "Commit workflow".into(),
        location: PathBuf::from("/tmp/skills/git-commit/SKILL.md"),
        body: "Do commit workflow".into(),
        extra: HashMap::new(),
    };
    let s = expand_skill_message("/skill:git-commit commit all staged", &[skill]);
    assert!(s.contains("commit all staged"));
    assert!(!s.contains("Do not only acknowledge loading the skill"));
}
