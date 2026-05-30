use std::path::Path;
use theta::system_prompt::{
    EXTENSION_CREATION_GUARDRAILS, RESPONSE_CONTRACT, apply_system_prompt_overrides,
    build_tools_prompt, discover_nested_agents, is_ignorable_dir,
};

#[test]
fn tools_prompt_uses_function_calling_not_xml() {
    let s = build_tools_prompt(Path::new("."));
    assert!(s.contains("native function-calling"));
    assert!(!s.contains("XML invocation format"));
}

#[test]
fn response_contract_contains_execution_continuity() {
    assert!(RESPONSE_CONTRACT.contains("DONE"));
    assert!(RESPONSE_CONTRACT.contains("BLOCKED"));
    assert!(RESPONSE_CONTRACT.contains("FAILED"));
}

#[test]
fn response_contract_contains_analysis_guard() {
    assert!(RESPONSE_CONTRACT.contains("do not implement changes"));
    assert!(RESPONSE_CONTRACT.contains("Summarize findings"));
}

#[test]
fn response_contract_contains_tool_discipline() {
    assert!(RESPONSE_CONTRACT.contains("Read files before editing"));
    assert!(RESPONSE_CONTRACT.contains("Do not repeat identical tool calls"));
}

#[test]
fn response_contract_no_longer_has_skill_auto_loading_section() {
    assert!(
        !RESPONSE_CONTRACT.contains("## Skill Auto-Loading"),
        "Skill Auto-Loading section must not be in response contract"
    );
}

#[test]
fn response_contract_no_longer_has_theta_extensions_docs() {
    assert!(
        !RESPONSE_CONTRACT.contains("tool.before(name, callback)"),
        "Rhai extension docs must not be in response contract"
    );
    assert!(
        !RESPONSE_CONTRACT.contains("create an extension"),
        "Extension creation docs must not be in response contract"
    );
}

#[test]
fn response_contract_has_resource_section() {
    assert!(
        RESPONSE_CONTRACT.contains("## Resources"),
        "Response contract must reference resources"
    );
}

#[test]
fn extension_guardrails_scopes_to_theta() {
    assert!(EXTENSION_CREATION_GUARDRAILS.contains("their own project"));
    assert!(EXTENSION_CREATION_GUARDRAILS.contains("extend Theta's behavior"));
}

#[test]
fn extension_guardrails_rejects_general_language() {
    assert!(EXTENSION_CREATION_GUARDRAILS.contains("Do not create an extension"));
    assert!(EXTENSION_CREATION_GUARDRAILS.contains("working on their own project"));
}

#[test]
fn extension_guardrails_has_ambiguity_fallback() {
    assert!(EXTENSION_CREATION_GUARDRAILS.contains("extend theta"));
    assert!(EXTENSION_CREATION_GUARDRAILS.contains("skill, extension, or code change"));
}

#[test]
fn ignorable_dirs_reject_vcs_and_build() {
    assert!(is_ignorable_dir(".git"));
    assert!(is_ignorable_dir("target"));
    assert!(is_ignorable_dir("node_modules"));
    assert!(is_ignorable_dir(".theta"));
    assert!(is_ignorable_dir(".github"));
    assert!(is_ignorable_dir(".vscode"));
    assert!(is_ignorable_dir(".idea"));
    assert!(is_ignorable_dir(".DS_Store"));
}

#[test]
fn ignorable_dirs_reject_dot_prefix() {
    assert!(is_ignorable_dir(".cache"));
    assert!(is_ignorable_dir(".config"));
    assert!(is_ignorable_dir(".local"));
}

#[test]
fn ignorable_dirs_accept_normal_names() {
    assert!(!is_ignorable_dir("src"));
    assert!(!is_ignorable_dir("crates"));
    assert!(!is_ignorable_dir("docs"));
    assert!(!is_ignorable_dir("tests"));
}

#[tokio::test]
async fn discover_nested_agents_finds_crate_files() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let nested = discover_nested_agents(root).await;

    let found_crates: Vec<&str> = nested.iter().map(|(p, _)| p.as_str()).collect();

    assert!(
        found_crates.contains(&"theta-ai"),
        "missing theta-ai: {:?}",
        found_crates
    );
    assert!(
        found_crates.contains(&"theta-agent-core"),
        "missing theta-agent-core: {:?}",
        found_crates
    );
    assert!(
        found_crates.contains(&"theta-tui"),
        "missing theta-tui: {:?}",
        found_crates
    );
    assert!(
        found_crates.contains(&"theta-models"),
        "missing theta-models: {:?}",
        found_crates
    );
    assert!(
        found_crates.contains(&"theta-script"),
        "missing theta-script: {:?}",
        found_crates
    );

    for (_path, content) in &nested {
        assert!(!content.is_empty(), "empty AGENTS.md found");
    }

    assert!(
        !found_crates.contains(&""),
        "root AGENTS.md should not be in nested results"
    );
}

#[tokio::test]
async fn apply_overrides_no_files_returns_original() {
    let dir = tempfile::tempdir().unwrap();
    let original = "base prompt content".to_string();
    let result = apply_system_prompt_overrides(dir.path(), original.clone()).await;
    assert_eq!(result, original);
}

#[tokio::test]
async fn apply_overrides_system_prompt_replaces() {
    let dir = tempfile::tempdir().unwrap();
    tokio::fs::write(dir.path().join("SYSTEM_PROMPT.md"), "replacement prompt")
        .await
        .unwrap();
    let original = "base prompt content".to_string();
    let result = apply_system_prompt_overrides(dir.path(), original).await;
    assert_eq!(result, "replacement prompt");
}

#[tokio::test]
async fn apply_overrides_append_adds_to_original() {
    let dir = tempfile::tempdir().unwrap();
    tokio::fs::write(
        dir.path().join("APPEND_SYSTEM_PROMPT.md"),
        "extra instructions",
    )
    .await
    .unwrap();
    let original = "base prompt".to_string();
    let result = apply_system_prompt_overrides(dir.path(), original).await;
    assert!(result.contains("base prompt"));
    assert!(result.contains("extra instructions"));
    assert!(result.contains("\n\n"));
}

#[tokio::test]
async fn apply_overrides_both_files_system_prompt_wins() {
    let dir = tempfile::tempdir().unwrap();
    tokio::fs::write(dir.path().join("SYSTEM_PROMPT.md"), "system wins")
        .await
        .unwrap();
    tokio::fs::write(dir.path().join("APPEND_SYSTEM_PROMPT.md"), "append ignored")
        .await
        .unwrap();
    let original = "base".to_string();
    let result = apply_system_prompt_overrides(dir.path(), original).await;
    assert_eq!(result, "system wins");
    assert!(!result.contains("append ignored"));
}

#[tokio::test]
async fn apply_overrides_append_ignored_when_system_present() {
    let dir = tempfile::tempdir().unwrap();
    tokio::fs::write(dir.path().join("SYSTEM_PROMPT.md"), "only system here")
        .await
        .unwrap();
    tokio::fs::write(
        dir.path().join("APPEND_SYSTEM_PROMPT.md"),
        "should not appear",
    )
    .await
    .unwrap();
    let original = "base".to_string();
    let result = apply_system_prompt_overrides(dir.path(), original).await;
    assert_eq!(result, "only system here");
    assert!(!result.contains("should not appear"));
    assert!(!result.contains("base"));
}

#[tokio::test]
async fn apply_overrides_append_empty_file_no_op() {
    let dir = tempfile::tempdir().unwrap();
    tokio::fs::write(dir.path().join("APPEND_SYSTEM_PROMPT.md"), "")
        .await
        .unwrap();
    let original = "just the base".to_string();
    let result = apply_system_prompt_overrides(dir.path(), original).await;
    assert_eq!(result, "just the base");
}

#[tokio::test]
async fn apply_overrides_system_prompt_handles_missing_file_gracefully() {
    let dir = tempfile::tempdir().unwrap();
    let original = "stays intact".to_string();
    let result = apply_system_prompt_overrides(dir.path(), original).await;
    assert_eq!(result, "stays intact");
}
