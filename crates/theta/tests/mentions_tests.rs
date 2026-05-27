use tempfile::TempDir;
use theta::mentions::expand_file_mentions;
use theta::mentions::extract_mentions;
use theta_ai::ContentBlock;

#[test]
fn extracts_plain_and_quoted_mentions() {
    assert_eq!(
        extract_mentions(r#"read @src/main.rs and @"docs/user guide.md""#),
        vec!["docs/user guide.md".to_string(), "src/main.rs".to_string()]
    );
}

#[test]
fn ignores_email_addresses() {
    assert!(extract_mentions("a@b.com").is_empty());
}

#[tokio::test]
async fn expands_existing_file_mentions() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("src")).unwrap();
    std::fs::write(tmp.path().join("src/main.rs"), "fn main() {}\n").unwrap();

    let blocks = expand_file_mentions(tmp.path(), "read @src/main.rs").await;
    let ContentBlock::Text { text } = &blocks[0] else {
        panic!("expected text block");
    };
    assert!(text.contains("# Referenced files"));
    assert!(text.contains("## @src/main.rs"));
    assert!(text.contains("fn main() {}"));
}
