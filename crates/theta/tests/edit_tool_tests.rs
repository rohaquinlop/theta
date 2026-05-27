use theta::tools::edit::make_diff_preview;

#[test]
fn diff_preview_uses_git_like_headers() {
    let before = "a\nb\nc\n";
    let after = "a\nB\nc\n";
    let diff = make_diff_preview("src/lib.rs", before, after);
    assert!(diff.contains("--- a/src/lib.rs"));
    assert!(diff.contains("+++ b/src/lib.rs"));
    assert!(diff.contains("@@"));
}
