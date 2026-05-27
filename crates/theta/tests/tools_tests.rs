use std::path::{Path, PathBuf};
use theta::tools::{ToolContext, resolve_path, shorten_path};

#[test]
fn resolve_path_keeps_absolute_path() {
    let ctx = ToolContext::new(PathBuf::from("/tmp/theta-workdir"));
    let resolved = resolve_path(&ctx, "/Users/rhafid/.theta");
    assert_eq!(resolved, PathBuf::from("/Users/rhafid/.theta"));
}

#[test]
fn resolve_path_resolves_relative_from_workdir() {
    let ctx = ToolContext::new(PathBuf::from("/tmp/theta-workdir"));
    let resolved = resolve_path(&ctx, "src/main.rs");
    assert_eq!(resolved, PathBuf::from("/tmp/theta-workdir/src/main.rs"));
}

#[test]
fn shorten_path_replaces_home_with_tilde() {
    let home = dirs::home_dir().unwrap();
    let path = home.join("projects/theta");
    let result = shorten_path(&path);
    assert!(result.starts_with("~/"));
    assert!(result.ends_with("projects/theta"));
}

#[test]
fn shorten_path_leaves_non_home_path_unchanged() {
    let result = shorten_path(Path::new("/tmp/theta-workdir"));
    assert_eq!(result, "/tmp/theta-workdir");
}
