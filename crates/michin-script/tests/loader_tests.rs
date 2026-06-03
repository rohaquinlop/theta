use michin_script::ScriptLoader;

#[tokio::test]
async fn test_discover_project_local_only() {
    let dir = tempfile::tempdir().unwrap();
    let ext_dir = dir.path().join(".theta").join("extensions");
    tokio::fs::create_dir_all(&ext_dir).await.unwrap();
    tokio::fs::write(ext_dir.join("test.rhai"), "// test")
        .await
        .unwrap();

    let loader = ScriptLoader::discover(dir.path()).await;
    assert!(
        loader.len() >= 1,
        "expected at least 1 script, got {}",
        loader.len()
    );
}

#[tokio::test]
async fn test_discover_no_project_local_scripts() {
    let dir = tempfile::tempdir().unwrap();
    let loader = ScriptLoader::discover(dir.path()).await;
    let _ = loader.len();
}
