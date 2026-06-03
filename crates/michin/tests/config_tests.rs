use michin::config::AuthConfig;
use michin::config::to_agent_config;
use michin::config::{MichiNConfig, ProfileOverrides, RuntimeProfileSetting};

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

#[tokio::test]
async fn test_auth_token_storage() {
    let mut auth = AuthConfig::default();
    auth.set_token("openai", "sk-test-key", None);
    assert_eq!(auth.get_token("openai"), Some("sk-test-key".into()));

    auth.set_token("openai-codex", "codex-token", Some(now_ms() + 3600_000));
    assert!(auth.get_token("openai-codex").is_some());
}

#[tokio::test]
async fn test_auth_env_fallback() {
    let auth = AuthConfig::default();
    assert_eq!(auth.get_token("nonexistent"), None);
}

#[tokio::test]
async fn test_save_auth_preserves_unrelated_provider_credentials() {
    let dir = tempfile::tempdir().expect("tempdir should be created");
    let path = dir.path().join("auth.json");

    let mut existing = AuthConfig::default();
    existing.set_oauth_token(
        "openai-codex",
        "codex-access",
        "codex-refresh",
        now_ms() + 3600_000,
    );
    michin::config::save_auth(&existing, Some(&path))
        .await
        .expect("initial auth should save");

    let mut incoming = AuthConfig::default();
    incoming.set_token("deepseek", "deepseek-key", None);
    michin::config::save_auth(&incoming, Some(&path))
        .await
        .expect("merged auth should save");

    let saved = michin::config::load_auth(Some(&path))
        .await
        .expect("merged auth should load");
    assert_eq!(saved.get_token("deepseek"), Some("deepseek-key".into()));
    assert_eq!(saved.get_token("openai-codex"), Some("codex-access".into()));
}

#[tokio::test]
async fn test_save_auth_replaces_same_provider_across_auth_kinds() {
    let dir = tempfile::tempdir().expect("tempdir should be created");
    let path = dir.path().join("auth.json");

    let mut existing = AuthConfig::default();
    existing.set_oauth_token(
        "openai-codex",
        "old-access",
        "old-refresh",
        now_ms() + 3600_000,
    );
    michin::config::save_auth(&existing, Some(&path))
        .await
        .expect("initial auth should save");

    let mut incoming = AuthConfig::default();
    incoming.set_token("openai-codex", "manual-token", None);
    michin::config::save_auth(&incoming, Some(&path))
        .await
        .expect("replacement auth should save");

    let saved = michin::config::load_auth(Some(&path))
        .await
        .expect("replacement auth should load");
    assert_eq!(saved.get_token("openai-codex"), Some("manual-token".into()));
    assert!(saved.oauth_tokens.is_empty());
}

#[test]
fn test_profile_dev_defaults_are_applied() {
    let cfg = MichiNConfig {
        profile: RuntimeProfileSetting::Dev,
        ..Default::default()
    };
    let ac = to_agent_config(&cfg);
    assert_eq!(ac.retry.max_retries, 1);
    assert_eq!(ac.retry.base_delay_ms, 250);
    assert!(!ac.command_policy_strict);
}

#[test]
fn test_profile_prod_defaults_are_applied() {
    let cfg = MichiNConfig {
        profile: RuntimeProfileSetting::Prod,
        ..Default::default()
    };
    let ac = to_agent_config(&cfg);
    assert_eq!(ac.retry.max_retries, 4);
    assert!(ac.command_policy_strict);
    assert_eq!(ac.provider_circuit_breaker.failure_threshold, 2);
}

#[test]
fn test_profile_overrides_take_precedence() {
    let cfg = MichiNConfig {
        profile: RuntimeProfileSetting::Prod,
        profile_overrides: ProfileOverrides {
            max_retries: Some(7),
            command_policy_strict: Some(false),
            ..Default::default()
        },
        ..Default::default()
    };
    let ac = to_agent_config(&cfg);
    assert_eq!(ac.retry.max_retries, 7);
    assert!(!ac.command_policy_strict);
}
