use theta::config::{AuthConfig, ProviderToken, ThetaConfig};
use theta::rpc::resolve_auth_for_model;
use theta_models::BuiltInCatalog;

#[tokio::test]
async fn resolve_auth_for_model_falls_back_to_authenticated_provider() {
    let catalog = BuiltInCatalog::new();
    let mut cfg = ThetaConfig::default();
    cfg.auth = AuthConfig {
        tokens: vec![ProviderToken {
            provider: "openai-codex".into(),
            token: "codex-token".into(),
            expires_at: None,
            obtained_at: 1,
        }],
        oauth_tokens: vec![],
    };

    let (model, key) = resolve_auth_for_model(&cfg, &catalog, "gpt-5.5")
        .await
        .expect("fallback should resolve");
    assert_eq!(model.provider, theta_ai::Provider::OpenAiCodex);
    assert_eq!(key, "codex-token");
}

#[tokio::test]
async fn resolve_auth_for_model_returns_explicit_error_when_no_auth() {
    let catalog = BuiltInCatalog::new();
    let cfg = ThetaConfig::default();
    let err = resolve_auth_for_model(&cfg, &catalog, "gpt-5.5")
        .await
        .expect_err("expected missing auth error");
    assert!(
        err.to_string().contains("no auth token for 'openai'"),
        "unexpected error: {err}"
    );
}
