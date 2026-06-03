//! Provider authentication: browser-based login with stdin token capture
//! for API keys, and OAuth 2.0 + PKCE flow for subscription providers.

use crate::config::{load_auth, save_auth};
use crate::oauth::codex;

/// All known login-able providers.
const PROVIDERS: &[(&str, &str, &str)] = &[
    ("openai", "OpenAI", "API Key"),
    (
        "openai-codex",
        "OpenAI Codex (ChatGPT Plus)",
        "Subscription (OAuth)",
    ),
    ("deepseek", "DeepSeek", "API Key"),
    ("opencode", "OpenCode", "API Key"),
    ("xiaomi", "Xiaomi MiMo", "API Key"),
];

/// Login to a provider.
///
/// For subscription providers ("openai-codex"), uses OAuth 2.0 + PKCE
/// flow with automatic browser callback. No manual token extraction needed.
///
/// For API key providers, opens the provider's key page and prompts for
/// a key via stdin.
///
/// If `provider` is None, shows an interactive provider list.
pub async fn login_provider(provider: Option<&str>) -> anyhow::Result<()> {
    let provider = match provider {
        Some(p) if !p.is_empty() => p.to_string(),
        _ => select_provider_interactive()?,
    };

    // Codex uses real OAuth — no stdin token needed.
    if provider == "openai-codex" {
        return login_codex_oauth().await;
    }

    // API key providers: open browser, prompt for key.
    let url = provider_token_url(&provider);
    println!("Opening browser to: {url}");
    println!("If the browser doesn't open, visit the URL manually.");
    let _ = open::that(url);

    println!();
    println!("Paste your API key for '{provider}' below:");
    let mut token = String::new();
    std::io::stdin().read_line(&mut token)?;
    let token = token.trim().to_string();

    if token.is_empty() {
        anyhow::bail!("no token provided");
    }

    let mut auth = load_auth(None).await?;
    auth.set_token(&provider, &token, None);
    save_auth(&auth, None).await?;

    println!("Token saved for '{provider}'.");
    Ok(())
}

/// Run the Codex OAuth login flow.
async fn login_codex_oauth() -> anyhow::Result<()> {
    println!("Starting ChatGPT Plus OAuth login...");
    println!("A browser window will open. Sign in to your ChatGPT account.");
    println!();

    let credentials = codex::login_codex()
        .await
        .map_err(|e| anyhow::anyhow!("Codex OAuth login failed: {e}"))?;

    // Store as an OAuth credential (not a plain token).
    let mut auth = load_auth(None).await?;
    auth.set_oauth_token(
        "openai-codex",
        &credentials.access_token,
        &credentials.refresh_token,
        credentials.expires_at,
    );
    save_auth(&auth, None).await?;

    println!();
    println!("Successfully logged in to ChatGPT Plus.");
    println!("Account ID: {}", credentials.account_id);
    Ok(())
}

/// Show an interactive provider list on the CLI.
fn select_provider_interactive() -> anyhow::Result<String> {
    println!("Select a provider to login to:");
    for (i, (_id, name, auth_type)) in PROVIDERS.iter().enumerate() {
        println!("  {}. {name} ({auth_type})", i + 1);
    }
    println!();
    print!("Enter number (1-{}): ", PROVIDERS.len());
    use std::io::Write;
    std::io::stdout().flush()?;

    let mut choice = String::new();
    std::io::stdin().read_line(&mut choice)?;
    let choice = choice.trim();

    let index: usize = choice
        .parse::<usize>()
        .ok()
        .filter(|&n| n >= 1 && n <= PROVIDERS.len())
        .map(|n| n - 1)
        .ok_or_else(|| anyhow::anyhow!("Invalid choice. Expected 1-{}.", PROVIDERS.len()))?;

    Ok(PROVIDERS[index].0.to_string())
}

/// Get the token/API key page URL for a provider.
fn provider_token_url(provider: &str) -> String {
    match provider {
        "openai" => "https://platform.openai.com/api-keys",
        "openai-codex" => "https://chatgpt.com",
        "deepseek" => "https://platform.deepseek.com/api_keys",
        "opencode" => "https://api.opencode.ai/settings",
        "xiaomi" => "https://platform.xiaomimimo.com/console/plan-manage",
        other => {
            eprintln!("Unknown provider '{other}'. Opening generic URL.");
            "https://google.com"
        }
    }
    .to_string()
}
