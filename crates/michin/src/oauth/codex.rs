//! OpenAI Codex (ChatGPT OAuth) flow.
//!
//! Implements OAuth 2.0 Authorization Code + PKCE flow against
//! OpenAI's auth server. Uses a local HTTP server on port 1455
//! to receive the callback. No manual cookie extraction needed.

use std::net::SocketAddr;
use std::time::{SystemTime, UNIX_EPOCH};

use base64::Engine;
use rand::RngCore;
use sha2::{Digest, Sha256};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

use crate::oauth::{
    CALLBACK_PORT, CODEX_AUTHORIZE_URL, CODEX_CLIENT_ID, CODEX_REDIRECT_URI, CODEX_SCOPE,
    CODEX_TOKEN_URL, JWT_CLAIM_PATH, OAuthError, OAuthHtml,
};

/// Result of a successful login.
#[derive(Debug, Clone)]
pub struct CodexCredentials {
    pub access_token: String,
    pub refresh_token: String,
    pub expires_at: u64,
    pub account_id: String,
}

/// Result of the token exchange step.
enum TokenExchangeResult {
    Success {
        access_token: String,
        refresh_token: String,
        expires_in: u64,
    },
    Failure {
        status: u16,
        message: String,
    },
}

/// Start the full Codex OAuth login flow.
///
/// 1. Generate PKCE challenge and random state
/// 2. Start local HTTP server on port {CALLBACK_PORT}
/// 3. Open browser to authorize URL
/// 4. Wait for callback with authorization code
/// 5. Exchange code for access + refresh tokens
/// 6. Extract account_id from JWT access token
pub async fn login_codex() -> Result<CodexCredentials, OAuthError> {
    let (verifier, challenge) = generate_pkce();
    let state = generate_state();

    let auth_url = build_authorize_url(&challenge, &state);

    // Start HTTP server before opening browser.
    // Try IPv6 first (macOS default for "localhost"), fall back to IPv4.
    let listener = match TcpListener::bind(SocketAddr::from((
        std::net::Ipv6Addr::LOCALHOST,
        CALLBACK_PORT,
    )))
    .await
    {
        Ok(l) => {
            tracing::info!("OAuth callback server listening on [::1]:{CALLBACK_PORT}");
            l
        }
        Err(_) => {
            let l = TcpListener::bind(SocketAddr::from((
                std::net::Ipv4Addr::LOCALHOST,
                CALLBACK_PORT,
            )))
            .await
            .map_err(|e| {
                tracing::error!(
                    "Failed to bind OAuth callback server on port {CALLBACK_PORT}: {e}"
                );
                OAuthError::Io(e)
            })?;
            tracing::info!("OAuth callback server listening on 127.0.0.1:{CALLBACK_PORT}");
            l
        }
    };

    // Open browser.
    tracing::info!("Opening browser for Codex OAuth: {auth_url}");
    let _ = open::that(&auth_url);

    // Wait for callback (30s timeout).
    let code = tokio::time::timeout(
        std::time::Duration::from_secs(120),
        wait_for_callback(listener, &state),
    )
    .await
    .map_err(|_| OAuthError::Timeout)?
    .map_err(|e| OAuthError::Callback(e.to_string()))?;

    // Exchange code for tokens.
    let token_result = exchange_code(&code, &verifier).await?;
    let (access_token, refresh_token, expires_in) = match token_result {
        TokenExchangeResult::Success {
            access_token,
            refresh_token,
            expires_in,
        } => (access_token, refresh_token, expires_in),
        TokenExchangeResult::Failure { status, message } => {
            return Err(OAuthError::TokenExchange(format!(
                "token exchange failed (HTTP {status}): {message}"
            )));
        }
    };

    // Extract account_id from JWT.
    let account_id = extract_account_id(&access_token)?;
    let expires_at = now_ms() + expires_in * 1000;

    Ok(CodexCredentials {
        access_token,
        refresh_token,
        expires_at,
        account_id,
    })
}

/// Refresh an expired access token using the refresh token.
pub async fn refresh_codex_token(refresh_token: &str) -> Result<CodexCredentials, OAuthError> {
    let client = reqwest::Client::new();
    let params = [
        ("grant_type", "refresh_token"),
        ("refresh_token", refresh_token),
        ("client_id", CODEX_CLIENT_ID),
    ];

    let resp = client
        .post(CODEX_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&params)
        .send()
        .await
        .map_err(|e| OAuthError::TokenExchange(e.to_string()))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(OAuthError::TokenExchange(format!(
            "refresh failed ({}): {text}",
            status.as_u16()
        )));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| OAuthError::TokenExchange(e.to_string()))?;

    let access_token = json["access_token"].as_str().ok_or_else(|| {
        OAuthError::TokenExchange("missing access_token in refresh response".into())
    })?;
    let refresh_token = json["refresh_token"].as_str().ok_or_else(|| {
        OAuthError::TokenExchange("missing refresh_token in refresh response".into())
    })?;
    let expires_in = json["expires_in"].as_u64().ok_or_else(|| {
        OAuthError::TokenExchange("missing expires_in in refresh response".into())
    })?;

    let account_id = extract_account_id(access_token)?;
    let expires_at = now_ms() + expires_in * 1000;

    Ok(CodexCredentials {
        access_token: access_token.to_string(),
        refresh_token: refresh_token.to_string(),
        expires_at,
        account_id,
    })
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Generate a PKCE code_verifier and SHA-256 code_challenge.
fn generate_pkce() -> (String, String) {
    let mut verifier_bytes = [0u8; 32];
    rand::rngs::OsRng.fill_bytes(&mut verifier_bytes);
    let verifier = base64url_encode(&verifier_bytes);

    let mut hasher = Sha256::new();
    hasher.update(verifier.as_bytes());
    let challenge_bytes = hasher.finalize();
    let challenge = base64url_encode(&challenge_bytes);

    (verifier, challenge)
}

/// Generate a random hex state for CSRF protection.
fn generate_state() -> String {
    let mut bytes = [0u8; 16];
    rand::rngs::OsRng.fill_bytes(&mut bytes);
    hex::encode(&bytes)
}

/// Build the OAuth authorize URL with all parameters.
fn build_authorize_url(challenge: &str, state: &str) -> String {
    format!(
        "{CODEX_AUTHORIZE_URL}?\
         response_type=code&\
         client_id={CODEX_CLIENT_ID}&\
         redirect_uri={CODEX_REDIRECT_URI}&\
         scope={CODEX_SCOPE}&\
         code_challenge={challenge}&\
         code_challenge_method=S256&\
         state={state}&\
         id_token_add_organizations=true&\
         codex_cli_simplified_flow=true&\
         originator=theta"
    )
}

/// Start an HTTP server, accept one connection, parse the callback.
async fn wait_for_callback(
    listener: TcpListener,
    expected_state: &str,
) -> Result<String, OAuthError> {
    tracing::info!("Waiting for OAuth callback on port {CALLBACK_PORT}...");
    let (mut stream, _addr) = listener.accept().await.map_err(|e| {
        tracing::error!("Failed to accept OAuth callback: {e}");
        OAuthError::Callback(format!("failed to accept connection: {e}"))
    })?;
    tracing::info!("OAuth callback connection received");

    let (reader, mut writer) = stream.split();
    let mut buf_reader = BufReader::new(reader);
    let mut request_line = String::new();
    buf_reader
        .read_line(&mut request_line)
        .await
        .map_err(|e| OAuthError::Callback(format!("failed to read request: {e}")))?;

    // Parse GET /auth/callback?code=...&state=...
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        send_response(&mut writer, &OAuthHtml::error("Invalid request")).await;
        return Err(OAuthError::Callback("invalid HTTP request".into()));
    }

    let path_and_query = parts[1];
    let (path, query) = match path_and_query.split_once('?') {
        Some((p, q)) => (p, Some(q)),
        None => (path_and_query, None),
    };

    if path != "/auth/callback" {
        send_response(&mut writer, &OAuthHtml::error("Callback route not found.")).await;
        return Err(OAuthError::Callback("wrong callback path".into()));
    }

    let query = query.unwrap_or("");
    let params: std::collections::HashMap<String, String> =
        url::form_urlencoded::parse(query.as_bytes())
            .map(|(k, v)| (k.into_owned(), v.into_owned()))
            .collect();

    // Validate state.
    if params.get("state").map(String::as_str) != Some(expected_state) {
        send_response(&mut writer, &OAuthHtml::error("State mismatch.")).await;
        return Err(OAuthError::Callback("state mismatch".into()));
    }

    // Extract code.
    let code = params
        .get("code")
        .cloned()
        .ok_or_else(|| OAuthError::Callback("missing authorization code".into()))?;

    send_response(
        &mut writer,
        &OAuthHtml::success("Authentication completed. You can close this window."),
    )
    .await;

    Ok(code)
}

/// Send an HTTP response with the OAuth success/error page.
async fn send_response(writer: &mut (impl AsyncWriteExt + Unpin), html: &OAuthHtml) {
    let (status, title, heading, message, details, icon, icon_class) = match html {
        OAuthHtml::Success { message } => (
            "200 OK",
            "Authentication successful",
            "Authentication successful",
            message.as_str(),
            None,
            "\u{2713}",
            "success",
        ),
        OAuthHtml::Error { message, details } => (
            "400 Bad Request",
            "Authentication failed",
            "Authentication failed",
            message.as_str(),
            details.as_deref(),
            "\u{2717}",
            "error",
        ),
    };

    let body = build_html_page(title, heading, message, details, icon, icon_class);
    let response = format!(
        "HTTP/1.1 {status}\r\n\
         Content-Type: text/html; charset=utf-8\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         \r\n\
         {body}",
        body.len()
    );
    let _ = writer.write_all(response.as_bytes()).await;
}

/// Build a minimal styled HTML page for the OAuth callback.
fn build_html_page(
    title: &str,
    heading: &str,
    message: &str,
    details: Option<&str>,
    icon: &str,
    icon_class: &str,
) -> String {
    let details_html = details
        .map(|d| format!("<div class=\"details\">{d}</div>"))
        .unwrap_or_default();

    HTML_PAGE_TEMPLATE
        .replace("{{TITLE}}", title)
        .replace("{{HEADING}}", heading)
        .replace("{{MESSAGE}}", message)
        .replace("{{DETAILS}}", &details_html)
        .replace("{{ICON}}", icon)
        .replace("{{ICON_CLASS}}", icon_class)
}

const HTML_PAGE_TEMPLATE: &str = r##"<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{{TITLE}}</title>
  <style>
    :root { --text: #fafafa; --text-dim: #a1a1aa; --page-bg: #09090b; --accent: #22c55e; }
    * { box-sizing: border-box; }
    html { color-scheme: dark; }
    body {
      margin: 0; min-height: 100vh; display: flex; align-items: center; justify-content: center;
      padding: 24px; background: var(--page-bg); color: var(--text);
      font-family: ui-sans-serif, system-ui, sans-serif; text-align: center;
    }
    main { max-width: 560px; display: flex; flex-direction: column; align-items: center; }
    .icon { width: 72px; height: 72px; display: flex; align-items: center; justify-content: center;
            border-radius: 50%; margin-bottom: 24px; font-size: 36px; }
    .success { background: rgba(34,197,94,0.15); color: var(--accent); }
    .error   { background: rgba(239,68,68,0.15);  color: #ef4444; }
    h1 { margin: 0 0 10px; font-size: 28px; font-weight: 650; }
    p { margin: 0; line-height: 1.7; color: var(--text-dim); font-size: 15px; }
    .details { margin-top: 16px; font-size: 13px; color: var(--text-dim); white-space: pre-wrap; word-break: break-word; }
  </style>
</head>
<body>
  <main>
    <div class="icon {{ICON_CLASS}}">{{ICON}}</div>
    <h1>{{HEADING}}</h1>
    <p>{{MESSAGE}}</p>
    {{DETAILS}}
  </main>
</body>
</html>"##;

/// Exchange authorization code for access + refresh tokens.
async fn exchange_code(code: &str, verifier: &str) -> Result<TokenExchangeResult, OAuthError> {
    let client = reqwest::Client::new();
    let params = [
        ("grant_type", "authorization_code"),
        ("client_id", CODEX_CLIENT_ID),
        ("code", code),
        ("code_verifier", verifier),
        ("redirect_uri", CODEX_REDIRECT_URI),
    ];

    let resp = client
        .post(CODEX_TOKEN_URL)
        .header("Content-Type", "application/x-www-form-urlencoded")
        .form(&params)
        .send()
        .await
        .map_err(|e| OAuthError::TokenExchange(e.to_string()))?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Ok(TokenExchangeResult::Failure {
            status: status.as_u16(),
            message: format!("token exchange failed ({status}): {text}"),
        });
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| OAuthError::TokenExchange(e.to_string()))?;

    let access_token = json["access_token"]
        .as_str()
        .ok_or_else(|| OAuthError::TokenExchange("missing access_token".into()))?;
    let refresh_token = json["refresh_token"]
        .as_str()
        .ok_or_else(|| OAuthError::TokenExchange("missing refresh_token".into()))?;
    let expires_in = json["expires_in"]
        .as_u64()
        .ok_or_else(|| OAuthError::TokenExchange("missing expires_in".into()))?;

    Ok(TokenExchangeResult::Success {
        access_token: access_token.to_string(),
        refresh_token: refresh_token.to_string(),
        expires_in,
    })
}

/// Extract the ChatGPT account_id from the JWT access token.
fn extract_account_id(access_token: &str) -> Result<String, OAuthError> {
    let parts: Vec<&str> = access_token.split('.').collect();
    if parts.len() != 3 {
        return Err(OAuthError::TokenExchange("invalid JWT format".into()));
    }

    let payload = parts[1];
    let decoded = base64::engine::general_purpose::URL_SAFE_NO_PAD
        .decode(payload)
        .map_err(|e| OAuthError::TokenExchange(format!("JWT decode: {e}")))?;
    let json: serde_json::Value = serde_json::from_slice(&decoded)
        .map_err(|e| OAuthError::TokenExchange(format!("JWT parse: {e}")))?;

    let account_id = json[JWT_CLAIM_PATH]["chatgpt_account_id"]
        .as_str()
        .ok_or_else(|| OAuthError::TokenExchange("missing chatgpt_account_id in JWT".into()))?;

    Ok(account_id.to_string())
}

/// Base64url-encode bytes (no padding).
fn base64url_encode(bytes: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes)
}

/// Current time in milliseconds.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

// Need hex for state generation.
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }
}
