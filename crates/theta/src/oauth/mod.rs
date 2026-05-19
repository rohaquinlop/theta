//! OAuth authentication for subscription-based providers.
//!
//! Handles OAuth 2.0 Authorization Code + PKCE flows with local
//! HTTP callback server.

pub mod codex;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// OpenAI Codex OAuth authorize endpoint.
const CODEX_AUTHORIZE_URL: &str = "https://auth.openai.com/oauth/authorize";

/// OpenAI Codex OAuth token endpoint.
const CODEX_TOKEN_URL: &str = "https://auth.openai.com/oauth/token";

/// OpenAI Codex OAuth client ID (ChatGPT desktop app).
const CODEX_CLIENT_ID: &str = "app_EMoamEEZ73f0CkXaXp7hrann";

/// OAuth callback URI (local server).
/// Must match the registered redirect URI for the Codex OAuth client.
const CODEX_REDIRECT_URI: &str = "http://localhost:1455/auth/callback";

/// OAuth scopes requested.
const CODEX_SCOPE: &str = "openid profile email offline_access";

/// JWT claim path for extracting the ChatGPT account ID.
const JWT_CLAIM_PATH: &str = "https://api.openai.com/auth";

/// Localhost port for the OAuth callback server.
const CALLBACK_PORT: u16 = 1455;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Errors that can occur during OAuth flow.
#[derive(Debug, thiserror::Error)]
pub enum OAuthError {
    #[error("OAuth callback error: {0}")]
    Callback(String),

    #[error("OAuth token exchange error: {0}")]
    TokenExchange(String),

    #[error("OAuth timed out waiting for browser callback")]
    Timeout,

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

/// HTML pages returned by the local callback server.
#[derive(Debug, Clone)]
pub enum OAuthHtml {
    Success {
        message: String,
    },
    Error {
        message: String,
        details: Option<String>,
    },
}

impl OAuthHtml {
    pub fn success(message: impl Into<String>) -> Self {
        Self::Success {
            message: message.into(),
        }
    }

    pub fn error(message: impl Into<String>) -> Self {
        Self::Error {
            message: message.into(),
            details: None,
        }
    }
}
