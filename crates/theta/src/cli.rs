//! CLI argument parsing with clap.

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

/// Theta: minimal terminal coding agent harness in Rust.
///
/// Running `theta` without a subcommand starts interactive TUI multi-turn mode.
#[derive(Debug, Parser)]
#[command(
    name = "theta",
    version,
    about = "Minimal terminal coding agent harness",
    after_help = "Running `theta` without a subcommand starts interactive TUI mode."
)]
pub struct Cli {
    /// Subcommand (defaults to interactive TUI mode if omitted).
    #[command(subcommand)]
    pub command: Option<Command>,

    /// Model to use (e.g. gpt-5.5, o4, deepseek-v4-pro).
    #[arg(short, long, global = true)]
    pub model: Option<String>,

    /// Thinking level (off, minimal, low, medium, high, xhigh).
    #[arg(short = 't', long, global = true)]
    pub thinking: Option<String>,

    /// Working directory (defaults to current directory).
    #[arg(short = 'C', long, global = true)]
    pub working_dir: Option<PathBuf>,

    /// Config file path (defaults to ~/.theta/config.toml).
    #[arg(long, global = true)]
    pub config: Option<PathBuf>,

    /// Run in print (non-interactive) mode. Default for `prompt`.
    #[arg(long, global = true)]
    pub print: bool,

    /// Run in interactive TUI mode. Default for no subcommand.
    #[arg(long, global = true)]
    pub tui: bool,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Start a new session with a prompt (print mode by default).
    Prompt(PromptArgs),

    /// Continue the last active session.
    Continue(ContinueArgs),

    /// Resume a specific session by ID.
    Resume(ResumeArgs),

    /// Fork an existing session.
    Fork(ForkArgs),

    /// List all sessions.
    Sessions,

    /// Login to a provider (opens browser for OAuth).
    Login(LoginArgs),

    /// JSON-RPC over stdin/stdout.
    Rpc,

    /// Start interactive TUI mode with an optional initial prompt.
    #[command(name = "tui")]
    Tui(TuiArgs),
}

#[derive(Debug, Args)]
pub struct PromptArgs {
    /// The initial prompt text.
    pub text: Vec<String>,

    /// Create a new session even if one exists.
    #[arg(long)]
    pub new: bool,

    /// Continue the latest session instead of starting a new one.
    #[arg(long = "continue")]
    pub continue_latest: bool,
}

#[derive(Debug, Args)]
pub struct ContinueArgs {
    /// Optional follow-up text.
    pub text: Vec<String>,
}

#[derive(Debug, Args)]
pub struct ResumeArgs {
    /// Session ID to resume.
    pub id: String,

    /// Optional initial text.
    pub text: Vec<String>,
}

#[derive(Debug, Args)]
pub struct ForkArgs {
    /// Session ID to fork from.
    pub id: String,

    /// Optional initial text for the forked session.
    pub text: Vec<String>,
}

#[derive(Debug, Args)]
pub struct LoginArgs {
    /// Provider to login to (e.g. openai-codex).
    /// If omitted, shows an interactive provider list.
    pub provider: Option<String>,
}

#[derive(Debug, Args)]
pub struct TuiArgs {
    /// Optional initial prompt text.
    pub text: Vec<String>,
}
