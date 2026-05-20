//! Theta entry point: parse CLI, load config, dispatch.

use std::path::{Path, PathBuf};

use clap::Parser;

use theta::cli::{Cli, Command};
use theta::config::{ThetaConfig, load_config};
use theta::interactive::run_tui;
use theta::login::login_provider;
use theta::print_mode::{run_continue_print_mode, run_prompt_print_mode, run_resume_print_mode};
use theta::session::SessionManager;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging — write to ~/.theta/theta.log so it doesn't
    // corrupt the TUI display. Errors in TUI mode are surfaced via
    // TuiEvent::Error messages.
    let log_dir = dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".theta");
    let _ = std::fs::create_dir_all(&log_dir);
    let log_file = std::fs::File::create(log_dir.join("theta.log"))
        .unwrap_or_else(|_| tempfile::tempfile().unwrap());
    tracing_subscriber::fmt()
        .with_writer(std::sync::Mutex::new(log_file))
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    let working_dir = cli
        .working_dir
        .clone()
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let config = load_config(cli.config.as_deref()).await?;

    match &cli.command {
        Some(Command::Prompt(args)) => {
            handle_prompt(&config, &working_dir, &cli, args).await?;
        }
        Some(Command::Continue(args)) => {
            handle_continue(&config, &working_dir, &cli, args).await?;
        }
        Some(Command::Resume(args)) => {
            handle_resume(&config, &working_dir, &cli, args).await?;
        }
        Some(Command::Fork(args)) => {
            handle_fork(&config, &working_dir, &cli, args).await?;
        }
        Some(Command::Sessions) => {
            handle_list_sessions(&working_dir).await?;
        }
        Some(Command::Login(args)) => {
            handle_login(&config, &working_dir, args).await?;
        }
        Some(Command::Rpc) => {
            theta::rpc::run_rpc(&config, &working_dir).await?;
        }
        Some(Command::Tui(args)) => {
            handle_tui(&config, &working_dir, &cli, args).await?;
        }
        None => {
            // Default: interactive TUI mode.
            let tui_args = theta::cli::TuiArgs { text: vec![] };
            handle_tui(&config, &working_dir, &cli, &tui_args).await?;
        }
    }

    Ok(())
}

async fn handle_prompt(
    config: &ThetaConfig,
    working_dir: &Path,
    cli: &Cli,
    args: &theta::cli::PromptArgs,
) -> anyhow::Result<()> {
    let text = args.text.join(" ");
    let model = cli.model.as_deref().or(config.model.default.as_deref());

    let session_mgr = SessionManager::new(working_dir);
    let session = if args.continue_latest && !args.new {
        match session_mgr.resume().await {
            Ok(s) => s,
            Err(_) => session_mgr.create(model).await?,
        }
    } else {
        session_mgr.create(model).await?
    };

    let sid = session
        .meta
        .as_ref()
        .map(|m| m.id.clone())
        .unwrap_or_default();
    run_prompt_print_mode(config, working_dir, model.unwrap_or("gpt-5.5"), &text, &sid).await?;

    Ok(())
}

async fn handle_continue(
    config: &ThetaConfig,
    working_dir: &Path,
    cli: &Cli,
    args: &theta::cli::ContinueArgs,
) -> anyhow::Result<()> {
    let model = cli
        .model
        .as_deref()
        .or(config.model.default.as_deref())
        .unwrap_or("gpt-5.5");
    let follow_up = if args.text.is_empty() {
        None
    } else {
        Some(args.text.join(" "))
    };
    let follow_up_ref: Option<&str> = follow_up.as_deref();
    run_continue_print_mode(config, working_dir, model, follow_up_ref).await?;
    Ok(())
}

async fn handle_resume(
    config: &ThetaConfig,
    working_dir: &Path,
    _cli: &Cli,
    args: &theta::cli::ResumeArgs,
) -> anyhow::Result<()> {
    let follow_up = if args.text.is_empty() {
        None
    } else {
        Some(args.text.join(" "))
    };
    let follow_up_ref: Option<&str> = follow_up.as_deref();
    run_resume_print_mode(config, working_dir, &args.id, follow_up_ref).await?;
    Ok(())
}

async fn handle_fork(
    _config: &ThetaConfig,
    working_dir: &Path,
    _cli: &Cli,
    args: &theta::cli::ForkArgs,
) -> anyhow::Result<()> {
    let session_mgr = SessionManager::new(working_dir);
    let source = session_mgr.open_by_id(&args.id).await?;
    let forked = session_mgr.fork(&source, None).await?;
    println!(
        "Forked session {} -> {}",
        args.id,
        forked.meta.as_ref().map(|m| m.id.as_str()).unwrap_or("?")
    );
    Ok(())
}

async fn handle_list_sessions(working_dir: &Path) -> anyhow::Result<()> {
    let session_mgr = SessionManager::new(working_dir);
    let sessions = session_mgr.list().await?;
    if sessions.is_empty() {
        println!("No sessions found.");
    } else {
        for meta in &sessions {
            println!(
                "  {id}  {model}  {branch}  {count} msgs  ~{tokens} tok  {time}  {title}",
                id = meta.id,
                model = meta.model.as_deref().unwrap_or("?"),
                branch = meta.branch.as_deref().unwrap_or("-"),
                count = meta.message_count,
                tokens = meta.token_count,
                time = humantime_ms(meta.last_active_at),
                title = meta.title.as_deref().unwrap_or("")
            );
        }
    }
    Ok(())
}

async fn handle_login(
    _config: &ThetaConfig,
    _working_dir: &Path,
    args: &theta::cli::LoginArgs,
) -> anyhow::Result<()> {
    login_provider(args.provider.as_deref()).await?;
    Ok(())
}

async fn handle_tui(
    config: &ThetaConfig,
    working_dir: &Path,
    cli: &Cli,
    args: &theta::cli::TuiArgs,
) -> anyhow::Result<()> {
    // Load persisted settings (last model + thinking from prior sessions).
    let settings = theta::settings::load_settings().await;

    // Priority: CLI arg > config file > persisted settings > built-in default.
    let model = cli
        .model
        .as_deref()
        .or(config.model.default.as_deref())
        .or(settings.last_model.as_deref())
        .unwrap_or("gpt-5.5");
    let thinking = cli
        .thinking
        .as_deref()
        .or(config.thinking.default.as_deref())
        .or(settings.last_thinking.as_deref())
        .unwrap_or("medium");

    let prompt = if args.text.is_empty() {
        None
    } else {
        Some(args.text.join(" "))
    };
    let prompt = prompt.as_deref();
    run_tui(config, working_dir, model, thinking, prompt).await?;
    Ok(())
}

/// Format a millisecond timestamp as a human-readable string.
fn humantime_ms(ts: u64) -> String {
    let secs = ts / 1000;
    let mins = secs / 60;
    let hours = mins / 60;
    let days = hours / 24;

    if days > 0 {
        format!("{days}d ago")
    } else if hours > 0 {
        format!("{hours}h ago")
    } else if mins > 0 {
        format!("{mins}m ago")
    } else {
        format!("{secs}s ago")
    }
}
