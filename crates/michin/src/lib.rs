//! MichiN: extensible terminal coding agent harness.

/// Build the terminal window title: cat emoji + working directory name.
pub fn window_title(working_dir: &std::path::Path) -> String {
    let dir_name = working_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "?".to_string());
    format!("🐈‍⬛ - {dir_name}")
}

pub mod cli;
pub mod config;
pub mod interactive;
pub mod login;
pub mod mentions;
pub mod mimo_cluster;
pub mod oauth;
pub mod print_mode;
pub mod prompts;
pub mod rpc;
pub mod scripts;
pub mod session;
pub mod settings;
pub mod skills;
pub mod system_prompt;
pub mod tools;
