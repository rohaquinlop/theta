//! Script loading and discovery.
//!
//! Auto-discovers `.rhai` scripts from:
//! - `~/.theta/extensions/*.rhai` (global)
//! - `./.theta/extensions/*.rhai` (project-local)

use std::path::{Path, PathBuf};

/// A discovered script definition.
#[derive(Debug, Clone)]
pub struct ScriptDef {
    /// Script name (filename without extension).
    pub name: String,
    /// Full path to the script file.
    pub location: PathBuf,
    /// Script source code.
    pub source: String,
}

/// Discover and load all `.rhai` scripts from standard locations.
pub struct ScriptLoader {
    /// Loaded script definitions.
    scripts: Vec<ScriptDef>,
}

impl ScriptLoader {
    /// Create a new loader and discover scripts.
    pub async fn discover(working_dir: &Path) -> Self {
        let mut scripts = Vec::new();

        // Global: ~/.theta/extensions/*.rhai
        if let Some(home) = dirs::home_dir() {
            let global_dir = home.join(".theta").join("extensions");
            Self::discover_in_dir(&global_dir, &mut scripts).await;
        }

        // Project-local: ./.theta/extensions/*.rhai
        let project_dir = working_dir.join(".theta").join("extensions");
        Self::discover_in_dir(&project_dir, &mut scripts).await;

        Self { scripts }
    }

    async fn discover_in_dir(dir: &Path, scripts: &mut Vec<ScriptDef>) {
        if !dir.exists() {
            return;
        }

        let Ok(mut entries) = tokio::fs::read_dir(dir).await else {
            return;
        };

        while let Ok(Some(entry)) = entries.next_entry().await {
            let path = entry.path();
            if path.extension().map(|e| e == "rhai").unwrap_or(false) {
                let name = path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string();

                match tokio::fs::read_to_string(&path).await {
                    Ok(source) => {
                        tracing::info!(
                            name = %name,
                            location = %path.display(),
                            "discovered script"
                        );
                        scripts.push(ScriptDef {
                            name,
                            location: path,
                            source,
                        });
                    }
                    Err(e) => {
                        tracing::warn!(
                            location = %path.display(),
                            error = %e,
                            "failed to read script"
                        );
                    }
                }
            }
        }
    }

    /// Iterate over discovered script definitions.
    pub fn scripts(&self) -> &[ScriptDef] {
        &self.scripts
    }

    /// Number of scripts discovered.
    pub fn len(&self) -> usize {
        self.scripts.len()
    }

    /// Whether no scripts were found.
    pub fn is_empty(&self) -> bool {
        self.scripts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_discover_project_local_only() {
        let dir = tempfile::tempdir().unwrap();
        // Create a .theta/extensions/ subdir with a script.
        let ext_dir = dir.path().join(".theta").join("extensions");
        tokio::fs::create_dir_all(&ext_dir).await.unwrap();
        tokio::fs::write(ext_dir.join("test.rhai"), "// test")
            .await
            .unwrap();

        // Note: global discovery also happens, so we just check
        // that *at least* the project-local script is found.
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
        // No .theta/extensions/ in this temp dir.
        // Global scripts may exist, so we just check it doesn't crash.
        let loader = ScriptLoader::discover(dir.path()).await;
        // This is fine — we just check the loader doesn't panic.
        let _ = loader.len();
    }
}
