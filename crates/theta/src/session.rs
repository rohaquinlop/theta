//! Session management: create, open, fork, resume, and append to Pi-compatible JSONL sessions.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use theta_ai::Message;

/// Result type for session operations.
pub type SessionResult<T> = Result<T, SessionError>;

/// Errors that can occur during session operations.
#[derive(Debug, thiserror::Error)]
pub enum SessionError {
    #[error("session not found: {path}")]
    NotFound { path: String },

    #[error("failed to read session file: {0}")]
    Read(std::io::Error),

    #[error("failed to write session file: {0}")]
    Write(std::io::Error),

    #[error("failed to parse session entry at line {line}: {error}")]
    Parse { line: usize, error: String },

    #[error("session directory error: {0}")]
    Dir(std::io::Error),
}

impl From<std::io::Error> for SessionError {
    fn from(e: std::io::Error) -> Self {
        match e.kind() {
            std::io::ErrorKind::NotFound => SessionError::NotFound {
                path: "unknown".into(),
            },
            _ => SessionError::Read(e),
        }
    }
}

/// Metadata for a session stored in the index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub id: String,
    pub path: String,
    pub title: Option<String>,
    pub model: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub branch: Option<String>,
    pub created_at: u64,
    pub last_active_at: u64,
    pub message_count: usize,
    #[serde(default)]
    pub token_count: u32,
}

/// Index of all sessions in a project.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SessionIndex {
    pub sessions: Vec<SessionMeta>,
}

/// A loaded session: path to the JSONL file plus its parsed messages.
#[derive(Debug, Clone)]
pub struct Session {
    /// Absolute path to the session JSONL file.
    pub file_path: PathBuf,

    /// All messages parsed from the session file.
    pub messages: Vec<Message>,

    /// Metadata from the index (if found).
    pub meta: Option<SessionMeta>,
}

/// Manages session creation, loading, forking, and persistence.
///
/// Sessions are stored as JSONL files in `<working_dir>/.theta/sessions/`.
/// An `index.json` maps session IDs to metadata.
pub struct SessionManager {
    sessions_dir: PathBuf,
    index_path: PathBuf,
    working_dir: PathBuf,
}

impl SessionManager {
    /// Create a new session manager.
    ///
    /// If `sessions_dir` is provided, uses that path. Otherwise defaults to
    /// `~/.theta/sessions/`. The `working_dir` parameter is kept for API
    /// compatibility but is only used when `sessions_dir` is set.
    pub fn new(working_dir: &Path) -> Self {
        let sessions_dir = dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join(".theta")
            .join("sessions");
        let index_path = sessions_dir.join("index.json");
        Self {
            sessions_dir,
            index_path,
            working_dir: working_dir.to_path_buf(),
        }
    }

    /// Create a session manager with a custom sessions directory (for testing).
    pub fn with_dir(sessions_dir: PathBuf) -> Self {
        let index_path = sessions_dir.join("index.json");
        Self {
            sessions_dir,
            index_path,
            working_dir: PathBuf::new(),
        }
    }

    /// Ensure the sessions directory exists.
    async fn ensure_dir(&self) -> SessionResult<()> {
        tokio::fs::create_dir_all(&self.sessions_dir)
            .await
            .map_err(SessionError::Dir)?;
        Ok(())
    }

    /// Load the index file. Returns empty index if file doesn't exist.
    async fn load_index(&self) -> SessionResult<SessionIndex> {
        self.ensure_dir().await?;
        match tokio::fs::read_to_string(&self.index_path).await {
            Ok(contents) => {
                let index: SessionIndex =
                    serde_json::from_str(&contents).map_err(|e| SessionError::Parse {
                        line: 0,
                        error: format!("index.json: {e}"),
                    })?;
                Ok(index)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(SessionIndex::default()),
            Err(e) => Err(SessionError::Read(e)),
        }
    }

    /// Save the index file.
    async fn save_index(&self, index: &SessionIndex) -> SessionResult<()> {
        self.ensure_dir().await?;
        let contents = serde_json::to_string_pretty(index).map_err(|e| SessionError::Parse {
            line: 0,
            error: e.to_string(),
        })?;
        tokio::fs::write(&self.index_path, contents)
            .await
            .map_err(SessionError::Write)?;
        Ok(())
    }

    /// Create a new session file. Returns the session with zero messages.
    pub async fn create(&self, model: Option<&str>) -> SessionResult<Session> {
        self.ensure_dir().await?;

        let id = generate_session_id();
        let filename = format!("{id}.jsonl");
        let file_path = self.sessions_dir.join(&filename);
        let now = now_ms();

        // Create an empty file.
        tokio::fs::write(&file_path, "")
            .await
            .map_err(SessionError::Write)?;

        let mut index = self.load_index().await?;
        let meta = SessionMeta {
            id: id.clone(),
            path: file_path.to_string_lossy().to_string(),
            title: None,
            model: model.map(|m| m.to_string()),
            project: if self.working_dir.as_os_str().is_empty() {
                None
            } else {
                Some(self.working_dir.to_string_lossy().to_string())
            },
            branch: current_git_branch(&self.working_dir),
            created_at: now,
            last_active_at: now,
            message_count: 0,
            token_count: 0,
        };
        index.sessions.push(meta.clone());
        self.save_index(&index).await?;

        Ok(Session {
            file_path,
            messages: vec![],
            meta: Some(meta),
        })
    }

    /// Open an existing session by file path.
    pub async fn open(&self, path: &Path) -> SessionResult<Session> {
        let file_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.sessions_dir.join(path)
        };

        if !file_path.exists() {
            return Err(SessionError::NotFound {
                path: file_path.to_string_lossy().to_string(),
            });
        }

        let messages = parse_session_file(&file_path).await?;

        // Try to find matching metadata in the index.
        let index = self.load_index().await?;
        let meta = index
            .sessions
            .iter()
            .find(|m| m.path == file_path.to_string_lossy())
            .cloned();

        Ok(Session {
            file_path,
            messages,
            meta,
        })
    }

    /// Open a session by its index ID.
    pub async fn open_by_id(&self, id: &str) -> SessionResult<Session> {
        let index = self.load_index().await?;
        let meta = index
            .sessions
            .iter()
            .find(|m| m.id == id)
            .cloned()
            .ok_or_else(|| SessionError::NotFound {
                path: id.to_string(),
            })?;
        self.open(PathBuf::from(&meta.path).as_path()).await
    }

    /// Resume the most recently active session.
    pub async fn resume(&self) -> SessionResult<Session> {
        let index = self.load_index().await?;
        let latest = index
            .sessions
            .iter()
            .max_by_key(|m| m.last_active_at)
            .cloned()
            .ok_or_else(|| SessionError::NotFound {
                path: "no sessions found".to_string(),
            })?;
        self.open(PathBuf::from(&latest.path).as_path()).await
    }

    /// Fork a session: copy the JSONL file and create a new session with a fresh ID.
    /// All existing messages are preserved.
    pub async fn fork(&self, session: &Session, model: Option<&str>) -> SessionResult<Session> {
        self.ensure_dir().await?;

        let id = generate_session_id();
        let filename = format!("{id}.jsonl");
        let new_path = self.sessions_dir.join(&filename);
        let now = now_ms();

        // Copy the session file.
        tokio::fs::copy(&session.file_path, &new_path)
            .await
            .map_err(SessionError::Write)?;

        let mut index = self.load_index().await?;
        let meta = SessionMeta {
            id: id.clone(),
            path: new_path.to_string_lossy().to_string(),
            title: session.meta.as_ref().and_then(|m| m.title.clone()),
            model: model
                .map(|m| m.to_string())
                .or_else(|| session.meta.as_ref().and_then(|m| m.model.clone())),
            project: session.meta.as_ref().and_then(|m| m.project.clone()),
            branch: current_git_branch(&self.working_dir)
                .or_else(|| session.meta.as_ref().and_then(|m| m.branch.clone())),
            created_at: now,
            last_active_at: now,
            message_count: session.messages.len(),
            token_count: session.messages.iter().map(Message::token_count).sum(),
        };
        index.sessions.push(meta.clone());
        self.save_index(&index).await?;

        Ok(Session {
            file_path: new_path,
            messages: session.messages.clone(),
            meta: Some(meta),
        })
    }

    /// Append a message to a session's JSONL file and update the index.
    pub async fn append_entry(
        &self,
        session: &mut Session,
        message: &Message,
    ) -> SessionResult<()> {
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .create(true)
            .open(&session.file_path)
            .map_err(SessionError::Write)?;

        let line = serde_json::to_string(message).map_err(|e| SessionError::Parse {
            line: 0,
            error: e.to_string(),
        })?;
        writeln!(file, "{line}").map_err(SessionError::Write)?;

        session.messages.push(message.clone());

        // Update the index.
        let mut index = self.load_index().await?;
        let now = now_ms();
        if let Some(ref mut meta) = index
            .sessions
            .iter_mut()
            .find(|m| m.id == session.meta.as_ref().map(|x| x.id.as_str()).unwrap_or(""))
        {
            meta.last_active_at = now;
            meta.message_count = session.messages.len();
            meta.token_count = session.messages.iter().map(Message::token_count).sum();
            if meta.title.is_none()
                && let Some(title) = message_title(message)
            {
                meta.title = Some(title);
            }
            if meta.branch.is_none() {
                meta.branch = current_git_branch(&self.working_dir);
            }
            session.meta = Some(meta.clone());
        }
        self.save_index(&index).await?;

        Ok(())
    }

    /// List all sessions from the index.
    pub async fn list(&self) -> SessionResult<Vec<SessionMeta>> {
        let index = self.load_index().await?;
        Ok(index.sessions.clone())
    }
}

/// Parse a JSONL session file into a vector of Messages.
async fn parse_session_file(path: &Path) -> SessionResult<Vec<Message>> {
    let file = std::fs::File::open(path).map_err(SessionError::Read)?;
    let reader = BufReader::new(file);
    let mut messages = Vec::new();

    for (line_num, line_result) in reader.lines().enumerate() {
        let line = line_result.map_err(SessionError::Read)?;
        if line.trim().is_empty() {
            continue;
        }
        let msg: Message = serde_json::from_str(&line).map_err(|e| SessionError::Parse {
            line: line_num + 1,
            error: e.to_string(),
        })?;
        messages.push(msg);
    }

    Ok(messages)
}

/// Generate a unique session ID based on timestamp + random suffix.
fn generate_session_id() -> String {
    let ts = now_ms();
    let random: u32 = (ts as u32).wrapping_mul(2654435761); // Knuth's multiplicative hash
    format!("{ts:x}-{random:x}")
}

fn current_git_branch(working_dir: &Path) -> Option<String> {
    if working_dir.as_os_str().is_empty() {
        return None;
    }
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(working_dir)
        .arg("branch")
        .arg("--show-current")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let branch = String::from_utf8_lossy(&output.stdout).trim().to_string();
    if branch.is_empty() {
        None
    } else {
        Some(branch)
    }
}

fn message_title(message: &Message) -> Option<String> {
    let Message::User { content, .. } = message else {
        return None;
    };
    let text = content
        .iter()
        .filter_map(|block| match block {
            theta_ai::ContentBlock::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join(" ");
    let title = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if title.is_empty() {
        None
    } else {
        Some(title.chars().take(80).collect())
    }
}

/// Current time in milliseconds since epoch.
fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;
    use theta_ai::ContentBlock;

    fn make_user_msg(text: &str) -> Message {
        Message::User {
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
            timestamp: now_ms(),
        }
    }

    #[tokio::test]
    async fn test_create_and_open_session() {
        let tmp = TempDir::new().unwrap();
        let mgr = SessionManager::with_dir(tmp.path().to_path_buf());

        let session = mgr.create(Some("test-model")).await.unwrap();
        assert_eq!(session.messages.len(), 0);
        assert!(session.file_path.exists());

        // Re-open by path.
        let reopened = mgr.open(&session.file_path).await.unwrap();
        assert_eq!(reopened.messages.len(), 0);

        // Re-open by ID.
        let reopened2 = mgr
            .open_by_id(&session.meta.as_ref().unwrap().id)
            .await
            .unwrap();
        assert_eq!(reopened2.messages.len(), 0);
    }

    #[tokio::test]
    async fn test_append_and_reload() {
        let tmp = TempDir::new().unwrap();
        let mgr = SessionManager::with_dir(tmp.path().to_path_buf());

        let mut session = mgr.create(None).await.unwrap();

        let msg = make_user_msg("hello");
        mgr.append_entry(&mut session, &msg).await.unwrap();
        assert_eq!(session.messages.len(), 1);

        // Reopen and verify message is persisted.
        let reopened = mgr.open(&session.file_path).await.unwrap();
        assert_eq!(reopened.messages.len(), 1);
    }

    #[tokio::test]
    async fn test_fork() {
        let tmp = TempDir::new().unwrap();
        let mgr = SessionManager::with_dir(tmp.path().to_path_buf());

        let mut session = mgr.create(None).await.unwrap();
        mgr.append_entry(&mut session, &make_user_msg("original"))
            .await
            .unwrap();

        let forked = mgr.fork(&session, Some("forked-model")).await.unwrap();
        assert_eq!(forked.messages.len(), 1);
        assert_ne!(forked.file_path, session.file_path);
    }

    #[tokio::test]
    async fn test_resume_latest() {
        let tmp = TempDir::new().unwrap();
        let mgr = SessionManager::with_dir(tmp.path().to_path_buf());

        let _session_a = mgr.create(None).await.unwrap();
        let session_b = mgr.create(None).await.unwrap();

        let resumed = mgr.resume().await.unwrap();
        assert_eq!(
            resumed.meta.as_ref().unwrap().id,
            session_b.meta.as_ref().unwrap().id
        );
    }

    #[tokio::test]
    async fn test_list_sessions() {
        let tmp = TempDir::new().unwrap();
        let mgr = SessionManager::with_dir(tmp.path().to_path_buf());

        mgr.create(None).await.unwrap();
        mgr.create(Some("gpt-5.5")).await.unwrap();

        let list = mgr.list().await.unwrap();
        assert_eq!(list.len(), 2);
    }
}
