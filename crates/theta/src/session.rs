//! Session management: create, open, fork, resume, and append JSONL sessions.
//!
//! Writes Theta-native message keys while remaining backward-compatible when
//! reading legacy Pi-compatible entries.

use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use theta_ai::{ContentBlock, Message};

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
    #[serde(default)]
    pub in_progress: bool,
    #[serde(default)]
    pub active_run_id: Option<String>,
    #[serde(default)]
    pub active_turn_id: Option<String>,
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
            in_progress: false,
            active_run_id: None,
            active_turn_id: None,
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
        let mut session = self.open(PathBuf::from(&latest.path).as_path()).await?;
        if latest.in_progress {
            let repair = Message::Assistant {
                content: vec![theta_ai::ContentBlock::text(
                    "Previous run was interrupted before turn completion. Session resumed with explicit runtime-constraint termination.",
                )],
                api: None,
                provider: None,
                model: None,
                usage: None,
                stop_reason: None,
                error_message: None,
                timestamp: now_ms(),
            };
            self.append_entry(&mut session, &repair).await?;
            self.mark_run_completed(
                &latest.id,
                Some("BlockedRuntimeConstraint: interrupted run on resume"),
            )
            .await?;
        }
        Ok(session)
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
            in_progress: false,
            active_run_id: None,
            active_turn_id: None,
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
        if matches!(message, Message::Assistant { content, .. } if content.is_empty()) {
            return Ok(());
        }

        let path = session.file_path.clone();
        let line = serde_json::to_string(message).map_err(|e| SessionError::Parse {
            line: 0,
            error: e.to_string(),
        })?;

        tokio::task::spawn_blocking(move || {
            let mut file = std::fs::OpenOptions::new()
                .append(true)
                .create(true)
                .open(&path)
                .map_err(SessionError::Write)?;
            writeln!(file, "{line}").map_err(SessionError::Write)?;
            Ok::<_, SessionError>(())
        })
        .await
        .expect("spawn_blocking panicked")?;

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

    pub async fn append_missing_entries(
        &self,
        session: &mut Session,
        state_messages: &[Message],
    ) -> SessionResult<usize> {
        let mut persisted: HashSet<(u64, u8, u64)> = session
            .messages
            .iter()
            .map(message_fingerprint)
            .collect::<HashSet<_>>();

        let mut appended = 0usize;
        for message in state_messages {
            if matches!(message, Message::Assistant { content, .. } if content.is_empty()) {
                continue;
            }

            let fingerprint = message_fingerprint(message);
            if persisted.contains(&fingerprint) {
                continue;
            }

            self.append_entry(session, message).await?;
            persisted.insert(fingerprint);
            appended += 1;
        }

        tracing::debug!(
            appended,
            latest_timestamp = state_messages.last().map(message_timestamp),
            "appended missing session entries"
        );

        Ok(appended)
    }

    /// List sessions from the index, filtered by the current working directory.
    ///
    /// When `working_dir` is set (non-empty), only sessions whose `project`
    /// matches the working directory are returned. When `working_dir` is empty
    /// (e.g., in tests), all sessions are returned.
    pub async fn list(&self) -> SessionResult<Vec<SessionMeta>> {
        let index = self.load_index().await?;
        let sessions = if self.working_dir.as_os_str().is_empty() {
            index.sessions.clone()
        } else {
            let wd = self.working_dir.to_string_lossy();
            index
                .sessions
                .iter()
                .filter(|m| m.project.as_deref().is_some_and(|p| p == wd.as_ref()))
                .cloned()
                .collect()
        };
        Ok(sessions)
    }

    /// Mark a session run as in-progress.
    pub async fn mark_run_in_progress(
        &self,
        session_id: &str,
        run_id: &str,
        turn_id: &str,
    ) -> SessionResult<()> {
        let mut index = self.load_index().await?;
        if let Some(meta) = index.sessions.iter_mut().find(|m| m.id == session_id) {
            meta.in_progress = true;
            meta.active_run_id = Some(run_id.to_string());
            meta.active_turn_id = Some(turn_id.to_string());
            meta.last_active_at = now_ms();
            self.save_index(&index).await?;
        }
        Ok(())
    }

    /// Mark a session run as completed.
    pub async fn mark_run_completed(
        &self,
        session_id: &str,
        _end_reason: Option<&str>,
    ) -> SessionResult<()> {
        let mut index = self.load_index().await?;
        if let Some(meta) = index.sessions.iter_mut().find(|m| m.id == session_id) {
            meta.in_progress = false;
            meta.active_run_id = None;
            meta.active_turn_id = None;
            meta.last_active_at = now_ms();
            self.save_index(&index).await?;
        }
        Ok(())
    }
}

pub fn message_fingerprint(message: &Message) -> (u64, u8, u64) {
    let tag: u8 = match message {
        Message::User { .. } => 1,
        Message::Assistant { .. } => 2,
        Message::ToolResult { .. } => 3,
        Message::ModelChange { .. } => 4,
        Message::ThinkingLevelChange { .. } => 5,
    };
    let content_hash = hash_message_content(message);
    (message_timestamp(message), tag, content_hash)
}

fn hash_message_content(message: &Message) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    // Hash key semantically-distinguishing fields only, avoiding full
    // serde_json serialization of the message (which includes timestamps,
    // usage, stop_reason, etc. that don't affect content identity).
    match message {
        Message::User { content, .. } => {
            for block in content {
                hash_content_block(block, &mut h);
            }
        }
        Message::Assistant { content, .. } => {
            for block in content {
                hash_content_block(block, &mut h);
            }
        }
        Message::ToolResult {
            tool_call_id,
            content,
            ..
        } => {
            tool_call_id.hash(&mut h);
            for block in content {
                hash_content_block(block, &mut h);
            }
        }
        Message::ModelChange {
            model_id, provider, ..
        } => {
            model_id.hash(&mut h);
            provider.hash(&mut h);
        }
        Message::ThinkingLevelChange { .. } => {
            // No semantically distinguishing fields; rely on timestamp+tag
            // alone. If you have two consecutive thinking-level changes at
            // the same ms, the second is a duplicate by definition.
        }
    }
    h.finish()
}

/// Hash the semantically-distinguishing fields of a ContentBlock.
/// More efficient than full serde serialization.
fn hash_content_block(block: &ContentBlock, h: &mut std::collections::hash_map::DefaultHasher) {
    use std::hash::Hash;
    match block {
        ContentBlock::Text { text } => {
            text.hash(h);
        }
        ContentBlock::ToolCall {
            id,
            name,
            arguments,
        } => {
            id.hash(h);
            name.hash(h);
            // serde_json::Value doesn't impl Hash — hash its string form.
            arguments.to_string().hash(h);
        }
        ContentBlock::Image { media_type, data } => {
            media_type.hash(h);
            data.hash(h);
        }
        ContentBlock::Thinking { thinking, .. } => {
            thinking.hash(h);
        }
        ContentBlock::ToolResult {
            tool_call_id,
            tool_name,
            content,
            ..
        } => {
            tool_call_id.hash(h);
            tool_name.hash(h);
            for block in content {
                hash_content_block(block, h);
            }
        }
    }
}

fn message_timestamp(message: &Message) -> u64 {
    match message {
        Message::User { timestamp, .. }
        | Message::Assistant { timestamp, .. }
        | Message::ToolResult { timestamp, .. }
        | Message::ModelChange { timestamp, .. }
        | Message::ThinkingLevelChange { timestamp, .. } => *timestamp,
    }
}

/// Parse a JSONL session file into a vector of Messages.
async fn parse_session_file(path: &Path) -> SessionResult<Vec<Message>> {
    let path = path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let file = std::fs::File::open(&path).map_err(SessionError::Read)?;
        let reader = BufReader::new(file);
        let mut messages = Vec::new();
        let mut corrupted_lines = 0usize;

        for (line_num, line_result) in reader.lines().enumerate() {
            let line = line_result.map_err(SessionError::Read)?;
            if line.trim().is_empty() {
                continue;
            }
            match serde_json::from_str::<Message>(&line) {
                Ok(msg) => messages.push(msg),
                Err(e) => {
                    corrupted_lines += 1;
                    tracing::warn!(
                        line = line_num + 1,
                        error = %e,
                        "skipping corrupted session line"
                    );
                }
            }
        }

        if corrupted_lines > 0 {
            messages.push(Message::Assistant {
                content: vec![ContentBlock::text(format!(
                    "Session repair: skipped {corrupted_lines} corrupted JSONL entr{} during load.",
                    if corrupted_lines == 1 { "y" } else { "ies" }
                ))],
                api: None,
                provider: None,
                model: None,
                usage: None,
                stop_reason: None,
                error_message: None,
                timestamp: now_ms(),
            });
        }

        Ok(messages)
    })
    .await
    .expect("spawn_blocking panicked")
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
