//! FFF integration — in-process file search with frecency ranking.
//!
//! Wrap `fff_search` so the agent and TUI editor share a single indexed
//! view of the working directory. FilePicker is initialized once at agent
//! startup and reused across all search operations (tool calls, @-mention
//! autocomplete) without process spawns.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use fff_search::file_picker::{FilePicker, FilePickerOptions};
use fff_search::shared::{SharedFilePicker, SharedFrecency, SharedQueryTracker};
use fff_search::{
    self, FFFMode, FileSearchConfig, FuzzySearchOptions, PaginationArgs, QueryParser,
};

/// Handle to a running FFF index. FilePicker is `!Send`, so we wrap in
/// a thread-local or spawn a dedicated thread. For MichiN, we run FilePicker
/// on a background thread and communicate via the shared handles.
#[derive(Debug)]
pub struct FffHandle {
    pub picker: SharedFilePicker,
    pub frecency: SharedFrecency,
    pub query_tracker: SharedQueryTracker,
    /// Background scan + watcher thread handle (taken on drop).
    bg_thread: Option<std::thread::JoinHandle<()>>,
    /// Shutdown signal.
    shutdown: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl Drop for FffHandle {
    fn drop(&mut self) {
        tracing::debug!("fff: shutting down FilePicker background thread");
        self.shutdown
            .store(true, std::sync::atomic::Ordering::Relaxed);
        if let Some(handle) = self.bg_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Initialize FFF for a working directory. Returns None if FFF setup fails
/// (non-fatal — falls back to `git ls-files` / shell tools).
pub fn init(working_dir: &Path) -> Option<FffHandle> {
    let base_path = working_dir.to_path_buf();
    let frecency_db = dirs::cache_dir()
        .or_else(dirs::data_local_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("michin")
        .join("fff_frecency");
    let history_db = dirs::data_local_dir()
        .or_else(dirs::cache_dir)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("michin")
        .join("fff_queries");

    if let Some(p) = frecency_db.parent() {
        let _ = std::fs::create_dir_all(p);
    }
    if let Some(p) = history_db.parent() {
        let _ = std::fs::create_dir_all(p);
    }

    let frecency = match fff_search::frecency::FrecencyTracker::open(&frecency_db) {
        Ok(f) => f,
        Err(e) => {
            tracing::warn!("fff: failed to open frecency DB: {e}");
            return None;
        }
    };

    let shared_frecency = SharedFrecency::default();
    if let Err(e) = shared_frecency.init(frecency) {
        tracing::warn!("fff: failed to init shared frecency: {e}");
        return None;
    }

    let query_tracker = match fff_search::query_tracker::QueryTracker::open(&history_db) {
        Ok(q) => q,
        Err(e) => {
            tracing::warn!("fff: failed to open query tracker DB: {e}");
            return None;
        }
    };

    let shared_query_tracker = SharedQueryTracker::default();
    if let Err(e) = shared_query_tracker.init(query_tracker) {
        tracing::warn!("fff: failed to init shared query tracker: {e}");
        return None;
    }

    let shared_picker = SharedFilePicker::default();
    let shutdown = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let shutdown_clone = shutdown.clone();
    let shared_picker_clone = shared_picker.clone();
    let shared_frecency_clone = shared_frecency.clone();
    let base_path_clone = base_path.clone();

    let bg_thread = std::thread::Builder::new()
        .name("fff-bg".into())
        .spawn(move || {
            let mode = FFFMode::Ai;

            match FilePicker::new_with_shared_state(
                shared_picker_clone.clone(),
                shared_frecency_clone.clone(),
                FilePickerOptions {
                    base_path: base_path_clone.to_string_lossy().to_string(),
                    mode,
                    ..Default::default()
                },
            ) {
                Ok(()) => {
                    tracing::info!("fff: FilePicker initialized");
                    if shared_picker_clone.wait_for_scan(Duration::from_secs(120)) {
                        let guard = shared_picker_clone.read().ok();
                        let count = guard
                            .and_then(|g| g.as_ref().map(|p| p.get_files().len()))
                            .unwrap_or(0);
                        tracing::info!("fff: initial scan complete — {count} files indexed");
                    }
                    while !shutdown_clone.load(std::sync::atomic::Ordering::Relaxed) {
                        std::thread::sleep(Duration::from_secs(1));
                    }
                }
                Err(e) => {
                    tracing::warn!("fff: FilePicker init failed: {e}");
                }
            }
        })
        .ok()?;

    let handle = FffHandle {
        picker: shared_picker,
        frecency: shared_frecency,
        query_tracker: shared_query_tracker,
        bg_thread: Some(bg_thread),
        shutdown,
    };

    tracing::info!("fff: initialized for {}", base_path.display());
    Some(handle)
}

/// Fuzzy file search. Returns relative paths ranked by frecency.
pub fn fuzzy_find(handle: &FffHandle, query: &str, max_results: usize) -> Vec<String> {
    let picker = match handle.picker.read() {
        Ok(guard) => guard,
        Err(e) => {
            tracing::warn!("fff: picker read lock error: {e}");
            return Vec::new();
        }
    };

    let picker = match picker.as_ref() {
        Some(p) => p,
        None => return Vec::new(),
    };

    let parser = QueryParser::new(FileSearchConfig);
    let fff_query = parser.parse(query);

    let results = picker.fuzzy_search(
        &fff_query,
        None,
        FuzzySearchOptions {
            max_threads: 0,
            current_file: None,
            pagination: PaginationArgs {
                offset: 0,
                limit: max_results,
            },
            ..Default::default()
        },
    );

    results
        .items
        .iter()
        .map(|item| {
            let path = item.relative_path(picker);
            path.replace('\\', "/")
        })
        .collect()
}

/// Fuzzy file search for editor autocomplete. Uses SharedFilePicker directly
/// without FffHandle (editor runs on main thread, can hold read lock).
pub fn fuzzy_find_shared(
    picker: &SharedFilePicker,
    query: &str,
    max_results: usize,
) -> Vec<String> {
    let guard = match picker.read() {
        Ok(g) => g,
        Err(_) => return Vec::new(),
    };
    let picker = match guard.as_ref() {
        Some(p) => p,
        None => return Vec::new(),
    };

    let parser = QueryParser::new(FileSearchConfig);
    let fff_query = parser.parse(query);

    let results = picker.fuzzy_search(
        &fff_query,
        None,
        FuzzySearchOptions {
            max_threads: 0,
            current_file: None,
            pagination: PaginationArgs {
                offset: 0,
                limit: max_results,
            },
            ..Default::default()
        },
    );

    results
        .items
        .iter()
        .map(|item| {
            let path = item.relative_path(picker);
            path.replace('\\', "/")
        })
        .collect()
}

/// Grep (content search) across indexed files.
///
/// Path filtering is done post-hoc so `"compiler/rustc_parse/"` prefix-matches
/// only files under that directory, not non-code files that happen to live there
/// (FFF's PathSegment constraint means "contains-in-path", not "under-this-dir").
pub fn grep(
    handle: &FffHandle,
    query: &str,
    path_constraint: Option<&str>,
    max_results: usize,
    case_sensitive: bool,
    use_regex: bool,
) -> Option<Vec<GrepMatch>> {
    let guard = match handle.picker.read() {
        Ok(g) => g,
        Err(_) => return None,
    };

    let picker = guard.as_ref()?;

    // Parse content query only — path filtering is done post-hoc.
    let parser = fff_search::QueryParser::new(fff_search::AiGrepConfig);
    let fff_query = parser.parse(query);

    let grep_text = fff_query.grep_text();
    if grep_text.is_empty() {
        tracing::debug!("fff::grep: empty grep_text after parsing query={query:?}");
        return None;
    }

    let mode = if use_regex {
        fff_search::grep::GrepMode::Regex
    } else {
        fff_search::grep::GrepMode::PlainText
    };

    let options = fff_search::grep::GrepSearchOptions {
        smart_case: !case_sensitive,
        page_limit: max_results,
        file_offset: 0,
        max_file_size: 10 * 1024 * 1024,
        max_matches_per_file: 50,
        trim_whitespace: true,
        mode,
        abort_signal: None,
        ..Default::default()
    };

    let result = picker.grep(&fff_query, &options);

    // IMPORTANT: file_index in GrepMatch indexes into result.files (the
    // deduplicated filtered list), NOT picker.get_files() (the full index).
    let result_files = &result.files;

    let matches: Vec<GrepMatch> = result
        .matches
        .iter()
        .filter_map(|m| {
            let path = result_files
                .get(m.file_index)
                .map(|f| f.relative_path(picker).replace('\\', "/"))
                .unwrap_or_else(|| "?".into());

            if !matches_path_constraint(&path, path_constraint) {
                return None;
            }

            Some(GrepMatch {
                path,
                line_number: m.line_number as u32,
                column: m.col as u32,
                line_content: m.line_content.clone(),
                match_ranges: m.match_byte_offsets.iter().map(|r| (r.0, r.1)).collect(),
            })
        })
        .collect();

    tracing::debug!(
        "fff::grep: query={query:?} path={path_constraint:?} raw={} filtered={} ",
        result.matches.len(),
        matches.len()
    );

    if matches.is_empty() {
        None
    } else {
        Some(matches)
    }
}

/// Check if a file path matches the user's path constraint.
///
/// Three forms:
/// - Directory prefix: `"compiler/rustc_parse/"` or `"compiler/rustc_parse"`
/// - Glob: `"*.rs"`, `"src/**/*.rs"`
/// - Plain segment: `"parser"` (any file with "parser" in its path at a dir boundary)
fn matches_path_constraint(path: &str, constraint: Option<&str>) -> bool {
    let Some(c) = constraint.filter(|c| !c.is_empty()) else {
        return true;
    };

    let prefix = c.strip_suffix('/').unwrap_or(c);

    // Glob patterns (*.rs, **/*.rs, {src,lib}/**)
    if c.contains('*') || c.contains('{') || c.contains('?') || c.contains('[') {
        return glob_match(c, path);
    }

    // Exact prefix match: "compiler/rustc_parse" matches "compiler/rustc_parse/src/foo.rs"
    if path == prefix || path.starts_with(prefix) {
        let after = &path[prefix.len()..];
        if after.is_empty() || after.starts_with('/') {
            return true;
        }
    }

    // Substring at a path boundary: "parser" matches "compiler/parser/foo.rs"
    let seg = format!("/{prefix}/");
    if format!("/{path}/").contains(&seg) {
        return true;
    }

    // Ends with suffix at a dir boundary: "parser.rs" matches "compiler/parser.rs"
    path.ends_with(prefix) && {
        let before = &path[..path.len() - prefix.len()];
        before.is_empty() || before.ends_with('/')
    }
}

/// Match a path against a glob pattern.
fn glob_match(pattern: &str, path: &str) -> bool {
    if !pattern.contains('/') {
        let filename = path.rsplit('/').next().unwrap_or(path);
        return glob::Pattern::new(pattern).is_ok_and(|p| p.matches(filename));
    }
    glob::Pattern::new(pattern).is_ok_and(|p| p.matches(path))
}

/// Thread-safe reference to an optional FffHandle. Tools and editor share this.
pub type FffHandleRef = Arc<Mutex<Option<FffHandle>>>;

/// A single grep match.
#[derive(Debug, Clone)]
pub struct GrepMatch {
    pub path: String,
    pub line_number: u32,
    pub column: u32,
    pub line_content: String,
    #[allow(dead_code)]
    pub match_ranges: Vec<(u32, u32)>,
}

/// Touch a file's frecency (called after reading/editing).
/// Updates the "access count" for ranking.
pub fn touch_frecency(handle: &FffHandle, relative_path: &Path) {
    let mut picker = match handle.picker.write() {
        Ok(guard) => guard,
        Err(e) => {
            tracing::warn!("fff: picker write lock poisoned: {e}");
            return;
        }
    };

    let picker = match picker.as_mut() {
        Some(p) => p,
        None => return,
    };

    let frecency_guard = match handle.frecency.read() {
        Ok(guard) => guard,
        Err(e) => {
            tracing::warn!("fff: frecency read lock poisoned: {e}");
            return;
        }
    };

    let frecency = match frecency_guard.as_ref() {
        Some(f) => f,
        None => return,
    };

    if let Err(e) = picker.update_single_file_frecency(relative_path, frecency) {
        tracing::debug!(
            "fff: frecency touch failed for {}: {e}",
            relative_path.display()
        );
    }
}
