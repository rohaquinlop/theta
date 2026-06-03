use michin::session::SessionManager;
use michin_ai::{ContentBlock, Message};

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

fn make_user_msg(text: &str) -> Message {
    Message::User {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        timestamp: now_ms(),
    }
}

fn make_assistant_msg(text: &str) -> Message {
    Message::Assistant {
        content: vec![ContentBlock::Text {
            text: text.to_string(),
        }],
        api: None,
        provider: None,
        model: None,
        usage: None,
        stop_reason: None,
        error_message: None,
        timestamp: now_ms(),
    }
}

fn make_empty_assistant_msg() -> Message {
    Message::Assistant {
        content: vec![],
        api: None,
        provider: None,
        model: None,
        usage: None,
        stop_reason: None,
        error_message: None,
        timestamp: now_ms(),
    }
}

#[tokio::test]
async fn test_create_and_open_session() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mgr = SessionManager::with_dir(tmp.path().to_path_buf());

    let session = mgr.create(Some("test-model")).await.unwrap();
    assert_eq!(session.messages.len(), 0);
    assert!(session.file_path.exists());

    let reopened = mgr.open(&session.file_path).await.unwrap();
    assert_eq!(reopened.messages.len(), 0);

    let reopened2 = mgr
        .open_by_id(&session.meta.as_ref().unwrap().id)
        .await
        .unwrap();
    assert_eq!(reopened2.messages.len(), 0);
}

#[tokio::test]
async fn test_append_and_reload() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mgr = SessionManager::with_dir(tmp.path().to_path_buf());

    let mut session = mgr.create(None).await.unwrap();

    let msg = make_user_msg("hello");
    mgr.append_entry(&mut session, &msg).await.unwrap();
    assert_eq!(session.messages.len(), 1);

    let reopened = mgr.open(&session.file_path).await.unwrap();
    assert_eq!(reopened.messages.len(), 1);
}

#[tokio::test]
async fn test_open_parses_legacy_pi_tool_result_entry() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mgr = SessionManager::with_dir(tmp.path().to_path_buf());
    let session = mgr.create(None).await.unwrap();
    let legacy = r#"{"type":"toolResult","toolCallId":"c1","toolName":"read","content":[{"type":"text","text":"ok"}],"details":null,"isError":false,"timestamp":1}"#;
    std::fs::write(&session.file_path, format!("{legacy}\n")).unwrap();
    let reopened = mgr.open(&session.file_path).await.unwrap();
    assert_eq!(reopened.messages.len(), 1);
    assert!(matches!(reopened.messages[0], Message::ToolResult { .. }));
}

#[tokio::test]
async fn test_open_skips_corrupted_lines_and_appends_repair_message() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mgr = SessionManager::with_dir(tmp.path().to_path_buf());
    let session = mgr.create(None).await.unwrap();
    let mixed = concat!(
        "{\"type\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"ok\"}],\"timestamp\":1}\n",
        "{not-valid-json}\n"
    );
    std::fs::write(&session.file_path, mixed).unwrap();
    let reopened = mgr.open(&session.file_path).await.unwrap();
    assert_eq!(reopened.messages.len(), 2);
    assert!(matches!(reopened.messages[0], Message::User { .. }));
    assert!(matches!(reopened.messages[1], Message::Assistant { .. }));
}

#[tokio::test]
async fn test_fork() {
    let tmp = tempfile::TempDir::new().unwrap();
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
    let tmp = tempfile::TempDir::new().unwrap();
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
async fn test_resume_repairs_in_progress_run() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mgr = SessionManager::with_dir(tmp.path().to_path_buf());

    let session = mgr.create(None).await.unwrap();
    let sid = session.meta.as_ref().unwrap().id.clone();
    mgr.mark_run_in_progress(&sid, "run-1", "turn-1")
        .await
        .unwrap();

    let resumed = mgr.resume().await.unwrap();
    let has_repair = resumed.messages.iter().any(|m| {
        matches!(
            m,
            Message::Assistant { content, .. }
            if content.iter().any(|b| matches!(b, ContentBlock::Text { text } if text.contains("interrupted before turn completion")))
        )
    });
    assert!(
        has_repair,
        "resume should append explicit interruption repair"
    );

    let list = mgr.list().await.unwrap();
    let meta = list.iter().find(|m| m.id == sid).unwrap();
    assert!(
        !meta.in_progress,
        "resume repair should clear in-progress flag"
    );
}

#[tokio::test]
async fn test_list_sessions() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mgr = SessionManager::with_dir(tmp.path().to_path_buf());

    mgr.create(None).await.unwrap();
    mgr.create(Some("gpt-5.5")).await.unwrap();

    let list = mgr.list().await.unwrap();
    assert_eq!(list.len(), 2);
}

#[tokio::test]
async fn test_append_missing_entries_skips_exact_duplicates() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mgr = SessionManager::with_dir(tmp.path().to_path_buf());

    let mut session = mgr.create(None).await.unwrap();
    let a = make_user_msg("A");
    let b = make_assistant_msg("B");
    let c = make_user_msg("C");

    mgr.append_entry(&mut session, &a).await.unwrap();
    mgr.append_entry(&mut session, &b).await.unwrap();

    let appended = mgr
        .append_missing_entries(&mut session, &[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();
    assert_eq!(appended, 1);

    let reopened = mgr.open(&session.file_path).await.unwrap();
    assert_eq!(reopened.messages.len(), 3);
    assert!(matches!(reopened.messages[2], Message::User { .. }));
}

#[tokio::test]
async fn test_append_missing_entries_skips_empty_assistant() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mgr = SessionManager::with_dir(tmp.path().to_path_buf());

    let mut session = mgr.create(None).await.unwrap();
    let empty = make_empty_assistant_msg();
    let user = make_user_msg("real");

    let appended = mgr
        .append_missing_entries(&mut session, &[empty, user])
        .await
        .unwrap();
    assert_eq!(appended, 1);

    let reopened = mgr.open(&session.file_path).await.unwrap();
    assert_eq!(reopened.messages.len(), 1);
    assert!(matches!(reopened.messages[0], Message::User { .. }));
}

#[tokio::test]
async fn test_append_missing_entries_preserves_order() {
    let tmp = tempfile::TempDir::new().unwrap();
    let mgr = SessionManager::with_dir(tmp.path().to_path_buf());

    let mut session = mgr.create(None).await.unwrap();
    let a = make_user_msg("1");
    let b = make_assistant_msg("2");
    let c = make_user_msg("3");

    let appended = mgr
        .append_missing_entries(&mut session, &[a.clone(), b.clone(), c.clone()])
        .await
        .unwrap();
    assert_eq!(appended, 3);

    let reopened = mgr.open(&session.file_path).await.unwrap();
    assert_eq!(reopened.messages.len(), 3);
    assert_eq!(
        michin::session::message_fingerprint(&reopened.messages[0]),
        michin::session::message_fingerprint(&a)
    );
    assert_eq!(
        michin::session::message_fingerprint(&reopened.messages[1]),
        michin::session::message_fingerprint(&b)
    );
    assert_eq!(
        michin::session::message_fingerprint(&reopened.messages[2]),
        michin::session::message_fingerprint(&c)
    );
}
