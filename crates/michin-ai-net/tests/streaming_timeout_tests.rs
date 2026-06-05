//! Streaming timeout semantics tests.
//!
//! Verifies that `options.timeout_ms` bounds request setup (connect +
//! response headers) only, and never kills a healthy stream that takes
//! longer than the timeout to finish generating.

use std::time::Duration;

use futures::StreamExt;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;

use michin_ai::event::AssistantMessageEvent;
use michin_ai::model::{Model, ModelCompat};
use michin_ai::provider::Provider as LlmProvider;
use michin_ai::types::{Api, ContentBlock, Context, Message, Modality, Provider, StreamOptions};
use michin_ai_net::providers::openai_compat::OpenAiCompatProvider;

fn test_model(base_url: &str) -> Model {
    Model {
        id: "test-model".into(),
        name: "Test".into(),
        api: Api::OpenAiCompletions,
        provider: Provider::OpenAI,
        base_url: base_url.into(),
        reasoning: false,
        thinking_level_map: Default::default(),
        input: vec![Modality::Text],
        context_window: 128_000,
        max_tokens: 16_384,
        compat: ModelCompat::for_openai(),
    }
}

fn test_context() -> Context {
    let mut ctx = Context::new();
    ctx.messages = vec![Message::User {
        content: vec![ContentBlock::text("hi")],
        timestamp: 0,
    }];
    ctx
}

fn sse_chunk(text: &str) -> String {
    format!(
        "data: {{\"choices\":[{{\"delta\":{{\"content\":\"{text}\"}},\"finish_reason\":null}}]}}\n\n"
    )
}

const SSE_HEADERS: &str =
    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n";

/// A streaming response that takes longer than `timeout_ms` end-to-end must
/// complete, because the timeout bounds request setup only — not the body.
#[tokio::test]
async fn long_stream_survives_past_provider_timeout() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        // Drain the request head; we don't care about its contents.
        let mut buf = [0u8; 4096];
        use tokio::io::AsyncReadExt;
        let _ = socket.read(&mut buf).await;

        socket.write_all(SSE_HEADERS.as_bytes()).await.unwrap();
        // Stream chunks for ~1s total — well past the 300ms timeout below.
        for i in 0..5 {
            socket
                .write_all(sse_chunk(&format!("chunk{i}")).as_bytes())
                .await
                .unwrap();
            socket.flush().await.unwrap();
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        let done =
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n";
        socket.write_all(done.as_bytes()).await.unwrap();
        socket.flush().await.unwrap();
    });

    let provider = OpenAiCompatProvider::new();
    provider.set_token("test-key-1234567890");
    let model = test_model(&format!("http://{addr}"));
    let options = StreamOptions {
        timeout_ms: Some(300),
        ..Default::default()
    };

    let mut stream = provider
        .stream(&model, &test_context(), &options)
        .await
        .expect("headers arrive immediately; setup must succeed");

    let mut text = String::new();
    let mut saw_done = false;
    let mut errors = Vec::new();
    while let Some(event) = stream.next().await {
        match event {
            AssistantMessageEvent::TextDelta { text: t } => text.push_str(&t),
            AssistantMessageEvent::Done { .. } => saw_done = true,
            AssistantMessageEvent::Error { code, message } => {
                errors.push(format!("{code}: {message}"))
            }
            _ => {}
        }
    }

    assert!(
        errors.is_empty(),
        "stream longer than timeout_ms must not be aborted, got: {errors:?}"
    );
    assert!(saw_done, "stream should complete with a Done event");
    assert_eq!(text, "chunk0chunk1chunk2chunk3chunk4");
}

/// A server that accepts the connection but never sends response headers
/// must fail with a 408 after `timeout_ms` — not hang forever.
#[tokio::test]
async fn slow_headers_time_out_with_408() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();

    tokio::spawn(async move {
        let (socket, _) = listener.accept().await.unwrap();
        // Hold the connection open without ever responding.
        tokio::time::sleep(Duration::from_secs(30)).await;
        drop(socket);
    });

    let provider = OpenAiCompatProvider::new();
    provider.set_token("test-key-1234567890");
    let model = test_model(&format!("http://{addr}"));
    let options = StreamOptions {
        timeout_ms: Some(300),
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let result = provider.stream(&model, &test_context(), &options).await;
    let elapsed = start.elapsed();

    match result {
        Err(michin_ai::error::MichiNError::ApiError { status, .. }) => {
            assert_eq!(status, 408, "header timeout should map to 408");
        }
        Ok(_) => panic!("expected setup timeout, got a stream"),
        Err(e) => panic!("expected 408 ApiError, got: {e}"),
    }
    assert!(
        elapsed < Duration::from_secs(5),
        "should fail at ~timeout_ms, took {elapsed:?}"
    );
}
