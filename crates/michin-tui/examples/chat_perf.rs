#![cfg(debug_assertions)]

use std::time::{Duration, Instant};
use tokio::sync::mpsc;

use michin_tui::app::TuiEvent;
use michin_tui::components::chat::{Chat, ChatMessage, ChatRole};
use michin_tui::theme::Theme;

fn main() {
    let mut chat = Chat::new(Theme::default());
    seed_history(&mut chat, 3_000);

    let width = 120usize;
    let rounds = 40usize;

    let baseline = bench("baseline_full_rebuild", rounds, || {
        let _ = chat.benchmark_full_rebuild_no_cache(width);
    });

    let cached = bench("cached_rebuild", rounds, || {
        let _ = chat.benchmark_cached_rebuild(width);
    });

    let append_cached = bench("append_cached_rebuild", rounds, || {
        chat.add_message(ChatMessage {
            role: ChatRole::Assistant,
            text: "incremental assistant delta line for cache append benchmark".into(),
            tool_call_id: None,
            is_streaming: false,

            is_error: false,
        });
        let _ = chat.benchmark_cached_rebuild(width);
    });

    println!();
    println!("Summary");
    println!("baseline avg:  {:?}", baseline / rounds as u32);
    println!("cached avg:    {:?}", cached / rounds as u32);
    println!("append avg:    {:?}", append_cached / rounds as u32);

    progress_burst_sim();
}

fn seed_history(chat: &mut Chat, count: usize) {
    for i in 0..count {
        let role = if i % 5 == 0 {
            ChatRole::Tool
        } else if i % 2 == 0 {
            ChatRole::User
        } else {
            ChatRole::Assistant
        };
        chat.add_message(ChatMessage {
            role,
            text: format!(
                "message {i}: {} {}",
                "lorem ipsum dolor sit amet".repeat(2),
                "x".repeat(80)
            ),
            tool_call_id: None,
            is_streaming: false,

            is_error: false,
        });
    }
}

fn bench(label: &str, rounds: usize, mut f: impl FnMut()) -> Duration {
    let start = Instant::now();
    for _ in 0..rounds {
        f();
    }
    let elapsed = start.elapsed();
    println!("{label}: total={elapsed:?} rounds={rounds}");
    elapsed
}

fn progress_burst_sim() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .build()
        .expect("build runtime");
    rt.block_on(async {
        let raw_events = 200_000usize;
        let tools = 8usize;
        let (raw_tx, mut raw_rx) = mpsc::unbounded_channel::<TuiEvent>();
        let (bounded_tx, mut bounded_rx) = mpsc::channel::<TuiEvent>(1024);

        let forwarder = tokio::spawn(async move {
            let mut pending = std::collections::HashMap::<String, String>::new();
            while let Some(event) = raw_rx.recv().await {
                match event {
                    TuiEvent::ToolProgress { name, message } => {
                        pending.insert(name, message);
                    }
                    evt => {
                        if !pending.is_empty() {
                            for (name, message) in pending.drain() {
                                let _ =
                                    bounded_tx.try_send(TuiEvent::ToolProgress { name, message });
                            }
                        }
                        let _ = bounded_tx.try_send(evt);
                    }
                }
            }
            if !pending.is_empty() {
                for (name, message) in pending.drain() {
                    let _ = bounded_tx.try_send(TuiEvent::ToolProgress { name, message });
                }
            }
        });

        let produce_start = Instant::now();
        for i in 0..raw_events {
            let tool = format!("tool-{}", i % tools);
            let msg = format!("progress-{i}");
            let _ = raw_tx.send(TuiEvent::ToolProgress {
                name: tool,
                message: msg,
            });
        }
        let _ = raw_tx.send(TuiEvent::AgentEnd { aborted: false });
        drop(raw_tx);
        let produce_elapsed = produce_start.elapsed();

        let mut drained = 0usize;
        let drain_start = Instant::now();
        while let Some(evt) = bounded_rx.recv().await {
            drained += 1;
            if matches!(evt, TuiEvent::AgentEnd { .. }) {
                break;
            }
        }
        let drain_elapsed = drain_start.elapsed();
        let _ = forwarder.await;

        println!();
        println!("Progress Burst Simulation");
        println!("raw events sent:   {raw_events}");
        println!("tools:             {tools}");
        println!("bounded drained:   {drained}");
        println!("produce elapsed:   {produce_elapsed:?}");
        println!("drain elapsed:     {drain_elapsed:?}");
    });
}
