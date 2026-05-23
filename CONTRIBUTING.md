# Contributing

Keep Theta small and terminal-first.

## Rules

- Rust 2024 across all crates.
- `tokio` for async.
- `tracing` for logs.
- `anyhow` in the binary, `thiserror` in libraries.
- No `unwrap()` in library code.
- No dynamic provider or tool loading in MVP.

## Before Sending Changes

```bash
cargo fmt --check
cargo clippy -- -D warnings
cargo test
cargo test -p theta-agent-core --test policy_scenario_matrix
```

Stage only files you changed. Do not commit generated or unrelated files.
