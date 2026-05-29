# theta-ai — Agent Rules

> Rules for working on the theta-ai crate: unified LLM API.

## Crate Purpose

Provides the unified LLM abstraction layer: types, provider trait, streaming, replay, and two provider implementations. All model interaction flows through this crate.

## Public API

- `theta_ai::types` — `ContentBlock`, `Message`, `Tool`, `Provider`, `Model`, `Context`, `StopReason`.
- `theta_ai::event` — `EventAccumulator`, `AssistantMessageEvent` for streaming.
- `theta_ai::Provider` trait — the extension point for new LLM backends.

## Provider Implementations

Two providers in `crates/theta-ai/src/providers/`:

1. **`OpenAiCompatProvider`** — handles OpenAI, DeepSeek, OpenCode, Xiaomi MiMo via `/v1/chat/completions`. Models fetched dynamically at runtime.
2. **`OpenAiCodexProvider`** — ChatGPT Plus session-token auth targeting `chatgpt.com/backend-api`, WebSocket + SSE fallback.

### Per-Model Compat Flags

| Flag                                      | Purpose                                                            |
| ----------------------------------------- | ------------------------------------------------------------------ |
| `thinking_format`                         | `"openai"` (reasoning_effort) vs `"deepseek"` (thinking: { type }) |
| `supports_developer_role`                 | o-series models need `developer` instead of `system`               |
| `requires_reasoning_content_on_assistant` | DeepSeek needs empty `reasoning_content` on replayed messages      |
| `max_tokens_field`                        | `max_completion_tokens` vs `max_tokens`                            |

### Codex Transport

- WebSocket TLS via `tokio-tungstenite` with `rustls-tls-webpki-roots`.
- WS fails → fallback to SSE.
- Don't emit duplicate synthetic `Done(stop)` after parser already emitted `Done(toolUse)`.

## Error Handling

- Use `thiserror` for library errors.
- No panics. Return `Result`.
- No `unwrap()` — use `?` or explicit error handling.
- Comments follow root `AGENTS.md` Comment Style rules.

## Testing

- Faux provider available for testing without real APIs.
- Token estimation tests must not depend on exact model tokenizers.
