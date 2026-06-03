# michin-models — Agent Rules

> Rules for working on the michin-models crate: built-in model catalog.

## Crate Purpose

Compile-time model definitions plus runtime OpenCode model fetch. Implements the `ModelCatalog` trait. No network calls except for OpenCode Zen dynamic fetch.

## Key Files

| File                         | Purpose                                    |
| ---------------------------- | ------------------------------------------ |
| `src/lib.rs`                 | `BuiltInCatalog` — implements `ModelCatalog` |
| `src/openai.rs`              | Static OpenAI model definitions            |
| `src/deepseek.rs`            | Static DeepSeek model definitions          |
| `src/opencode.rs`            | Dynamic OpenCode Zen fetch + fallback      |
| `src/codex.rs`               | Static Codex model definitions             |
| `src/xiaomi.rs`              | Dynamic Xiaomi MiMo fetch + fallback       |

## Current Models

- **OpenAI**: `gpt-5.5`, `gpt-5.5-instant`, `gpt-5`, `gpt-5-mini`, `gpt-5-nano`, `gpt-5-chat-latest`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4.1-nano`, `gpt-4o`, `gpt-4o-mini`, `o4`, `o4-mini`, `o3`, `o3-mini`, `o1`, `o1-mini`
- **OpenAI Codex**: `gpt-5.5`, `gpt-5.3-codex`, `gpt-5.5-instant`, `gpt-5`, `gpt-5-mini`, `gpt-5-chat-latest`, `gpt-4.1`, `gpt-4.1-mini`, `gpt-4o`, `gpt-4o-mini`, `o4`, `o4-mini`, `o3`, `o3-mini`, `o1`, `o1-mini`
- **DeepSeek**: `deepseek-v4-pro` (1M ctx), `deepseek-v4-flash` (1M ctx)
- **OpenCode Zen**: fetched from `opencode.ai/zen/v1/models` at runtime
- **Xiaomi MiMo**: fetched from `api.xiaomimimo.com/v1/models` at runtime (fallback: `mimo-v2.5-pro`, `mimo-v2-pro`, `mimo-v2.5`, `mimo-v2-omni`, `mimo-v2-flash`)

## Adding a Model

1. Add definition in `michin-models/src/<provider>.rs`
2. Register in `BuiltInCatalog::new()` in `src/lib.rs`
3. If new provider, update `Provider` enum in `michin-ai/src/types.rs`
4. Add env var in `config.rs::provider_env_var()`
5. Update `AGENTS.md` files

## Conventions

- Cost calculation uses `known_cost()` for OpenCode models.
- Free/rate-limited OpenCode models excluded from catalog.
- Token limits must be accurate (used for compaction calculations).
- Comments follow root `AGENTS.md` Comment Style rules.
