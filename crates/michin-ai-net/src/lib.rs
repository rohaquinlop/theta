//! Network-backed LLM provider implementations for MichiN.
//!
//! Contains the OpenAI-compatible and Codex providers along with
//! registry helpers. Separated from `michin-ai` so that crates that
//! only need types/traits don't pull in ring/rustls/reqwest.

pub mod providers;

pub use providers::default_registry;
pub use providers::openai_codex::OpenAiCodexProvider;
pub use providers::openai_compat::OpenAiCompatProvider;
