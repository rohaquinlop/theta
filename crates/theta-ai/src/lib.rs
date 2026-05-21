//! Unified multi-provider LLM API for Theta.
//!
//! Provides types, traits, and a provider registry for
//! streaming LLM requests. Supports OpenAI, DeepSeek, and
//! OpenCode through a single OpenAI-compatible provider.

pub mod error;
pub mod event;
pub mod model;
pub mod provider;
pub mod providers;
pub mod replay;
pub mod types;

pub use error::*;
pub use event::*;
pub use model::*;
pub use replay::*;
pub use types::*;
// Re-export Provider trait with different name to avoid ambiguity with types::Provider
pub use provider::{EventStream, Provider as LlmProvider};
