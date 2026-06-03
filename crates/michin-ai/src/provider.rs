//! Provider trait and type aliases for streaming LLM calls.

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use super::error::MichiNError;
use super::event::AssistantMessageEvent;
use super::model::Model;
use super::types::{Context, SimpleStreamOptions, StreamOptions};

/// Type alias for a boxed, pinned stream of assistant message events.
pub type EventStream<'a> = Pin<Box<dyn Stream<Item = AssistantMessageEvent> + Send + 'a>>;

/// An LLM provider that can stream completions.
#[async_trait]
pub trait Provider: Send + Sync {
    /// Stream a full conversation with tool calling support.
    async fn stream<'a>(
        &'a self,
        model: &Model,
        context: &Context,
        options: &StreamOptions,
    ) -> Result<EventStream<'a>, MichiNError>;

    /// Stream a simple completion (no tool calling, simpler options).
    async fn stream_simple<'a>(
        &'a self,
        model: &Model,
        context: &Context,
        options: &SimpleStreamOptions,
    ) -> Result<EventStream<'a>, MichiNError>;

    /// Set an authentication token for this provider.
    /// Default is a no-op; providers that need tokens (e.g. Codex OAuth)
    /// override this.
    fn set_token(&self, _token: &str) {}

    /// Return a reference to `self` as `&dyn Any` for downcasting.
    fn as_any(&self) -> &dyn std::any::Any;
}
