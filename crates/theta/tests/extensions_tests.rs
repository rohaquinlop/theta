use async_trait::async_trait;
use std::sync::Arc;
use theta::extensions::{Extension, ExtensionContext, ExtensionRegistry};
use theta::tools::ToolContext;

struct TestExtension;

#[async_trait]
impl Extension for TestExtension {
    fn name(&self) -> &str {
        "test-ext"
    }
    fn version(&self) -> &str {
        "0.1.0"
    }
}

#[tokio::test]
async fn test_registry_startup_shutdown() {
    let mut reg = ExtensionRegistry::new();
    reg.register(Arc::new(TestExtension));
    assert_eq!(reg.len(), 1);

    let ctx = ExtensionContext {
        working_dir: std::path::PathBuf::from("."),
        tool_context: ToolContext::new(std::path::PathBuf::from(".")),
    };

    reg.startup(&ctx).await.unwrap();
    reg.shutdown(&ctx).await.unwrap();
}
