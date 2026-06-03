// Consolidated tests: catalog, openai, codex, deepseek model validations.
// Merged to reduce test binary count and improve test suite startup time.

mod catalog {
    use michin_ai::{ModelCatalog, Provider};
    use michin_models::BuiltInCatalog;

    #[test]
    fn test_catalog_has_models() {
        let catalog = BuiltInCatalog::new();
        let all = catalog.list();
        assert!(!all.is_empty(), "Catalog should have models");

        let openai_models = catalog.list_by_provider(Provider::OpenAI);
        assert!(!openai_models.is_empty(), "Should have OpenAI models");

        let codex_models = catalog.list_by_provider(Provider::OpenAiCodex);
        assert!(!codex_models.is_empty(), "Should have Codex models");

        let deepseek_models = catalog.list_by_provider(Provider::DeepSeek);
        assert!(!deepseek_models.is_empty(), "Should have DeepSeek models");
    }

    #[test]
    fn test_find_model() {
        let catalog = BuiltInCatalog::new();
        let gpt55 = catalog.find(Provider::OpenAI, "gpt-5.5");
        assert!(gpt55.is_some(), "gpt-5.5 should exist");
        assert_eq!(gpt55.unwrap().base_url, "https://api.openai.com");
    }

    #[test]
    fn test_find_nonexistent() {
        let catalog = BuiltInCatalog::new();
        assert!(catalog.find(Provider::OpenAI, "nonexistent").is_none());
    }
}

mod openai_models {
    use michin_ai::Provider;
    use michin_models::openai;

    #[test]
    fn test_all_models_valid() {
        for m in openai::models() {
            assert!(!m.id.is_empty());
            assert_eq!(m.provider, Provider::OpenAI);
            assert_eq!(m.base_url, "https://api.openai.com");
            assert!(m.context_window > 0);
            assert!(m.max_tokens > 0);
        }
    }
}

mod codex_models {
    use michin_ai::{Api, Provider};
    use michin_models::codex;

    #[test]
    fn test_all_models_valid() {
        for m in codex::models() {
            assert!(!m.id.is_empty());
            assert_eq!(m.provider, Provider::OpenAiCodex);
            assert_eq!(m.api, Api::OpenAiCodexResponses);
            assert_eq!(m.base_url, "https://chatgpt.com/backend-api");
            assert!(m.context_window > 0);
            assert!(m.max_tokens > 0);
        }
    }
}

mod deepseek_models {
    use michin_ai::{Provider, ThinkingLevel};
    use michin_models::deepseek;

    #[test]
    fn test_all_models_valid() {
        for m in deepseek::models() {
            assert!(!m.id.is_empty());
            assert_eq!(m.provider, Provider::DeepSeek);
            assert_eq!(m.base_url, "https://api.deepseek.com");
            assert!(m.context_window > 0);
            assert!(m.max_tokens > 0);
            assert!(
                m.requires_reasoning_on_replay(),
                "DeepSeek models must strip reasoning content on replay"
            );
        }
    }

    #[test]
    fn test_all_have_thinking_map() {
        for m in deepseek::models() {
            assert!(m.reasoning);
            assert!(
                m.thinking_param(ThinkingLevel::Off).is_none(),
                "Off should be None"
            );
            assert!(
                m.thinking_param(ThinkingLevel::High).is_some(),
                "High should have a value"
            );
        }
    }
}
