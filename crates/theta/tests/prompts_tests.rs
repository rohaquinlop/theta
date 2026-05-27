use std::collections::HashMap;
use theta::prompts::PromptTemplate;

#[test]
fn test_template_resolve() {
    let tpl = PromptTemplate {
        name: "test".into(),
        body: "Hello {{name}}, your project is {{project}}.".into(),
    };
    let mut vars = HashMap::new();
    vars.insert("name".into(), "Alice".into());
    vars.insert("project".into(), "Theta".into());

    let resolved = tpl.resolve(&vars);
    assert_eq!(resolved, "Hello Alice, your project is Theta.");
}

#[test]
fn test_template_unresolved() {
    let tpl = PromptTemplate {
        name: "test".into(),
        body: "Hello {{name}}.".into(),
    };
    let vars = HashMap::new();
    let resolved = tpl.resolve(&vars);
    assert_eq!(resolved, "Hello {{name}}.");
}
