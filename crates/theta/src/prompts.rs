//! Prompt templates: parameterized instruction files with variable substitution.
//!
//! Templates are Markdown files in `.theta/prompts/` that contain
//! `{{variable}}` placeholders. When loaded with values, the placeholders
//! are replaced.

use std::collections::HashMap;
use std::path::Path;

/// A loaded prompt template.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    /// Template name derived from the filename (without extension).
    pub name: String,
    /// Raw template body with `{{placeholders}}`.
    pub body: String,
}

impl PromptTemplate {
    /// Resolve all `{{variable}}` placeholders with the given values.
    /// Unknown variables are left unchanged.
    pub fn resolve(&self, vars: &HashMap<String, String>) -> String {
        let mut result = self.body.clone();
        for (key, value) in vars {
            let placeholder = format!("{{{{{key}}}}}");
            result = result.replace(&placeholder, value);
        }
        result
    }
}

/// Discover all prompt templates from `.theta/prompts/` (project-local only for now).
pub fn discover_templates(working_dir: &Path) -> Vec<PromptTemplate> {
    let prompts_dir = working_dir.join(".theta").join("prompts");
    let mut templates = Vec::new();

    let entries = match std::fs::read_dir(&prompts_dir) {
        Ok(e) => e,
        Err(_) => return templates,
    };

    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_file() && path.extension().map(|e| e == "md").unwrap_or(false) {
            let name = path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string();

            if let Ok(body) = std::fs::read_to_string(&path) {
                templates.push(PromptTemplate { name, body });
            }
        }
    }

    templates
}
