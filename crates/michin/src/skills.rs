//! Skills system: discover and load Markdown skill files with YAML frontmatter.
//!
//! Compatible with Pi's SKILL.md format. Skills are discovered from:
//! - `~/.michin/skills/` (global)
//! - `.michin/skills/` (project-local)
//!
//! Each skill is a Markdown file with YAML frontmatter between `---` delimiters.
//! The frontmatter must contain `name` and `description` fields.

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A loaded skill from a SKILL.md file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Skill {
    /// Unique skill identifier (from frontmatter `name`).
    pub name: String,
    /// Human-readable description for the LLM (from frontmatter `description`).
    pub description: String,
    /// Where the skill file lives on disk.
    pub location: PathBuf,
    /// The full Markdown body (everything after the frontmatter).
    pub body: String,
    /// Extra frontmatter fields as key-value pairs.
    #[serde(default)]
    pub extra: HashMap<String, serde_json::Value>,
}

/// YAML frontmatter parsed from a skill file.
#[derive(Debug, Clone, Deserialize)]
struct SkillFrontmatter {
    name: String,
    description: String,
    #[serde(flatten)]
    extra: HashMap<String, serde_json::Value>,
}

impl Skill {
    /// Parse a SKILL.md file. Returns `None` if the file has no valid frontmatter.
    pub fn from_file(path: &Path) -> Option<Self> {
        let content = std::fs::read_to_string(path).ok()?;
        let trimmed = content.trim_start();
        if !trimmed.starts_with("---") {
            return None;
        }

        let body_after_first_delim = &trimmed[3..];
        let end_idx = body_after_first_delim.find("\n---")?;
        let yaml_str = &body_after_first_delim[..end_idx];
        let body = body_after_first_delim[end_idx + 4..]
            .trim_start()
            .to_string();

        let fm: SkillFrontmatter = serde_yaml::from_str(yaml_str).ok()?;

        Some(Skill {
            name: fm.name,
            description: fm.description,
            location: path.to_path_buf(),
            body,
            extra: fm.extra,
        })
    }

    /// Build the `<available_skill>` XML block for the system prompt.
    pub fn to_prompt_block(&self) -> String {
        format!(
            r#"  <skill>
    <name>{name}</name>
    <description>{desc}</description>
    <location>{loc}</location>
  </skill>"#,
            name = self.name,
            desc = self.description,
            loc = self.location.display(),
        )
    }
}

/// Discover all skills from global and project directories.
///
/// Search roots:
/// - ~/.agents/skills
/// - ~/.michin/skills
/// - ./.agents/skills
/// - ./.michin/skills
pub async fn discover_skills(working_dir: &Path) -> Vec<Skill> {
    let mut skills = Vec::new();
    let mut seen_names: HashSet<String> = HashSet::new();

    let mut roots = Vec::new();
    if let Some(home) = dirs::home_dir() {
        roots.push(home.join(".agents").join("skills"));
        roots.push(home.join(".michin").join("skills"));
    }
    roots.push(working_dir.join(".agents").join("skills"));
    roots.push(working_dir.join(".michin").join("skills"));

    for root in roots {
        load_skills_from_dir_recursive(&root, &mut skills, &mut seen_names);
    }

    skills
}

/// Recursively load skills.
/// If a directory contains SKILL.md, treat that directory as one skill root and do not recurse deeper.
fn load_skills_from_dir_recursive(
    dir: &Path,
    skills: &mut Vec<Skill>,
    seen_names: &mut HashSet<String>,
) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.filter_map(|e| e.ok()) {
        let path = entry.path();
        if path.is_dir() {
            let skill_md = path.join("SKILL.md");
            if skill_md.exists() {
                if let Some(skill) = Skill::from_file(&skill_md)
                    && seen_names.insert(skill.name.clone())
                {
                    skills.push(skill);
                }
                continue;
            }
            load_skills_from_dir_recursive(&path, skills, seen_names);
            continue;
        }

        if path.file_name().map(|n| n == "SKILL.md").unwrap_or(false)
            && let Some(skill) = Skill::from_file(&path)
            && seen_names.insert(skill.name.clone())
        {
            skills.push(skill);
        }
    }
}

/// Build the `<available_skills>` block for injection into the system prompt.
pub fn build_skills_prompt_block(skills: &[Skill]) -> Option<String> {
    if skills.is_empty() {
        return None;
    }

    let mut block = String::from("\n<available_skills>\n");

    for skill in skills {
        block.push_str(&skill.to_prompt_block());
        block.push('\n');
    }
    block.push_str("</available_skills>");

    Some(block)
}
