//! Always-on command safety policy engine.
//!
//! Checks tool calls for destructive operations regardless of "mode".
//! The system prompt guides the model on when to use tools; the command
//! policy is the safety net that blocks truly dangerous operations
//! (rm -rf, git push --force, destructive sed, etc.).

use crate::types::{SafetyDecisionKind, ToolCall};

#[derive(Debug, Clone)]
pub struct SafetyDecision {
    pub decision: SafetyDecisionKind,
    pub details: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthorizationClass {
    FileMutation,
    VcsMutation,
    Commit,
    DependencyMutation,
}

#[derive(Debug, Clone)]
pub struct CommandSegment {
    pub raw: String,
    pub argv: Vec<String>,
}

/// Evaluate a tool call for safety. Returns Allowed or Rejected.
///
/// The command policy is always-on — it does not depend on turn modes.
/// The system prompt is responsible for guiding the model on *when* to
/// use mutation tools vs read-only tools. The policy only blocks
/// operations that are inherently dangerous regardless of context.
pub fn evaluate_tool_call(tc: &ToolCall, strict: bool) -> SafetyDecision {
    match tc.name.as_str() {
        "bash" => evaluate_bash(tc, strict),
        _ => SafetyDecision {
            decision: SafetyDecisionKind::Allowed,
            details: format!("tool '{}' allowed", tc.name),
        },
    }
}

pub fn required_user_authorization(tc: &ToolCall) -> Option<AuthorizationClass> {
    match tc.name.as_str() {
        "write" | "edit" => Some(AuthorizationClass::FileMutation),
        "bash" => classify_bash_authorization(tc),
        _ => None,
    }
}

fn classify_bash_authorization(tc: &ToolCall) -> Option<AuthorizationClass> {
    let command = tc.arguments.get("command").and_then(|v| v.as_str())?;
    let segments = parse_command_segments(command);
    let mut saw_file_mutation = false;
    let mut saw_vcs_mutation = false;
    let mut saw_dependency_mutation = false;
    for segment in &segments {
        let argv = segment
            .argv
            .iter()
            .map(|s| s.to_lowercase())
            .collect::<Vec<_>>();
        if argv.is_empty() {
            continue;
        }
        let toks = argv.iter().map(String::as_str).collect::<Vec<_>>();
        if contains_token_sequence(&toks, &["git", "commit"]) {
            return Some(AuthorizationClass::Commit);
        }
        if is_vcs_mutating_command(&toks) {
            saw_vcs_mutation = true;
        }
        if is_dependency_mutating_command(&toks) {
            saw_dependency_mutation = true;
        }
        if is_file_mutating_command(&toks) {
            saw_file_mutation = true;
        }
    }
    if saw_vcs_mutation {
        return Some(AuthorizationClass::VcsMutation);
    }
    if saw_dependency_mutation {
        return Some(AuthorizationClass::DependencyMutation);
    }
    if saw_file_mutation {
        return Some(AuthorizationClass::FileMutation);
    }
    None
}

fn is_vcs_mutating_command(tokens: &[&str]) -> bool {
    contains_token_sequence(tokens, &["git", "push"])
        || contains_token_sequence(tokens, &["git", "merge"])
        || contains_token_sequence(tokens, &["git", "rebase"])
        || contains_token_sequence(tokens, &["git", "reset"])
        || contains_token_sequence(tokens, &["git", "revert"])
        || contains_token_sequence(tokens, &["git", "cherry", "pick"])
        || contains_token_sequence(tokens, &["git", "stash"])
        || contains_token_sequence(tokens, &["git", "checkout"])
        || contains_token_sequence(tokens, &["git", "switch"])
        || contains_token_sequence(tokens, &["git", "branch"])
        || contains_token_sequence(tokens, &["git", "tag"])
        || contains_token_sequence(tokens, &["git", "worktree", "add"])
        || contains_token_sequence(tokens, &["git", "add"])
        || contains_token_sequence(tokens, &["git", "rm"])
        || contains_token_sequence(tokens, &["git", "mv"])
}

fn is_dependency_mutating_command(tokens: &[&str]) -> bool {
    contains_token_sequence(tokens, &["cargo", "add"])
        || contains_token_sequence(tokens, &["cargo", "install"])
        || contains_token_sequence(tokens, &["npm", "install"])
        || contains_token_sequence(tokens, &["npm", "i"])
        || contains_token_sequence(tokens, &["npm", "add"])
        || contains_token_sequence(tokens, &["pnpm", "add"])
        || contains_token_sequence(tokens, &["pnpm", "install"])
        || contains_token_sequence(tokens, &["yarn", "add"])
        || contains_token_sequence(tokens, &["yarn", "install"])
        || contains_token_sequence(tokens, &["bun", "add"])
        || contains_token_sequence(tokens, &["bun", "install"])
        || contains_token_sequence(tokens, &["pip", "install"])
        || contains_token_sequence(tokens, &["uv", "add"])
}

fn is_file_mutating_command(tokens: &[&str]) -> bool {
    contains_token_sequence(tokens, &["rm"])
        || contains_token_sequence(tokens, &["mv"])
        || contains_token_sequence(tokens, &["cp"])
        || contains_token_sequence(tokens, &["mkdir"])
        || contains_token_sequence(tokens, &["rmdir"])
        || contains_token_sequence(tokens, &["touch"])
        || contains_token_sequence(tokens, &["truncate"])
        || contains_token_sequence(tokens, &["chmod"])
        || contains_token_sequence(tokens, &["chown"])
        || contains_token_sequence(tokens, &["ln"])
        || contains_token_sequence(tokens, &["sed", "-i"])
        || contains_token_sequence(tokens, &["patch"])
        || contains_token_sequence(tokens, &["tee"])
}

fn contains_token_sequence(tokens: &[&str], phrase_tokens: &[&str]) -> bool {
    if phrase_tokens.is_empty() || tokens.is_empty() || phrase_tokens.len() > tokens.len() {
        return false;
    }
    tokens
        .windows(phrase_tokens.len())
        .any(|w| w.iter().copied().eq(phrase_tokens.iter().copied()))
}

fn evaluate_bash(tc: &ToolCall, strict: bool) -> SafetyDecision {
    let Some(command) = tc.arguments.get("command").and_then(|v| v.as_str()) else {
        return SafetyDecision {
            decision: SafetyDecisionKind::Rejected,
            details: "bash command is missing".to_string(),
        };
    };

    let segments = parse_command_segments(command);
    if segments.is_empty() {
        return SafetyDecision {
            decision: SafetyDecisionKind::Rejected,
            details: "bash command is empty".to_string(),
        };
    }

    for segment in &segments {
        if strict && is_destructive_command(segment) {
            return SafetyDecision {
                decision: SafetyDecisionKind::Rejected,
                details: format!("destructive command blocked: '{}'", segment.raw),
            };
        }
    }

    SafetyDecision {
        decision: SafetyDecisionKind::Allowed,
        details: "bash command allowed".to_string(),
    }
}

fn is_destructive_command(segment: &CommandSegment) -> bool {
    let command = effective_command(segment);
    match command {
        "mkfs" | "shutdown" | "reboot" | "poweroff" | "halt" | "diskutil" => true,
        "rm" => contains_recursive_delete(segment),
        _ => false,
    }
}

fn effective_command(segment: &CommandSegment) -> &str {
    let mut iter = segment.argv.iter().map(String::as_str);
    loop {
        let Some(arg) = iter.next() else {
            return "";
        };
        match arg {
            "sudo" | "command" | "builtin" => continue,
            "env" => {
                for env_arg in iter.by_ref() {
                    if !env_arg.contains('=') {
                        return env_arg;
                    }
                }
                return "";
            }
            _ if arg.contains('=') => continue,
            _ => return arg,
        }
    }
}

fn contains_recursive_delete(segment: &CommandSegment) -> bool {
    let has_recursive_flag = segment
        .argv
        .iter()
        .any(|a| a == "-r" || a == "-rf" || a == "-fr" || (a.starts_with('-') && a.contains('r')));
    if !has_recursive_flag {
        return false;
    }
    let targets = segment
        .argv
        .iter()
        .skip(1)
        .filter(|a| !a.starts_with('-'))
        .collect::<Vec<_>>();
    targets.is_empty()
        || targets
            .iter()
            .any(|t| matches!(t.as_str(), "/" | "~" | "*" | ".*" | ".."))
}

pub fn parse_command_segments(command: &str) -> Vec<CommandSegment> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    let chars: Vec<char> = command.chars().collect();
    let mut i = 0usize;

    while i < chars.len() {
        let ch = chars[i];
        if escaped {
            cur.push(ch);
            escaped = false;
            i += 1;
            continue;
        }
        match ch {
            '\\' => {
                escaped = true;
                cur.push(ch);
            }
            '\'' if !in_double => {
                in_single = !in_single;
                cur.push(ch);
            }
            '"' if !in_single => {
                in_double = !in_double;
                cur.push(ch);
            }
            ';' if !in_single && !in_double => flush_segment(&mut out, &mut cur),
            '|' if !in_single && !in_double => {
                let is_double = chars.get(i + 1) == Some(&'|');
                flush_segment(&mut out, &mut cur);
                if is_double {
                    i += 1;
                }
            }
            '&' if !in_single && !in_double && chars.get(i.wrapping_sub(1)) != Some(&'>') => {
                let is_double = chars.get(i + 1) == Some(&'&');
                flush_segment(&mut out, &mut cur);
                if is_double {
                    i += 1;
                }
            }
            _ => cur.push(ch),
        }
        i += 1;
    }
    flush_segment(&mut out, &mut cur);
    out
}

fn flush_segment(out: &mut Vec<CommandSegment>, cur: &mut String) {
    let raw = cur.trim();
    if raw.is_empty() {
        cur.clear();
        return;
    }
    out.push(CommandSegment {
        raw: raw.to_string(),
        argv: tokenize_segment(raw),
    });
    cur.clear();
}

fn tokenize_segment(segment: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut cur = String::new();
    let mut in_single = false;
    let mut in_double = false;
    let mut escaped = false;
    for ch in segment.chars() {
        if escaped {
            cur.push(ch);
            escaped = false;
            continue;
        }
        match ch {
            '\\' => escaped = true,
            '\'' if !in_double => in_single = !in_single,
            '"' if !in_single => in_double = !in_double,
            c if c.is_whitespace() && !in_single && !in_double => {
                if !cur.is_empty() {
                    tokens.push(cur.clone());
                    cur.clear();
                }
            }
            _ => cur.push(ch),
        }
    }
    if !cur.is_empty() {
        tokens.push(cur);
    }
    tokens
}
