//! Fuzzy matching via nucleo-matcher — fast SIMD-accelerated matching.
//! Lower score = better match (inverted from nucleo's higher-is-better for
//! backward compat with the previous scoring convention).

use nucleo_matcher::{Config, Matcher, Utf32Str};

/// Result of a fuzzy match.
#[derive(Debug, Clone)]
pub struct FuzzyMatch {
    pub matches: bool,
    pub score: f64,
}

/// Check if `query` fuzzy-matches `text`. Returns match result with score.
/// All query chars must appear in `text` in order (not necessarily consecutive).
/// Rewards: consecutive matches, word-boundary matches, camelCase, path separators.
pub fn fuzzy_match(query: &str, text: &str) -> FuzzyMatch {
    if query.is_empty() {
        return FuzzyMatch {
            matches: true,
            score: 0.0,
        };
    }
    // FILE_PATH config adds bonuses for path separators, underscores, dots, etc.
    let mut matcher = Matcher::new(Config::DEFAULT.match_paths());
    // Separate buffers for needle and haystack to avoid overwrite (Unicode case).
    let mut needle_buf = Vec::new();
    let needle = Utf32Str::new(query, &mut needle_buf);
    let mut haystack_buf = Vec::new();
    let haystack = Utf32Str::new(text, &mut haystack_buf);
    match matcher.fuzzy_match(haystack, needle) {
        Some(score) => FuzzyMatch {
            matches: true,
            // Invert so lower = better (backward compat).
            score: -(score as f64),
        },
        None => FuzzyMatch {
            matches: false,
            score: 0.0,
        },
    }
}

/// Filter and sort items by fuzzy match quality (best first).
/// Supports space-separated tokens: all tokens must match.
pub fn fuzzy_filter<'a, T>(
    items: &'a [T],
    query: &str,
    get_text: impl Fn(&T) -> &str,
) -> Vec<&'a T> {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return items.iter().collect();
    }

    let tokens: Vec<&str> = trimmed
        .split_whitespace()
        .filter(|t| !t.is_empty())
        .collect();
    if tokens.is_empty() {
        return items.iter().collect();
    }

    // Reuse one matcher across all items (scratch buffers reused).
    let mut matcher = Matcher::new(Config::DEFAULT.match_paths());
    let mut results: Vec<(&T, u32)> = Vec::new();

    for item in items {
        let text = get_text(item);
        let mut total: u32 = 0;
        let mut all_match = true;
        for token in &tokens {
            let mut needle_buf = Vec::new();
            let needle = Utf32Str::new(token, &mut needle_buf);
            let mut haystack_buf = Vec::new();
            let haystack = Utf32Str::new(text, &mut haystack_buf);
            match matcher.fuzzy_match(haystack, needle) {
                Some(score) => total += score as u32,
                None => {
                    all_match = false;
                    break;
                }
            }
        }
        if all_match {
            results.push((item, total));
        }
    }

    // nucleo: higher score = better match. Sort descending.
    results.sort_by_key(|b| std::cmp::Reverse(b.1));
    results.into_iter().map(|(item, _)| item).collect()
}
