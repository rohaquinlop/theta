//! Fuzzy matching — matches if all query chars appear in order.
//! Lower score = better match. Mirrors Pi's fuzzy match algorithm.

/// Result of a fuzzy match.
#[derive(Debug, Clone)]
pub struct FuzzyMatch {
    pub matches: bool,
    pub score: f64,
}

/// Check if `query` fuzzy-matches `text`. Returns match result with score.
/// All query chars must appear in `text` in order (not necessarily consecutive).
/// Rewards: consecutive matches, word-boundary matches, exact match.
pub fn fuzzy_match(query: &str, text: &str) -> FuzzyMatch {
    let query_lower = query.to_lowercase();
    let text_lower = text.to_lowercase();
    match_query(&query_lower, &text_lower)
}

fn match_query(query: &str, text: &str) -> FuzzyMatch {
    if query.is_empty() {
        return FuzzyMatch {
            matches: true,
            score: 0.0,
        };
    }
    if query.len() > text.len() {
        return FuzzyMatch {
            matches: false,
            score: 0.0,
        };
    }

    let mut query_idx = 0usize;
    let mut score = 0.0f64;
    let mut last_match: isize = -1;
    let mut consecutive = 0u32;
    let text_bytes = text.as_bytes();

    for (i, tc) in text.char_indices() {
        if query_idx >= query.len() {
            break;
        }
        let qc = query.as_bytes()[query_idx] as char;
        if tc.eq_ignore_ascii_case(&qc) {
            let is_word_boundary = i == 0
                || text_bytes
                    .get(i.wrapping_sub(1))
                    .is_none_or(|&b| matches!(b as char, ' ' | '-' | '_' | '.' | '/' | ':'));

            if last_match == i as isize - 1 {
                consecutive += 1;
                score -= consecutive as f64 * 5.0;
            } else {
                consecutive = 0;
                if last_match >= 0 {
                    score += (i as isize - last_match - 1) as f64 * 2.0;
                }
            }

            if is_word_boundary {
                score -= 10.0;
            }

            score += i as f64 * 0.1;
            last_match = i as isize;
            query_idx += 1;
        }
    }

    if query_idx < query.len() {
        return FuzzyMatch {
            matches: false,
            score: 0.0,
        };
    }

    if query == text {
        score -= 100.0;
    }

    FuzzyMatch {
        matches: true,
        score,
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

    let mut results: Vec<(&T, f64)> = Vec::new();
    for item in items {
        let text = get_text(item);
        let mut total = 0.0;
        let mut all_match = true;
        for token in &tokens {
            let m = fuzzy_match(token, text);
            if m.matches {
                total += m.score;
            } else {
                all_match = false;
                break;
            }
        }
        if all_match {
            results.push((item, total));
        }
    }

    results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    results.into_iter().map(|(item, _)| item).collect()
}
