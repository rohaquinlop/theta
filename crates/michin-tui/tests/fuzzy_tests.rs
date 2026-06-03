use michin_tui::components::fuzzy::{fuzzy_filter, fuzzy_match};

#[test]
fn test_exact_match() {
    let m = fuzzy_match("main", "main");
    assert!(m.matches);
    assert!(m.score < 0.0);
}

#[test]
fn test_subsequence() {
    let m = fuzzy_match("mn", "main");
    assert!(m.matches);
}

#[test]
fn test_no_match() {
    let m = fuzzy_match("xyz", "main");
    assert!(!m.matches);
}

#[test]
fn test_word_boundary_bonus() {
    let m1 = fuzzy_match("sr", "src/main.rs");
    let m2 = fuzzy_match("sr", "asr");
    assert!(m1.matches && m2.matches);
    assert!(m1.score < m2.score);
}

#[test]
fn test_consecutive_bonus() {
    let m_cons = fuzzy_match("abc", "abc");
    let m_gap = fuzzy_match("ac", "abc");
    assert!(m_cons.matches && m_gap.matches);
    assert!(m_cons.score < m_gap.score);
}

#[test]
fn test_filter() {
    let items = [
        ("src/main.rs", "source"),
        ("src/lib.rs", "library"),
        ("Cargo.toml", "config"),
    ];
    let result = fuzzy_filter(&items, "main", |(name, _)| name);
    assert_eq!(result.len(), 1);
    assert_eq!(result[0].0, "src/main.rs");
}
