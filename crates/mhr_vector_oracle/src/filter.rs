//! Filter semantics (REQUIREMENTS.md, "Filters").
//!
//! A record matches a filter when its path matches **any** of the path prefixes (or the
//! list is empty) **and** its language equals **any** of the languages (or that list is
//! empty). Path prefixes match whole path components: `src/foo` matches `src/foo` and
//! `src/foo/bar.rs`, never `src/foobar.rs`. A trailing `/` on a prefix is ignored, and an
//! empty prefix (or `/`) matches every path. Comparison is on bytes; no other
//! normalization happens.

use serde::{Deserialize, Serialize};

/// A conjunction of a path-prefix disjunction and a language disjunction.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Filter {
    /// Matches when the path matches any of these (empty: any path).
    #[serde(default)]
    pub path_prefixes: Vec<String>,
    /// Matches when the language equals any of these (empty: any language).
    #[serde(default)]
    pub languages: Vec<String>,
}

impl Filter {
    /// The filter that matches everything.
    pub fn all() -> Filter {
        Filter::default()
    }

    /// Whether this filter restricts nothing.
    pub fn is_empty(&self) -> bool {
        self.path_prefixes.is_empty() && self.languages.is_empty()
    }

    /// Whether a record with `path` and `language` matches.
    pub fn matches(&self, path: &str, language: &str) -> bool {
        let path_ok = self.path_prefixes.is_empty()
            || self
                .path_prefixes
                .iter()
                .any(|prefix| path_has_prefix(path, prefix));
        let language_ok = self.languages.is_empty() || self.languages.iter().any(|l| l == language);
        path_ok && language_ok
    }
}

/// Component-wise prefix match (see the module docs).
pub fn path_has_prefix(path: &str, prefix: &str) -> bool {
    let prefix = prefix.strip_suffix('/').unwrap_or(prefix);
    if prefix.is_empty() {
        return true;
    }
    if path == prefix {
        return true;
    }
    path.len() > prefix.len() && path.starts_with(prefix) && path.as_bytes()[prefix.len()] == b'/'
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefixes_match_whole_components() {
        assert!(path_has_prefix("src/foo", "src/foo"));
        assert!(path_has_prefix("src/foo/bar.rs", "src/foo"));
        assert!(path_has_prefix("src/foo/bar.rs", "src/foo/"));
        assert!(!path_has_prefix("src/foobar.rs", "src/foo"));
        assert!(!path_has_prefix("src", "src/foo"));
        assert!(path_has_prefix("anything", ""));
        assert!(path_has_prefix("anything", "/"));
        assert!(!path_has_prefix("lib/src/foo", "src"));
    }

    #[test]
    fn conjunction_of_disjunctions() {
        let f = Filter {
            path_prefixes: vec!["a".into(), "b/c".into()],
            languages: vec!["rust".into(), "go".into()],
        };
        assert!(f.matches("a/x.rs", "rust"));
        assert!(f.matches("b/c/d.go", "go"));
        assert!(!f.matches("b/x.go", "go"));
        assert!(!f.matches("a/x.py", "python"));
        assert!(Filter::all().matches("whatever", "x"));
    }
}
