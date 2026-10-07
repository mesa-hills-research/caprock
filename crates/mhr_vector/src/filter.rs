//! Filters (unit U04; REQUIREMENTS.md, "Filters").
//!
//! A record matches a [`Filter`] when its path matches **any** of the path prefixes (or
//! the list is empty) **and** its language equals **any** of the languages (or that list
//! is empty). Prefixes match whole path components: `src/foo` matches `src/foo` and
//! `src/foo/bar.rs`, never `src/foobar.rs`; a trailing `/` is ignored and an empty prefix
//! matches every path. Comparison is on bytes.
//!
//! Sealed segments order their records by (path, key), so the records matching a prefix
//! form at most two ordinal ranges (the path equal to the prefix, and the paths under
//! `prefix/`). [`SegmentFilter`] compiles a filter against a segment's [`PathTable`] into
//! ranges and a language set.

use std::ops::Range;

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

    /// Records under one path prefix.
    pub fn path(prefix: &str) -> Filter {
        Filter {
            path_prefixes: vec![prefix.to_string()],
            languages: Vec::new(),
        }
    }

    /// Records of one language.
    pub fn language(language: &str) -> Filter {
        Filter {
            path_prefixes: Vec::new(),
            languages: vec![language.to_string()],
        }
    }

    /// Whether the filter restricts nothing.
    pub fn is_empty(&self) -> bool {
        self.path_prefixes.is_empty() && self.languages.is_empty()
    }

    /// Whether a record with `path` and `language` matches.
    pub fn matches(&self, path: &str, language: &str) -> bool {
        todo!("U04 filter: Filter::matches")
    }
}

/// Component-wise prefix match (see the module docs).
pub fn path_has_prefix(path: &str, prefix: &str) -> bool {
    todo!("U04 filter: path_has_prefix")
}

/// What a sealed segment exposes for filter compilation. Its ordinals are sorted by
/// (path, key).
pub trait PathTable {
    /// Number of ordinals.
    fn ordinals(&self) -> u32;
    /// The (at most two) ordinal ranges whose paths match `prefix`, in increasing order,
    /// non-empty and disjoint.
    fn prefix_ranges(&self, prefix: &str) -> Vec<Range<u32>>;
    /// The segment's id for `language`, if any record has it.
    fn language_id(&self, language: &str) -> Option<u16>;
    /// The language id of an ordinal.
    fn language_of(&self, ordinal: u32) -> u16;
}

/// A filter compiled against one segment: sorted, disjoint ordinal ranges (the path
/// part) and an optional set of language ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentFilter {
    /// Sorted, disjoint, non-empty ranges; all ordinals when the filter has no prefixes.
    pub ranges: Vec<Range<u32>>,
    /// `None`: any language. `Some(ids)`, sorted: one of these (empty: nothing matches).
    pub languages: Option<Vec<u16>>,
}

impl SegmentFilter {
    /// Compiles `filter` for a segment.
    pub fn compile(filter: &Filter, table: &dyn PathTable) -> SegmentFilter {
        todo!("U04 filter: SegmentFilter::compile")
    }

    /// Whether ordinal `ordinal` matches.
    pub fn allows(&self, ordinal: u32, table: &dyn PathTable) -> bool {
        todo!("U04 filter: SegmentFilter::allows")
    }

    /// The number of matching ordinals (live or not), exactly.
    pub fn count(&self, table: &dyn PathTable) -> u32 {
        todo!("U04 filter: SegmentFilter::count")
    }

    /// Whether nothing can match.
    pub fn is_empty(&self) -> bool {
        todo!("U04 filter: SegmentFilter::is_empty")
    }
}
