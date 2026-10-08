//! Filters (unit U04; docs/requirements.md, "Filters").
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
        (self.path_prefixes.is_empty()
            || self
                .path_prefixes
                .iter()
                .any(|prefix| path_has_prefix(path, prefix)))
            && (self.languages.is_empty() || self.languages.iter().any(|name| name == language))
    }
}

/// Component-wise prefix match (see the module docs).
pub fn path_has_prefix(path: &str, prefix: &str) -> bool {
    // Strip exactly one trailing slash, without normalizing any other components.
    let prefix = prefix.strip_suffix('/').unwrap_or(prefix);
    prefix.is_empty()
        || path == prefix
        || (path.starts_with(prefix) && path.as_bytes().get(prefix.len()) == Some(&b'/'))
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
        let mut ranges = Vec::new();
        if filter.path_prefixes.is_empty() {
            let end = table.ordinals();
            if end != 0 {
                ranges.push(0..end);
            }
        } else {
            for prefix in &filter.path_prefixes {
                ranges.extend(table.prefix_ranges(prefix));
            }
            ranges.sort_unstable_by_key(|range| (range.start, range.end));
        }

        // Prefixes can be repeated or nested. Merge their union, including adjacent
        // ranges, so membership and counting never visit an ordinal twice.
        let mut merged: Vec<Range<u32>> = Vec::with_capacity(ranges.len());
        for range in ranges {
            if range.is_empty() {
                continue;
            }
            match merged.last_mut() {
                Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
                _ => merged.push(range),
            }
        }

        let languages = if filter.languages.is_empty() {
            None
        } else {
            let mut ids: Vec<u16> = filter
                .languages
                .iter()
                .filter_map(|language| table.language_id(language))
                .collect();
            ids.sort_unstable();
            ids.dedup();
            Some(ids)
        };
        SegmentFilter {
            ranges: merged,
            languages,
        }
    }

    /// Whether ordinal `ordinal` matches.
    pub fn allows(&self, ordinal: u32, table: &dyn PathTable) -> bool {
        let index = self.ranges.partition_point(|range| range.end <= ordinal);
        if !self
            .ranges
            .get(index)
            .is_some_and(|range| range.start <= ordinal)
        {
            return false;
        }
        match &self.languages {
            None => true,
            Some(ids) => !ids.is_empty() && ids.binary_search(&table.language_of(ordinal)).is_ok(),
        }
    }

    /// The number of matching ordinals (live or not), exactly.
    pub fn count(&self, table: &dyn PathTable) -> u32 {
        match &self.languages {
            None => self
                .ranges
                .iter()
                .map(|range| range.end - range.start)
                .sum(),
            Some(ids) if ids.is_empty() => 0,
            Some(ids) => self
                .ranges
                .iter()
                .flat_map(|range| range.clone())
                .map(|ordinal| u32::from(ids.binary_search(&table.language_of(ordinal)).is_ok()))
                .sum(),
        }
    }

    /// Whether nothing can match.
    pub fn is_empty(&self) -> bool {
        self.ranges.is_empty() || self.languages.as_ref().is_some_and(Vec::is_empty)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Table(&'static [(&'static str, u16)]);

    impl PathTable for Table {
        fn ordinals(&self) -> u32 {
            self.0.len() as u32
        }

        fn prefix_ranges(&self, prefix: &str) -> Vec<Range<u32>> {
            let mut ranges: Vec<Range<u32>> = Vec::new();
            for (ordinal, (path, _)) in self.0.iter().enumerate() {
                if path_has_prefix(path, prefix) {
                    let ordinal = ordinal as u32;
                    match ranges.last_mut() {
                        Some(last) if last.end == ordinal => last.end += 1,
                        _ => ranges.push(ordinal..ordinal + 1),
                    }
                }
            }
            ranges
        }

        fn language_id(&self, language: &str) -> Option<u16> {
            let id = match language {
                "rust" => u16::MAX,
                "go" => 0,
                _ => return None,
            };
            self.0.iter().any(|&(_, value)| value == id).then_some(id)
        }

        fn language_of(&self, ordinal: u32) -> u16 {
            self.0[ordinal as usize].1
        }
    }

    #[test]
    fn only_one_trailing_slash_is_ignored() {
        assert!(path_has_prefix("x/", "x//"));
        assert!(path_has_prefix("x//child", "x//"));
        assert!(!path_has_prefix("x/child", "x//"));
        assert!(!path_has_prefix("x", "x//"));
        assert!(!path_has_prefix("x", "//"));
        assert!(path_has_prefix("//x", "//"));
        assert!(!path_has_prefix("éclair", "é"));
        assert!(path_has_prefix("é/child", "é"));
    }

    #[test]
    fn compilation_unions_ranges_and_deduplicates_ids() {
        let table = Table(&[
            ("x", 0),
            ("x", u16::MAX),
            ("x-", 0),
            ("x.", 0),
            ("x/a", u16::MAX),
            ("x/a/z", 0),
            ("x/b", u16::MAX),
            ("y", 0),
        ]);
        let filter = Filter {
            path_prefixes: vec!["x/a".into(), "y".into(), "x".into(), "x".into()],
            languages: vec!["rust".into(), "go".into(), "absent".into(), "rust".into()],
        };
        let mut compiled = SegmentFilter::compile(&filter, &table);
        assert_eq!(compiled.ranges, vec![0..2, 4..8]);
        assert_eq!(compiled.languages, Some(vec![0, u16::MAX]));
        assert_eq!(compiled.count(&table), 6);
        assert!(!compiled.is_empty());
        for ordinal in 0..table.ordinals() {
            assert_eq!(compiled.allows(ordinal, &table), !(2..4).contains(&ordinal));
        }
        assert!(!compiled.allows(table.ordinals(), &table));
        assert!(!compiled.allows(u32::MAX, &table));

        compiled.languages = Some(vec![u16::MAX]);
        assert_eq!(compiled.count(&table), 3);
        for ordinal in 0..table.ordinals() {
            assert_eq!(
                compiled.allows(ordinal, &table),
                [1, 4, 6].contains(&ordinal)
            );
        }
    }

    #[test]
    fn empty_tables_and_missing_filters_match_nothing() {
        let empty = Table(&[]);
        let one = Table(&[("x", 0)]);
        for (table, filter) in [
            (&empty, Filter::all()),
            (&empty, Filter::path("/")),
            (&empty, Filter::language("go")),
            (&one, Filter::path("absent")),
            (&one, Filter::language("absent")),
        ] {
            let compiled = SegmentFilter::compile(&filter, table);
            assert!(compiled.is_empty());
            assert_eq!(compiled.count(table), 0);
            assert!(!compiled.allows(0, table));
        }
        let all = SegmentFilter::compile(&Filter::all(), &one);
        assert!(!all.is_empty());
        assert_eq!(all.count(&one), 1);
        assert!(all.allows(0, &one));
    }
}
