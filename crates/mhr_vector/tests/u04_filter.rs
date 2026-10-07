//! Acceptance tests of unit U04 (filter): the oracle's semantics, and compilation
//! against a segment's path table.
#![cfg(feature = "internals")]

use std::ops::Range;

use mhr_vector::Filter;
use mhr_vector::filter::{PathTable, SegmentFilter, path_has_prefix};
use mhr_vector_oracle::filter as oracle_filter;
use mhr_vector_oracle::generate::{path_for, standard_filters};

const PREFIXES: [&str; 18] = [
    "",
    "/",
    "p10",
    "p10/",
    "p1",
    "p10/d1",
    "p10/d1/",
    "p10/d17",
    "p50/d7",
    "rest",
    "rest/d3/f0001.rs",
    "zzz",
    "p",
    "p001",
    "p01",
    "é",
    "p10//",
    "src/foo",
];

const PATHS: [&str; 12] = [
    "src/foo",
    "src/foo/bar.rs",
    "src/foobar.rs",
    "src/foo-bar.rs",
    "src/foo.rs",
    "src",
    "lib/src/foo",
    "é/x",
    "p10//x",
    "p10/",
    "",
    "a/b/c",
];

fn to_oracle(f: &Filter) -> oracle_filter::Filter {
    oracle_filter::Filter {
        path_prefixes: f.path_prefixes.clone(),
        languages: f.languages.clone(),
    }
}

#[test]
fn prefix_matching_equals_the_oracle() {
    let mut paths: Vec<String> = PATHS.iter().map(|s| s.to_string()).collect();
    paths.extend((0..300).map(|i| path_for(i, 1).0));
    for p in &paths {
        for prefix in PREFIXES {
            assert_eq!(
                path_has_prefix(p, prefix),
                oracle_filter::path_has_prefix(p, prefix),
                "path {p:?} prefix {prefix:?}"
            );
        }
    }
}

#[test]
fn filters_equal_the_oracle() {
    let mut filters: Vec<Filter> = standard_filters()
        .into_iter()
        .map(|(_, f, _)| Filter {
            path_prefixes: f.path_prefixes,
            languages: f.languages,
        })
        .collect();
    filters.push(Filter::path("p10/d1"));
    filters.push(Filter::language("rust"));
    filters.push(Filter {
        path_prefixes: vec!["p50".into(), "p10".into()],
        languages: vec!["go".into(), "nosuch".into()],
    });
    for i in 0..2000u64 {
        let (path, language) = path_for(i, 3);
        for f in &filters {
            assert_eq!(
                f.matches(&path, &language),
                to_oracle(f).matches(&path, &language),
                "{f:?} on {path} {language}"
            );
        }
    }
    assert!(Filter::all().is_empty());
    assert!(!Filter::path("x").is_empty());
}

/// A path table over records sorted by (path, key), answering by brute force.
struct Table {
    paths: Vec<String>,
    languages: Vec<String>,
    names: Vec<String>,
}

impl Table {
    fn new(n: u64) -> Table {
        let mut records: Vec<(String, String, String)> = (0..n)
            .map(|i| {
                let (p, l) = path_for(i, 5);
                (p, mhr_vector_oracle::generate::key(i), l)
            })
            .collect();
        // A few special paths: a path equal to a prefix and its siblings.
        for (i, p) in ["p10/d1", "p10/d1-x.rs", "p10/d1.rs", "p10/d1/z.go"]
            .iter()
            .enumerate()
        {
            records.push((p.to_string(), format!("special{i}"), "go".into()));
        }
        records.sort();
        let mut names: Vec<String> = records.iter().map(|r| r.2.clone()).collect();
        names.sort();
        names.dedup();
        Table {
            paths: records.iter().map(|r| r.0.clone()).collect(),
            languages: records.iter().map(|r| r.2.clone()).collect(),
            names,
        }
    }
}

impl PathTable for Table {
    fn ordinals(&self) -> u32 {
        self.paths.len() as u32
    }

    fn prefix_ranges(&self, prefix: &str) -> Vec<Range<u32>> {
        let mut out: Vec<Range<u32>> = Vec::new();
        for (i, p) in self.paths.iter().enumerate() {
            if oracle_filter::path_has_prefix(p, prefix) {
                let i = i as u32;
                match out.last_mut() {
                    Some(r) if r.end == i => r.end = i + 1,
                    _ => out.push(i..i + 1),
                }
            }
        }
        assert!(out.len() <= 2, "sorted paths give at most two ranges");
        out
    }

    fn language_id(&self, language: &str) -> Option<u16> {
        self.names
            .iter()
            .position(|n| n == language)
            .map(|p| p as u16)
    }

    fn language_of(&self, ordinal: u32) -> u16 {
        self.language_id(&self.languages[ordinal as usize]).unwrap()
    }
}

#[test]
fn compiled_filters_allow_exactly_the_matching_ordinals() {
    let table = Table::new(3000);
    let mut filters: Vec<Filter> = standard_filters()
        .into_iter()
        .map(|(_, f, _)| Filter {
            path_prefixes: f.path_prefixes,
            languages: f.languages,
        })
        .collect();
    for prefix in PREFIXES {
        filters.push(Filter::path(prefix));
    }
    filters.push(Filter {
        path_prefixes: vec!["p10/d1".into(), "p10".into(), "p50/d7".into()],
        languages: vec![],
    });
    filters.push(Filter::language("nosuch"));
    filters.push(Filter {
        path_prefixes: vec!["p01".into()],
        languages: vec!["go".into(), "rust".into()],
    });
    for f in &filters {
        let compiled = SegmentFilter::compile(f, &table);
        assert!(
            compiled.ranges.windows(2).all(|w| w[0].end <= w[1].start),
            "sorted, disjoint"
        );
        assert!(compiled.ranges.iter().all(|r| r.start < r.end), "non-empty");
        let oracle = to_oracle(f);
        let mut count = 0;
        for o in 0..table.ordinals() {
            let want = oracle.matches(&table.paths[o as usize], &table.languages[o as usize]);
            assert_eq!(compiled.allows(o, &table), want, "{f:?} ordinal {o}");
            count += want as u32;
        }
        assert_eq!(compiled.count(&table), count, "{f:?}");
        if compiled.is_empty() {
            assert_eq!(count, 0, "{f:?}");
        }
        if count == 0 && (f.languages == ["nosuch"] || f.path_prefixes == ["zzz"]) {
            assert!(compiled.is_empty(), "{f:?}");
        }
    }
}
