//! The README's example: a flat index with int8 codes, two records, a filtered search, a
//! commit and a reader. Run it with
//! `cargo run -p caprock --example quickstart`.

use caprock::{
    AnnKind, Filter, IndexConfig, Metric, Op, Reader, Record, SearchParams, StoreOptions, Writer,
};

fn main() -> caprock::Result<()> {
    let dir = std::env::temp_dir().join("caprock_quickstart");
    let _ = std::fs::remove_dir_all(&dir);

    // int8 codes find candidates, full-precision vectors rescore them.
    let mut config = IndexConfig::new(4, Metric::Cosine);
    config.ann.kind = AnnKind::Flat; // without this, large segments get a Vamana graph
    let mut writer = Writer::create(&dir, config, StoreOptions::default())?;

    writer.apply(&[
        Op::Upsert(Record {
            key: "lib-0",
            vector: &[0.9, 0.1, 0.0, 0.2],
            path: "src/lib.rs",
            language: "rust",
        }),
        Op::Upsert(Record {
            key: "intro-0",
            vector: &[0.8, 0.0, 0.2, 0.1],
            path: "docs/intro.md",
            language: "markdown",
        }),
    ])?;

    // Searchable at once, before any commit.
    let query = [1.0, 0.1, 0.0, 0.1];
    let filter = Filter::path("src");
    let hits = writer
        .snapshot()
        .search(&query, 10, Some(&filter), &SearchParams::default())?;
    for hit in &hits {
        println!("{:.4}  {}  {}", hit.score, hit.key, hit.path);
    }

    // Durable, and visible to readers in other processes.
    writer.commit(None)?;
    let reader = Reader::open(&dir, StoreOptions::default())?;
    assert_eq!(reader.snapshot().len(), 2);
    Ok(())
}
