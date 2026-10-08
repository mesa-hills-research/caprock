# Datasets

What mhr_vector is measured on, why, and how the Voyage sets are built. Every set is a
folder in the raw format of `crates/mhr_vector_oracle/src/dataset.rs`, under
`$MHRV_DATASETS` (default `datasets/`).

| Set | Content | Vectors | Queries | Used by |
|---|---|---|---|---|
| `code-voyagecode4-200k` | arko_index chunks of 12 code checkouts in 8 languages, voyage-code-4, 1024 dims | 199,999 | 2,000 doc-comment sentences | the score, guards, full suite |
| `docs-voyage4-200k` | arko_index chunks of the Cloudflare docs and MDN, voyage-4, 1024 dims | 34,601 (all there is) | 16 Cloudflare evaluation questions, 1,984 page titles and headings | the score, guards, full suite |
| `code-voyage4-1024` | arko_index chunks of 11 small pinned repositories, voyage-code-4 | 11,467 | 1,000 doc-comment sentences | guards (searched flat), full suite |
| `glove-100-angular`, `nytimes-256-angular` | ann-benchmarks | 1.18M, 290k | 10,000 | guards (the hard sets) |
| `sift-128-euclidean` | ann-benchmarks | 1M | 10,000 | guards (L2) |
| `synth-1024-6m` | `mhr_vector_bench synth`, clustered | 6M | 1,000 | full suite (scale, memory) |

## Why the score uses Voyage sets

The score switched to Voyage-only data on 2026-10-07 (score-v2). arko_index embeds with
Voyage, voyage-code-4 for code and voyage-4 for documentation text, so the store is tuned
on the vectors it will actually hold. Code and documentation embeddings spread
differently, and a score over both keeps an optimization from fitting one of them. The
two parts combine by geometric mean: a 10% gain on either part moves the score by the
same amount, and a part that misses its recall target makes the whole score 0.

score-v1 used 500,000 OpenAI text-embedding-3-large vectors of DBpedia cut to 1024
dimensions (`dbp3l-1024`) as a stand-in. Its download script, `scripts/fetch_openai3.py`,
is in the history at c8ccae9.

score-v2 measured the default Vamana configuration (`vamana-int8`) at the smallest beam
reaching recall@10 0.95. score-v3 (2026-10-08) measures flat search with int8 codes
(`flat-int8`), the configuration arko_index uses by default, on the same two sets.

The Cloudflare docs and MDN hold 34,601 chunks between them, so the docs set stays well
under its 200,000 target. The name records the target.

## Sources

Code checkouts (under `$REPOS`), listed in `scripts/sources/code.json`:

| Source | Checkout | Kept | Also excluded |
|---|---|---|---|
| kubernetes | `kubernetes__kubernetes` | Go | |
| elasticsearch | `elastic__elasticsearch` | Java | |
| typescript | `microsoft__TypeScript` | TypeScript | `tests/baselines/` (generated compiler output) |
| postgres | `postgres__postgres` | C | |
| redis | `redis__redis` | C | `deps/` (vendored libraries) |
| opencv | `opencv__opencv` | C++ | |
| django | `django__django` | Python | |
| react | `facebook__react` | JavaScript | |
| wordpress | `WordPress__wordpress-develop` | JavaScript | |
| tokio, bevy | `tokio-rs__tokio`, `bevyengine__bevy` | Rust | |
| protobuf | `protocolbuffers__protobuf` | all 8 languages | |

The 8 languages are arko_index's grammars: Rust, Python, TypeScript, JavaScript, Go,
Java, C and C++. arko_index files `.h` headers under C, and PHP is plain text to it, so
WordPress contributes its JavaScript. Every source also skips `node_modules/`, `vendor/`,
`third_party/`, `3rdparty/`, `*.min.js` and gitignored files.

Documentation, listed in `scripts/sources/docs.json`:

- **cloudflare**: `cloudflare/cloudflare-docs` at 791e4a8 (`production`,
  `src/content/docs`), a copy of that folder at `$CF_DOCS/docs`, its `*.md` and
  `*.mdx` files. Legal and license pages are left out, as in arko's earlier index of the
  same folder: 6,872 files, 11,626 chunks.
- **mdn**: `mdn__content`, `files/en-us/**/*.md` (about 14,700 pages).

## How a set is built

`scripts/build_voyage_set.py` does it in four steps.

1. **Chunks.** `arko_index_cli chunk --respect-ignore` on every source, the chunker arko
   itself uses. A chunk's text is its path relative to the source folder, a newline, and
   its bytes (`path-newline-source/1`). Documents are embedded with input type
   `document`, queries with `query`.
2. **Selection** of `--target-chunks N` with `--seed S`. Code chunks fall into one
   stratum per language and documentation chunks into one per source. N is shared
   equally among the strata. A stratum with fewer chunks than its share keeps them all
   and leaves the rest to the others (water-filling), and each stratum's quota is shared
   among its sources the same way. Files are taken whole, in the order of
   SHA-256(seed, source, path), skipping any file that would pass the quota. The base
   rows follow that hash order across all sources with each file's chunks together, so
   any prefix of the base is a fair sample (the updates guard loads a prefix).
3. **Queries.** A question file (`docs-questions:<file>`) goes in whole and first.
   Doc comments (`code-doc-comments`: the first sentence of a chunk's leading comment or
   docstring, five words or more, a symbol phrase when there is none) and headings
   (`docs-headings`: front-matter titles and Markdown headings of 3 to 20 words, outside
   fenced code) fill the rest. Each stratum gives queries in proportion to its share of
   the base, in seeded-hash order, and a text already taken is skipped.
4. **Embedding.** Requests of at most 1,000 texts and 280,000 bytes, four at a time.
   Every vector is cached by SHA-256 of its text, so an interrupted run bills nothing
   twice. `--max-tokens` caps what the set may bill, checked against an estimate (bytes
   divided by `--bytes-per-token`, 3.0) before anything is sent and against Voyage's
   reported usage while running. A ledger in `<work>/billing/` keeps the total across
   runs. Caps above 3,000,000 tokens need `--allow-over`.

A dry run (`--dry-run`) stops after step 3. It prints the counts per stratum and source,
the estimated tokens and cost, and whether the cap covers them, and it writes the plan
(every base row and query) to `<work>/plans/`.

`dataset.json` records the model, the dimensions, the input types, the text format, the
tokens billed (in total and in this run), the selection rule and its statistics, the
query specs, and per source its folder, commit (`git rev-parse HEAD`, `git describe`,
whether tracked files changed), declared commit, file rules and counts. A source outside
git, or with changes, also gets a SHA-256 manifest of its chunked files. The chunker's
path and commit and the full command are there too.

## Cost

| Set | Chunks | Tokens billed | Price per million | Cost at standard rates |
|---|---|---|---|---|
| code-voyagecode4-200k | 199,999 | 35.5M | $0.12 (voyage-code-4) | $4.26 |
| docs-voyage4-200k | 34,601 | 22.5M | $0.06 (voyage-4) | $1.35 |

The bills include the queries. Before any call the builders estimate tokens at 3 bytes per
token, which came out about 35% above these bills, and `--max-tokens` caps each set (55M
tokens for code and 70M for docs in `scripts/voyage_sets.sh`). voyage-code-4 comes with
200 million free tokens per account.

## Building them

```sh
scripts/voyage_sets.sh --dry-run             # counts, estimates, plans
VOYAGE_API_KEY=... scripts/voyage_sets.sh    # embed both sets
```

The script needs `REPOS` (the code checkouts and `mdn__content`), `CF_DOCS` (the
Cloudflare docs) and `ARKO_CLI` (arko_index's chunker). `scripts/setup_datasets.sh`
computes the score's and guards' ground truth once the sets exist.
`build_voyage_set.py --help` lists every option.
