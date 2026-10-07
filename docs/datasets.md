# Datasets

What mhr_vector is measured on, why, and how the Voyage sets are built. Every set is a
folder in the raw format of `crates/mhr_vector_oracle/src/dataset.rs`, under
`$MHRV_DATASETS` (`/work/mhr-vector/datasets` on the box).

| Set | Content | Vectors | Queries | Used by |
|---|---|---|---|---|
| `code-voyagecode4-200k` | arko_index chunks of 12 code checkouts in 8 languages, voyage-code-4, 1024 dims | ~200,000 | 2,000 doc-comment sentences | the score, guards, full suite |
| `docs-voyage4-200k` | arko_index chunks of the Cloudflare docs and MDN, voyage-4, 1024 dims | ~57,000 (all there is) | 16 Cloudflare evaluation questions, 1,984 page headings | the score, guards, full suite |
| `code-voyage4-1024` | arko_index chunks of 11 small pinned repositories, voyage-code-4 | ~11,500 | 1,000 doc-comment sentences | guards (searched flat), full suite |
| `glove-100-angular`, `nytimes-256-angular` | ann-benchmarks | 1.18M, 290k | 10,000 | guards (the hard sets) |
| `sift-128-euclidean` | ann-benchmarks | 1M | 10,000 | guards (L2) |
| `synth-1024-6m` | `mhr_vector_bench synth`, clustered | 6M | 1,000 | full suite (scale, memory) |

## Why the score uses Voyage sets (score-v2)

The owner switched the score to Voyage-only data on 2026-10-07. arko_index embeds with
Voyage, voyage-code-4 for code and voyage-4 for documentation text, so the store is tuned
on the vectors it will actually hold. Code and documentation embeddings spread
differently, and a score over both keeps an optimization from fitting one of them. The
two parts combine by geometric mean: a 10% gain on either part moves the score by the
same amount, and a part that misses its recall target makes the whole score 0.

score-v1 used 500,000 OpenAI text-embedding-3-large vectors of DBpedia cut to 1024
dimensions (`dbp3l-1024`) as a stand-in. Its download script, `scripts/fetch_openai3.py`,
is in the history at c8ccae9.

The Cloudflare docs and MDN hold about 57,000 chunks between them, so the docs set stays
well under its 200,000 target. The name records the target the owner set.

## Sources

Code checkouts (`/work/safety/repos/` on the box), listed in `scripts/sources/code.json`:

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
  `src/content/docs`), the folder at `~/Desktop/cloudflare-docs/docs`, its `*.md` and
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

| Set | Chunks | Expected tokens | Price per million | Expected cost | Cap (`--max-tokens`) |
|---|---|---|---|---|---|
| code-voyagecode4-200k | 200,000 | ~36M (31M-42M) | $0.12 (voyage-code-4) | ~$4.35 ($3.70-$5.00) | 55M ($6.60) |
| docs-voyage4-200k | ~57,000 | ~45M (43M-48M) | $0.06 (voyage-4) | ~$2.75 ($2.60-$2.90) | 70M ($4.20) |

These are standard rates, before free tokens. voyage-code-4 comes with 200 million free
tokens per account, which would cover the code set if they are unused.

How the estimates were made (2026-10-07, no Voyage calls):

1. Dry runs on local samples. For code, 20,000 chunks of 14 local repositories in the
   same 8 languages (caddy, jsoup, gson, type-fest, vite, redis, zlib, leveldb, spdlog,
   scrapy, flask, jquery, tokio, serde_json). For docs, all of the Cloudflare docs.
2. Exact token counts of the planned texts with Voyage's tokenizer: the
   `voyageai/voyage-4` `tokenizer.json` from Hugging Face (the same file as
   voyage-code-4's), a Qwen2-style byte-level BPE, run in Python. Code came to 181 tokens
   per chunk with the languages weighted equally, 3.47 bytes per token (C is denser at
   2.67, since digits are single tokens). The Cloudflare chunks came to 7.12M tokens,
   4.34 bytes per token. Queries average 16 tokens (code) and 6 (docs).
3. Extrapolation. Code is 200,000 chunks at 181 tokens, with ±15% for the box's
   repositories. MDN is taken as 153 MB of Markdown (the owner's figure) at 3,300 to
   3,500 bytes per chunk (Cloudflare's chunk sizes for pages over 4 KB and over 8 KB,
   MDN pages averaging 10 KB), so about 45,000 chunks, at 3.8 to 4.3 bytes per token.
4. A billing check: arko's earlier build of the Cloudflare index was billed 99.3% of the
   tokens its tokenizer counted beforehand.

The dry run on the box gives the real counts. Its `estimated_tokens` uses 3 bytes per
token, so it reads about 15% (code) and 35% (docs) above the bill.

## Building them

```sh
scripts/voyage_sets.sh --dry-run             # on the box: counts, estimates, plans
VOYAGE_API_KEY=... scripts/voyage_sets.sh    # embed both sets
```

The script's variables (`MHRV_ROOT`, `REPOS`, `CF_DOCS`, `ARKO_CLI`) default to the box's
layout. `scripts/box_setup.sh` computes the score's and guards' ground truth once the sets
exist. `build_voyage_set.py --help` lists every option.
