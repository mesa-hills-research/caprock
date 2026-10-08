#!/usr/bin/env python3
"""Build a Voyage embedding dataset for mhr_vector_bench from arko_index chunks of code
checkouts and documentation folders: a seeded sample of whole files spread evenly over
languages (code) and sources (docs), with queries from doc comments, page headings or
question files. docs/datasets.md describes the rules; scripts/voyage_sets.sh runs the
score-v2 sets.

    # 1. See what it would embed and cost (no API calls; needs arko_index_cli):
    python3 scripts/build_voyage_set.py --name code-voyagecode4-200k --model voyage-code-4 \\
        --sources scripts/sources/code.json --target-chunks 200000 \\
        --queries code-doc-comments --query-count 2000 \\
        --work voyage-work --out datasets --arko-cli /path/to/arko_index_cli --dry-run
    # 2. Embed: the same command without --dry-run, with VOYAGE_API_KEY in the
    #    environment (never printed) and --max-tokens/--allow-over covering the estimate.

Sources (--sources FILE): a JSON list of {"name", "path" | "url" + "tag", "kind": "code" |
"docs", "language": NAME or [NAMES], "include": [PATTERNS], "exclude": [PATTERNS],
"commit": DECLARED}. "path" is a local folder used as it is ($VARS expand), "url" + "tag"
a pinned shallow clone in <work>/repos/. Patterns use gitignore syntax and go to
arko_index_cli. A code source keeps chunks in arko_index's grammar languages (all of
them, or "language"), a docs source keeps every chunk.

Selection (--target-chunks N, --seed S): chunks fall into strata, "lang:<language>" for
code and "source:<name>" for docs. N is shared equally among the strata, and a stratum
with fewer chunks than its share keeps them all and leaves the rest to the others
(water-filling). Each stratum's quota is shared among its sources the same way. Within
one source and stratum, files are taken whole in the order of SHA-256(seed, source, path),
skipping a file that would pass the quota. Base rows follow the same hash order across
all sources, each file's chunks in file order, so any prefix is a fair sample.

Queries (--queries SPEC[,SPEC...], --query-count Q): "docs-questions:<file>" takes every
question of a JSON or JSONL file (Cloudflare's evaluation format {"id", "query"}, or plain
strings) first. "code-doc-comments" (first doc-comment sentences, symbol phrases as a
fallback) and "docs-headings" (front-matter titles and Markdown headings of 3-20 words)
fill the rest from the selected chunks, each stratum in proportion to its share of the
base, in seeded-hash order, without duplicate texts.

Text: the source-relative path, a newline, the chunk's bytes ("path-newline-source/1").
Documents use input_type "document", queries "query". Budget: --max-tokens caps the
dataset's billed tokens (estimated as bytes / --bytes-per-token before any call, then as
Voyage reports them, with a ledger in <work>/billing/ across interrupted runs). Above
3,000,000 it needs --allow-over. Embeddings are cached by SHA-256 of their text in
<work>/embed-cache-<model>-<dim>.bin, shared with build_code_set.py.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

from voyage_common import (HARD_CEILING, TEXT_FORMAT, Budget, Cache, clone, cost, doc_sentence, embed_all, git_info,
                           log, price, symbol_query, write_dataset)

GRAMMAR_LANGUAGES = {"rust", "python", "typescript", "javascript", "go", "java", "c", "cpp"}
SELECTION_RULE = "water-filling/1"


# ----- Sources and chunks -----

def load_sources(path, work):
    with open(path) as f:
        specs = json.load(f)
    sources, names = [], set()
    for spec in specs:
        name = spec["name"]
        if name in names or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
            raise SystemExit(f"{path}: source name {name!r} is repeated or not [A-Za-z0-9._-]+")
        names.add(name)
        kind = spec.get("kind", "code")
        if kind not in ("code", "docs"):
            raise SystemExit(f"{path}: source {name}: kind must be code or docs")
        languages = spec.get("language")
        if isinstance(languages, str):
            languages = [languages]
        if "url" in spec:
            root = clone(work, name, spec["url"], spec["tag"])
        else:
            root = os.path.abspath(os.path.expanduser(os.path.expandvars(spec["path"])))
            if "$" in root or not os.path.isdir(root):
                raise SystemExit(f"{path}: source {name}: {spec['path']} is not a folder (set the variables it names)")
        sources.append({
            "name": name, "kind": kind, "root": root, "languages": languages,
            "include": spec.get("include", []), "exclude": spec.get("exclude", []),
            "url": spec.get("url"), "tag": spec.get("tag"), "declared_commit": spec.get("commit"),
        })
    return sources


def chunk_list(work, source, cli, global_exclude):
    """arko_index_cli's chunks of a source (cached in <work>/chunks/, keyed by the folder
    and the file rules)."""
    flags = ["--respect-ignore"]
    for p in global_exclude + source["exclude"]:
        flags += ["--exclude", p]
    for p in source["include"]:
        flags += ["--include", p]
    tag = hashlib.sha256(json.dumps([source["root"], flags]).encode()).hexdigest()[:12]
    out = os.path.join(work, "chunks", f"{source['name']}-{tag}.jsonl")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".tmp", "w") as w, open(out + ".stderr", "w") as e:
            subprocess.run([cli, "chunk", *flags, source["root"]], stdout=w, stderr=e, check=True)
        os.replace(out + ".tmp", out)
    chunks = []
    with open(out) as f:
        for line in f:
            if line.startswith("{"):
                c = json.loads(line)
                chunks.append({k: c[k] for k in ("id", "path", "start_byte", "end_byte", "language", "kind", "symbol_path")})
    return chunks


def candidates(source, chunks):
    if source["kind"] == "docs":
        keep = (lambda lang: lang in source["languages"]) if source["languages"] else (lambda lang: True)
    else:
        allowed = set(source["languages"] or GRAMMAR_LANGUAGES) & GRAMMAR_LANGUAGES
        keep = lambda lang: lang in allowed  # noqa: E731
    return [c for c in chunks if keep(c["language"])]


def stratum_of(source, chunk):
    return f"lang:{chunk['language']}" if source["kind"] == "code" else f"source:{source['name']}"


def file_rank(seed, source, path):
    return hashlib.sha256(f"{seed}\0{source}\0{path}".encode()).hexdigest()


# ----- Selection -----

def water_fill(total, capacities):
    """Equal shares of `total` over the keys; a key that can't use its share keeps its
    capacity and the rest is shared by the others."""
    quotas = {k: 0 for k in capacities}
    remaining = total
    open_keys = sorted(k for k, c in capacities.items() if c > 0)
    while open_keys and remaining > 0:
        share = remaining // len(open_keys)
        if share == 0:
            for k in open_keys[:remaining]:
                quotas[k] += 1
            break
        still_open = []
        for k in open_keys:
            take = min(share, capacities[k] - quotas[k])
            quotas[k] += take
            remaining -= take
            if quotas[k] < capacities[k]:
                still_open.append(k)
        open_keys = still_open
    return quotas


def select(sources, chunks_by_source, target, seed):
    """Files chosen by the rule in the module docs: [(rank, source, path, [chunks])] in
    row order, and per-stratum statistics."""
    cells = {}  # (stratum, source) -> {path: [chunks]}
    for source in sources:
        for c in candidates(source, chunks_by_source[source["name"]]):
            cell = cells.setdefault((stratum_of(source, c), source["name"]), {})
            cell.setdefault(c["path"], []).append(c)
    strata = sorted({s for s, _ in cells})
    available = {s: sum(len(cs) for (t, _), files in cells.items() if t == s for cs in files.values()) for s in strata}
    stratum_quota = water_fill(target, available)
    chosen, stats = [], {}
    for s in strata:
        names = sorted(n for t, n in cells if t == s)
        source_quota = water_fill(stratum_quota[s], {n: sum(len(cs) for cs in cells[(s, n)].values()) for n in names})
        stats[s] = {"available": available[s], "quota": stratum_quota[s], "chunks": 0, "files": 0, "sources": {}}
        for n in names:
            files = cells[(s, n)]
            taken = 0
            for rank, path in sorted((file_rank(seed, n, p), p) for p in files):
                size = len(files[path])
                if taken + size > source_quota[n]:
                    continue
                taken += size
                chosen.append((rank, n, path, sorted(files[path], key=lambda c: c["start_byte"])))
                stats[s]["files"] += 1
            stats[s]["sources"][n] = taken
            stats[s]["chunks"] += taken
    chosen.sort()
    return chosen, stats


# ----- Texts and queries -----

def read_texts(sources, chosen):
    """Each chosen chunk as a row: key, source, path, language, text."""
    roots = {s["name"]: s["root"] for s in sources}
    rows, files = [], {}
    for _, name, path, chunks in chosen:
        with open(os.path.join(roots[name], path), "rb") as f:
            data = f.read()
        files[(name, path)] = data
        for c in chunks:
            body = data[c["start_byte"]:c["end_byte"]].decode("utf-8", "replace")
            rows.append({
                "key": f"{name}:{c['id']}", "source": name, "path": path, "language": c["language"],
                "kind": c["kind"], "symbol_path": c["symbol_path"], "start": c["start_byte"], "end": c["end_byte"],
                "text": path + "\n" + body,
            })
    return rows, files


HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
FENCE = re.compile(r"^ {0,3}(```+|~~~+)")
GENERIC_HEADINGS = {
    "before you begin", "frequently asked questions", "how it works", "what's next", "what is next",
    "related resources", "next steps", "see also", "browser compatibility", "get started", "getting started",
}


def clean_heading(text):
    text = re.sub(r"\{#[^}]*\}", "", text)  # {#anchor}
    text = re.sub(r"\{\{.*?\}\}", "", text)  # MDN macros
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)  # links and images
    text = re.sub(r"<[^>]+>", "", text)  # HTML and MDX tags
    text = text.replace("`", "").replace("**", "").replace("__", "")
    text = text.strip().strip("\"'").strip()
    return re.sub(r"\s+", " ", text)


def good_heading(text):
    words = re.findall(r"[^\W\d_]{2,}", text)
    if not 3 <= len(words) <= 20 or text.lower().rstrip("?.!:") in GENERIC_HEADINGS:
        return False
    letters = sum(ch.isalpha() for ch in text)
    return letters >= 0.6 * sum(not ch.isspace() for ch in text)


def headings_of(data):
    """(byte offset, kind, text) of a Markdown file's front-matter title and headings,
    skipping fenced code."""
    lines = data.split(b"\n")
    starts, at = [], 0
    for raw in lines:
        starts.append(at)
        at += len(raw) + 1
    out, body = [], 0
    if lines and lines[0].strip() == b"---":
        for i in range(1, len(lines)):
            line = lines[i].decode("utf-8", "replace")
            if line.strip() == "---":
                body = i + 1
                break
            m = re.match(r"^title:\s*(.+?)\s*$", line)
            if m:
                out.append((starts[i], "title", clean_heading(m.group(1))))
        else:
            out = []  # no closing line: not front matter
    fence = None
    for i in range(body, len(lines)):
        line = lines[i].decode("utf-8", "replace")
        f = FENCE.match(line)
        if f:
            run = f.group(1)
            if fence is None:
                fence = run
            elif run[0] == fence[0] and len(run) >= len(fence) and line.strip() == run:
                fence = None
        elif fence is None:
            m = HEADING.match(line)
            if m:
                out.append((starts[i], "heading", clean_heading(m.group(2))))
    return [h for h in out if good_heading(h[2])]


def load_questions(path):
    with open(os.path.expanduser(os.path.expandvars(path))) as f:
        text = f.read()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, dict):
        data = data.get("queries") or data.get("questions") or []
    out = []
    for i, item in enumerate(data):
        if isinstance(item, str):
            q, ident = item, str(i)
        else:
            q = item.get("query") or item.get("question") or item.get("text")
            ident = str(item.get("id", i))
        if q and q.strip():
            out.append((q.strip(), ident))
    return out


RUST_ATTRIBUTE = re.compile(r"^\s*#!?\[.*$", re.M)


def doc_query(row):
    """A doc-comment sentence (doc_sentence), with Rust attributes not taken for "#"
    comments; a symbol phrase when there is none."""
    if row["language"] == "rust":
        row = dict(row, text=RUST_ATTRIBUTE.sub("", row["text"]))
    q = doc_sentence(row)
    if q:
        return "doc", q
    q = symbol_query(row)
    return ("symbol", q) if q else None


def norm(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def make_queries(specs, count, rows, files, sources, seed):
    kinds = {s["name"]: s["kind"] for s in sources}
    picked, seen = [], set()
    generated = set()
    for spec in specs:
        if spec.startswith("docs-questions:"):
            path = spec.split(":", 1)[1]
            for q, ident in load_questions(path):
                if len(picked) < count and norm(q) not in seen:
                    seen.add(norm(q))
                    picked.append({"text": q, "origin": f"questions:{os.path.basename(path)}#{ident}", "source": None,
                                   "kind": "question", "stratum": None})
        elif spec in ("code-doc-comments", "docs-headings"):
            generated.add(spec)
        else:
            raise SystemExit(f"unknown --queries spec {spec!r}")
    lists, weights = {}, {}
    headings = {}
    if "docs-headings" in generated:
        for (name, path), data in files.items():
            if kinds[name] == "docs":
                headings[(name, path)] = headings_of(data)
    for row in rows:
        stratum = f"lang:{row['language']}" if kinds[row["source"]] == "code" else f"source:{row['source']}"
        weights[stratum] = weights.get(stratum, 0) + 1
        found = []
        if kinds[row["source"]] == "code" and "code-doc-comments" in generated:
            found = [q for q in [doc_query(row)] if q]
        elif kinds[row["source"]] == "docs" and "docs-headings" in generated:
            found = [(k, t) for off, k, t in headings[(row["source"], row["path"])] if row["start"] <= off < row["end"]]
        for i, (kind, q) in enumerate(found):
            rank = hashlib.sha256(f"{seed}\0{row['key']}\0{i}".encode()).hexdigest()
            lists.setdefault(stratum, []).append((kind == "symbol", rank, q, row["key"], row["source"], kind, stratum))
    for items in lists.values():
        items.sort()
    taken = {s: 0 for s in lists}
    at = {s: 0 for s in lists}
    while len(picked) < count:
        open_strata = [s for s in lists if at[s] < len(lists[s])]
        if not open_strata:
            break
        s = min(open_strata, key=lambda s: (taken[s] / weights[s], s))
        _, _, q, origin, source, kind, stratum = lists[s][at[s]]
        at[s] += 1
        if norm(q) in seen:
            continue
        seen.add(norm(q))
        taken[s] += 1
        picked.append({"text": q, "origin": origin, "source": source, "kind": kind, "stratum": stratum})
    return picked


def manifest_hash(root, paths):
    """SHA-256 over the sorted (path, SHA-256 of bytes) of a folder's chunked files."""
    h = hashlib.sha256()
    for p in sorted(paths):
        with open(os.path.join(root, p), "rb") as f:
            h.update(p.encode() + b"\0" + hashlib.sha256(f.read()).hexdigest().encode() + b"\n")
    return h.hexdigest()


# ----- Main -----

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="dataset name (its folder under --out)")
    p.add_argument("--model", required=True, help="voyage-code-4 for code, voyage-4 for documentation")
    p.add_argument("--dim", type=int, default=1024, choices=[256, 512, 1024, 2048])
    p.add_argument("--sources", required=True, help="JSON list of sources (see above)")
    p.add_argument("--target-chunks", type=int, required=True)
    p.add_argument("--seed", default="1", help="the sample's seed (recorded)")
    p.add_argument("--queries", default="code-doc-comments", help="comma-separated: code-doc-comments, docs-headings, docs-questions:<file>")
    p.add_argument("--query-count", type=int, default=2000)
    p.add_argument("--work", required=True, help="clones, chunk lists, plans, billing ledgers and the embedding cache")
    p.add_argument("--out", required=True, help="datasets root")
    p.add_argument("--arko-cli", required=True, help="arko_index_cli (cargo build --release -p arko_index --example arko_index_cli)")
    p.add_argument("--exclude", action="append", default=[], help="a gitignore-style pattern excluded in every source")
    p.add_argument("--max-tokens", type=int, default=2_800_000, help="the dataset's billed tokens, documents plus queries")
    p.add_argument("--allow-over", action="store_true", help=f"allow --max-tokens above {HARD_CEILING:,}")
    p.add_argument("--bytes-per-token", type=float, default=3.0, help="for estimates before any call (3.0 overestimates code and prose)")
    p.add_argument("--concurrency", type=int, default=4, help="Voyage requests at once")
    p.add_argument("--dry-run", action="store_true", help="select and estimate only: no API calls")
    args = p.parse_args()
    ceiling = args.max_tokens if args.allow_over else min(args.max_tokens, HARD_CEILING)
    os.makedirs(args.work, exist_ok=True)

    sources = load_sources(args.sources, args.work)
    chunks_by_source = {s["name"]: chunk_list(args.work, s, args.arko_cli, args.exclude) for s in sources}
    chosen, strata = select(sources, chunks_by_source, args.target_chunks, args.seed)
    rows, files = read_texts(sources, chosen)
    if len({r["key"] for r in rows}) != len(rows):
        raise SystemExit("two chunks share a key")
    specs = [s for s in args.queries.split(",") if s]
    queries = make_queries(specs, args.query_count, rows, files, sources, args.seed)

    doc_bytes = sum(len(r["text"].encode("utf-8")) for r in rows)
    query_bytes = sum(len(q["text"].encode("utf-8")) for q in queries)
    est = int(doc_bytes / args.bytes_per_token) + len(rows) + int(query_bytes / args.bytes_per_token) + len(queries)
    per_source = {}
    for s in sources:
        selected = [r for r in rows if r["source"] == s["name"]]
        per_source[s["name"]] = {
            "available_chunks": len(candidates(s, chunks_by_source[s["name"]])),
            "selected_chunks": len(selected),
            "selected_files": len({r["path"] for r in selected}),
            "bytes": sum(len(r["text"].encode("utf-8")) for r in selected),
        }
    by_kind = {}
    for q in queries:
        by_kind[q["kind"]] = by_kind.get(q["kind"], 0) + 1
    summary = {
        "name": args.name, "model": args.model, "dim": args.dim, "seed": args.seed,
        "target_chunks": args.target_chunks, "chunks": len(rows), "files": len(chosen),
        "strata": strata, "sources": per_source,
        "queries": len(queries), "queries_by_kind": by_kind,
        "bytes": {"documents": doc_bytes, "queries": query_bytes},
        "bytes_per_token": args.bytes_per_token, "estimated_tokens": est,
        "price_per_million_usd": price(args.model), "estimated_cost_usd": cost(est, args.model),
        "ceiling": ceiling, "fits": est <= ceiling,
    }
    if est > ceiling:
        summary["needs"] = f"--max-tokens {int(est * 1.05) + 1} --allow-over" if est > HARD_CEILING else f"--max-tokens {int(est * 1.05) + 1}"
    print(json.dumps(summary, indent=2), flush=True)

    plans = os.path.join(args.work, "plans")
    os.makedirs(plans, exist_ok=True)
    with open(os.path.join(plans, f"{args.name}-base.jsonl"), "w") as w:
        for r in rows:
            w.write(json.dumps({k: r[k] for k in ("key", "source", "path", "language", "start", "end")}) + "\n")
    with open(os.path.join(plans, f"{args.name}-queries.jsonl"), "w") as w:
        for q in queries:
            w.write(json.dumps(q) + "\n")
    if args.dry_run:
        return
    if est > ceiling:
        raise SystemExit(f"the estimate ({est:,} tokens) passes the ceiling ({ceiling:,}): pass {summary['needs']}")
    if len(queries) < args.query_count:
        log(f"note: {len(queries)} queries found, fewer than {args.query_count}")

    key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not key:
        raise SystemExit("VOYAGE_API_KEY is not set")
    cache = Cache(os.path.join(args.work, f"embed-cache-{args.model}-{args.dim}.bin"), args.dim)
    cached_before = sum(1 for t in {r["text"] for r in rows} | {q["text"] for q in queries}
                        if hashlib.sha256(t.encode("utf-8")).digest() in cache.items)
    budget = Budget(ceiling, args.bytes_per_token, os.path.join(args.work, "billing", f"{args.name}.jsonl"))
    doc_vectors = embed_all([r["text"] for r in rows], "document", args.model, args.dim, key, cache, budget, args.concurrency)
    query_vectors = embed_all([q["text"] for q in queries], "query", args.model, args.dim, key, cache, budget, args.concurrency)

    provenance = []
    for s in sources:
        chunked = {c["path"] for c in chunks_by_source[s["name"]]}
        git = git_info(s["root"])
        entry = {
            "name": s["name"], "kind": s["kind"], "path": s["root"], "languages": s["languages"],
            "include": s["include"], "exclude": args.exclude + s["exclude"], "git": git,
            "url": s["url"], "tag": s["tag"], "declared_commit": s["declared_commit"],
            "files_chunked": len(chunked), **per_source[s["name"]],
        }
        if git is None or git["dirty"]:
            entry["manifest_sha256"] = manifest_hash(s["root"], chunked)
        provenance.append(entry)
    cli = os.path.abspath(args.arko_cli)
    meta_rows = [{"key": r["key"], "path": f"{r['source']}/{r['path']}", "language": r["language"]} for r in rows]
    out = write_dataset(args.out, args.name, args.dim, doc_vectors, query_vectors, meta_rows, queries, {
        "provider": "voyage",
        "model": args.model,
        "output_dimension": args.dim,
        "input_types": {"base": "document", "queries": "query"},
        "text_format": TEXT_FORMAT,
        "tokens_billed": budget.used,
        "tokens_billed_this_run": budget.this_run,
        "texts_cached_before_run": cached_before,
        "estimated_tokens": est,
        "bytes_per_token_estimate": args.bytes_per_token,
        "price_per_million_usd": price(args.model),
        "selection": {"rule": SELECTION_RULE, "seed": args.seed, "target_chunks": args.target_chunks, "strata": strata},
        "queries": {"specs": specs, "requested": args.query_count, "by_kind": by_kind},
        "sources": provenance,
        "chunker": {"path": cli, "flags": ["--respect-ignore"], "git": git_info(os.path.dirname(cli))},
        "built_by": "scripts/build_voyage_set.py",
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "command": sys.argv[1:],
    })
    log(f"{out}: {len(doc_vectors)} chunks, {len(query_vectors)} queries; {budget.used} tokens billed (${cost(budget.used, args.model)})")


if __name__ == "__main__":
    main()
