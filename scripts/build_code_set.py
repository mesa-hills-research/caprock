#!/usr/bin/env python3
"""Build a real code-embedding dataset for mhr_vector_bench: arko_index chunks of pinned
open-source repositories, embedded with Voyage (voyage-code-4), plus natural-language
queries taken from the code's own doc comments.

    # 1. See what it would embed and cost (no API calls; needs git and arko_index_cli):
    python3 scripts/build_code_set.py --work /work/mhr-vector/code-work --out datasets \\
        --arko-cli /path/to/arko_index_cli --dry-run
    # 2. Embed (reads VOYAGE_API_KEY from the environment; never prints it):
    VOYAGE_API_KEY=... python3 scripts/build_code_set.py --work ... --out datasets --arko-cli ...

Writes <out>/code-voyage4-<dim>/: base.f32 (one vector per chunk, as returned),
queries.f32, meta.jsonl ({"key", "path", "language"}: key "<repo>:<chunk id>", path
"<repo>/<path>"), queries.jsonl ({"text", "origin", "repo"}), dataset.json (model,
token usage, cost, repositories and tags).

Budget: --max-tokens (default 2,800,000) caps documents plus queries by estimate before
any call (bytes / --bytes-per-token, 3.0 by default, an overestimate for code), and by
the tokens Voyage reports while running; a hard ceiling of 3,000,000 applies unless
--allow-over. voyage-code-4 costs $0.12 per million tokens (docs/voyage-api-notes.md in
arko), so the default budget costs at most about $0.34 (and nothing while the account's
200M free tokens last). Every embedding is cached by SHA-256 of its text in the work
directory, so a rerun after an interruption bills nothing twice.

Each repository gets an equal share of the budget (unused shares pass to later ones);
a repository over its share keeps a deterministic subset of whole files. Chunks are
rendered as arko_index's dense evaluation did: the repository-relative path, a newline,
the chunk's source bytes ("text_format": "path-newline-source/1").
"""

import argparse
import array
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request

USER_AGENT = "mhr-vector-dataset-builder/0.1"
API = "https://api.voyageai.com/v1/embeddings"
PRICE_PER_MILLION = 0.12
HARD_CEILING = 3_000_000
MAX_INPUTS = 1000
MAX_REQUEST_TOKENS_EST = 100_000

# name, clone URL, tag, language note. Pinned: the dataset must not drift.
REPOS = [
    ("ripgrep", "https://github.com/BurntSushi/ripgrep.git", "15.2.0", "rust"),
    ("flask", "https://github.com/pallets/flask.git", "3.1.3", "python"),
    ("requests", "https://github.com/psf/requests.git", "v2.34.2", "python"),
    ("cobra", "https://github.com/spf13/cobra.git", "v1.10.2", "go"),
    ("gin", "https://github.com/gin-gonic/gin.git", "v1.12.0", "go"),
    ("express", "https://github.com/expressjs/express.git", "v5.2.1", "javascript"),
    ("zod", "https://github.com/colinhacks/zod.git", "v4.6.5", "typescript"),
    ("gson", "https://github.com/google/gson.git", "gson-parent-2.14.0", "java"),
    ("lua", "https://github.com/lua/lua.git", "v5.5.1", "c"),
    ("jq", "https://github.com/jqlang/jq.git", "jq-1.8.2", "c"),
    ("fmt", "https://github.com/fmtlib/fmt.git", "12.2.0", "cpp"),
]


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).digest()


def clone(work, name, url, tag):
    dest = os.path.join(work, "repos", name)
    if not os.path.isdir(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--depth", "1", "--branch", tag, url, dest], check=True)
    return dest


def chunks_of(work, name, root, cli):
    out = os.path.join(work, "chunks", f"{name}.jsonl")
    if not os.path.exists(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out + ".tmp", "w") as w:
            subprocess.run([cli, "chunk", root], stdout=w, stderr=subprocess.DEVNULL, check=True)
        os.replace(out + ".tmp", out)
    chunks = []
    files = {}
    with open(out) as f:
        for line in f:
            if not line.startswith("{"):
                continue
            c = json.loads(line)
            p = c["path"]
            if p not in files:
                try:
                    with open(os.path.join(root, p), "rb") as fh:
                        files[p] = fh.read()
                except OSError:
                    files[p] = None
            data = files[p]
            if data is None:
                continue
            source = data[c["start_byte"]:c["end_byte"]].decode("utf-8", "replace")
            c["text"] = p + "\n" + source
            chunks.append(c)
    return chunks


COMMENT_PREFIXES = ("///", "//!", "//", "--", "#", "*", "/**", "/*")


def doc_sentence(chunk):
    """The first sentence of the chunk's first comment block or docstring, if it reads
    like documentation (at least five words, not a license header)."""
    lines = chunk["text"].split("\n")[1:]
    block = []
    in_doc = False
    for raw in lines[:40]:
        s = raw.strip()
        if chunk["language"] == "python" and (s.startswith('"""') or s.startswith("'''")):
            body = s[3:]
            if body.endswith('"""') or body.endswith("'''"):
                block.append(body[:-3])
                break
            block.append(body)
            in_doc = True
            continue
        if in_doc:
            if '"""' in s or "'''" in s:
                block.append(s.split('"""')[0].split("'''")[0])
                break
            block.append(s)
            continue
        if s.startswith("#") and chunk["language"] in ("c", "cpp"):
            continue  # preprocessor
        if s.startswith(COMMENT_PREFIXES):
            block.append(s.lstrip("/!*-# ").rstrip("*/ "))
        elif block:
            break
    text = " ".join(t for t in block if t).strip()
    text = re.sub(r"\s+", " ", text)
    if not text or re.search(r"copyright|license|spdx", text, re.I):
        return None
    sentence = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0][:240]
    # Commented-out code is not documentation.
    if re.search(r"[;{}]|=>|\s=\s|\+=|==", sentence):
        return None
    letters = sum(ch.isalpha() for ch in sentence)
    if letters < 0.7 * sum(not ch.isspace() for ch in sentence):
        return None
    words = re.findall(r"[A-Za-z]{2,}", sentence)
    return sentence if len(words) >= 5 else None


def symbol_query(chunk):
    sym = chunk.get("symbol_path") or ""
    if not sym:
        return None
    parts = [p for p in re.split(r"::|\.|/", sym) if p]
    words = " ".join(re.sub(r"([a-z])([A-Z])", r"\1 \2", p).replace("_", " ") for p in parts[-2:])
    kind = (chunk.get("kind") or "code").replace("_", " ")
    return f"{kind} {words}".strip() if len(words) >= 3 else None


def estimate(text, bytes_per_token):
    return int(len(text.encode("utf-8")) / bytes_per_token) + 1


def select(repo_chunks, budget, bytes_per_token):
    """Equal shares per repository, unused share carried forward; whole files only."""
    chosen = []
    total = 0
    remaining_repos = len(repo_chunks)
    for name, chunks in repo_chunks:
        share = max((budget - total) // max(remaining_repos, 1), 0)
        by_file = {}
        for c in chunks:
            by_file.setdefault(c["path"], []).append(c)
        files = sorted(by_file, key=lambda p: hashlib.sha256(f"{name}/{p}".encode()).hexdigest())
        used = 0
        for p in files:
            cost = sum(estimate(c["text"], bytes_per_token) for c in by_file[p])
            if used + cost > share:
                continue
            used += cost
            for c in by_file[p]:
                chosen.append((name, c))
        total += used
        remaining_repos -= 1
    return chosen


class Cache:
    """Embeddings by SHA-256 of their text: records of 32 bytes + dim float32."""

    def __init__(self, path, dim):
        self.path, self.dim, self.items = path, dim, {}
        if os.path.exists(path):
            data = open(path, "rb").read()
            rec = 32 + 4 * dim
            for at in range(0, len(data) - len(data) % rec, rec):
                v = array.array("f")
                v.frombytes(data[at + 32:at + rec])
                self.items[data[at:at + 32]] = v
        self.out = open(path, "ab")

    def put(self, key, vector):
        v = array.array("f", vector)
        self.out.write(key)
        self.out.write(v.tobytes())
        self.items[key] = v

    def flush(self):
        self.out.flush()
        os.fsync(self.out.fileno())


def embed(texts, input_type, model, dim, key):
    body = json.dumps({
        "input": texts, "model": model, "input_type": input_type,
        "truncation": False, "output_dimension": dim, "output_dtype": "float",
    }).encode()
    for attempt in range(7):
        request = urllib.request.Request(API, data=body, headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json", "User-Agent": USER_AGENT,
        })
        try:
            with urllib.request.urlopen(request, timeout=300) as r:
                out = json.load(r)
            vectors = [d["embedding"] for d in sorted(out["data"], key=lambda d: d["index"])]
            if len(vectors) != len(texts):
                raise SystemExit("Voyage returned the wrong number of vectors")
            return vectors, out["usage"]["total_tokens"]
        except urllib.error.HTTPError as e:
            detail = e.read()[:300].decode(errors="replace")
            if e.code in (429, 500, 502, 503, 504) and attempt < 6:
                time.sleep(min(60, 2 ** attempt))
                continue
            raise SystemExit(f"Voyage HTTP {e.code}: {detail}")
        except urllib.error.URLError as e:
            if attempt < 6:
                time.sleep(min(60, 2 ** attempt))
                continue
            raise SystemExit(f"Voyage unreachable: {e}")
    raise SystemExit("gave up after retries")


def embed_all(items, input_type, args, cache, key, used, bytes_per_token):
    """items: list of texts. Returns vectors (cached ones are not billed)."""
    todo = [t for t in dict.fromkeys(items) if sha(t) not in cache.items]
    batch, size = [], 0

    def flush(batch):
        nonlocal used
        est = sum(estimate(t, bytes_per_token) for t in batch)
        if used + est > args.ceiling:
            raise SystemExit(f"stopping: {used} tokens used, the next request (~{est}) would pass {args.ceiling}")
        vectors, billed = embed(batch, input_type, args.model, args.dim, key)
        used += billed
        for t, v in zip(batch, vectors):
            cache.put(sha(t), v)
        cache.flush()
        log(f"{input_type}: {len(cache.items)} cached, {used} tokens billed so far")

    for t in todo:
        est = estimate(t, bytes_per_token)
        if batch and (len(batch) >= MAX_INPUTS or size + est > MAX_REQUEST_TOKENS_EST):
            flush(batch)
            batch, size = [], 0
        batch.append(t)
        size += est
    if batch:
        flush(batch)
    return [cache.items[sha(t)] for t in items], used


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", required=True, help="clones, chunk lists and the embedding cache")
    p.add_argument("--out", required=True, help="datasets root")
    p.add_argument("--arko-cli", required=True, help="arko_index_cli (cargo build --release -p arko_index --example arko_index_cli)")
    p.add_argument("--model", default="voyage-code-4")
    p.add_argument("--dim", type=int, default=1024, choices=[256, 512, 1024, 2048])
    p.add_argument("--max-tokens", type=int, default=2_800_000, help="documents + queries, estimated")
    p.add_argument("--allow-over", action="store_true", help=f"allow more than {HARD_CEILING:,} tokens")
    p.add_argument("--queries", type=int, default=1000)
    p.add_argument("--bytes-per-token", type=float, default=3.0)
    p.add_argument("--repos", help="JSON list of [name, url, tag, language] replacing the default list")
    p.add_argument("--extra-repo", action="append", default=[], help="NAME=PATH of a local checkout to include (not cloned)")
    p.add_argument("--name", help="dataset name (default code-voyage4-<dim>)")
    p.add_argument("--dry-run", action="store_true", help="select and estimate only; no API calls")
    args = p.parse_args()
    args.ceiling = args.max_tokens if args.allow_over else min(args.max_tokens, HARD_CEILING)

    repos = REPOS
    if args.repos:
        repos = [tuple(r) for r in json.load(open(args.repos))]
    repo_chunks, provenance = [], []
    for name, url, tag, _ in repos:
        root = clone(args.work, name, url, tag)
        repo_chunks.append((name, chunks_of(args.work, name, root, args.arko_cli)))
        provenance.append({"name": name, "url": url, "tag": tag})
    for spec in args.extra_repo:
        name, path = spec.split("=", 1)
        repo_chunks.append((name, chunks_of(args.work, name, os.path.abspath(path), args.arko_cli)))
        provenance.append({"name": name, "local": True})

    query_budget = args.queries * 40
    chosen = select(repo_chunks, args.ceiling - query_budget, args.bytes_per_token)
    # Queries: doc sentences first, symbol phrases to fill, ordered by a hash of the key.
    candidates = []
    for name, c in chosen:
        q = doc_sentence(c)
        kind = "doc"
        if q is None:
            q, kind = symbol_query(c), "symbol"
        if q:
            rank = hashlib.sha256(f"{name}:{c['id']}".encode()).hexdigest()
            candidates.append((kind != "doc", rank, q, f"{name}:{c['id']}", name, kind))
    candidates.sort()
    queries = candidates[: args.queries]

    doc_tokens = sum(estimate(c["text"], args.bytes_per_token) for _, c in chosen)
    query_tokens = sum(estimate(q[2], args.bytes_per_token) for q in queries)
    per_repo = {}
    for name, c in chosen:
        r = per_repo.setdefault(name, {"chunks": 0, "files": set(), "est_tokens": 0})
        r["chunks"] += 1
        r["files"].add(c["path"])
        r["est_tokens"] += estimate(c["text"], args.bytes_per_token)
    summary = {
        "chunks": len(chosen),
        "queries": len(queries),
        "doc_queries": sum(1 for q in queries if q[5] == "doc"),
        "estimated_tokens": doc_tokens + query_tokens,
        "estimated_cost_usd": round((doc_tokens + query_tokens) / 1e6 * PRICE_PER_MILLION, 4),
        "ceiling": args.ceiling,
        "repos": {k: {"chunks": v["chunks"], "files": len(v["files"]), "est_tokens": v["est_tokens"]} for k, v in per_repo.items()},
        "languages": {},
    }
    for _, c in chosen:
        summary["languages"][c["language"]] = summary["languages"].get(c["language"], 0) + 1
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        return

    key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not key:
        raise SystemExit("VOYAGE_API_KEY is not set")
    cache = Cache(os.path.join(args.work, f"embed-cache-{args.model}-{args.dim}.bin"), args.dim)
    used = 0
    doc_vectors, used = embed_all([c["text"] for _, c in chosen], "document", args, cache, key, used, args.bytes_per_token)
    query_vectors, used = embed_all([q[2] for q in queries], "query", args, cache, key, used, args.bytes_per_token)

    name = args.name or f"code-voyage4-{args.dim}"
    out = os.path.join(args.out, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "base.f32"), "wb") as w:
        for v in doc_vectors:
            w.write(v.tobytes())
    with open(os.path.join(out, "queries.f32"), "wb") as w:
        for v in query_vectors:
            w.write(v.tobytes())
    with open(os.path.join(out, "meta.jsonl"), "w") as w:
        for repo, c in chosen:
            w.write(json.dumps({"key": f"{repo}:{c['id']}", "path": f"{repo}/{c['path']}", "language": c["language"]}) + "\n")
    with open(os.path.join(out, "queries.jsonl"), "w") as w:
        for q in queries:
            w.write(json.dumps({"text": q[2], "origin": q[3], "repo": q[4], "kind": q[5]}) + "\n")
    info = {
        "format": "mhr-vector-dataset/1",
        "name": name,
        "metric": "cosine",
        "dim": args.dim,
        "base_count": len(doc_vectors),
        "query_count": len(query_vectors),
        "base": "base.f32",
        "queries": "queries.f32",
        "meta": "meta.jsonl",
        "query_texts": "queries.jsonl",
        "source": {
            "provider": "voyage",
            "model": args.model,
            "output_dimension": args.dim,
            "input_types": {"base": "document", "queries": "query"},
            "text_format": "path-newline-source/1",
            "tokens_billed_this_run": used,
            "estimated_tokens": doc_tokens + query_tokens,
            "price_per_million_usd": PRICE_PER_MILLION,
            "repositories": provenance,
            "chunker": os.path.basename(args.arko_cli),
            "built_by": "scripts/build_code_set.py",
        },
    }
    with open(os.path.join(out, "dataset.json"), "w") as w:
        json.dump(info, w, indent=2)
        w.write("\n")
    log(f"{out}: {len(doc_vectors)} chunks, {len(query_vectors)} queries; {used} tokens billed (${used / 1e6 * PRICE_PER_MILLION:.2f})")


if __name__ == "__main__":
    main()
