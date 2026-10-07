"""Shared pieces of the Voyage dataset builders (build_code_set.py, build_voyage_set.py):
the embedding cache, the Voyage client with its budget, the doc-comment query extractor,
provenance, and the mhr-vector-dataset/1 writer. Standard library only."""

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
from concurrent.futures import ThreadPoolExecutor

USER_AGENT = "mhr-vector-dataset-builder/0.2"
API = "https://api.voyageai.com/v1/embeddings"
# Standard rates in USD per million tokens: voyage-code-4 from arko's
# docs/voyage-api-notes.md, the voyage-4 family from arko_ai/README.md.
PRICES = {"voyage-code-4": 0.12, "voyage-4": 0.06, "voyage-4-large": 0.12, "voyage-4-lite": 0.02}
HARD_CEILING = 3_000_000
MAX_INPUTS = 1000
# Voyage allows 320,000 tokens per request. A byte-level BPE token covers at least one
# UTF-8 byte, so 280,000 bytes stays under the limit with room for per-input prompts.
MAX_REQUEST_BYTES = 280_000
TEXT_FORMAT = "path-newline-source/1"
FORMAT = "mhr-vector-dataset/1"


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def sha(text):
    return hashlib.sha256(text.encode("utf-8")).digest()


def estimate(text, bytes_per_token):
    return int(len(text.encode("utf-8")) / bytes_per_token) + 1


def price(model):
    return PRICES.get(model)


def cost(tokens, model):
    p = price(model)
    return None if p is None else round(tokens / 1e6 * p, 4)


def clone(work, name, url, tag):
    """A shallow clone of `tag` in <work>/repos/<name> (kept between runs)."""
    dest = os.path.join(work, "repos", name)
    if not os.path.isdir(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "--quiet", "--depth", "1", "--branch", tag, url, dest], check=True)
    return dest


def git_info(path):
    """The checkout's commit, `git describe` and whether tracked files changed; None
    outside a git work tree."""
    def git(*args):
        r = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    if git("rev-parse", "--is-inside-work-tree") != "true":
        return None
    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "describe": git("describe", "--tags", "--always"),
        "subdir": git("rev-parse", "--show-prefix") or "",
        "dirty": bool(status),
    }


class Cache:
    """Embeddings by SHA-256 of their text: records of 32 bytes + dim float32."""

    def __init__(self, path, dim):
        self.path, self.dim, self.items = path, dim, {}
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = f.read()
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


class Budget:
    """Tokens billed against a ceiling. `ledger` (a JSONL path) records every request's
    billed tokens, so a dataset's total survives interrupted runs."""

    def __init__(self, ceiling, bytes_per_token, ledger=None):
        self.ceiling, self.bytes_per_token, self.ledger = ceiling, bytes_per_token, ledger
        self.this_run = 0
        self.before = 0
        if ledger and os.path.exists(ledger):
            with open(ledger) as f:
                self.before = sum(json.loads(line)["tokens"] for line in f if line.strip())

    @property
    def used(self):
        return self.before + self.this_run

    def record(self, model, dim, input_type, texts, tokens):
        self.this_run += tokens
        if self.ledger:
            os.makedirs(os.path.dirname(self.ledger), exist_ok=True)
            with open(self.ledger, "a") as f:
                f.write(json.dumps({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "model": model, "dim": dim,
                                    "input_type": input_type, "texts": texts, "tokens": tokens}) + "\n")


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


def batches(texts):
    """Requests of at most MAX_INPUTS texts and MAX_REQUEST_BYTES bytes."""
    out, batch, size = [], [], 0
    for t in texts:
        n = len(t.encode("utf-8"))
        if batch and (len(batch) >= MAX_INPUTS or size + n > MAX_REQUEST_BYTES):
            out.append(batch)
            batch, size = [], 0
        batch.append(t)
        size += n
    if batch:
        out.append(batch)
    return out


def embed_all(items, input_type, model, dim, key, cache, budget, concurrency=1):
    """Vectors for `items` (texts). Cached texts are reused unbilled; the rest are sent
    `concurrency` requests at a time, each wave checked against the budget first."""
    todo = [t for t in dict.fromkeys(items) if sha(t) not in cache.items]
    planned = batches(todo)
    with ThreadPoolExecutor(max(1, concurrency)) as pool:
        for at in range(0, len(planned), max(1, concurrency)):
            wave = planned[at:at + max(1, concurrency)]
            est = sum(estimate(t, budget.bytes_per_token) for b in wave for t in b)
            if budget.used + est > budget.ceiling:
                raise SystemExit(f"stopping: {budget.used} tokens billed, the next requests (~{est}) would pass {budget.ceiling}")
            futures = [pool.submit(embed, b, input_type, model, dim, key) for b in wave]
            failure = None
            for batch, future in zip(wave, futures):
                try:
                    vectors, billed = future.result()
                except BaseException as e:  # keep the other requests' (billed) vectors
                    failure = failure or e
                    continue
                for t, v in zip(batch, vectors):
                    cache.put(sha(t), v)
                budget.record(model, dim, input_type, len(batch), billed)
            cache.flush()
            if failure is not None:
                raise failure
            log(f"{input_type}: {len(cache.items)} cached, {budget.used} tokens billed in total")
    return [cache.items[sha(t)] for t in items]


COMMENT_PREFIXES = ("///", "//!", "//", "--", "#", "*", "/**", "/*")


def doc_sentence(chunk):
    """The first sentence of the chunk's first comment block or docstring, if it reads
    like documentation (at least five words, not a license header). `chunk` has "text"
    (path, newline, source) and "language"."""
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


def write_vectors(path, vectors):
    with open(path, "wb") as w:
        for v in vectors:
            if sys.byteorder != "little":
                v = array.array("f", v)
                v.byteswap()
            w.write(v.tobytes())


def write_dataset(out, name, dim, doc_vectors, query_vectors, meta_rows, query_rows, source):
    """Writes <out>/<name>/ in the mhr-vector-dataset/1 format (cosine): base.f32,
    queries.f32, meta.jsonl, queries.jsonl and dataset.json (`source` is its provenance)."""
    folder = os.path.join(out, name)
    os.makedirs(folder, exist_ok=True)
    write_vectors(os.path.join(folder, "base.f32"), doc_vectors)
    write_vectors(os.path.join(folder, "queries.f32"), query_vectors)
    with open(os.path.join(folder, "meta.jsonl"), "w") as w:
        for row in meta_rows:
            w.write(json.dumps(row) + "\n")
    with open(os.path.join(folder, "queries.jsonl"), "w") as w:
        for row in query_rows:
            w.write(json.dumps(row) + "\n")
    info = {
        "format": FORMAT,
        "name": name,
        "metric": "cosine",
        "dim": dim,
        "base_count": len(doc_vectors),
        "query_count": len(query_vectors),
        "base": "base.f32",
        "queries": "queries.f32",
        "meta": "meta.jsonl",
        "query_texts": "queries.jsonl",
        "source": source,
    }
    with open(os.path.join(folder, "dataset.json"), "w") as w:
        json.dump(info, w, indent=2)
        w.write("\n")
    return folder
