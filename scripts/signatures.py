#!/usr/bin/env python3
"""Extract the interface of crates/mhr_vector/src: every `pub` fn signature (with its
impl or trait context) and every pub struct/enum/trait/type/const/field line, whitespace
normalized. The skeleton's list is frozen in contract/signatures.txt; the gate checks
that a branch still has every frozen line (new items are fine).

    python3 scripts/signatures.py > contract/signatures.txt        # freeze (host only)
    python3 scripts/signatures.py --check contract/signatures.txt  # gate: exit 1 on a change
"""

import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "crates", "mhr_vector", "src")


def items(path):
    rel = os.path.relpath(path, ROOT)
    text = open(path).read()
    text = re.sub(r"//[^\n]*", "", text)
    out = []
    context = []
    depth = 0
    i = 0
    lines = text.split("\n")
    buf = None
    for line in lines:
        s = line.strip()
        if buf is not None:
            buf += " " + s
            if "{" in s or s.endswith(";"):
                sig = re.sub(r"\s+", " ", buf.split("{")[0]).strip().rstrip(";").strip()
                out.append(f"{rel}: {' > '.join(context)} :: {sig}" if context else f"{rel}: {sig}")
                buf = None
        elif re.match(r"(pub(\([a-z]+\))? )?(async )?(const )?fn ", s) and (s.startswith("pub") or (context and context[-1].startswith("trait"))):
            if "{" in s or s.endswith(";"):
                sig = re.sub(r"\s+", " ", s.split("{")[0]).strip().rstrip(";").strip()
                out.append(f"{rel}: {' > '.join(context)} :: {sig}" if context else f"{rel}: {sig}")
            else:
                buf = s
        elif re.match(r"pub (struct|enum|trait|type|const|mod|use) ", s):
            out.append(f"{rel}: {re.sub(r'[{;].*$', '', s).strip()}")
        elif re.match(r"pub [a-z_0-9]+: ", s):
            out.append(f"{rel}: {' > '.join(context)} :: field {s.rstrip(',')}")
        m = re.match(r"(impl(<[^>]*>)? .*?|pub trait [A-Za-z0-9_]+.*?|pub (struct|enum) [A-Za-z0-9_]+.*?)\s*\{\s*$", s)
        opens, closes = s.count("{"), s.count("}")
        if m and depth == 0:
            name = re.sub(r"^pub ", "", m.group(1)).strip()
            context = [name]
        depth += opens - closes
        if depth == 0:
            context = []
    return out


# Implementation-defined parts of the skeleton: a unit may reshape these freely.
FREE = [
    re.compile(r"^quant/(int8|binary)\.rs: .* :: field "),
    re.compile(r"^quant/(int8|binary)\.rs: pub struct (Int8|Binary)Query"),
]


def collect():
    result = []
    for dirpath, _, files in os.walk(ROOT):
        for f in sorted(files):
            if f.endswith(".rs"):
                result.extend(items(os.path.join(dirpath, f)))
    return sorted(set(r for r in result if not any(f.search(r) for f in FREE)))


def main():
    current = collect()
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        frozen = [l.rstrip("\n") for l in open(sys.argv[2]) if l.strip()]
        missing = [l for l in frozen if l not in set(current)]
        for l in missing:
            print(f"changed or removed: {l}")
        sys.exit(1 if missing else 0)
    for l in current:
        print(l)


if __name__ == "__main__":
    main()
