#!/usr/bin/env python3
"""Find duplicated code: runs of WINDOW consecutive meaningful lines that appear in two places.

  scripts/dup_check.py             # exit 1 if there are more duplicated runs than the recorded debt
  scripts/dup_check.py --update    # record today's count
  scripts/dup_check.py --report    # list the worst files

Blank lines, comments, `use`/`mod` lines and lines that are only brackets do not count as meaningful.
Tests are included: AGENTS.md asks for duplicates to be eliminated everywhere.
"""
import collections
import json
import os
import re
import subprocess
import sys

WINDOW = 8
CODE = re.compile(r"\.(rs|js|mjs|py|sh|wgsl)$")
SKIP = re.compile(r"^(//|#|use |mod |pub mod |pub use |import |from |[{}()\[\];,]*$)")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = "scripts/dup-baseline.json"


def meaningful(text):
    """[(line number, normalised line)] of the lines that can make up a duplicate."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        s = " ".join(line.split())
        if len(s) >= 12 and not SKIP.match(s):
            out.append((n, s))
    return out


def windows(path, text):
    m = meaningful(text)
    return [(hash(tuple(s for _, s in m[i:i + WINDOW])), path, m[i][0]) for i in range(len(m) - WINDOW + 1)]


def duplicates(files):
    """{path: duplicated runs starting in it} over {path: text}; each repeated run counts once per extra copy."""
    seen = collections.defaultdict(list)
    for path, text in files.items():
        for h, p, n in windows(path, text):
            seen[h].append((p, n))
    per_file = collections.Counter()
    for spots in seen.values():
        for p, _ in spots[1:]:
            per_file[p] += 1
    return per_file


def load(root):
    out = subprocess.run(["git", "-C", root, "ls-files", "--cached", "--others", "--exclude-standard"], capture_output=True, text=True).stdout
    files = {}
    for p in out.split("\n"):
        if p and CODE.search(p) and not p.startswith((".claude/", "fixtures/")):
            try:
                files[p] = open(f"{root}/{p}", encoding="utf-8").read()
            except (OSError, UnicodeDecodeError):
                continue
    return files


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    per_file = duplicates(load(ROOT))
    total = sum(per_file.values())
    path = f"{ROOT}/{BASELINE}"
    if "--update" in args:
        json.dump({"runs": total}, open(path, "w"))
        print(f"recorded {total} duplicated runs")
        return 0
    if "--report" in args:
        for p, n in per_file.most_common(15):
            print(f"{n:5d} {p}")
        print(f"{total} duplicated runs of {WINDOW} lines in {len(per_file)} files")
        return 0
    recorded = json.load(open(path))["runs"] if os.path.exists(path) else 0
    if total > recorded:
        print(f"duplicates: {total} duplicated runs (recorded {recorded}); run scripts/dup_check.py --report", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
