#!/usr/bin/env python3
"""List the functions (closures included) that never ran, from `cargo llvm-cov report --json`.

  cargo llvm-cov report --json --output-path target/cov.json
  scripts/missed_functions.py target/cov.json          # production code: file:line: the source line

The summary's "functions" and "lines" columns count a closure that never ran (an error branch inside
`map_err`) as missed even when the line around it ran. Instances of a generic are merged by source span: a
function is missed only when no instance ran. Test code is left out (tests/ directories, tests.rs, *_tests).
"""
import collections
import json
import sys

TESTISH = ("/tests/", "tests.rs", "_tests", "/tests", "benches/")


def missed(data, root_marker):
    """[(file, line, name)] of functions no instance of which ran, sorted."""
    best, name = collections.defaultdict(int), {}
    for fn in data["data"][0]["functions"]:
        f = fn["filenames"][0]
        if root_marker not in f:
            continue
        f = f.split(root_marker)[-1]
        r = fn["regions"][0]
        key = (f, r[0], r[1], r[2])
        best[key] = max(best[key], fn["count"])
        name[key] = fn["name"]
    return sorted((k[0], k[1], name[k]) for k, c in best.items() if c == 0)


def production(rows):
    return [r for r in rows if not any(t in r[0] for t in TESTISH)]


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    with open(args[0], encoding="utf-8") as f:
        data = json.load(f)
    marker = "/".join(__file__.split("/")[:-2]).split("/")[-1] + "/"
    rows = production(missed(data, marker))
    for path, line, _ in rows:
        try:
            with open(path, encoding="utf-8") as src:
                text = src.read().split("\n")[line - 1].strip()[:100]
        except OSError:
            text = "?"
        print(f"{path}:{line}: {text}")
    print(f"{len(rows)} production functions never ran", file=sys.stderr)
    return 1 if rows else 0


if __name__ == "__main__":
    sys.exit(main())
