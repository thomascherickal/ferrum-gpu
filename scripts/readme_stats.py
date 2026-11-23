#!/usr/bin/env python3
"""Keep the measured numbers in readme.md current: a block between markers, written from metrics.md.

  scripts/readme_stats.py        # rewrite the block in readme.md (the pre-commit hook does this)

Nothing is typed by hand: each figure is read from the generated `metrics.md`, and a figure it does not
have is left out rather than guessed.
"""
import os
import re
import sys

BEGIN, END = "<!-- stats:begin -->", "<!-- stats:end -->"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def facts(metrics):
    """{name: text} of the figures `metrics.md` states, for the ones it has."""
    found = {}
    pairs = {
        "snapshot": r"after `([0-9a-f]+)`",
        "tests": r"\*\*(\d+)\*\* `#\[test\]` functions",
        "production": r"Rust: \*\*(\d+)\*\* lines of production code",
        "test lines": r"and \*\*(\d+)\*\* lines of test code",
        "lines": r"\| lines \| \d+ \| \d+ \| \d+ \| ([\d.]+)%",
        "regions": r"\| regions \| \d+ \| \d+ \| \d+ \| ([\d.]+)%",
        "functions": r"\| functions \| \d+ \| \d+ \| \d+ \| ([\d.]+)%",
        "gate": r"\| \*\*total\*\* \| (\w+) \| ([^|]+)\|",
    }
    for key, pat in pairs.items():
        m = re.search(pat, metrics)
        if m:
            found[key] = " in ".join(g.strip() for g in m.groups())
    return found


def block(metrics):
    f = facts(metrics)
    rows = [("`#[test]` functions", f.get("tests")), ("Rust production lines", f.get("production")),
            ("Rust test lines", f.get("test lines")), ("line coverage", f.get("lines") and f["lines"] + "%"),
            ("function coverage", f.get("functions") and f["functions"] + "%"),
            ("region coverage", f.get("regions") and f["regions"] + "%"), ("last gate", f.get("gate"))]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows if v)
    return (f"{BEGIN}\nMeasured figures, written from [`metrics.md`](metrics.md) by `scripts/readme_stats.py` "
            f"(snapshot `{f.get('snapshot', '-')}`); do not edit by hand.\n\n| measure | value |\n|---|---|\n{body}\n{END}")


def apply(readme, metrics):
    new = block(metrics)
    a, b = readme.find(BEGIN), readme.find(END)
    if a >= 0 and b > a:
        return readme[:a] + new + readme[b + len(END):]
    return readme


def main(argv=None):
    root = ROOT if not argv else argv[0]
    paths = [f"{root}/readme.md", f"{root}/metrics.md"]
    if not all(os.path.exists(p) for p in paths):
        return 0
    readme, metrics = (open(p, encoding="utf-8").read() for p in paths)
    out = apply(readme, metrics)
    if out != readme:
        open(paths[0], "w", encoding="utf-8").write(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
