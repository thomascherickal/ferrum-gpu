#!/usr/bin/env python3
"""Exact line coverage from an lcov file: every instrumented line must have run at least once.

  scripts/lcov_check.py target/coverage.lcov     # prints the uncovered lines; exit 1 if there are any

`cargo llvm-cov --summary-only` counts lines per function, so a closure that never runs (an error
branch inside `map_err`) is a missed line there even when its line ran in the enclosing function. This
check is the plain question "did every line execute", and the gate enforces it at zero. Both figures are
in metrics.md.
"""
import collections
import sys

IGNORED = ("/target/", "/.cargo/", "/rustc/")


def uncovered(text):
    """({file: [line, ...]} of lines executed zero times, total instrumented lines)."""
    miss, total, cur = collections.defaultdict(list), 0, None
    for raw in text.splitlines():
        if raw.startswith("SF:"):
            cur = raw[3:]
        elif raw.startswith("DA:") and cur and not any(i in cur for i in IGNORED):
            line, count = raw[3:].split(",")[:2]
            total += 1
            if count == "0":
                miss[cur].append(int(line))
    return dict(miss), total


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__)
        return 2
    miss, total = uncovered(open(args[0], encoding="utf-8").read())
    for f, lines in sorted(miss.items()):
        print(f"{f}: {lines}")
    n = sum(len(v) for v in miss.values())
    print(f"lcov: {n} uncovered lines of {total} instrumented ({100 * (total - n) / max(total, 1):.3f}% covered)")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
