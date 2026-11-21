"""Regenerate metrics.md: code size, tests, structure, dependencies, history, the last gate's timing
and the last coverage run.

  scripts/metrics.py                       # rewrite metrics.md from the repository as it is now
  scripts/metrics.py --next-commit         # same, counting the commit being made (the pre-commit hook)
  scripts/metrics.py --coverage FILE       # also replace the coverage section with the llvm-cov
                                           # summary in FILE (scripts/coverage.sh does this)
  scripts/metrics.py --gate gate.txt       # also replace the gate section (scripts/gate.sh does this)

Everything is recomputed on each run except the two measured sections (gate, coverage), which only a
run of the gate or of coverage can change; they are kept between markers and carried over untouched.
Line counts are non-blank lines; "test" means a test file or the part of a file from its first
`#[cfg(test)]` on. Nothing here is estimated: a number that cannot be read is shown as "-".
"""
import argparse
import datetime
import os
import re
import sys
import metrics_collect as mc  # noqa: E402
from metrics_reports import (  # noqa: E402,F401
    COVERAGE_MARK, GATE_MARK, NONE_YET, coverage_block, gate_block, get_block, parse_summary, pct, seconds, table, wrap,
)
from metrics_render import *  # noqa: F401,F403


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--output", default=None)
    ap.add_argument("--next-commit", action="store_true")
    ap.add_argument("--coverage", metavar="SUMMARY_FILE")
    ap.add_argument("--floor", default=os.environ.get("FLOOR", "-"))
    ap.add_argument("--gate", metavar="GATE_TXT")
    ap.add_argument("--verdict", default="", help="PASS or FAIL for --gate")
    ap.add_argument("--commit", default=None, help="the commit a --coverage run measured (default: HEAD)")
    ap.add_argument("--when", default=None, help="when a --coverage run was made (default: now)")
    a = ap.parse_args(argv)
    out_path = a.output or os.path.join(a.root, "metrics.md")
    old = open(out_path, encoding="utf-8").read() if os.path.exists(out_path) else ""
    keep = {"gate": get_block(old, GATE_MARK), "coverage": get_block(old, COVERAGE_MARK)}
    commit = a.commit or mc.git(a.root, "rev-parse", "--short", "HEAD").strip() or "-"
    now = a.when or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    if a.coverage:
        keep["coverage"] = coverage_block(open(a.coverage, encoding="utf-8").read(), a.floor, commit, now, a.root)
    if a.gate:
        verdict = a.verdict
        done = os.path.join(os.path.dirname(a.gate), "gate.done")
        if not verdict and os.path.exists(done):
            verdict = open(done).read()
        keep["gate"] = gate_block(open(a.gate, encoding="utf-8").read(), verdict)
    new = render(a.root, a.next_commit, keep)
    if new != old:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(new)
    return 0
