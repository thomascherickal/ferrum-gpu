#!/usr/bin/env python3
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics_collect as mc  # noqa: E402

COVERAGE_MARK = ("<!-- coverage:begin -->", "<!-- coverage:end -->")
GATE_MARK = ("<!-- gate:begin -->", "<!-- gate:end -->")
NONE_YET = "_No run recorded yet._"


def pct(part, whole):
    return f"{100.0 * part / whole:.2f}%" if whole else "-"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def get_block(text, marks):
    """The text between the markers (without them), or None."""
    a, b = text.find(marks[0]), text.find(marks[1])
    return text[a + len(marks[0]):b].strip("\n") if 0 <= a < b else None


def wrap(marks, body):
    return f"{marks[0]}\n{body}\n{marks[1]}"


def parse_summary(text):
    """Rows of `cargo llvm-cov report --summary-only`: ([file rows], total). A row is a dict with
    name, regions, missed_regions, functions, missed_functions, lines, missed_lines."""
    rows, total = [], None
    for line in text.splitlines():
        p = line.split()
        if len(p) < 10 or not (p[0] == "TOTAL" or p[0].endswith(".rs")):
            continue
        try:
            r = dict(name=p[0], regions=int(p[1]), missed_regions=int(p[2]), functions=int(p[4]),
                     missed_functions=int(p[5]), lines=int(p[7]), missed_lines=int(p[8]))
        except ValueError:
            continue
        if p[0] == "TOTAL":
            total = r
        else:
            rows.append(r)
    return rows, total


def coverage_block(summary_text, floor, commit, when, root=""):
    """The markdown of the coverage section for one run (file names are made relative to `root`)."""
    rows, total = parse_summary(summary_text)
    for r in rows:
        r["name"] = r["name"].replace(root.rstrip("/") + "/", "", 1) if root else r["name"]
    if total is None:
        return NONE_YET
    by_crate = {}
    for r in rows:
        c = by_crate.setdefault(mc.crate_of(r["name"]), dict(lines=0, missed_lines=0, regions=0, missed_regions=0))
        for k in c:
            c[k] += r[k]
    lines = [
        f"Measured {when} on commit `{commit}` by `scripts/coverage.sh` (the whole workspace on the default "
        "adapter, then on no adapter, merged).",
        "",
        table(["measure", "covered", "total", "missed", "coverage"], [
            ["lines", total["lines"] - total["missed_lines"], total["lines"], total["missed_lines"], pct(total["lines"] - total["missed_lines"], total["lines"])],
            ["regions", total["regions"] - total["missed_regions"], total["regions"], total["missed_regions"], pct(total["regions"] - total["missed_regions"], total["regions"])],
            ["functions", total["functions"] - total["missed_functions"], total["functions"], total["missed_functions"], pct(total["functions"] - total["missed_functions"], total["functions"])],
        ]),
        "",
        f"Floor enforced by the gate: **{floor}%** of lines.",
        "",
        "### By crate",
        "",
        table(["crate", "lines", "missed", "line coverage", "region coverage"],
              [[k, v["lines"], v["missed_lines"], pct(v["lines"] - v["missed_lines"], v["lines"]),
                pct(v["regions"] - v["missed_regions"], v["regions"])] for k, v in sorted(by_crate.items())]),
        "",
        "### Least covered files",
        "",
    ]
    worst = sorted((r for r in rows if r["missed_lines"]), key=lambda r: (-r["missed_lines"], r["name"]))[:10]
    lines.append(table(["file", "missed lines", "of", "line coverage"],
                       [[r["name"], r["missed_lines"], r["lines"], pct(r["lines"] - r["missed_lines"], r["lines"])] for r in worst])
                 if worst else "Every file is fully covered.")
    return "\n".join(lines)


def seconds(hms):
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s


def gate_block(gate_text, verdict):
    """The markdown of the gate section from `gate.txt` ("<step> rc=<n> <HH:MM:SS>" per step, the
    first line being the start marker)."""
    steps = []
    for line in gate_text.splitlines():
        m = re.match(r"(\S+) rc=(\d+) (\d\d:\d\d:\d\d)$", line.strip())
        if m:
            steps.append((m.group(1), int(m.group(2)), seconds(m.group(3))))
    if len(steps) < 2:
        return NONE_YET
    rows, prev = [], steps[0][2]
    for name, rc, t in steps[1:]:
        d = (t - prev) % 86400
        rows.append([name, "pass" if rc == 0 else f"FAIL ({rc})", f"{d // 60}m {d % 60:02d}s"])
        prev = t
    total = (steps[-1][2] - steps[0][2]) % 86400
    rows.append(["**total**", verdict.strip() or "-", f"{total // 60}m {total % 60:02d}s"])
    return "Last run of `scripts/gate.sh` (target: under 20 minutes with warm builds).\n\n" + table(["step", "result", "time"], rows)


def render(root, next_commit, keep):
    files = mc.tracked(root)
    rust, sizes = mc.rust_table(root, files)
    langs = mc.language_table(root, files)
    hist = mc.history(root)
    if next_commit:
        hist["commits"] += 1
    head = mc.git(root, "rev-parse", "--short", "HEAD").strip() or "-"
    branch = mc.git(root, "rev-parse", "--abbrev-ref", "HEAD").strip() or "-"
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prod = sum(c["prod"] for c in rust.values())
    test = sum(c["test"] for c in rust.values())
    tests = sum(c["tests"] for c in rust.values())
    big = sorted(((n, p) for p, n in sizes.items() if n), reverse=True)
    over = [(p, n) for n, p in big if n > mc.MAX_PROD_LINES]
    lock = mc.lock_packages(mc.read(root, "Cargo.lock"))
    members = mc.workspace_members(mc.read(root, "Cargo.toml"))
    out = [
        "# Project metrics",
        "",
        "Generated by `scripts/metrics.py`; do not edit by hand. The git pre-commit hook (`scripts/githooks`) refreshes "
        "everything below except the two measured sections; `scripts/coverage.sh` refreshes the coverage section and "
        "`scripts/gate.sh` the gate section. See [DEVELOPING.md](docs/DEVELOPING.md).",
        "",
        f"Snapshot: {now}, branch `{branch}`, after `{head}`" + (" plus the commit being made." if next_commit else "."),
        "",
        "## Size",
        "",
        table(["language", "files", "lines"], [[k, v[0], v[1]] for k, v in sorted(langs.items(), key=lambda kv: -kv[1][1])]),
        "",
        f"Rust: **{prod}** lines of production code and **{test}** lines of test code (test lines are "
        f"{pct(test, prod + test)} of the Rust), non-blank lines.",
        "",
        "### Rust by crate",
        "",
        table(["crate", "files", "production lines", "test lines", "`#[test]` functions", "ignored"],
              [[k, v["files"], v["prod"], v["test"], v["tests"], v["ignored"]] for k, v in sorted(rust.items())]),
        "",
        "## Tests",
        "",
        f"**{tests}** `#[test]` functions counted in the source ({sum(c['ignored'] for c in rust.values())} marked "
        "`#[ignore]`, run by hand). The number of tests that ran, per run, is in `LAST-TEST-RESULTS.md`.",
        "",
        "## Structure",
        "",
        f"Rule 11 asks for files of about {mc.MAX_PROD_LINES} production lines or fewer: "
        f"**{len(over)}** Rust file(s) are over it.",
        "",
        table(["largest Rust files (production lines)", "lines"], [[p, n] for n, p in big[:10]]),
        "",
        "## Dependencies",
        "",
        f"{len(members)} workspace members; {lock} packages in `Cargo.lock` (every crate, ours included).",
        "",
        "## History",
        "",
        table(["commits", "first", "latest", "authors", "local branches", "tags"],
              [[hist["commits"], hist["first"], hist["last"], hist["authors"], hist["branches"], hist["tags"]]]),
        "",
        "## Gate",
        "",
        wrap(GATE_MARK, keep.get("gate") or NONE_YET),
        "",
        "## Test coverage",
        "",
        wrap(COVERAGE_MARK, keep.get("coverage") or NONE_YET),
        "",
    ]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--output", default=None)
    ap.add_argument("--next-commit", action="store_true")
    ap.add_argument("--coverage", metavar="SUMMARY_FILE")
    ap.add_argument("--floor", default=os.environ.get("FLOOR", "-"))
    ap.add_argument("--gate", metavar="GATE_TXT")
    ap.add_argument("--verdict", default="", help="PASS or FAIL for --gate")
    a = ap.parse_args(argv)
    out_path = a.output or os.path.join(a.root, "metrics.md")
    old = open(out_path, encoding="utf-8").read() if os.path.exists(out_path) else ""
    keep = {"gate": get_block(old, GATE_MARK), "coverage": get_block(old, COVERAGE_MARK)}
    commit = mc.git(a.root, "rev-parse", "--short", "HEAD").strip() or "-"
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
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


if __name__ == "__main__":
    sys.exit(main())
