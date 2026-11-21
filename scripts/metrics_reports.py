"""Formatting for metrics.md's measured sections: tables, the coverage block from an llvm-cov summary and the
gate block from gate.txt, plus the markers that keep those sections between regenerations."""
import re

import metrics_collect as mc

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
