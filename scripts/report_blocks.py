"""Part of metrics_reports.py."""
import re
import metrics_collect as mc
from report_tables import *  # noqa: F401,F403


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
