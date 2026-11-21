"""Part of metrics_reports.py."""
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
