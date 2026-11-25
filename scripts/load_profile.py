#!/usr/bin/env python3
"""Load-test figures in system-profile.md, from real runs only (never typed by hand).

  scripts/load_profile.py run [--heavy]   # run the load tests (debug build, one cargo command at a time), record their figures
  scripts/load_profile.py render FILE     # record the `LOAD ...` lines found in a saved test output

The tests (ferrum_gpu_core serving::boot::tests::load, ferrum_gpu_gui --test api *load*) print one
`LOAD name=... clients=... p50_ms=...` line each; this writes them as a table between the load markers of
system-profile.md, which scripts/system_profile.py keeps when it adds a gate row.
"""
import datetime
import os
import subprocess
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LBEGIN, LEND = "<!-- load:begin -->", "<!-- load:end -->"
COLUMNS = ["name", "clients", "requests", "ok", "refused", "seconds", "rps", "p50_ms", "p95_ms", "p99_ms", "max_ms"]
TESTS = [["-p", "ferrum_gpu_core", "--lib", "serving::boot::tests::load"],
         ["-p", "ferrum_gpu_core", "--lib", "http::tests::past_the_connection_cap"], ["-p", "ferrum_gpu_gui", "--test", "api", "load"]]


def parse(text):
    """{column: value} for every `LOAD k=v ...` line of `text`."""
    rows = []
    for line in text.splitlines():
        at = line.find("LOAD name=")  # libtest may print `test x ... ` before it
        if at >= 0:
            rows.append(dict(kv.split("=", 1) for kv in line[at:].split()[1:] if "=" in kv))
    return rows


def block(rows, when, commit, build):
    """The load section: when and how it was measured, then one table row per run."""
    head = (f"Load tests, measured {when} on `{commit}` ({build} build; `scripts/load_profile.py run`). "
            "Latency is per request, from connect to the last byte; a refusal is the documented 503 past the "
            "in-flight cap.")
    lines = [LBEGIN, head, "", "| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    lines += ["| " + " | ".join(r.get(c, "-") for c in COLUMNS) + " |" for r in rows]
    return "\n".join(lines + [LEND])


def splice(doc, section):
    """`doc` with its load section replaced by `section` (appended when there is none)."""
    a, b = doc.find(LBEGIN), doc.find(LEND)
    if 0 <= a < b:
        return doc[:a] + section + doc[b + len(LEND):]
    return doc.rstrip("\n") + "\n\n" + section + "\n"


def record(text, build, root=ROOT):
    rows = parse(text)
    if not rows:
        print("no LOAD lines", file=sys.stderr)
        return 1
    commit = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True).stdout.strip() or "-"
    when = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
    path = f"{root}/system-profile.md"
    old = open(path).read() if os.path.exists(path) else ""
    with open(path, "w") as f:
        f.write(splice(old, block(rows, when, commit, build)))
    return 0


def run(heavy):
    out = ""
    extra = ["--include-ignored"] if heavy else []  # the heavy runs as well
    for t in TESTS:
        cmd = ["cargo", "test", *t, "--", *extra, "--nocapture", "--test-threads", "1"]
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        out += p.stdout
        if p.returncode != 0:
            print(p.stdout[-2000:] + p.stderr[-2000:], file=sys.stderr)
            return p.returncode, out
    return 0, out


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) == 2 and a[0] == "render":
        return record(open(a[1]).read(), "saved output")
    if a and a[0] == "run" and set(a[1:]) <= {"--heavy"}:
        code, out = run("--heavy" in a)
        return code or record(out, "debug; heavy runs included" if "--heavy" in a else "debug")
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
