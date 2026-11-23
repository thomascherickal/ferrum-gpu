#!/usr/bin/env python3
"""Run every test of the workspace in its own process, with a hard time limit where one is set, and
FAIL if a test fails or exceeds its limit.

  scripts/time_limit.py [GENERAL_LIMIT_SECONDS] [extra `cargo test --no-run` arguments]

(`scripts/test-time-limit.sh` calls this.) Limits (user decision 2026-11-14): ordinary tests none by
default (0; pass a number to cap them), the `interop` test binary 600 s each or $INTEROP_LIMIT; 0 is
no limit. A process per test because libtest's `--ensure-time` never fires for workspace members and
nothing stops a hung test; the limit is a `subprocess` timeout. The binaries are built once by
cargo, then run directly. Doc-tests are not covered. $TIMES_OUT receives every test with its time
(for scripts/timing_report.py). Needs Python and cargo only.
"""
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runner  # noqa: E402
from time_limit_core import classify, limit_for, parse_artifacts, parse_limit, summary, listed_tests  # noqa: E402


def build(cargo_args):
    """Every test executable cargo builds, as (manifest dir, path)."""
    cmd = ["cargo", "test", "--no-run", "--message-format=json", *cargo_args]
    done = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, errors="replace")
    return parse_artifacts(done.stdout.splitlines())


def run_test(directory, binary, name, limit):
    env = {"CARGO_MANIFEST_DIR": directory}
    cmd = [binary, "--exact", name, "--test-threads", "1", "-q"]
    return runner.run(cmd, env, directory, subprocess.DEVNULL, subprocess.DEVNULL, limit or None)


def main(argv):
    if runner.help_requested(argv, __doc__):
        return 0
    general = parse_limit(argv[0]) if argv else 0
    interop = parse_limit(os.environ.get("INTEROP_LIMIT", "600"))
    if general is None or interop is None:
        print("limits must be whole numbers of seconds", file=sys.stderr)
        return 2
    os.environ.setdefault("CARGO_BUILD_JOBS", "3")
    binaries = build(argv[1:] or ["--workspace"])
    if not binaries:
        print("no test executables found", file=sys.stderr)
        return 2
    counts, failed, hung, times = {"pass": 0, "fail": 0, "hang": 0}, [], [], []
    for directory, binary in binaries:
        limit = limit_for(binary, general, interop)
        listing = subprocess.run([binary, "--list", "--format", "terse"], cwd=directory, capture_output=True,
                                 text=True, errors="replace").stdout
        for name in listed_tests(listing):
            began = time.time()
            rc = run_test(directory, binary, name, limit)
            times.append((time.time() - began, os.path.basename(binary), name))
            kind = classify(rc, limit)
            counts[kind] += 1
            if kind != "pass":
                (hung if kind == "hang" else failed).append(name)
                print(f"TIMEOUT (>{limit}s) {name}" if kind == "hang" else f"FAILED  {name} (exit {rc})")
    if os.environ.get("TIMES_OUT"):
        with open(os.environ["TIMES_OUT"], "w", encoding="utf-8") as f:
            f.writelines(f"{secs:.9f} {b} {n}\n" for secs, b, n in times)
    print("\n".join(summary(times, general, interop, counts, failed, hung)))
    return 0 if not failed and not hung else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
