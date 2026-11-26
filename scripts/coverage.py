#!/usr/bin/env python3
"""Line coverage of the whole workspace, merging a real-GPU run with a no-GPU run of the crates that open
an adapter (the "skip when there is no adapter" branches only execute in the second one). The
interoperability tests are not instrumented (the gate runs them in their own step, with their time limits).
  scripts/coverage.py            # print the per-file table (scripts/coverage.sh calls this)
  FLOOR=98 scripts/coverage.py   # also fail if total line coverage (the llvm-cov summary) is below FLOOR
Budget: under 20 minutes, cold or warm (the user's decision; the gate checks it). Where the time goes, and what keeps it down:
  * One instrumented build, kept in target/llvm-cov-target between runs: a rerun compiles only what changed.
    Never delete that directory unless files MOVED (the report would list the old paths at 0%).
  * `cargo nextest` runs every test in its own process in parallel; .config/nextest.toml makes the packages
    that open a GPU adapter run one at a time (several at once crash drivers) and the rest use all cores.
  * Two runs only: everything on the default adapter, then the adapter crates again with no adapter.
  * Each stage's seconds go to target/coverage-timing.txt and the end of the output.
Needs cargo-llvm-cov, cargo-nextest and the llvm-tools component. Run it alone, not next to other cargo jobs.
"""
import os
import sys
import time

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # tracked __pycache__ must stay as committed
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import coverage_plan as plan  # noqa: E402
import runner  # noqa: E402

CLEAN = ["cargo", "llvm-cov", "clean", "--workspace"]


def stage(name, cmd, env=None, quiet=False):
    """Run one stage and record its seconds; a failing stage ends the script with its status."""
    began = time.time()
    rc = runner.run_quiet(cmd, env) if quiet else runner.run(cmd, env)
    if rc != 0:
        sys.exit(rc or 1)
    line = plan.timing_line(name, time.time() - began)
    with open(plan.TIMING, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, file=sys.stderr)


def stored_layout():
    try:
        with open(plan.LAYOUT, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def main(argv=None):
    if runner.help_requested(sys.argv[1:] if argv is None else argv, __doc__):
        return 0
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.environ.setdefault("CARGO_BUILD_JOBS", "4")
    floor = os.environ.get("FLOOR", "")
    os.makedirs("target", exist_ok=True)
    open(plan.TIMING, "w", encoding="utf-8").close()
    start = time.time()
    # The old profile data goes, the instrumented build stays: only when the set of source files changed are the
    # workspace crates' own artifacts cleaned too (stale objects of moved files show up as bogus zero-count
    # entries). The dependencies stay built either way. `cargo llvm-cov clean` is cargo's cleaner, not `rm`.
    stage("clean-profraw", CLEAN + ["--profraw-only"], quiet=True)
    sources = runner.captured(["git", "ls-files", "-co", "--exclude-standard", "*.rs"]).splitlines()
    # A new compiler also invalidates the instrumented objects, so its version is part of the layout.
    layout = plan.layout_hash(sources + [runner.captured(["rustc", "-V"]).strip()])
    if stored_layout() != layout:
        stage("clean-workspace-artifacts", CLEAN, quiet=True)
        with open(plan.LAYOUT, "w", encoding="utf-8") as f:
            f.write(layout + "\n")
    stage("default-adapter", plan.run_one())
    stage("no-adapter", plan.run_two(), {"FERRUM_GPU_ADAPTER": "no-such-adapter"})
    # The summary is kept in target/ and given to metrics.md whether or not the floor is met, so the record of the
    # run is never lost to a failure. The floor decides the exit status.
    status = 0 if runner.run_tee(plan.summary_cmd(floor), plan.SUMMARY) == 0 else 1
    runner.run_quiet(plan.lcov_cmd())
    # the exact per-line check (every instrumented line executed); see scripts/lcov_check.py
    if runner.run([plan.PY, "scripts/lcov_check.py", plan.LCOV]) != 0:
        status = 1
    line = f"total {int(time.time() - start)}s (budget: under 1200s, 20 minutes)"
    with open(plan.TIMING, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, file=sys.stderr)
    if runner.run(plan.metrics_cmd(floor)) != 0:
        print("metrics.md was not updated", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
