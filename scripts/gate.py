#!/usr/bin/env python3
"""The full gate (AGENTS.md sections 1, 2 and 6; CLAUDE.md rules 3 and 15), one step at a time, results in
$GATE_OUT (default ./.gate):
  gate.txt   one line per step: "<step> rc=<exit status> <time>" (the first line is the start marker)
  gate.done  PASS or FAIL when everything has run (RUNNING while it does)
  key        what the pre-commit hook compares (scripts/gate_key.py)
  commit     the commit the gate ran on, or "dirty" if the tree had uncommitted changes (scripts/gate_ok.py)
  <step>.log the output of each step
Steps (scripts/gate_steps.py): fmt, clippy, the repository rules, duplicate code, the script tests, the GPU crates'
tests on llvmpipe and on the Intel Xe adapter, every test in its own process (the live timing for
test-time-report.md), coverage (floor 100 and zero uncovered lines), cargo deny, and the desktop app (its own
workspace): fmt, clippy, tests, deny. CPU and GPU usage is sampled throughout (system-profile.md).
  scripts/gate.py                 # or scripts/gate.sh; run from anywhere in the repository (or worktree)
One cargo command at a time on the low-memory box (rule 7). Target: under 30 minutes (AGENTS.md). No file is
deleted (the command ban list): files are overwritten, and a run is "not done" until gate.done says so.
Gotchas: a stale `target/llvm-cov-target` from before files moved reports the old paths at 0%; `cargo test` stops
at the first failing test binary; the no-unsafe guard is a text match, so a test must not contain the word.
"""
import glob
import os
import sys
import time

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"  # tracked __pycache__ must stay as committed
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate_steps  # noqa: E402
import runner  # noqa: E402

PY = gate_steps.PY


def script_tests(log):
    """Every scripts/test_*.py, stopping at the first that fails; their output goes to `log`."""
    for test in sorted(glob.glob("scripts/test_*.py")):
        if runner.run_logged([PY, test], log, mode="a") != 0:
            return 1
    return 0


def run_step(commands, env, cwd, log):
    """Run the commands in turn (stopping at the first failure); the status of the last one run."""
    if commands == "script_tests":
        open(log, "w", encoding="utf-8").close()
        return script_tests(log)
    rc = 0
    for i, cmd in enumerate(commands):
        rc = runner.run_logged(cmd, log, env, cwd, mode="w" if i == 0 else "a")
        if rc != 0:
            break
    return rc


def refresh(cmd, what):
    """Run a report generator; a failure is a warning, never a failed gate."""
    if runner.run(cmd) != 0:
        print(f"{what} was not updated", file=sys.stderr)


def write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main(argv=None):
    if runner.help_requested(sys.argv[1:] if argv is None else argv, __doc__):
        return 0
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = os.environ.get("GATE_OUT") or os.path.join(os.getcwd(), ".gate")
    os.makedirs(out, exist_ok=True)
    os.environ.setdefault("CARGO_BUILD_JOBS", "3")
    write(f"{out}/gate.done", "RUNNING\n")
    profile = f"{out}/profile-{os.getpid()}.csv"
    write(f"{out}/key", runner.captured([PY, "scripts/gate_key.py"]))
    clock = lambda: time.strftime("%H:%M:%S")  # noqa: E731
    write(f"{out}/gate.txt", gate_steps.step_line("start", 0, clock()) + "\n")
    head = runner.captured(["git", "rev-parse", "HEAD"])
    write(f"{out}/commit", gate_steps.commit_record(runner.captured(["git", *gate_steps.status_args()]), head) + "\n")
    sampler = runner.start([PY, "scripts/system_profile.py", "sample", profile])
    failed, took = False, {}
    for name, commands, env, cwd in gate_steps.steps(out):
        rc, took[name] = runner.timed(run_step, commands, env, cwd, f"{out}/{name}.log")
        failed = failed or rc != 0
        with open(f"{out}/gate.txt", "a", encoding="utf-8") as f:
            f.write(gate_steps.step_line(name, rc, clock()) + "\n")
    slow = gate_steps.over_budget(took)
    print(*slow, sep="\n", file=sys.stderr)
    open(f"{out}/gate.txt", "a", encoding="utf-8").write(gate_steps.step_line("budget", int(bool(slow)), clock()) + "\n")
    write(f"{profile}.stop", "")
    sampler.wait()
    refresh([PY, "scripts/system_profile.py", "render", profile], "system-profile.md")
    write(f"{out}/gate.done", gate_steps.verdict(failed or bool(slow)) + "\n")
    refresh([PY, "scripts/metrics.py", "--gate", f"{out}/gate.txt"], "metrics.md")
    refresh([PY, "scripts/next_steps.py"], "next-steps.md")
    print(open(f"{out}/gate.txt", encoding="utf-8").read() + open(f"{out}/gate.done", encoding="utf-8").read(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
