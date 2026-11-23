"""What gate.py runs, in order, and the small decisions around it. Nothing here runs anything."""
import sys

PY = sys.executable
ADAPTERS = ["-p", "ferrum_wgpu", "-p", "ferrum_gpu_core"]
GENERATED = ["metrics.md", "next-steps.md", "test-time-report.md", "system-profile.md"]
STUDIO = [["cargo", "fmt", "--check"], ["cargo", "clippy", "--all-targets", "--locked", "--", "-D", "warnings"],
          ["cargo", "test", "--locked", "--", "--test-threads", "1"]]


def steps(out):
    """(name, commands, env, cwd): the commands run in order and stop at the first failure."""
    cargo_test = ["cargo", "test", *ADAPTERS, "--", "--test-threads", "1"]
    return [
        ("fmt", [["cargo", "fmt", "--all", "--", "--check"]], {}, None),
        ("clippy", [["cargo", "clippy", "--workspace", "--all-targets", "--", "-D", "warnings"]], {}, None),
        ("rules", [[PY, "scripts/rules_check.py"]], {}, None),
        ("dups", [[PY, "scripts/dup_check.py"]], {}, None),
        ("scripts", "script_tests", {}, None),
        ("llvmpipe", [cargo_test], {"FERRUM_GPU_ADAPTER": "llvmpipe"}, None),
        ("xe", [cargo_test], {"FERRUM_GPU_ADAPTER": "Iris(R) Xe Graphics (TGL GT2)"}, None),
        ("timelimit", [[PY, "scripts/time_limit.py", "0"]], {"TIMES_OUT": f"{out}/test-times.txt"}, None),
        ("timing", [[PY, "scripts/timing_report.py", f"{out}/test-times.txt"]], {}, None),
        ("coverage", [[PY, "scripts/coverage.py"]], {"FLOOR": "100"}, None),
        ("deny", [["cargo", "deny", "check"]], {}, None),
        ("studio", STUDIO, {"FERRUM_GPU_ADAPTER": "no-such-adapter"}, "ferrum_gpu_studio"),
        ("studiodeny", [["cargo", "deny", "--manifest-path", "ferrum_gpu_studio/Cargo.toml", "check"]], {}, None),
    ]


def step_line(name, rc, clock):
    """One line of gate.txt; `clock` is the time of day as HH:MM:SS."""
    return f"{name} rc={rc} {clock}"


def commit_record(porcelain, head):
    """What .gate/commit holds: HEAD when the tree is clean apart from the generated reports, else 'dirty'."""
    return "dirty" if porcelain.strip() else head.strip()


def status_args():
    """The `git status` arguments that leave out the generated reports."""
    return ["status", "--porcelain", "--", ".", *[f":!{g}" for g in GENERATED]]


def verdict(failed):
    return "FAIL" if failed else "PASS"
