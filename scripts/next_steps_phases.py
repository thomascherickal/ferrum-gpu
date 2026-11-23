"""The phased plan in next-steps.md: each phase's tasks, how it is judged done, and which phase is current."""

def phase_list(inp):
    """[(title, done, [task], done_when)] from the same inputs as the checklist."""
    cov, (verdict, failed) = inp["coverage"], inp["gate"]
    covered = cov is not None and cov[1] == 0
    unmerged = inp["unmerged"]
    return [
        ("Phase 1: prove every error path with simulated failures", covered, [
            "List the functions that never ran: `cargo llvm-cov report --json --output-path target/cov.json`, then `python3 scripts/missed_functions.py target/cov.json`.",
            "Cover each with a simulated failure (a directory where a file is expected, corrupt input, the failing writers, the GPU fault injection, an injected environment).",
            "Re-measure with `scripts/coverage.sh` (" + (f"{cov[1]} lines missed in the summary now" if cov else "no run recorded yet") + "; zero lines uncovered by lcov is enforced there).",
        ], "the summary reports 0 missed lines and 0 missed functions"),
        ("Phase 2: a green gate on this tree", verdict == "PASS", [
            "Run `scripts/gate.sh`: fmt, clippy, rules, duplicates, script tests, llvmpipe and Xe, timing, coverage, deny, the desktop app.",
            "Fix what it reports" + (f" (failed: {', '.join(failed)})" if failed else "") + "; it also writes `test-time-report.md` and `system-profile.md`.",
        ], "`.gate/gate.done` says PASS for this commit"),
        ("Phase 3: merge and push", not unmerged, [
            *[f"Merge `{b}` ({n} commits ahead of main: {s})." for b, n, s in unmerged],
            "Merge into `main`, re-check the merged result, then `git push origin main` (pushing is automatic by the user's standing decision).",
        ], "no branch is ahead of `main` and `main` is pushed"),
        ("Phase 4: quality ratchets", inp["dups"] == 0, [
            f"Remove duplicated code ({inp['dups']} runs recorded; `scripts/dup_check.py --report`), then `--update`.",
            "Narrow visibility further (`cargo clippy --workspace --all-targets -- -W unreachable_pub`), then SOLID review of the largest modules.",
        ], "`scripts/dup_check.py` reports 0 duplicated runs"),
        ("Phase 5: the backlog", not inp["backlog"], inp["backlog"] or ["Pick the next item from docs/matcher-plan.md."], "`docs/backlog.md` has no open item"),
        ("Phase 6: housekeeping", False, [
            "Delete branches and worktrees that are fully merged (`git branch -d`, never `-D`).",
            "Refresh `handoff.md` and the readme's measured numbers.",
        ], "nothing stale is left behind"),
    ]


def render(phases):
    """The markdown for the plan; the first unfinished phase is marked as the current one."""
    out, current = [], next((i for i, p in enumerate(phases) if not p[1]), None)
    for i, (title, done, tasks, done_when) in enumerate(phases):
        mark = "[x]" if done else "[ ]"
        here = " **(current)**" if i == current else ""
        out += [f"### {mark} {title}{here}", ""] + [f"- {t}" for t in tasks] + [f"- Done when: {done_when}.", ""]
    return "\n".join(out)
