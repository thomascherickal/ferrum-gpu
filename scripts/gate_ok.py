#!/usr/bin/env python3
"""Did a gate pass on exactly the tree that is about to be pushed?

  scripts/gate_ok.py        # exit 0 if some worktree's .gate says PASS for a tree equal to HEAD's

`scripts/gate.sh` records the commit it ran on (or "dirty" when the tree had uncommitted changes) in
`.gate/commit`. A push is allowed when that commit and HEAD differ only in the generated reports
(metrics.md, next-steps.md, test-time-report.md, system-profile.md): those change after every gate by
design. The pre-push hook calls this; AGENTS.md: no push without a passing gate.
"""
import os
import subprocess
import sys

GENERATED = ["metrics.md", "next-steps.md", "test-time-report.md", "system-profile.md"]


def git(root, *args):
    r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True)
    return r.returncode, r.stdout


def worktrees(root):
    return [l[9:] for l in git(root, "worktree", "list", "--porcelain")[1].splitlines() if l.startswith("worktree ")]


def gate_state(path):
    """(verdict, commit) recorded in `path`/.gate, or (None, None)."""
    try:
        with open(f"{path}/.gate/gate.done") as f:
            done = f.read().strip()
        with open(f"{path}/.gate/commit") as f:
            commit = f.read().strip()
    except OSError:
        return None, None
    return done, commit


def passed_on_this_tree(root):
    for w in worktrees(root):
        done, commit = gate_state(w)
        if done != "PASS" or not commit or commit == "dirty":
            continue
        same, _ = git(root, "diff", "--quiet", commit, "HEAD", "--", ".", *[f":!{g}" for g in GENERATED])
        if same == 0:
            return True
    return False


def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gate_key  # the content key the pre-commit hook and the gate use

    if gate_key.check(root)[0] == 0 or passed_on_this_tree(root):
        return 0
    print("push refused: no passing `scripts/gate.sh` on this tree (run it, or set FERRUM_SKIP_GATE_CHECK=1 for a docs-only push)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
