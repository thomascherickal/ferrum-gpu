#!/usr/bin/env python3
"""Did the gate pass on exactly the content being committed?

  scripts/gate_key.py            # print the key of the staged content (scripts/gate.sh records it at its start)
  scripts/gate_key.py --check    # exit 0 if the last gate PASSED on that key; 1 = no passing gate; 3 = unstaged changes

The key is a hash of the index (mode, blob and path of every file) leaving out the generated reports, which
the gate and the commit hook rewrite by design. The gate tests the working tree, so the working tree must equal
the index: everything staged (`git add -A`) and nothing changed since. The pre-commit hook uses this to
enforce AGENTS.md: the gate passes before every commit.
"""
import hashlib
import os
import subprocess
import sys

# readme.md carries a block the commit hook rewrites from metrics.md; the gate does not test prose
GENERATED = {"metrics.md", "next-steps.md", "test-time-report.md", "system-profile.md", "readme.md"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STAGED_ONLY = set("MARCD")


def git(root, *args):
    return subprocess.run(["git", "-C", root, *args], capture_output=True, text=True).stdout


def key(root):
    """Hash of the staged content without the generated reports."""
    rows = [r for r in git(root, "ls-files", "-s").split("\n") if r and r.split("\t", 1)[-1] not in GENERATED]
    return hashlib.sha1("\n".join(rows).encode()).hexdigest()


def unstaged(root):
    """Paths whose working-tree content differs from the index (untracked included), generated ones apart."""
    out = []
    for line in git(root, "status", "--porcelain").split("\n"):
        if len(line) > 3 and line[3:] not in GENERATED and not line[3:].startswith(".gate/") and not (line[0] in STAGED_ONLY and line[1] == " "):
            out.append(line[3:])
    return out


def read(root, rel):
    try:
        with open(f"{root}/{rel}", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


def check(root):
    """(exit code, message)."""
    loose = unstaged(root)
    if loose:
        return 3, "unstaged changes (run `git add -A` first, the gate tests the working tree): " + ", ".join(loose[:5])
    if read(root, ".gate/gate.done") == "PASS" and read(root, ".gate/key") == key(root):
        return 0, "the gate passed on this content"
    return 1, "no passing gate for this content"


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    if "--check" in args:
        code, message = check(ROOT)
        print(f"gate: {message}", file=sys.stderr)
        return code
    print(key(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
