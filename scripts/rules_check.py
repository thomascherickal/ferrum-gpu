#!/usr/bin/env python3
"""Check the repository against AGENTS.md: file length, Markdown names, required files.

  scripts/rules_check.py                  # exit 1 on a violation that is not already recorded debt
  scripts/rules_check.py --update         # rewrite scripts/rules-baseline.json to today's debt
  scripts/rules_check.py --report         # print every violation, recorded or not

AGENTS.md allows no file over 100 lines. Existing files over the limit are recorded in the baseline
as debt; a file may only get shorter, and a new file may not be over at all. The baseline is the
list of work left, not a permission.
"""
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rules_docs import missing_diagrams  # noqa: E402

MAX_LINES = 100
CODE = re.compile(r"\.(rs|js|mjs|py|sh|wgsl|html|css)$")
REQUIRED = ["readme.md", "next-steps.md", "metrics.md", "test-time-report.md", "system-profile.md"]
UPPER_OK = {"AGENTS.md", "GEMINI.md", "CLAUDE.md", "SKILL.md", "LICENSE.md"}
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASELINE = "scripts/rules-baseline.json"


def tracked(root):
    out = subprocess.run(["git", "-C", root, "ls-files", "--cached", "--others", "--exclude-standard"], capture_output=True, text=True).stdout
    return [p for p in out.split("\n") if p and not p.startswith(".claude/") and os.path.isfile(f"{root}/{p}")]


def length(root, path):
    with open(f"{root}/{path}", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def bad_md_name(path):
    name = os.path.basename(path)
    return path.endswith(".md") and name not in UPPER_OK and name != name.lower().replace("_", "-")


def violations(root, files=None):
    """{'long': {path: lines}, 'names': [path], 'missing': [file]} for the tracked files."""
    files = tracked(root) if files is None else files
    long = {p: n for p in files if CODE.search(p) and (n := length(root, p)) > MAX_LINES}
    names = sorted(p for p in files if bad_md_name(p))
    present = {p.lower() for p in files}
    return {"long": long, "names": names, "missing": [r for r in REQUIRED if r not in present],
            "diagrams": missing_diagrams(root, files)}


def load_baseline(root):
    try:
        with open(f"{root}/{BASELINE}", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"long": {}, "names": [], "missing": [], "diagrams": {}}


def new_problems(found, base):
    """What is worse than the recorded debt."""
    out = []
    for p, n in sorted(found["long"].items()):
        if n > base["long"].get(p, MAX_LINES):
            out.append(f"{p}: {n} lines (limit {MAX_LINES}" + (f", recorded {base['long'][p]})" if p in base["long"] else ")"))
    out += [f"{p}: {n} Mermaid diagram(s) missing (one per 200 words)" for p, n in sorted(found.get("diagrams", {}).items())
            if n > base.get("diagrams", {}).get(p, 0)]
    out += [f"{p}: Markdown file names are lowercase kebab-case" for p in found["names"] if p not in base["names"]]
    out += [f"{m}: required by AGENTS.md section 4" for m in found["missing"] if m not in base["missing"]]
    return out


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    root = ROOT
    found = violations(root)
    if "--update" in args:
        with open(f"{root}/{BASELINE}", "w", encoding="utf-8") as f:
            json.dump(found, f, indent=1, sort_keys=True)
            f.write("\n")
        print(f"baseline written: {len(found['long'])} long files, {len(found['names'])} names, {len(found['missing'])} missing")
        return 0
    if "--report" in args:
        for p, n in sorted(found["long"].items(), key=lambda kv: -kv[1]):
            print(f"{n:5d} {p}")
        print(f"{len(found['long'])} files over {MAX_LINES} lines; names: {found['names']}; missing: {found['missing']}")
        return 0
    problems = new_problems(found, load_baseline(root))
    for p in problems:
        print(f"rules: {p}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
