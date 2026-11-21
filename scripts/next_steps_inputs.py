"""Part of next_steps.py."""
import json
import os
import re
import subprocess
import sys
import metrics_collect as mc  # noqa: E402


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(root, rel):
    try:
        with open(f"{root}/{rel}", encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def coverage(metrics):
    """(line coverage %, missed lines) from metrics.md, or None."""
    m = re.search(r"\| lines \| \d+ \| \d+ \| (\d+) \| ([\d.]+)%", metrics)
    return (float(m.group(2)), int(m.group(1))) if m else None


def least_covered(metrics, n=5):
    return re.findall(r"^\| (\S+\.rs) \| (\d+) \| \d+ \|", metrics, re.M)[:n]


def suite_minutes(report):
    m = re.search(r"\*\*([\d.]+) minutes\*\*", report)
    return float(m.group(1)) if m else None


def gate_state(gate_txt, done):
    """(verdict, failed steps) from the gate files."""
    failed = re.findall(r"^(\w+) rc=([1-9]\d*)", gate_txt, re.M)
    return (done.strip() or "-", [name for name, _ in failed])


def unmerged(root):
    out = mc.git(root, "for-each-ref", "--format=%(refname:short)", "refs/heads").split()
    rows = []
    for b in out:
        if b == "main":
            continue
        ahead = mc.git(root, "rev-list", "--count", f"main..{b}").strip()
        if ahead not in ("", "0"):
            rows.append((b, int(ahead), mc.git(root, "log", "-1", "--format=%s", b).strip()))
    return rows


def status(ok, text):
    return f"{'[x]' if ok else '[ ]'} {text}"
