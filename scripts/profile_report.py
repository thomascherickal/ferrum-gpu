"""CPU and GPU usage while a job runs, kept in system-profile.md.

  scripts/system_profile.py sample FILE      # sample every 2 s into FILE until FILE.stop exists
  scripts/system_profile.py render FILE      # add a row for that run to system-profile.md

The gate samples itself (scripts/gate.sh). A sample is: seconds, CPU busy %, memory in use (MiB), GPU
busy % and GPU memory (MiB); the GPU columns are empty without `nvidia-smi`. Nothing is estimated: a
column with no readings shows "-".
"""
import datetime
import os
import subprocess
import sys
import time
from profile_sampling import *  # noqa: F401,F403


def render(old, new_row):
    """The document with `new_row` first in the history table, keeping the last KEEP rows."""
    a, b = old.find(BEGIN), old.find(END)
    rows = [l for l in old[a:b].splitlines() if l.startswith("| 20")] if 0 <= a < b else []
    rows = ([new_row] + rows)[:KEEP]
    return "\n".join([
        "# System profile", "",
        "CPU and GPU usage during the gate (`scripts/gate.sh` samples every 2 s with `scripts/system_profile.py`); "
        "do not edit by hand. A GPU column is `-` when `nvidia-smi` reported nothing. Look for rising memory or CPU "
        "peaks across rows: that is a resource regression.", "",
        BEGIN, HEADER, "|" + "---|" * 10, *rows, END, ""])


def main(argv=None):
    a = sys.argv[1:] if argv is None else argv
    if len(a) == 2 and a[0] == "sample":
        sample(a[1])
        return 0
    if len(a) == 2 and a[0] == "render":
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        s = summarize(open(a[1]).read())
        if s is None:
            print("no samples", file=sys.stderr)
            return 1
        commit = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "-"
        when = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
        path = f"{root}/system-profile.md"
        old = open(path).read() if os.path.exists(path) else ""
        open(path, "w").write(render(old, row(s, commit, "gate", when)))
        return 0
    print(__doc__)
    return 2
