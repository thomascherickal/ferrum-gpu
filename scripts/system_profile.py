"""CPU and GPU usage while a job runs, kept in system-profile.md.

  scripts/system_profile.py sample FILE      # sample every 2 s into FILE until FILE.stop exists
  scripts/system_profile.py render FILE      # add a row for that run to system-profile.md

The gate samples itself (scripts/gate.sh). A sample is: seconds, CPU busy %, memory in use (MiB), GPU
busy % and GPU memory (MiB); the GPU columns are empty without `nvidia-smi`. Without /proc (macOS, Windows) CPU is the load average over the CPU count (none on Windows) and memory is empty. A
column with no readings shows "-".
"""
import datetime
import os
import subprocess
import sys
import time
from profile_report import *  # noqa: F401,F403


if __name__ == "__main__":
    sys.exit(main())
