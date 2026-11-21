"""Regenerate next-steps.md: AGENTS.md's post-commit/push checklist with its live status, and the way forward.

  scripts/next_steps.py            # rewrite next-steps.md from the repository and its generated reports

Every status comes from a file this repository already writes (metrics.md, test-time-report.md,
system-profile.md, .gate/, scripts/rules-baseline.json) or from git; nothing is guessed. The hooks run it
before each commit and each push (scripts/githooks), so the file is current after both.
"""
import json
import os
import re
import subprocess
import sys
import metrics_collect as mc  # noqa: E402
from next_steps_document import *  # noqa: F401,F403


if __name__ == "__main__":
    sys.exit(main())
