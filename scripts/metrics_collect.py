"""Collecting the numbers behind metrics.md from the repository: files, lines, tests, dependencies,
git history. Pure functions over text and paths where possible, so they can be tested without a
repository; the few that run git take the root directory."""
import collections
import re
import subprocess
from metrics_tables import *  # noqa: F401,F403



