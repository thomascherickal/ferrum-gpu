"""Formatting for metrics.md's measured sections: tables, the coverage block from an llvm-cov summary and the
gate block from gate.txt, plus the markers that keep those sections between regenerations."""
import re
import metrics_collect as mc
from report_blocks import *  # noqa: F401,F403



