"""Regenerate metrics.md: code size, tests, structure, dependencies, history, the last gate's timing
and the last coverage run.

  scripts/metrics.py                       # rewrite metrics.md from the repository as it is now
  scripts/metrics.py --next-commit         # same, counting the commit being made (the pre-commit hook)
  scripts/metrics.py --coverage FILE       # also replace the coverage section with the llvm-cov
                                           # summary in FILE (scripts/coverage.sh does this)
  scripts/metrics.py --gate gate.txt       # also replace the gate section (scripts/gate.sh does this)

Everything is recomputed on each run except the two measured sections (gate, coverage), which only a
run of the gate or of coverage can change; they are kept between markers and carried over untouched.
Line counts are non-blank lines; "test" means a test file or the part of a file from its first
`#[cfg(test)]` on. Nothing here is estimated: a number that cannot be read is shown as "-".
"""
import sys
from metrics_render_2 import *  # noqa: F401,F403


if __name__ == "__main__":
    sys.exit(main(usage=__doc__))
