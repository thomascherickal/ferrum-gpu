#!/usr/bin/env bash
# A wrapper kept for the old name and for the hooks, CI and docs that call it; the work is in all_tests.py.
exec python3 "$(dirname "$0")/all_tests.py" "$@"
