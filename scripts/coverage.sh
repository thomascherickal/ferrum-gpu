#!/usr/bin/env bash
# Line coverage of the whole workspace, merging a real-GPU run with a no-GPU run (the "skip when
# there is no adapter" branches only execute in the second one).
#   scripts/coverage.sh            # print the per-file table
#   FLOOR=99 scripts/coverage.sh   # also fail if total line coverage is below FLOOR
# The two runs are the two configurations the gate needs for the whole workspace (the default adapter
# and no adapter), so the gate does not run those separately; a failing test fails this script.
# Needs cargo-llvm-cov and the llvm-tools component. Run it alone, not next to other cargo jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"

# Only the old profile data goes: the instrumented build is kept, so a rerun compiles what changed and
# no more (cargo's own fingerprints rebuild a file that changed). After files MOVE or are deleted,
# remove target/llvm-cov-target by hand, or the report lists the old paths at 0%.
cargo llvm-cov clean --workspace --profraw-only
cargo llvm-cov --workspace --no-report -- --test-threads 1
FERRUM_GPU_ADAPTER=no-such-adapter cargo llvm-cov --workspace --no-report -- --test-threads 1

if [[ -n "${FLOOR:-}" ]]; then
  cargo llvm-cov report --summary-only --fail-under-lines "$FLOOR"
else
  cargo llvm-cov report --summary-only
fi
