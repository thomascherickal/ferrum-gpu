#!/usr/bin/env bash
# Line coverage of the whole workspace, merging a real-GPU run with a no-GPU run (the "skip when
# there is no adapter" branches only execute in the second one).
#   scripts/coverage.sh            # print the per-file table
#   FLOOR=99 scripts/coverage.sh   # also fail if total line coverage is below FLOOR
# Needs cargo-llvm-cov and the llvm-tools component. Run it alone, not next to other cargo jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"

cargo llvm-cov clean --workspace
cargo llvm-cov --workspace --no-report -- --test-threads 1
FERRUM_GPU_ADAPTER=no-such-adapter cargo llvm-cov --workspace --no-report -- --test-threads 1

if [[ -n "${FLOOR:-}" ]]; then
  cargo llvm-cov report --summary-only --fail-under-lines "$FLOOR"
else
  cargo llvm-cov report --summary-only
fi
