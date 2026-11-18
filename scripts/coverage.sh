#!/usr/bin/env bash
# Line coverage of the whole workspace, merging a real-GPU run with a no-GPU run of the crates that open
# an adapter (the "skip when there is no adapter" branches only execute in the second one). The
# interoperability tests are not instrumented (see the comment at the runs).
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
# What each run executes, and what it leaves out, measured 2026-11-18 (instrumented tests of each crate,
# warm build): interop 668 s and the browser tests of the GUI crate 316 s of 1172 s in all. The
# interoperability tests run the real llama.cpp and Ollama; the gate runs them in their own step (with
# their time limits), so the instrumented copy only repeated that work. Run 1 is everything else on the
# default adapter. Run 2, with no adapter, exists for the "skip when there is no adapter" branches, which
# only the crates that open an adapter in their tests have, so it runs those and nothing else.
cargo llvm-cov --workspace --exclude ferrum_gpu_cli --no-report -- --test-threads 1
cargo llvm-cov -p ferrum_gpu_cli --bins --test e2e --no-report -- --test-threads 1
export FERRUM_GPU_ADAPTER=no-such-adapter
cargo llvm-cov -p ferrum_wgpu -p ferrum_gpu_core --no-report -- --test-threads 1
cargo llvm-cov -p ferrum_gpu_gui --lib --bins --test api --test binary --no-report -- --test-threads 1
unset FERRUM_GPU_ADAPTER

# The summary is kept in target/ and given to metrics.md whether or not the floor is met, so the
# record of the run is never lost to a failure. The floor decides the exit status.
summary=target/coverage-summary.txt
mkdir -p target
set +e
if [[ -n "${FLOOR:-}" ]]; then
  cargo llvm-cov report --summary-only --fail-under-lines "$FLOOR" 2>&1 | tee "$summary"
else
  cargo llvm-cov report --summary-only 2>&1 | tee "$summary"
fi
status=${PIPESTATUS[0]}
set -e
python3 scripts/metrics.py --coverage "$summary" --floor "${FLOOR:-none}" || echo "metrics.md was not updated" >&2
exit "$status"
