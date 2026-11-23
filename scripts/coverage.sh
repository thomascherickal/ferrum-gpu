#!/usr/bin/env bash
# Line coverage of the whole workspace, merging a real-GPU run with a no-GPU run of the crates that open
# an adapter (the "skip when there is no adapter" branches only execute in the second one). The
# interoperability tests are not instrumented (the gate runs them in their own step, with their time limits).
#   scripts/coverage.sh            # print the per-file table
#   FLOOR=98 scripts/coverage.sh   # also fail if total line coverage (the llvm-cov summary) is below FLOOR
# Target: under 30 minutes, cold or warm (AGENTS.md). Where the time goes, and what keeps it down:
#   * One instrumented build, kept in target/llvm-cov-target between runs: a rerun compiles only what changed.
#     Never delete that directory unless files MOVED (the report would list the old paths at 0%).
#   * `cargo nextest` runs every test in its own process in parallel; .config/nextest.toml makes the packages
#     that open a GPU adapter run one at a time (several at once crash drivers) and everything else use all cores.
#   * Two runs only: everything on the default adapter, then the adapter crates again with no adapter.
#   * Each stage's seconds go to target/coverage-timing.txt and the end of the output.
# Needs cargo-llvm-cov, cargo-nextest and the llvm-tools component. Run it alone, not next to other cargo jobs.
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"
mkdir -p target
timing=target/coverage-timing.txt
: > "$timing"
start=$SECONDS
stage() { # stage <name> <command...>: run it and record its seconds
  local name=$1 began=$SECONDS
  shift
  "$@"
  echo "$name $((SECONDS - began))s" | tee -a "$timing" >&2
}

# The old profile data goes, and the instrumented build stays: only when the set of source files changed (a file
# moved, was added or removed) are the workspace crates' own instrumented artifacts cleaned too, because stale
# objects of moved files show up in the report as bogus zero-count entries. The dependencies (wgpu, naga, ...)
# stay built either way. `cargo llvm-cov clean` is cargo's own cleaner, not `rm`.
cargo llvm-cov clean --workspace --profraw-only 2> /dev/null
layout=$(git ls-files -co --exclude-standard '*.rs' | sort | sha1sum | cut -d' ' -f1)
if [[ "$(cat target/coverage-layout 2> /dev/null)" != "$layout" ]]; then
  stage clean-workspace-artifacts cargo llvm-cov clean --workspace 2> /dev/null
  echo "$layout" > target/coverage-layout
fi

# Run 1: everything on the default adapter, except the interoperability tests (real llama.cpp and Ollama).
stage default-adapter cargo llvm-cov nextest --workspace --no-report -E 'not binary(interop)'

# Run 2: no adapter, for the "skip when there is no adapter" branches. Only the crates that open an adapter in
# their tests have any, so it runs those: the engine and core crates, and the GUI's unit, binary and API tests.
export FERRUM_GPU_ADAPTER=no-such-adapter
stage no-adapter cargo llvm-cov nextest --no-report -p ferrum_wgpu -p ferrum_gpu_core -p ferrum_gpu_gui \
  -E 'package(ferrum_wgpu) | package(ferrum_gpu_core) | (package(ferrum_gpu_gui) & (kind(lib) | kind(bin) | binary(api) | binary(binary)))'
unset FERRUM_GPU_ADAPTER

# The summary is kept in target/ and given to metrics.md whether or not the floor is met, so the record of the
# run is never lost to a failure. The floor decides the exit status.
summary=target/coverage-summary.txt
set +e
if [[ -n "${FLOOR:-}" ]]; then
  cargo llvm-cov report --summary-only --fail-under-lines "$FLOOR" 2>&1 | tee "$summary"
else
  cargo llvm-cov report --summary-only 2>&1 | tee "$summary"
fi
status=${PIPESTATUS[0]}
set -e
# the exact per-line check (every instrumented line executed); see scripts/lcov_check.py
cargo llvm-cov report --lcov --output-path target/coverage.lcov > /dev/null 2>&1
python3 scripts/lcov_check.py target/coverage.lcov || status=1
echo "total $((SECONDS - start))s (target: under 1800s)" | tee -a "$timing" >&2
python3 scripts/metrics.py --coverage "$summary" --floor "${FLOOR:-none}" || echo "metrics.md was not updated" >&2
exit "$status"
