#!/usr/bin/env bash
# The full gate (CLAUDE.md rules 3 and 15), one step at a time, results in $GATE_OUT (default ./.gate):
#   gate.txt   one line per step: "<step> rc=<exit status> <time>"
#   gate.done  PASS or FAIL when everything has run
#   <step>.log the output of each step
# Steps: fmt, clippy, the workspace tests in four configurations (the default adapter, no GPU,
# llvmpipe, the Intel Xe adapter by name), the per-test time limits, coverage with floor 98,
# cargo deny, and the desktop app (its own workspace): fmt, clippy, tests, deny.
#   scripts/gate.sh                 # run from the repository (or worktree) root
# One cargo command at a time on the low-memory box (rule 7). A full run takes about 30 minutes.
#
# Gotchas learned: a stale `target/llvm-cov-target` from before files moved makes coverage report the
# old paths at 0% (delete it first if files moved); `cargo test` stops at the first failing test binary,
# so a red step can hide failures in later binaries; the repository's no-unsafe guard is a text match,
# so a test must not contain the word as a string literal.
cd "$(dirname "$0")/.." || exit 1
OUT="${GATE_OUT:-$PWD/.gate}"
mkdir -p "$OUT"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"
rm -f "$OUT"/gate.txt "$OUT"/gate.done "$OUT"/*.log
fail=0
step() {
  name=$1
  shift
  "$@" > "$OUT/$name.log" 2>&1
  rc=$?
  echo "$name rc=$rc $(date +%T)" >> "$OUT/gate.txt"
  [ $rc -ne 0 ] && fail=1
}
step fmt cargo fmt --all -- --check
step clippy cargo clippy --workspace --all-targets -- -D warnings
step gpu cargo test --workspace -- --test-threads 1
FERRUM_GPU_ADAPTER=no-such-adapter step nogpu cargo test --workspace -- --test-threads 1
FERRUM_GPU_ADAPTER=llvmpipe step llvmpipe cargo test --workspace -- --test-threads 1
FERRUM_GPU_ADAPTER="Iris(R) Xe Graphics (TGL GT2)" step xe cargo test --workspace -- --test-threads 1
step timelimit scripts/test-time-limit.sh 0
FLOOR=98 step coverage scripts/coverage.sh
step deny cargo deny check
FERRUM_GPU_ADAPTER=no-such-adapter step studio bash -c "cd ferrum_gpu_studio && cargo fmt --check && cargo clippy --all-targets --locked -- -D warnings && cargo test --locked -- --test-threads 1"
step studiodeny cargo deny --manifest-path ferrum_gpu_studio/Cargo.toml check
if [ $fail -eq 0 ]; then echo PASS > "$OUT/gate.done"; else echo FAIL > "$OUT/gate.done"; fi
cat "$OUT/gate.txt" "$OUT/gate.done"
