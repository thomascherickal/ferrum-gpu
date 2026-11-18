#!/usr/bin/env bash
# The full gate (CLAUDE.md rules 3 and 15), one step at a time, results in $GATE_OUT (default ./.gate):
#   gate.txt   one line per step: "<step> rc=<exit status> <time>"
#   gate.done  PASS or FAIL when everything has run
#   <step>.log the output of each step
# Steps: fmt, clippy, the GPU crates' tests on llvmpipe and on the Intel Xe adapter, the interop
# tests under their time limit, coverage with floor 98 (its two runs are the whole workspace on the
# default adapter and on no adapter), cargo deny, and the desktop app (its own workspace): fmt,
# clippy, tests, deny.
#   scripts/gate.sh                 # run from the repository (or worktree) root
# One cargo command at a time on the low-memory box (rule 7). Target: under 20 minutes with a warm
# target and llvm-cov-target (the first run in a new worktree compiles everything and takes longer).
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
# The default-adapter and no-adapter configurations over the whole workspace are the two runs inside
# `scripts/coverage.sh` (a failing test fails that step), so they are not run a second time here.
# The other two adapters can only change what the GPU crates compute, so they run those crates only.
ADAPTER_CRATES="-p ferrum_wgpu -p ferrum_gpu_core"
FERRUM_GPU_ADAPTER=llvmpipe step llvmpipe cargo test $ADAPTER_CRATES -- --test-threads 1
FERRUM_GPU_ADAPTER="Iris(R) Xe Graphics (TGL GT2)" step xe cargo test $ADAPTER_CRATES -- --test-threads 1
# Only the interoperability tests have a time limit (rule 17), so only they need a process each.
step timelimit scripts/test-time-limit.sh 0 -p ferrum_gpu_cli --test interop
FLOOR=98 step coverage scripts/coverage.sh
step deny cargo deny check
FERRUM_GPU_ADAPTER=no-such-adapter step studio bash -c "cd ferrum_gpu_studio && cargo fmt --check && cargo clippy --all-targets --locked -- -D warnings && cargo test --locked -- --test-threads 1"
step studiodeny cargo deny --manifest-path ferrum_gpu_studio/Cargo.toml check
if [ $fail -eq 0 ]; then echo PASS > "$OUT/gate.done"; else echo FAIL > "$OUT/gate.done"; fi
cat "$OUT/gate.txt" "$OUT/gate.done"
