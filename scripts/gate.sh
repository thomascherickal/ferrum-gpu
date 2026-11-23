#!/usr/bin/env bash
# The full gate (AGENTS.md sections 1, 2 and 6; CLAUDE.md rules 3 and 15), one step at a time, results in
# $GATE_OUT (default ./.gate):
#   gate.txt   one line per step: "<step> rc=<exit status> <time>" (the first line is the start marker)
#   gate.done  PASS or FAIL when everything has run
#   commit     the commit the gate ran on, or "dirty" if the tree had uncommitted changes (scripts/gate_ok.py)
#   <step>.log the output of each step
# Steps: fmt, clippy, the repository rules (file length, names, required files), duplicate code, the script
# tests, the GPU crates' tests on llvmpipe and on the Intel Xe adapter, every test in its own process (this
# is also the live timing for test-time-report.md; interop has its limit), coverage (summary floor 100, and zero lines uncovered by scripts/lcov_check.py; its two
# runs are the whole workspace on the default adapter and on no adapter), cargo deny, and the desktop app
# (its own workspace): fmt, clippy, tests, deny. CPU and GPU usage is sampled throughout (system-profile.md).
#   scripts/gate.sh                 # run from the repository (or worktree) root
# One cargo command at a time on the low-memory box (rule 7). Target: under 30 minutes (AGENTS.md), about 20
# with a warm target and llvm-cov-target; the first run in a new worktree compiles everything.
#
# Gotchas learned: a stale `target/llvm-cov-target` from before files moved makes coverage report the old
# paths at 0% (delete it first if files moved); `cargo test` stops at the first failing test binary, so a red
# step can hide failures in later binaries; the no-unsafe guard is a text match, so a test must not contain
# the word as a string literal.
cd "$(dirname "$0")/.." || exit 1
OUT="${GATE_OUT:-$PWD/.gate}"
mkdir -p "$OUT"
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"
# no `rm` (the command ban list): files are overwritten, and a run is "not done" until gate.done says so
echo RUNNING > "$OUT/gate.done"
PROFILE="$OUT/profile-$$.csv"
python3 scripts/gate_key.py > "$OUT/key"  # what the pre-commit hook compares (scripts/gate_key.py)
echo "start rc=0 $(date +%T)" > "$OUT/gate.txt"
if [ -z "$(git status --porcelain -- . ':!metrics.md' ':!next-steps.md' ':!test-time-report.md' ':!system-profile.md')" ]; then
  git rev-parse HEAD > "$OUT/commit"
else
  echo dirty > "$OUT/commit"
fi
python3 scripts/system_profile.py sample "$PROFILE" &
sampler=$!
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
step rules python3 scripts/rules_check.py
step dups python3 scripts/dup_check.py
step scripts bash -c 'for t in scripts/test_*.py; do python3 "$t" || exit 1; done'
ADAPTER_CRATES="-p ferrum_wgpu -p ferrum_gpu_core"
FERRUM_GPU_ADAPTER=llvmpipe step llvmpipe cargo test $ADAPTER_CRATES -- --test-threads 1
FERRUM_GPU_ADAPTER="Iris(R) Xe Graphics (TGL GT2)" step xe cargo test $ADAPTER_CRATES -- --test-threads 1
step timelimit env TIMES_OUT="$OUT/test-times.txt" scripts/test-time-limit.sh 0
step timing python3 scripts/timing_report.py "$OUT/test-times.txt"
FLOOR=100 step coverage bash scripts/coverage.sh
step deny cargo deny check
FERRUM_GPU_ADAPTER=no-such-adapter step studio bash -c "cd ferrum_gpu_studio && cargo fmt --check && cargo clippy --all-targets --locked -- -D warnings && cargo test --locked -- --test-threads 1"
step studiodeny cargo deny --manifest-path ferrum_gpu_studio/Cargo.toml check
touch "$PROFILE.stop"
wait "$sampler"
python3 scripts/system_profile.py render "$PROFILE" || echo "system-profile.md was not updated" >&2
if [ $fail -eq 0 ]; then echo PASS > "$OUT/gate.done"; else echo FAIL > "$OUT/gate.done"; fi
python3 scripts/metrics.py --gate "$OUT/gate.txt" || echo "metrics.md was not updated" >&2
python3 scripts/next_steps.py || echo "next-steps.md was not updated" >&2
cat "$OUT/gate.txt" "$OUT/gate.done"
