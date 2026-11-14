#!/usr/bin/env bash
# Run every test of the workspace in its own process, with a hard time limit where one is set, and
# FAIL if a test fails or exceeds its limit.
#
#   scripts/test-time-limit.sh [GENERAL_LIMIT_SECONDS] [extra `cargo test --no-run` arguments]
#
# Limits (user decision 2026-11-14):
#   - ordinary tests: none by default (GENERAL_LIMIT_SECONDS = 0). Pass a number to cap them again.
#   - the interoperability tests (the `interop` test binary): 600 s each, or $INTEROP_LIMIT.
# A limit of 0 means no limit.
#
# Why a process per test: libtest's `--ensure-time` never fires for workspace members (their tests
# are not classified as unit or integration tests), and no flag stops a test that hangs; GNU
# `timeout` does both. The test binaries are built once by cargo, then run directly, so the cost is
# process start-up only. Doc-tests are not covered. Needs GNU `timeout` and `jq`.
set -u
general=${1:-0}
shift || true
interop=${INTEROP_LIMIT:-600}
for v in "$general" "$interop"; do
  case $v in ''|*[!0-9]*) echo "limits must be whole numbers of seconds, got '$v'" >&2; exit 2;; esac
done
command -v timeout >/dev/null || { echo "needs GNU coreutils 'timeout'" >&2; exit 2; }
command -v jq >/dev/null || { echo "needs 'jq'" >&2; exit 2; }
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-3}
if [ $# -eq 0 ]; then set -- --workspace; fi

# every test executable cargo builds: "<manifest dir>\t<path>"
bins=$(cargo test --no-run --message-format=json "$@" 2>/dev/null \
  | jq -r 'select(.reason=="compiler-artifact" and .profile.test==true and .executable!=null)
           | "\(.manifest_path | rtrimstr("/Cargo.toml"))\t\(.executable)"' | sort -u)
[ -n "$bins" ] || { echo "no test executables found" >&2; exit 2; }

pass=0; fail=0; hang=0; failed=""; hung=""
times=$(mktemp)
trap 'rm -f "$times"' EXIT
while IFS=$'\t' read -r dir bin; do
  case $(basename "$bin") in interop-*) lim=$interop;; *) lim=$general;; esac
  if [ "$lim" -eq 0 ]; then run=(); else run=(timeout -k 5 "$lim"); fi
  for name in $("$bin" --list --format terse 2>/dev/null | sed -n 's/: test$//p'); do
    t0=$(date +%s.%N)
    ( cd "$dir" && CARGO_MANIFEST_DIR="$dir" "${run[@]}" "$bin" --exact "$name" --test-threads 1 -q >/dev/null 2>&1 )
    rc=$?
    t1=$(date +%s.%N)
    echo "$(echo "$t1 - $t0" | bc) $(basename "$bin") $name" >> "$times"
    if [ $rc -eq 0 ]; then
      pass=$((pass + 1))
    elif [ "$lim" -ne 0 ] && { [ $rc -eq 124 ] || [ $rc -eq 137 ]; }; then
      hang=$((hang + 1)); hung="$hung\n  $name"; echo "TIMEOUT (>${lim}s) $name"
    else
      fail=$((fail + 1)); failed="$failed\n  $name"; echo "FAILED  $name (exit $rc)"
    fi
  done
done <<< "$bins"

echo
echo "== slowest tests (limits: ordinary ${general}s [0 = none], interop ${interop}s) =="
sort -rn "$times" | head -8
echo
echo "RESULT: $pass passed, $fail failed, $hang over their limit"
[ -z "$failed" ] || printf "FAILED:%b\n" "$failed"
[ -z "$hung" ] || printf "OVER THE LIMIT:%b\n" "$hung"
[ $fail -eq 0 ] && [ $hang -eq 0 ]
