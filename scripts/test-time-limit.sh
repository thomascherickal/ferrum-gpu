#!/usr/bin/env bash
# Run every test of the workspace in its own process under a hard time limit (default 180 s) and
# FAIL if any single test exceeds it or fails.
#
#   scripts/test-time-limit.sh [LIMIT_SECONDS] [extra `cargo test --no-run` arguments, e.g. -p ferrum_wgpu]
#
# Why not libtest's own limit: `--ensure-time` never fires for workspace members (their tests are
# not classified as unit or integration tests), and no flag stops a test that hangs. A process per
# test with coreutils `timeout` does both: a hung test is killed at the limit and reported TIMEOUT,
# a slow one is reported with its time. The test binaries are built once by cargo, then run
# directly, so the cost is process start-up only. Doc-tests are not covered (the few there are
# compile-checks). Needs GNU `timeout` and `jq`.
set -u
limit=${1:-180}
shift || true
case $limit in ''|*[!0-9]*) echo "LIMIT_SECONDS must be a whole number, got '$limit'" >&2; exit 2;; esac
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
  for name in $("$bin" --list --format terse 2>/dev/null | sed -n 's/: test$//p'); do
    t0=$(date +%s.%N)
    ( cd "$dir" && CARGO_MANIFEST_DIR="$dir" timeout -k 5 "$limit" "$bin" --exact "$name" --test-threads 1 -q >/dev/null 2>&1 )
    rc=$?
    t1=$(date +%s.%N)
    echo "$(echo "$t1 - $t0" | bc) $(basename "$bin") $name" >> "$times"
    if [ $rc -eq 0 ]; then
      pass=$((pass + 1))
    elif [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
      hang=$((hang + 1)); hung="$hung\n  $name"; echo "TIMEOUT (>${limit}s) $name"
    else
      fail=$((fail + 1)); failed="$failed\n  $name"; echo "FAILED  $name (exit $rc)"
    fi
  done
done <<< "$bins"

echo
echo "== slowest tests (limit ${limit}s) =="
sort -rn "$times" | head -8
echo
echo "RESULT: $pass passed, $fail failed, $hang over ${limit}s"
[ -z "$failed" ] || printf "FAILED:%b\n" "$failed"
[ -z "$hung" ] || printf "OVER THE LIMIT:%b\n" "$hung"
[ $fail -eq 0 ] && [ $hang -eq 0 ]
