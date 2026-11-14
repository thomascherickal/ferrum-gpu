#!/usr/bin/env bash
# Run every ferrum_wgpu unit test on its own, each under a hard alarm, and report PASS / FAIL /
# TIMEOUT per test. Written to find out what the hosted macOS runner's paravirtual GPU does with our
# kernels: a hang in one test must not hide the others. macOS has no `timeout`, so perl's alarm
# (SIGALRM, default action: terminate) is the deadline. Always exits 0: the report is the result.
#   usage: scripts/mac_probe.sh [seconds-per-test]
set -u
limit=${1:-90}
export CARGO_TERM_COLOR=never
cargo test -p ferrum_wgpu --lib --no-run 2>&1 | tail -2
names=$(cargo test -p ferrum_wgpu --lib -- --list 2>/dev/null | sed -n 's/: test$//p')
total=$(echo "$names" | wc -l | tr -d ' ')
echo "probing $total tests, ${limit}s each"
pass=0; fail=0; hang=0
failed=""; hung=""
for n in $names; do
  out=$(perl -e 'alarm shift; exec @ARGV' "$limit" cargo test -q -p ferrum_wgpu --lib -- --exact "$n" --test-threads 1 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then
    pass=$((pass+1))
  elif [ $rc -eq 142 ] || [ $rc -eq 14 ]; then   # 128+SIGALRM
    hang=$((hang+1)); hung="$hung $n"; echo "TIMEOUT $n"
  else
    fail=$((fail+1)); failed="$failed $n"; echo "FAIL    $n"
    echo "$out" | grep -E "panicked|assert|left:|right:|error" | head -4 | sed 's/^/        /'
  fi
done
echo "RESULT pass=$pass fail=$fail timeout=$hang of $total"
[ -z "$failed" ] || echo "FAILED:$failed"
[ -z "$hung" ] || echo "TIMED OUT:$hung"
exit 0
