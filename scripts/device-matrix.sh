#!/usr/bin/env bash
# Run the GPU test suite once per adapter (FERRUM_GPU_ADAPTER) and print one DEVICES.md row each.
# Usage: scripts/device-matrix.sh [adapter-name-fragment ...]   (default: every adapter `info` lists)
set -uo pipefail
cd "$(dirname "$0")/.."
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}" CARGO_PROFILE_RELEASE_LTO="${CARGO_PROFILE_RELEASE_LTO:-off}" CARGO_PROFILE_RELEASE_CODEGEN_UNITS="${CARGO_PROFILE_RELEASE_CODEGEN_UNITS:-16}"
cargo build --release -q -p ferrum_gpu_cli || exit 1
if [ $# -gt 0 ]; then names=("$@"); else
  mapfile -t names < <(target/release/ferrum-gpu info 2>/dev/null | sed -n 's/^adapter: \(.*\) (\([A-Za-z]*\), \([A-Za-z0-9]*\))$/\1/p' | sort -u)
fi
os="$(uname -sr)"; wgpu="$(grep -A1 '^name = "wgpu"$' Cargo.lock | sed -n 's/version = "\(.*\)"/\1/p' | head -1)"
for n in "${names[@]}"; do
  log="${TMPDIR:-/tmp}/matrix-$$.log"
  FERRUM_GPU_ADAPTER="$n" cargo test --release -q -p ferrum_wgpu >"$log" 2>&1; rc=$?
  passed=$(grep -o '[0-9]* passed' "$log" | awk '{s+=$1} END{print s+0}')
  failed=$(grep -o '[0-9]* failed' "$log" | awk '{s+=$1} END{print s+0}')
  skipped=$(grep -ci 'skip' "$log")
  echo "| $n | $os | wgpu $wgpu | $(date +%F) | rc=$rc passed=$passed failed=$failed skip-lines=$skipped |"
done
