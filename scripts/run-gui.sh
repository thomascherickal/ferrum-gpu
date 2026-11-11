#!/usr/bin/env bash
# Run the ferrum-gpu GUI on port 8080 with --device gpu.
# The adapter and thread count are chosen once per architecture (OS + CPU) and stored in
# .gui-prefs (git-ignored); later runs just read them. Delete the line, or pass --retune, to redo.
# Usage: scripts/run-gui.sh [--retune] [extra ferrum_gpu_gui flags, e.g. --port 9000]
set -euo pipefail
cd "$(dirname "$0")/.."

PREFS=.gui-prefs
KEY="$(uname -s)-$(uname -m)"
retune=0
[ "${1:-}" = "--retune" ] && { retune=1; shift; }

# Low-memory box defaults from CLAUDE.md rule 7 (override by exporting your own).
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"
export CARGO_PROFILE_RELEASE_LTO="${CARGO_PROFILE_RELEASE_LTO:-off}"
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS="${CARGO_PROFILE_RELEASE_CODEGEN_UNITS:-16}"
cargo build --release -q -p ferrum_gpu_cli -p ferrum_gpu_gui

# ponytail: rank by adapter kind (Discrete > Integrated > Virtual/Other), never Software, and
# Vulkan/Metal/Dx12 over Gl. This is a heuristic, not a benchmark; add a timed probe if it picks wrong.
pick_adapter() {
  target/release/ferrum-gpu info 2>/dev/null | sed -n 's/^adapter: \(.*\) (\([A-Za-z]*\), \([A-Za-z0-9]*\))$/\2\t\3\t\1/p' |
    awk -F'\t' '$1!="Software" {
      s = ($1=="Discrete") ? 30 : ($1=="Integrated") ? 20 : 10
      if ($2=="Gl") s -= 5
      print s "\t" $3 }' | sort -t$'\t' -k1,1nr -s | head -n1 | cut -f2
}

line="$(grep -F "$KEY"$'\t' "$PREFS" 2>/dev/null | head -n1 || true)"
if [ -z "$line" ] || [ "$retune" = 1 ]; then
  adapter="$(pick_adapter)"
  [ -n "$adapter" ] || { echo "no hardware GPU adapter found (see: target/release/ferrum-gpu info)" >&2; exit 1; }
  threads="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
  { grep -vF "$KEY"$'\t' "$PREFS" 2>/dev/null || true; printf '%s\t%s\t%s\n' "$KEY" "$adapter" "$threads"; } >"$PREFS.tmp"
  mv "$PREFS.tmp" "$PREFS"
  echo "tuned for $KEY: adapter '$adapter', $threads threads (saved to $PREFS)"
  line="$KEY"$'\t'"$adapter"$'\t'"$threads"
fi
IFS=$'\t' read -r _ adapter threads <<<"$line"

exec target/release/ferrum_gpu_gui --port 8080 --device gpu --gpu-adapter "$adapter" --threads "$threads" "$@"
