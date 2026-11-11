#!/usr/bin/env bash
# Produce one golden fixture with a ferrum-cpu binary and copy it into fixtures/.
# This is NOT run by tests or CI and creates no dependency: the repository only keeps the output.
#
#   scripts/make-fixtures.sh <ferrum-cpu-train_transformer-binary> <name> -- <arguments for the binary>
#
# Example (M1, when the tokenizer fixture is made; the corpus file must already be in fixtures/):
#   scripts/make-fixtures.sh ~/scratch/train_transformer tiny-bpe -- \
#       train fixtures/corpus.txt OUT --vocab 300 --epochs 1 --seed 1
#   (the word OUT is replaced with the output path in the scratch directory)
#
# Build the binary with its output sent outside this repository, for example:
#   (cd ../ferrum-cpu && CARGO_TARGET_DIR=$SCRATCH/cpu-target cargo build --release -p slm_cli)
# and never modify ferrum-cpu. Set CPU_COMMIT to the commit it was built from. The script prints the ledger line for fixtures/README.md.
set -euo pipefail
cd "$(dirname "$0")/.."
bin="${1:?binary}"; name="${2:?fixture name}"; shift 2
[ "${1:-}" = "--" ] && shift
scratch="$(mktemp -d)"; trap 'rm -rf "$scratch"' EXIT
args=(); for a in "$@"; do args+=("${a/OUT/$scratch/$name.gguf}"); done
"$bin" "${args[@]}"
cp "$scratch/$name.gguf" "fixtures/$name.gguf"
src="${CPU_COMMIT:?set CPU_COMMIT to the ferrum-cpu commit the binary was built from}"
echo "| fixtures/$name.gguf | (describe) | \`train_transformer $*\` | $src | $(date +%F) | $(sha256sum "fixtures/$name.gguf" | cut -d' ' -f1) |"
