#!/usr/bin/env bash
# Build (and optionally run) the desktop app, Ferrum GPU Studio. It is its own workspace
# (ferrum_gpu_studio/Cargo.toml), so it is built from there.
#   scripts/build-studio.sh            # release build; the binary is ferrum_gpu_studio/target/release/ferrum_gpu_studio
#   scripts/build-studio.sh --run      # build, then start it (flags after --run go to the app)
#   scripts/build-studio.sh --debug    # a debug build instead
# Linux needs the Tauri system libraries first (Debian/Ubuntu names):
#   libwebkit2gtk-4.1-dev libgtk-3-dev libsoup-3.0-dev libayatana-appindicator3-dev librsvg2-dev
set -e
cd "$(dirname "$0")/../ferrum_gpu_studio"
profile=release
flags=--release
run=0
if [ "$1" = "--debug" ]; then profile=debug; flags=; shift; fi
if [ "$1" = "--run" ]; then run=1; shift; fi
# the low-memory development box: few jobs, no LTO (CLAUDE.md rule 7); override by setting them
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"
export CARGO_PROFILE_RELEASE_LTO="${CARGO_PROFILE_RELEASE_LTO:-off}"
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS="${CARGO_PROFILE_RELEASE_CODEGEN_UNITS:-16}"
cargo build $flags --locked
echo "built: $(pwd)/target/$profile/ferrum_gpu_studio"
if [ "$run" = 1 ]; then exec "target/$profile/ferrum_gpu_studio" "$@"; fi
