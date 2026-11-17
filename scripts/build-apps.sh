#!/usr/bin/env bash
# Build the command line and the desktop app and put them side by side in one folder:
#   dist/ferrum-gpu                the command line (and ferrum-gpu-server, the JSON server, beside it)
#   dist/ferrum-gpu-studio         the desktop app (a native window; it runs the same engine in-process)
# Both are thin programs over ferrum_gpu_core, so a command run from either does the same thing.
#   scripts/build-apps.sh          # release builds into ./dist
#   scripts/build-apps.sh --debug  # faster to build, slower to run
# Linux needs the Tauri system libraries for the desktop app (Debian/Ubuntu names):
#   libwebkit2gtk-4.1-dev libgtk-3-dev libsoup-3.0-dev libayatana-appindicator3-dev librsvg2-dev
set -e
cd "$(dirname "$0")/.."
profile=release
flags=--release
if [ "$1" = "--debug" ]; then profile=debug; flags=; fi
# the low-memory development box: few jobs, no LTO (CLAUDE.md rule 7); override by setting them
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}"
export CARGO_PROFILE_RELEASE_LTO="${CARGO_PROFILE_RELEASE_LTO:-off}"
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS="${CARGO_PROFILE_RELEASE_CODEGEN_UNITS:-16}"
ext=""
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) ext=".exe" ;; esac

cargo build $flags --locked -p ferrum_gpu_cli -p ferrum_gpu_server
(cd ferrum_gpu_studio && cargo build $flags --locked)

mkdir -p dist
cp "target/$profile/ferrum-gpu$ext" "dist/ferrum-gpu$ext"
cp "target/$profile/ferrum-gpu-server$ext" "dist/ferrum-gpu-server$ext"
cp "ferrum_gpu_studio/target/$profile/ferrum_gpu_studio$ext" "dist/ferrum-gpu-studio$ext"
echo "built, side by side in $(pwd)/dist:"
ls -l "dist"
