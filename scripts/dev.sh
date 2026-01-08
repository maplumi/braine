#!/usr/bin/env bash
# Unified dev helper: fmt, clippy, build release (Linux + Windows), portable zip.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

command -v cargo >/dev/null || { echo "cargo not found"; exit 1; }
command -v 7z >/dev/null || { echo "7z not found; install p7zip-full (Linux) or 7-Zip (Windows)"; exit 1; }

echo "[1/5] cargo fmt"
cargo fmt

echo "[2/5] cargo clippy --all-targets"
cargo clippy --workspace --all-targets -- -D warnings

echo "[3/5] cargo build --release (native)"
cargo build --release

echo "[4/5] cargo build --release --all --target x86_64-pc-windows-gnu"
cargo build --release --all --target x86_64-pc-windows-gnu

echo "[5/5] Create portable zip"
mkdir -p dist/windows
cp target/x86_64-pc-windows-gnu/release/braine_viz.exe dist/windows/
cp target/x86_64-pc-windows-gnu/release/brained.exe dist/windows/
cp target/x86_64-pc-windows-gnu/release/braine.exe dist/windows/
cp braine_vis/assets/braine.ico dist/windows/
cat > dist/windows/run_braine.bat <<'EOF'
@echo off
rem Ensure we run from the folder containing this script.
rem This also makes UNC paths (e.g. \\wsl.localhost\...) work by mapping a temp drive.
pushd "%~dp0" || (echo Failed to enter script directory & exit /b 1)

start "brained" "%~dp0brained.exe"
timeout /t 1 >NUL
start "braine_viz" "%~dp0braine_viz.exe"

popd
EOF
cat > dist/windows/README.txt <<'EOF'
Braine - Brain-like Learning Substrate

SETUP:
1. Run run_braine.bat (or create a shortcut to it)
   - Starts brained daemon in background
   - Launches braine_viz UI after 1 second

NOTE (WSL / UNC PATHS):
  If you run this from a path like \\wsl.localhost\..., cmd.exe may not treat it as a normal working directory.
  run_braine.bat uses pushd to handle UNC paths, but the most reliable approach is:
  - Copy dist/braine-portable.zip to a normal Windows folder (e.g. Downloads)
  - Extract it, then run run_braine.bat from there

CONTROLS:
- Start/Stop: Toggle learning loop
- Dream/Burst/Sync/Imprint: Learning helpers
- Save: Persist brain state
- Load: Restore from saved brain.bbi
- Reset: Clear brain to fresh state

CLI (Command Line):
  braine-cli status          Show daemon state
  braine-cli start/stop      Control learning
  braine-cli save/load       Persistence
  braine-cli shutdown        Graceful exit
  braine-cli paths           Show data directory

DATA LOCATION:
  Brain saves to: %APPDATA%\Braine\brain.bbi

AUTO-SAVE:
  Brain auto-saves every 10 trials.
  It also saves on explicit Stop or Shutdown.
  Spot runtime metrics persist to: %APPDATA%\Braine\runtime.json
EOF
(
  cd dist/windows
  7z a -tzip ../braine-portable.zip braine_viz.exe brained.exe braine.exe braine.ico run_braine.bat README.txt >/dev/null
)
echo "Portable bundle: dist/braine-portable.zip"
