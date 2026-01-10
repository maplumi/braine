#!/usr/bin/env bash
# Unified dev helper: fmt, clippy, build release (Linux + Windows), portable zip, (optional) web build + sync.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

command -v cargo >/dev/null || { echo "cargo not found"; exit 1; }
command -v 7z >/dev/null || { echo "7z not found; install p7zip-full (Linux) or 7-Zip (Windows)"; exit 1; }

echo "[1/7] cargo fmt"
cargo fmt

echo "[2/7] cargo clippy --all-targets"
cargo clippy --workspace --all-targets -- -D warnings

echo "[3/7] cargo build --release (native)"
cargo build --release

echo "[4/7] cargo build --release --all --target x86_64-pc-windows-gnu"
cargo build --release --all --target x86_64-pc-windows-gnu

echo "[5/7] Create portable zip"
mkdir -p dist/windows
cp target/x86_64-pc-windows-gnu/release/braine_desktop.exe dist/windows/
cp target/x86_64-pc-windows-gnu/release/brained.exe dist/windows/
cp target/x86_64-pc-windows-gnu/release/braine.exe dist/windows/
cp crates/braine_desktop/assets/braine.ico dist/windows/
cat > dist/windows/run_braine.bat <<'EOF'
@echo off
rem Ensure we run from the folder containing this script.
rem This also makes UNC paths (e.g. \\wsl.localhost\...) work by mapping a temp drive.
pushd "%~dp0" || (echo Failed to enter script directory & exit /b 1)

start "brained" "%~dp0brained.exe"
timeout /t 1 >NUL
start "braine_desktop" "%~dp0braine_desktop.exe"

popd
EOF
cat > dist/windows/README.txt <<'EOF'
Braine - Brain-like Learning Substrate

SETUP:
1. Run run_braine.bat (or create a shortcut to it)
   - Starts brained daemon in background
  - Launches braine_desktop UI after 1 second

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
  7z a -tzip ../braine-portable.zip braine_desktop.exe brained.exe braine.exe braine.ico run_braine.bat README.txt >/dev/null
)
echo "Portable bundle: dist/braine-portable.zip"

echo "[6/7] trunk build --release (braine_web)"
WEB_DIST_DIR="$ROOT/crates/braine_web/dist"
if command -v trunk >/dev/null; then
  (
    cd "$ROOT/crates/braine_web"
    trunk build --release --features web
  )
  echo "braine_web dist: $WEB_DIST_DIR"
else
  echo "trunk not found; skipping braine_web build (install with: cargo install trunk)"
fi

echo "[7/7] Sync braine_web dist to ~/projects/research/braine-web"
WEB_DEPLOY_DIR="$HOME/projects/research/braine-web"
if [[ -d "$WEB_DIST_DIR" ]]; then
  mkdir -p "$WEB_DEPLOY_DIR"
  if command -v rsync >/dev/null; then
    rsync -a --delete "$WEB_DIST_DIR"/ "$WEB_DEPLOY_DIR"/
  else
    rm -rf "$WEB_DEPLOY_DIR"/*
    cp -r "$WEB_DIST_DIR"/* "$WEB_DEPLOY_DIR"/
  fi
  echo "Synced web assets to: $WEB_DEPLOY_DIR"

  # Also copy native release assets for convenience.
  # Keep them in a subfolder so they don't conflict with the web dist.
  mkdir -p "$WEB_DEPLOY_DIR/release"
  if [[ -f "$ROOT/dist/braine-portable.zip" ]]; then
    cp -f "$ROOT/dist/braine-portable.zip" "$WEB_DEPLOY_DIR/release/"
    echo "Copied release bundle to: $WEB_DEPLOY_DIR/release/braine-portable.zip"
  fi
else
  echo "Skipping sync (missing dist or target dir)."
  echo "  dist:   $WEB_DIST_DIR"
  echo "  target: $WEB_DEPLOY_DIR"
fi
