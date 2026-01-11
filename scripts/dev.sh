#!/usr/bin/env bash
# Unified dev helper: fmt, clippy, build release (Linux + Windows), portable zip, web build + sync.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

MODE="full"          # full|web
SKIP_DESKTOP=0
SKIP_WINDOWS=0

for arg in "$@"; do
  case "$arg" in
    --web-only)
      MODE="web"
      ;;
    --skip-desktop)
      SKIP_DESKTOP=1
      ;;
    --skip-windows)
      SKIP_WINDOWS=1
      ;;
    -h|--help)
      cat <<'EOF'
Usage: ./scripts/dev.sh [--web-only] [--skip-desktop] [--skip-windows]

Modes:
  --web-only       Only builds braine_web via trunk and syncs dist (no fmt/clippy/native/Windows/zip).

Flags (full mode):
  --skip-desktop   Skip braine_desktop build and skip portable zip step.
  --skip-windows   Skip the Windows cross-build and portable zip step.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg"
      echo "Run: ./scripts/dev.sh --help"
      exit 2
      ;;
  esac
done

command -v cargo >/dev/null || { echo "cargo not found"; exit 1; }

WEB_DIST_DIR="$ROOT/crates/braine_web/dist"
WEB_DEPLOY_DIR="$HOME/projects/research/braine-web"

if [[ "$MODE" == "web" ]]; then
  # Manual build instead of trunk due to wasm-bindgen reference-types issues
  # Using Rust 1.84 to avoid reference-types issues in newer Rust versions
  echo "[1/3] cargo build braine_web (wasm32) with Rust 1.84"
  cargo +1.84.0 build -p braine_web --target wasm32-unknown-unknown --features braine_web/web --release
  
  echo "[2/3] wasm-bindgen post-processing"
  WASM_FILE="$ROOT/target/wasm32-unknown-unknown/release/braine_web.wasm"
  if ! command -v wasm-bindgen >/dev/null; then
    echo "wasm-bindgen not found; install with: cargo install wasm-bindgen-cli --version 0.2.99"
    exit 1
  fi
  mkdir -p "$WEB_DIST_DIR"
  wasm-bindgen \
    --target=web \
    --out-dir="$WEB_DIST_DIR" \
    --out-name=braine_web \
    "$WASM_FILE" \
    --no-typescript
  
  # app.css is checked into the repo (full class-based styling). Don't overwrite it.
  if [[ ! -f "$WEB_DIST_DIR/app.css" ]]; then
    echo "ERROR: Missing $WEB_DIST_DIR/app.css (expected to be checked in).";
    exit 1
  fi

  # Generate index.html
  cat > "$WEB_DIST_DIR/index.html" << 'HTMLEOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Braine Web</title>
    <link rel="stylesheet" href="./app.css" />
    <script type="module">
      import init from './braine_web.js';
      init().catch((e) => console.error('wasm init failed', e));
    </script>
  </head>
  <body>
    <div id="app"></div>
  </body>
</html>
HTMLEOF
  echo "braine_web dist: $WEB_DIST_DIR"

  echo "[3/3] Sync braine_web dist to $WEB_DEPLOY_DIR"
  if [[ -d "$WEB_DIST_DIR" ]]; then
    mkdir -p "$WEB_DEPLOY_DIR"
    if command -v rsync >/dev/null; then
      # NOTE: $WEB_DEPLOY_DIR is often a git checkout. Protect .git from deletion.
      rsync -a --delete \
        --filter='protect .git/' \
        --filter='protect .git/***' \
        --exclude='.git/' \
        "$WEB_DIST_DIR"/ "$WEB_DEPLOY_DIR"/
    else
      rm -rf "$WEB_DEPLOY_DIR"/*
      cp -r "$WEB_DIST_DIR"/* "$WEB_DEPLOY_DIR"/
    fi
    echo "Synced web assets to: $WEB_DEPLOY_DIR"
  else
    echo "Missing dist dir: $WEB_DIST_DIR"
    exit 1
  fi

  exit 0
fi

command -v 7z >/dev/null || { echo "7z not found; install p7zip-full (Linux) or 7-Zip (Windows)"; exit 1; }

echo "[1/7] cargo fmt"
cargo fmt

echo "[2/7] cargo clippy --all-targets"
cargo clippy --workspace --all-targets -- -D warnings

echo "[3/7] cargo build --release (native)"
if [[ "$SKIP_DESKTOP" == "1" ]]; then
  cargo build --release -p braine -p brained
else
  cargo build --release -p braine -p brained -p braine_desktop
fi

if [[ "$SKIP_WINDOWS" == "1" ]]; then
  echo "[4/7] Skipping Windows build"
else
  echo "[4/7] cargo build --release --target x86_64-pc-windows-gnu"
  if [[ "$SKIP_DESKTOP" == "1" ]]; then
    cargo build --release --target x86_64-pc-windows-gnu -p braine -p brained
  else
    cargo build --release --target x86_64-pc-windows-gnu -p braine -p brained -p braine_desktop
  fi
fi

if [[ "$SKIP_WINDOWS" == "1" || "$SKIP_DESKTOP" == "1" ]]; then
  echo "[5/7] Skipping portable zip (requires Windows build + braine_desktop)"
else
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
- Load: Restore from saved braine.bbi
- Reset: Clear brain to fresh state

CLI (Command Line):
  braine-cli status          Show daemon state
  braine-cli start/stop      Control learning
  braine-cli save/load       Persistence
  braine-cli shutdown        Graceful exit
  braine-cli paths           Show data directory

DATA LOCATION:
  Brain saves to: %APPDATA%\Braine\braine.bbi

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
fi

echo "[6/7] cargo build braine_web (wasm32) + wasm-bindgen"
# Manual build with Rust 1.84 to avoid wasm-bindgen reference-types issues
cargo +1.84.0 build -p braine_web --target wasm32-unknown-unknown --features braine_web/web --release
if command -v wasm-bindgen >/dev/null; then
  WASM_FILE="$ROOT/target/wasm32-unknown-unknown/release/braine_web.wasm"
  mkdir -p "$WEB_DIST_DIR"
  wasm-bindgen \
    --target=web \
    --out-dir="$WEB_DIST_DIR" \
    --out-name=braine_web \
    "$WASM_FILE" \
    --no-typescript

  # app.css is checked into the repo (full class-based styling). Don't overwrite it.
  if [[ ! -f "$WEB_DIST_DIR/app.css" ]]; then
    echo "ERROR: Missing $WEB_DIST_DIR/app.css (expected to be checked in).";
    exit 1
  fi

  # Generate index.html
  cat > "$WEB_DIST_DIR/index.html" << 'HTMLEOF'
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Braine Web</title>
    <link rel="stylesheet" href="./app.css" />
    <script type="module">
      import init from './braine_web.js';
      init().catch((e) => console.error('wasm init failed', e));
    </script>
  </head>
  <body>
    <div id="app"></div>
  </body>
</html>
HTMLEOF
  echo "braine_web dist: $WEB_DIST_DIR"
else
  echo "wasm-bindgen not found; skipping braine_web build (install with: cargo install wasm-bindgen-cli --version 0.2.99)"
fi

echo "[7/7] Sync braine_web dist to $WEB_DEPLOY_DIR"
if [[ -d "$WEB_DIST_DIR" ]]; then
  mkdir -p "$WEB_DEPLOY_DIR"
  if command -v rsync >/dev/null; then
    # NOTE: $WEB_DEPLOY_DIR is often a git checkout. Protect .git from deletion.
    rsync -a --delete \
      --filter='protect .git/' \
      --filter='protect .git/***' \
      --exclude='.git/' \
      "$WEB_DIST_DIR"/ "$WEB_DEPLOY_DIR"/
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
