#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI_BIN="${CLI_BIN:-$ROOT_DIR/target/debug/braine-cli}"
DAEMON_BIN="${DAEMON_BIN:-$ROOT_DIR/target/debug/brained}"
LOG_FILE="${1:-/tmp/game_learning_validation.log}"

need_bin() {
  local path="$1"
  local pkg="$2"
  if [[ ! -x "$path" ]]; then
    echo "missing $path; building $pkg" >&2
    (cd "$ROOT_DIR" && cargo build -p "$pkg")
  fi
}

need_bin "$CLI_BIN" core
need_bin "$DAEMON_BIN" brained

# Start daemon if it doesn't respond.
if ! "$CLI_BIN" status >/dev/null 2>&1; then
  echo "starting daemon..." >&2
  "$DAEMON_BIN" >/tmp/brained.log 2>&1 &
  DAEMON_PID=$!
  trap 'kill "$DAEMON_PID" 2>/dev/null || true' EXIT
  # Give it a moment to bind.
  for _ in {1..50}; do
    if "$CLI_BIN" status >/dev/null 2>&1; then
      break
    fi
    sleep 0.1
  done
fi

status_one_line() {
  # Flatten multi-line status output and keep a stable subset of fields.
  "$CLI_BIN" status | tr '\n' ' ' | tr -s ' ' \
    | sed -E 's/.*\btrials=([0-9]+).*\bacc=([^ ]+).*\blast100=([^ ]+).*/trials=\1 acc=\2 last100=\3/'
}

run_game() {
  local game="$1"
  local trialms="$2"
  local seconds="$3"

  "$CLI_BIN" stop >/dev/null || true
  "$CLI_BIN" reset >/dev/null || true
  "$CLI_BIN" game "$game" >/dev/null
  "$CLI_BIN" trialms "$trialms" >/dev/null

  printf '%s\n' "== $game (trialms=$trialms, seconds=$seconds) ==" | tee -a "$LOG_FILE"

  "$CLI_BIN" start >/dev/null
  sleep "$seconds"
  "$CLI_BIN" stop >/dev/null || true

  printf '%s %s\n' "$game" "$(status_one_line)" | tee -a "$LOG_FILE"
}

# Default sweep parameters (tuned for episode length / credit assignment).
# - Spot/Bandit/SpotReversal/SpotXY: moderate tick OK (single-step trials)
# - Maze: needs many steps per episode → smaller trialms
# - Pong: benefits from faster credit and more interactions → smaller trialms and longer run

: > "$LOG_FILE"
run_game spot 100 25
run_game bandit 100 25
run_game spot_reversal 100 50
run_game spotxy 100 30

# Maze/Pong are step-heavy; run them at higher FPS to get more interactions per wall-clock.
"$CLI_BIN" fps 240 >/dev/null || true
run_game maze 10 60
run_game pong 20 60
"$CLI_BIN" fps 60 >/dev/null || true

# Demos (best-effort): they may depend on local assets/datasets.
# Leave commented by default; uncomment if you want them in the sweep.
# "$CLI_BIN" demo replay >/dev/null
# "$CLI_BIN" stop >/dev/null || true
# "$CLI_BIN" demo text >/dev/null
# "$CLI_BIN" stop >/dev/null || true

printf '%s\n' "wrote $LOG_FILE" >&2
