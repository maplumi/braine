#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLI_BIN="${CLI_BIN:-$ROOT_DIR/target/debug/braine-cli}"
DAEMON_BIN="${DAEMON_BIN:-$ROOT_DIR/target/debug/brained}"
LOG_FILE="${1:-/tmp/game_learning_validation.log}"
RESTART_DAEMON="${RESTART_DAEMON:-1}"

# Duration defaults are kept short so the script finishes quickly in CI/automation.
SPOT_SECS="${SPOT_SECS:-15}"
BANDIT_SECS="${BANDIT_SECS:-15}"
SPOT_REV_SECS="${SPOT_REV_SECS:-30}"
SPOTXY_SECS="${SPOTXY_SECS:-20}"
MAZE_SECS="${MAZE_SECS:-60}"
PONG_SECS="${PONG_SECS:-60}"

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

DAEMON_PID=""

start_daemon() {
  if [[ "$RESTART_DAEMON" == "1" ]]; then
    # Best-effort shutdown of any existing daemon.
    "$CLI_BIN" shutdown >/dev/null 2>&1 || true
    pkill -f "${DAEMON_BIN}" >/dev/null 2>&1 || true
    sleep 0.2
  fi

  if ! "$CLI_BIN" status >/dev/null 2>&1; then
    echo "starting daemon..." >&2
    "$DAEMON_BIN" >/tmp/brained.log 2>&1 &
    DAEMON_PID=$!
    trap 'kill "$DAEMON_PID" 2>/dev/null || true' EXIT
    for _ in {1..50}; do
      if "$CLI_BIN" status >/dev/null 2>&1; then
        return 0
      fi
      sleep 0.1
    done
    echo "daemon did not respond; see /tmp/brained.log" >&2
    return 1
  fi
}

start_daemon

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
run_game spot 100 "$SPOT_SECS"
run_game bandit 100 "$BANDIT_SECS"
run_game spot_reversal 100 "$SPOT_REV_SECS"
run_game spotxy 100 "$SPOTXY_SECS"
run_game maze 50 "$MAZE_SECS"
run_game pong 20 "$PONG_SECS"

# Demos (best-effort): they may depend on local assets/datasets.
# Leave commented by default; uncomment if you want them in the sweep.
# "$CLI_BIN" demo replay >/dev/null
# "$CLI_BIN" stop >/dev/null || true
# "$CLI_BIN" demo text >/dev/null
# "$CLI_BIN" stop >/dev/null || true

printf '%s\n' "wrote $LOG_FILE" >&2

if [[ "$RESTART_DAEMON" == "1" ]]; then
  "$CLI_BIN" shutdown >/dev/null 2>&1 || true
fi
