#!/usr/bin/env bash

set -euo pipefail

ROOT="$(realpath "$(dirname "$0")/..")"
APP="$ROOT/app"
NOTEBOOK="$ROOT/notebooks/MultiTaxi-5x5-Benchmark-dqn.ipynb"
OUTPUT="$ROOT/notebooks/MultiTaxi-5x5-Benchmark-dqn.executed.ipynb"
LOG="$ROOT/logs/MultiTaxi-5x5-Benchmark-dqn.log"
LOCK="$ROOT/logs/MultiTaxi-5x5-Benchmark-dqn.lock"
PID_FILE="$ROOT/logs/MultiTaxi-5x5-Benchmark-dqn.pid"
STATUS_FILE="$ROOT/logs/MultiTaxi-5x5-Benchmark-dqn.status"

mkdir -p "$ROOT/logs"

if [[ "${1:-}" == "--worker" ]]; then
    exec 9>"$LOCK"
    if ! flock -n 9; then
        printf 'already running\n' >>"$LOG"
        exit 75
    fi

    printf '%s\n' "$$" >"$PID_FILE"
    printf 'running pid=%s started=%s\n' "$$" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
    cleanup() {
        exit_code=$?
        printf 'finished pid=%s status=%s ended=%s\n' "$$" "$exit_code" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
        rm -f "$PID_FILE"
    }
    trap cleanup EXIT

    cd "$APP"
    jupyter nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=-1 \
        --output "$(basename "$OUTPUT")" \
        --output-dir "$ROOT/notebooks" \
        "$NOTEBOOK"
    exit 0
fi

if [[ -s "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
    printf 'Notebook already running (PID %s).\n' "$(<"$PID_FILE")" >&2
    exit 1
fi

nohup bash "$ROOT/scripts/lauch_notebook.sh" --worker \
    >>"$LOG" 2>&1 < /dev/null &

worker_pid=$!
for _ in {1..20}; do
    if [[ -s "$PID_FILE" ]]; then
        printf 'Notebook started in background (PID %s).\n' "$(<"$PID_FILE")"
        exit 0
    fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
        printf 'Notebook failed to start; inspect %s.\n' "$LOG" >&2
        exit 1
    fi
    sleep 0.1
done

printf 'Notebook launch submitted (PID %s); inspect %s.\n' "$worker_pid" "$STATUS_FILE"
