#!/usr/bin/env bash

set -euo pipefail

ROOT="$(realpath "$(dirname "$0")/..")"
APP="$ROOT/app"

BENCHMARK="${1:-dqn}"
WORKER=false
if [[ "${1:-}" == "--worker" ]]; then
    BENCHMARK="${2:-dqn}"
    WORKER=true
elif [[ "${2:-}" == "--worker" ]]; then
    WORKER=true
fi

case "$BENCHMARK" in
    q)
        NOTEBOOK="$ROOT/notebooks/MultiTaxi-5x5-Benchmark-q.ipynb"
        OUTPUT="$ROOT/notebooks/MultiTaxi-5x5-Benchmark-q.executed.ipynb"
        LOG="$ROOT/logs/MultiTaxi-5x5-Benchmark-q.log"
        LOCK="$ROOT/logs/MultiTaxi-5x5-Benchmark-q.lock"
        PID_FILE="$ROOT/logs/MultiTaxi-5x5-Benchmark-q.pid"
        STATUS_FILE="$ROOT/logs/MultiTaxi-5x5-Benchmark-q.status"
        ;;
    dqn)
        NOTEBOOK="$ROOT/notebooks/MultiTaxi-5x5-Benchmark-dqn.ipynb"
        OUTPUT="$ROOT/notebooks/MultiTaxi-5x5-Benchmark-dqn.executed.ipynb"
        LOG="$ROOT/logs/MultiTaxi-5x5-Benchmark-dqn.log"
        LOCK="$ROOT/logs/MultiTaxi-5x5-Benchmark-dqn.lock"
        PID_FILE="$ROOT/logs/MultiTaxi-5x5-Benchmark-dqn.pid"
        STATUS_FILE="$ROOT/logs/MultiTaxi-5x5-Benchmark-dqn.status"
        ;;
    *)
        printf 'Usage: %s [q|dqn]\n' "$0" >&2
        exit 2
        ;;
esac

mkdir -p "$ROOT/logs"

if [[ "$WORKER" == true ]]; then
    exec 9>"$LOCK"
    if ! flock -n 9; then
        printf 'already running\n' >>"$LOG"
        exit 75
    fi

    printf '%s\n' "$$" >"$PID_FILE"
    printf 'running pid=%s started=%s\n' "$$" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
    printf 'started benchmark=%s pid=%s notebook=%s output=%s\n' "$BENCHMARK" "$$" "$NOTEBOOK" "$OUTPUT" >>"$LOG"
    cleanup() {
        exit_code=$?
        printf 'finished pid=%s status=%s ended=%s\n' "$$" "$exit_code" "$(date --iso-8601=seconds)" >"$STATUS_FILE"
        rm -f "$PID_FILE"
    }
    trap cleanup EXIT

    cd "$APP"
    conda run --no-capture-output -n TFM_env python -m nbconvert \
        --to notebook \
        --execute \
        --ExecutePreprocessor.timeout=-1 \
        --ExecutePreprocessor.kernel_name=tfm_env \
        --output "$(basename "$OUTPUT")" \
        --output-dir "$ROOT/notebooks" \
        "$NOTEBOOK"
    exit 0
fi

if [[ -s "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null; then
    printf 'Notebook already running (PID %s).\n' "$(<"$PID_FILE")" >&2
    exit 1
fi

nohup bash "$ROOT/scripts/lauch_notebook.sh" "$BENCHMARK" --worker \
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
