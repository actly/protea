#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PID_FILE=".protea.pid"
GRACEFUL_TIMEOUT=15   # seconds to wait for graceful shutdown

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found — Protea is not running."
    exit 0
fi

pid=$(cat "$PID_FILE")

if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "Process $pid not running — cleaned up stale PID file."
    exit 0
fi

# Send SIGTERM for graceful shutdown (sentinel cleans up Ring 2, skills, bot).
echo "Stopping Protea (pid=$pid) — waiting up to ${GRACEFUL_TIMEOUT}s for graceful shutdown..."
kill "$pid"

# Wait for process to exit gracefully.
elapsed=0
while kill -0 "$pid" 2>/dev/null; do
    if [ "$elapsed" -ge "$GRACEFUL_TIMEOUT" ]; then
        echo "Process did not exit in time — sending SIGKILL..."
        kill -9 "$pid" 2>/dev/null || true
        break
    fi
    sleep 1
    elapsed=$((elapsed + 1))
done

rm -f "$PID_FILE"
echo "Protea stopped (pid=$pid)"
