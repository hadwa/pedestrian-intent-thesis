#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/run_with_nohup.sh \"COMMAND\""
  exit 1
fi

COMMAND="$*"
mkdir -p logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/nohup_${TIMESTAMP}.log"

nohup bash -lc "$COMMAND" > "$LOG_FILE" 2>&1 &
PID="$!"

echo "Started nohup job on the remote server."
echo "PID: $PID"
echo "Command: $COMMAND"
echo "Log file: $LOG_FILE"
echo "The job continues if VS Code disconnects or the laptop sleeps, as long as the remote server keeps running."
echo "Tail logs: tail -f $LOG_FILE"
