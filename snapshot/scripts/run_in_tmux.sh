#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/run_in_tmux.sh SESSION_NAME \"COMMAND\""
  exit 1
fi

SESSION_NAME="$1"
shift
COMMAND="$*"
mkdir -p logs
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="logs/${SESSION_NAME}_${TIMESTAMP}.log"

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not installed or not on PATH. Use scripts/run_with_nohup.sh or install tmux on the remote server."
  exit 1
fi

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "tmux session already exists: $SESSION_NAME"
  echo "Attach with: tmux attach -t $SESSION_NAME"
  exit 1
fi

tmux new-session -d -s "$SESSION_NAME" "bash -lc '$COMMAND' > '$LOG_FILE' 2>&1"

echo "Started tmux session: $SESSION_NAME"
echo "Command: $COMMAND"
echo "Log file: $LOG_FILE"
echo "The job is running on the remote server and continues if VS Code disconnects or the laptop sleeps."
echo "Reattach: tmux attach -t $SESSION_NAME"
echo "Tail logs: tail -f $LOG_FILE"
