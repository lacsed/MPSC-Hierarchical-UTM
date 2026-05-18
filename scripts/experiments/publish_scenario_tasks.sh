#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${UTM_RUN_DIR:-}" ]]; then
  echo "[ERROR] Source scenario.env first."
  exit 1
fi

TASK_FILE="$UTM_RUN_DIR/task_list.active.txt"
if [[ ! -f "$TASK_FILE" ]]; then
  echo "[ERROR] Task file not found: $TASK_FILE"
  exit 1
fi

SLEEP_S="${UTM_TASK_SLEEP:-1.0}"
while IFS= read -r task; do
  [[ -z "$task" ]] && continue
  ros2 topic pub /task_todo std_msgs/msg/String "{data: '$task'}" --once
  sleep "$SLEEP_S"
done < "$TASK_FILE"
