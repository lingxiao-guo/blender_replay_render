#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BLENDER_BIN="${BLENDER_BIN:-${ROOT}/blender}"
SCRIPT_PATH="${SCRIPT_PATH:-${ROOT}/debug/replay_4d_fast.py}"
TASK="${1:-${TASK:-picking_up_trash}}"
PARALLEL="${PARALLEL:-0}"

cd "$ROOT"

if [[ ! -x "$BLENDER_BIN" ]]; then
  echo "Blender binary not found or not executable: $BLENDER_BIN" >&2
  exit 1
fi

if [[ ! -f "$SCRIPT_PATH" ]]; then
  echo "Replay script not found: $SCRIPT_PATH" >&2
  exit 1
fi

if [[ $# -gt 1 ]]; then
  CAMERAS=("${@:2}")
else
  CAMERAS=(left_wrist right_wrist head)
fi

run_cam() {
  local cam="$1"
  echo "[render_all] task=${TASK} camera=${cam}"
  "${BLENDER_BIN}" --factory-startup --python "${SCRIPT_PATH}" -- \
    --task_name "${TASK}" \
    --camera_name "${cam}"
}

if [[ "$PARALLEL" == "1" ]]; then
  pids=()
  for cam in "${CAMERAS[@]}"; do
    run_cam "${cam}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
else
  for cam in "${CAMERAS[@]}"; do
    run_cam "${cam}"
  done
fi

echo "All renders finished."
