#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lingxiao/Downloads/blender-5.0.1-linux-x64"
BLENDER="${ROOT}/blender"
SCRIPT="${ROOT}/blender_replay_render/4d_replay/replay_4d_fast.py"
TASK="${TASK:-picking_up_trash}"

cd "$ROOT"

run_cam() {
  local cam="$1"
  "${BLENDER}" --factory-startup --python "${SCRIPT}" -- \
    --task_name "${TASK}" \
    --camera_name "${cam}" &
}

run_cam left_wrist
# run_cam right_wrist
# run_cam head

wait
echo "All renders finished."
