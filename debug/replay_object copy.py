"""
Replay object poses in world frame from object-in-camera trajectories.

Primary usage (from replay_4d_fast.py):
    setup_object_replay(...)
    register_replay_handler(...)

Standalone usage (requires EXTERNAL_CAMERA_POSES_BY_NAME to be set by caller):
    blender --python debug/replay_object.py -- --task_name picking_up_trash
"""

import argparse
import json
import os
import sys

import bpy
from mathutils import Matrix, Quaternion


CAMERA_NAMES = {"head", "left_wrist", "right_wrist"}
OBJECT_COLLECTION_NAME = "ReplayObjects"
DEFAULT_TASK_NAME = "picking_up_trash"
DEFAULT_MISSING_POSE_POLICY = "hold_last"

# Optional external camera pose input for standalone mode.
EXTERNAL_CAMERA_POSES_BY_NAME = None

STATE = {}
OPENGL_TO_BLENDER_CAMERA_FRAME = Matrix(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)


def _parse_argv():
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def parse_args():
    parser = argparse.ArgumentParser(description="Replay object poses.")
    parser.add_argument(
        "--task_name",
        default=DEFAULT_TASK_NAME,
        help="Task folder name or absolute path to task directory.",
    )
    parser.add_argument(
        "--parquet_path",
        default=None,
        help="Unused compatibility argument (accepted for workflow compatibility).",
    )
    argv = _parse_argv()
    if not argv and DEFAULT_TASK_NAME:
        return argparse.Namespace(task_name=DEFAULT_TASK_NAME, parquet_path=None)
    return parser.parse_args(argv)


def normalize_camera_name(name):
    if not isinstance(name, str):
        return None
    cand = name.strip().lower()
    if cand in CAMERA_NAMES:
        return cand
    if "left" in cand and "wrist" in cand:
        return "left_wrist"
    if "right" in cand and "wrist" in cand:
        return "right_wrist"
    if "head" in cand:
        return "head"
    return None


def resolve_task_dir(task_name, base_dir):
    if os.path.isabs(task_name):
        if os.path.isdir(task_name):
            return task_name
        raise FileNotFoundError(f"Task directory not found: {task_name}")

    candidates = []
    if base_dir:
        candidates.append(base_dir)
    cwd = os.getcwd()
    if cwd not in candidates:
        candidates.append(cwd)
    blend_path = bpy.data.filepath
    if blend_path:
        blend_dir = os.path.dirname(blend_path)
        if blend_dir and blend_dir not in candidates:
            candidates.append(blend_dir)

    tried = []
    for base in candidates:
        candidate = os.path.join(base, task_name)
        tried.append(candidate)
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError(
        f"Task directory not found: {task_name} (tried: {', '.join(tried)})"
    )


def ensure_collection(name):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def find_object_mesh_dirs(task_dir):
    mesh_dirs = []
    for root, _dirs, files in os.walk(task_dir):
        if "textured_mesh.obj" in files:
            mesh_dirs.append(root)
    mesh_dirs.sort()
    return mesh_dirs


def load_camera_name(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing camera_name.txt: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    name = normalize_camera_name(raw)
    if name is None:
        raise ValueError(f"Unsupported camera name '{raw}' in {path}")
    return name


def load_ob_in_cam_matrix(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            rows.append([float(v) for v in parts])
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise ValueError(f"Invalid 4x4 transform in {path}")
    return Matrix(rows)


def pose_matrix(position, quat_wxyz):
    q = Quaternion(quat_wxyz).normalized()
    return Matrix.Translation(position) @ q.to_matrix().to_4x4()


def import_obj_mesh(mesh_path, object_name, collection):
    if not os.path.isfile(mesh_path):
        raise FileNotFoundError(mesh_path)

    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=mesh_path)
    imported = [obj for name, obj in bpy.data.objects.items() if name not in before]
    if not imported:
        raise RuntimeError(f"No objects imported from {mesh_path}")

    target_obj = None
    for obj in imported:
        if obj.type == "MESH":
            target_obj = obj
            break
    if target_obj is None:
        target_obj = imported[0]

    target_obj.name = object_name
    target_obj.rotation_mode = "XYZ"

    for obj in imported:
        for user_collection in list(obj.users_collection):
            user_collection.objects.unlink(obj)
        collection.objects.link(obj)

    return target_obj


def load_object_json_frame_hint(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    objects = data.get("objects")
    if not isinstance(objects, list):
        return None

    max_end = None
    for obj in objects:
        segments = obj.get("segments", []) if isinstance(obj, dict) else []
        if not isinstance(segments, list):
            continue
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            try:
                end_id = int(seg.get("end id"))
            except Exception:
                continue
            if max_end is None or end_id > max_end:
                max_end = end_id
    return max_end


def build_pose_map(obj_dir, camera_name, cam_pose_dict):
    ob_in_cam_dir = os.path.join(obj_dir, "ob_in_cam")
    if not os.path.isdir(ob_in_cam_dir):
        raise FileNotFoundError(f"Missing ob_in_cam directory: {ob_in_cam_dir}")

    cam_poses = cam_pose_dict.get(camera_name)
    if not isinstance(cam_poses, dict) or not cam_poses:
        available = sorted(cam_pose_dict.keys())
        raise RuntimeError(
            f"Missing external camera poses for '{camera_name}'. Available: {available}"
        )

    pose_map = {}
    for name in sorted(os.listdir(ob_in_cam_dir)):
        if not name.endswith(".txt"):
            continue
        stem = os.path.splitext(name)[0]
        if not stem.isdigit():
            continue
        image_id = int(stem)
        if image_id not in cam_poses:
            continue
        cam_pos, cam_quat = cam_poses[image_id]
        t_world_cam = pose_matrix(cam_pos, cam_quat)
        t_cam_obj = load_ob_in_cam_matrix(os.path.join(ob_in_cam_dir, name))
        t_world_cam_blender = t_world_cam @ OPENGL_TO_BLENDER_CAMERA_FRAME
        pose_map[image_id] = t_world_cam_blender @ t_cam_obj

    return pose_map


def _set_object_visible(obj, visible):
    hidden = not visible
    obj.hide_render = hidden
    if hasattr(obj, "hide_viewport"):
        obj.hide_viewport = hidden
    else:
        obj.hide_set(hidden)


def _apply_pose(entry, image_id, policy):
    fresh_pose = entry["pose_map"].get(image_id)
    if fresh_pose is not None:
        entry["last_pose"] = fresh_pose
        pose = fresh_pose
    elif policy == "hold_last":
        pose = entry["last_pose"]
    else:
        pose = None

    obj = entry["obj"]
    if pose is None:
        _set_object_visible(obj, False)
        return

    loc, rot, _scale = pose.decompose()
    obj.location = loc
    obj.rotation_mode = "XYZ"
    obj.rotation_euler = rot.to_euler("XYZ")
    _set_object_visible(obj, True)


def replay_object_frame_handler(scene):
    state = STATE
    if not state.get("ready"):
        return

    start_frame = state["start_frame"]
    num_frames = state["num_frames"]
    image_id = scene.frame_current - start_frame + 1

    for entry in state["objects"]:
        if image_id < 1 or image_id > num_frames:
            _set_object_visible(entry["obj"], False)
            continue
        _apply_pose(entry, image_id, state["missing_pose_policy"])


def register_replay_handler(state_dict):
    handlers = bpy.app.handlers.frame_change_pre
    for handler in list(handlers):
        if getattr(handler, "__name__", "") == "replay_object_frame_handler":
            handlers.remove(handler)

    STATE.clear()
    STATE.update(state_dict)
    STATE["ready"] = True
    handlers.append(replay_object_frame_handler)


def setup_object_replay(
    task_dir,
    cam_pose_dict,
    start_frame=1,
    num_frames=None,
    missing_pose_policy=DEFAULT_MISSING_POSE_POLICY,
):
    if missing_pose_policy != "hold_last":
        raise ValueError(
            f"Unsupported missing_pose_policy='{missing_pose_policy}'. Only 'hold_last' is supported."
        )

    if not isinstance(cam_pose_dict, dict) or not cam_pose_dict:
        raise RuntimeError("cam_pose_dict must be a non-empty dict keyed by camera name.")

    mesh_dirs = find_object_mesh_dirs(task_dir)
    if not mesh_dirs:
        raise FileNotFoundError(
            f"No textured_mesh.obj files found under task directory: {task_dir}"
        )

    collection = ensure_collection(OBJECT_COLLECTION_NAME)

    objects = []
    max_pose_id = 0
    max_hint_id = 0

    for obj_dir in mesh_dirs:
        object_name = os.path.basename(obj_dir)
        mesh_path = os.path.join(obj_dir, "textured_mesh.obj")
        camera_name = load_camera_name(os.path.join(obj_dir, "camera_name.txt"))
        if camera_name not in cam_pose_dict:
            available = sorted(cam_pose_dict.keys())
            raise RuntimeError(
                f"camera_name.txt requests '{camera_name}' for '{object_name}', "
                f"but no camera poses were provided for it. Available: {available}"
            )

        pose_map = build_pose_map(obj_dir, camera_name, cam_pose_dict)
        sorted_ids = sorted(pose_map.keys())
        if sorted_ids:
            max_pose_id = max(max_pose_id, sorted_ids[-1])

        frame_hint = load_object_json_frame_hint(os.path.join(obj_dir, "object.json"))
        if frame_hint is not None:
            max_hint_id = max(max_hint_id, int(frame_hint))

        obj = import_obj_mesh(mesh_path, object_name, collection)
        _set_object_visible(obj, False)

        objects.append(
            {
                "object_name": object_name,
                "camera_name": camera_name,
                "obj": obj,
                "pose_map": pose_map,
                "sorted_pose_ids": sorted_ids,
                "last_pose": None,
                "frame_hint_max": frame_hint,
            }
        )

    if num_frames is None:
        num_frames = max(1, max_pose_id, max_hint_id)

    state = {
        "ready": True,
        "task_dir": task_dir,
        "start_frame": int(start_frame),
        "num_frames": int(num_frames),
        "missing_pose_policy": missing_pose_policy,
        "objects": objects,
    }

    scene = bpy.context.scene
    scene.frame_start = int(start_frame)
    scene.frame_end = int(start_frame) + int(num_frames) - 1

    return state


def main():
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    task_dir = resolve_task_dir(args.task_name, project_root)

    external_camera_poses = EXTERNAL_CAMERA_POSES_BY_NAME
    if not isinstance(external_camera_poses, dict) or not external_camera_poses:
        raise RuntimeError(
            "Standalone replay_object.py requires EXTERNAL_CAMERA_POSES_BY_NAME to be set "
            "before calling main()."
        )

    state = setup_object_replay(
        task_dir=task_dir,
        cam_pose_dict=external_camera_poses,
        start_frame=1,
        num_frames=None,
        missing_pose_policy=DEFAULT_MISSING_POSE_POLICY,
    )
    register_replay_handler(state)
    bpy.context.scene.frame_set(state["start_frame"])

    print(
        f"Object replay ready: objects={len(state['objects'])} "
        f"frames={state['num_frames']} task_dir={task_dir}"
    )
    return state


if __name__ == "__main__":
    main()
