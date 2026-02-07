"""
Replay object poses in world frame from object-in-camera trajectories.

Primary usage (from replay_4d_fast.py):
    setup_object_replay(...)
    register_replay_handler(...)

Standalone usage (requires EXTERNAL_CAMERA_POSES_BY_NAME to be set by caller):
    blender --python debug/replay_object.py -- --task_name picking_up_trash
"""

import argparse
import bisect
import json
import multiprocessing
import os
import sys

import bpy
from mathutils import Matrix, Quaternion
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


CAMERA_NAMES = {"head", "left_wrist", "right_wrist"}
OBJECT_COLLECTION_NAME = "ReplayObjects"
DEFAULT_TASK_NAME = "picking_up_trash"
DEFAULT_MISSING_POSE_POLICY = "hold_last"
CONTACT_SEGMENTS_KEY = "contact rich segments"
SOURCE_DEMO_DIRNAME = "source_demo"
OVERLAY_PARALLEL = True
OVERLAY_WORKERS = 0

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


def parse_segment_list(raw_segments):
    parsed = []
    if not isinstance(raw_segments, list):
        return parsed
    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        try:
            start_id = int(seg.get("start id"))
            end_id = int(seg.get("end id"))
        except Exception:
            continue
        if end_id < start_id:
            continue
        parsed.append(
            {
                "start_id": start_id,
                "end_id": end_id,
                "type": str(seg.get("type", "")).strip(),
            }
        )
    parsed.sort(key=lambda x: (x["start_id"], x["end_id"]))
    return parsed


def load_object_segments(path):
    if not os.path.isfile(path):
        return [], None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return [], None

    objects = data.get("objects")
    if not isinstance(objects, list) or not objects:
        return [], None

    obj0 = objects[0]
    if not isinstance(obj0, dict):
        return [], None

    segments = parse_segment_list(obj0.get("segments", []))
    max_end = None
    for seg in segments:
        if max_end is None or seg["end_id"] > max_end:
            max_end = seg["end_id"]
    return segments, max_end


def segment_type_for_id(segments, image_id):
    for seg in segments:
        if seg["start_id"] <= image_id <= seg["end_id"]:
            return seg["type"]
    return None


def nearest_id(existing_ids_sorted, target_id):
    pos = bisect.bisect_left(existing_ids_sorted, target_id)
    if pos <= 0:
        return existing_ids_sorted[0]
    if pos >= len(existing_ids_sorted):
        return existing_ids_sorted[-1]
    before = existing_ids_sorted[pos - 1]
    after = existing_ids_sorted[pos]
    return before if (target_id - before) <= (after - target_id) else after


def nearest_id_before_or_equal(existing_ids_sorted, target_id):
    pos = bisect.bisect_right(existing_ids_sorted, target_id)
    if pos <= 0:
        return None
    return existing_ids_sorted[pos - 1]


def world_cam_matrix(cam_poses, image_id):
    pose = cam_poses.get(image_id)
    if pose is None:
        return None
    cam_pos, cam_quat = pose
    return pose_matrix(cam_pos, cam_quat) @ OPENGL_TO_BLENDER_CAMERA_FRAME


def fill_missing_pose_map(base_pose_map, segments, cam_pose_dict, num_frames):
    if not base_pose_map:
        return {}
    if not segments:
        return dict(base_pose_map)

    filled = dict(base_pose_map)
    existing_ids_sorted = sorted(base_pose_map.keys())

    for image_id in range(1, int(num_frames) + 1):
        if image_id in filled:
            continue

        seg_type = segment_type_for_id(segments, image_id)
        if seg_type is None:
            continue

        if seg_type == "still_in_world":
            ref_id = nearest_id(existing_ids_sorted, image_id)
            filled[image_id] = base_pose_map[ref_id]
            continue

        if seg_type == "tracking":
            # Keep current object pose for tracking gaps; runtime handler will hold last pose.
            continue

        if seg_type in ("still_in_right_wrist", "still_in_left_wrist"):
            cam_name = "right_wrist" if seg_type == "still_in_right_wrist" else "left_wrist"
            cam_poses = cam_pose_dict.get(cam_name)
            if not isinstance(cam_poses, dict) or not cam_poses:
                continue

            candidate_ids = [i for i in existing_ids_sorted if i <= image_id and i in cam_poses]
            if not candidate_ids:
                continue
            ref_id = nearest_id_before_or_equal(candidate_ids, image_id)
            if ref_id is None:
                continue

            t_world_obj_ref = base_pose_map.get(ref_id)
            t_world_cam_ref = world_cam_matrix(cam_poses, ref_id)
            t_world_cam_now = world_cam_matrix(cam_poses, image_id)
            if t_world_obj_ref is None or t_world_cam_ref is None or t_world_cam_now is None:
                continue

            t_cam_obj_const = t_world_cam_ref.inverted() @ t_world_obj_ref
            filled[image_id] = t_world_cam_now @ t_cam_obj_const
            continue

    return filled


def load_contact_overlay_ids(path):
    if not os.path.isfile(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return set()

    objects = data.get("objects")
    if not isinstance(objects, list) or not objects:
        return set()
    obj0 = objects[0]
    if not isinstance(obj0, dict):
        return set()

    ids = set()
    raw_segments = obj0.get(CONTACT_SEGMENTS_KEY, [])
    if not isinstance(raw_segments, list):
        return ids

    for seg in raw_segments:
        if not isinstance(seg, dict):
            continue
        seg_type = str(seg.get("type", "")).strip().lower()
        if seg_type != "overlay":
            continue
        try:
            start_id = int(seg.get("start id"))
            end_id = int(seg.get("end id"))
        except Exception:
            continue
        if end_id < start_id:
            continue
        for image_id in range(start_id, end_id + 1):
            ids.add(image_id)
    return ids


def build_numeric_stem_map(directory):
    mapping = {}
    if not os.path.isdir(directory):
        return mapping
    for name in os.listdir(directory):
        stem, _ext = os.path.splitext(name)
        if stem.isdigit():
            mapping[int(stem)] = os.path.join(directory, name)
    return mapping


def build_demo_index(source_demo_dir):
    rgb_dir = os.path.join(source_demo_dir, "rgb")
    mask_dir = os.path.join(source_demo_dir, "masks")
    if not os.path.isdir(rgb_dir) or not os.path.isdir(mask_dir):
        return None
    return {
        "rgb_dir": rgb_dir,
        "mask_dir": mask_dir,
        "rgb": build_numeric_stem_map(rgb_dir),
        "mask": build_numeric_stem_map(mask_dir),
    }


def resolve_render_camera_name(scene):
    if scene is None:
        return None
    scene_hint = scene.get("render_camera_name") if hasattr(scene, "get") else None
    name = normalize_camera_name(scene_hint)
    if name:
        return name
    cam = scene.camera
    if cam is None:
        return None
    for target in (cam, getattr(cam, "data", None)):
        if target is None or not hasattr(target, "get"):
            continue
        name = normalize_camera_name(target.get("camera_name"))
        if name:
            return name
    for cand in (cam.name, getattr(getattr(cam, "data", None), "name", None)):
        name = normalize_camera_name(cand)
        if name:
            return name
    return None


def _overlay_frame_cv2(task):
    image_id, render_path, overlays = task
    if cv2 is None or np is None:
        return image_id, False, "opencv_unavailable"
    render = cv2.imread(render_path, cv2.IMREAD_UNCHANGED)
    if render is None:
        return image_id, False, "render_missing"
    render_h, render_w = render.shape[:2]
    render_has_alpha = render.ndim == 3 and render.shape[2] == 4

    for rgb_path, mask_path in overlays:
        rgb = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if rgb is None or mask is None:
            continue
        if (rgb.shape[1], rgb.shape[0]) != (render_w, render_h):
            rgb = cv2.resize(rgb, (render_w, render_h), interpolation=cv2.INTER_LINEAR)
        if (mask.shape[1], mask.shape[0]) != (render_w, render_h):
            mask = cv2.resize(mask, (render_w, render_h), interpolation=cv2.INTER_NEAREST)

        if mask.ndim == 3:
            mask_ch = mask[:, :, 0]
        else:
            mask_ch = mask
        mask_bool = mask_ch > 127

        if render_has_alpha:
            if rgb.ndim == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGRA)
            elif rgb.shape[2] == 3:
                alpha = np.full((rgb.shape[0], rgb.shape[1], 1), 255, dtype=rgb.dtype)
                rgb = np.concatenate([rgb, alpha], axis=2)
            elif rgb.shape[2] != 4:
                rgb = rgb[:, :, :4]
        else:
            if rgb.ndim == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            elif rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]

        render[mask_bool] = rgb[mask_bool]
        if render_has_alpha:
            render[mask_bool, 3] = 255

    ok = cv2.imwrite(render_path, render)
    return image_id, bool(ok), None


def _build_overlay_tasks(state_dict, scene, render_paths_by_frame=None):
    overlay_state = state_dict.get("overlay_state", {})
    overlay_frames = overlay_state.get("overlay_frames", {})
    if not overlay_frames:
        return []

    start_frame = int(overlay_state.get("start_frame", state_dict.get("start_frame", 1)))
    objects_by_name = overlay_state.get("objects_by_name", {})
    tasks = []

    for image_id, object_names in overlay_frames.items():
        frame = start_frame + int(image_id) - 1
        if render_paths_by_frame and frame in render_paths_by_frame:
            render_path = render_paths_by_frame[frame]
        else:
            render_path = bpy.path.abspath(scene.render.frame_path(frame=frame))
        overlays = []
        for object_name in sorted(object_names):
            entry = objects_by_name.get(object_name)
            if entry is None:
                continue
            demo_index = entry.get("demo_index") or {}
            rgb_path = demo_index.get("rgb", {}).get(image_id)
            mask_path = demo_index.get("mask", {}).get(image_id)
            if rgb_path and mask_path:
                overlays.append((rgb_path, mask_path))
        if overlays:
            tasks.append((image_id, render_path, overlays))
    return tasks


def reset_contact_overlay_state(state_dict, start_frame=None):
    if not isinstance(state_dict, dict):
        return
    overlay_state = state_dict.get("overlay_state")
    if not isinstance(overlay_state, dict):
        return
    overlay_state["overlay_frames"] = {}
    if start_frame is not None:
        overlay_state["start_frame"] = int(start_frame)


def apply_contact_overlays_after_render(
    scene,
    state_dict=None,
    render_paths_by_frame=None,
    parallel=OVERLAY_PARALLEL,
    workers=OVERLAY_WORKERS,
):
    if state_dict is None:
        state_dict = STATE
    if not isinstance(state_dict, dict):
        return False
    overlay_state = state_dict.get("overlay_state")
    if not isinstance(overlay_state, dict):
        return False

    tasks = _build_overlay_tasks(state_dict, scene, render_paths_by_frame)
    total = len(tasks)
    if total == 0:
        return False
    if cv2 is None or np is None:
        print("Contact overlay skipped: OpenCV/Numpy unavailable in Blender Python.")
        return False

    print(f"Contact overlay: applying source_demo on {total} frame(s).")

    if parallel and cv2 is not None and np is not None:
        proc_count = workers or max(1, (os.cpu_count() or 2) - 1)
        try:
            ctx = multiprocessing.get_context("fork") if hasattr(multiprocessing, "get_context") else multiprocessing
            with ctx.Pool(processes=proc_count) as pool:
                for idx, result in enumerate(pool.imap_unordered(_overlay_frame_cv2, tasks), start=1):
                    image_id, ok, err = result
                    msg = f"Overlay {idx}/{total}: image_id={image_id}"
                    if not ok and err:
                        msg += f" error={err}"
                    print(msg)
        except Exception as exc:
            print(f"Contact overlay parallel pass failed ({exc}), falling back to serial.")
            for idx, task in enumerate(tasks, start=1):
                image_id, ok, err = _overlay_frame_cv2(task)
                msg = f"Overlay {idx}/{total}: image_id={image_id}"
                if not ok and err:
                    msg += f" error={err}"
                print(msg)
    else:
        for idx, task in enumerate(tasks, start=1):
            image_id, ok, err = _overlay_frame_cv2(task)
            msg = f"Overlay {idx}/{total}: image_id={image_id}"
            if not ok and err:
                msg += f" error={err}"
            print(msg)

    for entry in overlay_state.get("entries", []):
        obj = entry.get("obj")
        if obj is None:
            continue
        _set_object_visible(obj, True)

    return True


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
    render_camera_name = resolve_render_camera_name(scene)
    overlay_state = state.get("overlay_state")
    overlay_frames = {}
    if isinstance(overlay_state, dict):
        overlay_frames = overlay_state.setdefault("overlay_frames", {})

    for entry in state["objects"]:
        if image_id < 1 or image_id > num_frames:
            _set_object_visible(entry["obj"], False)
            continue
        _apply_pose(entry, image_id, state["missing_pose_policy"])

        if not isinstance(overlay_state, dict):
            continue
        contact_ids = entry.get("contact_overlay_ids", set())
        demo_index = entry.get("demo_index") or {}
        has_demo = (
            image_id in demo_index.get("rgb", {})
            and image_id in demo_index.get("mask", {})
        )
        should_overlay = (
            render_camera_name == entry.get("camera_name")
            and image_id in contact_ids
            and has_demo
        )
        if should_overlay:
            _set_object_visible(entry["obj"], False)
            frame_map = overlay_frames.setdefault(image_id, set())
            frame_map.add(entry.get("object_name"))


def register_replay_handler(state_dict):
    handlers = bpy.app.handlers.frame_change_pre
    for handler in list(handlers):
        if getattr(handler, "__name__", "") == "replay_object_frame_handler":
            handlers.remove(handler)

    reset_contact_overlay_state(state_dict, state_dict.get("start_frame"))
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

        object_json_path = os.path.join(obj_dir, "object.json")
        segments, segment_hint = load_object_segments(object_json_path)
        contact_overlay_ids = load_contact_overlay_ids(object_json_path)
        frame_hint = segment_hint
        if frame_hint is None:
            frame_hint = load_object_json_frame_hint(object_json_path)
        if frame_hint is not None:
            max_hint_id = max(max_hint_id, int(frame_hint))

        demo_index = None
        if contact_overlay_ids:
            source_demo_dir = os.path.join(obj_dir, SOURCE_DEMO_DIRNAME)
            if os.path.isdir(source_demo_dir):
                demo_index = build_demo_index(source_demo_dir)
                if demo_index is None:
                    print(
                        f"Warning: source_demo missing rgb/masks for {object_name}; "
                        "contact overlay disabled."
                    )
            else:
                print(
                    f"Warning: contact rich segments found but no {SOURCE_DEMO_DIRNAME} "
                    f"directory for {object_name}; overlay disabled."
                )

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
                "segments": segments,
                "contact_overlay_ids": contact_overlay_ids,
                "demo_index": demo_index,
            }
        )

    if num_frames is None:
        num_frames = max(1, max_pose_id, max_hint_id)

    for entry in objects:
        entry["pose_map"] = fill_missing_pose_map(
            base_pose_map=entry["pose_map"],
            segments=entry.get("segments", []),
            cam_pose_dict=cam_pose_dict,
            num_frames=num_frames,
        )
        entry["sorted_pose_ids"] = sorted(entry["pose_map"].keys())

    overlay_entries = [
        entry
        for entry in objects
        if entry.get("contact_overlay_ids") and entry.get("demo_index") is not None
    ]

    state = {
        "ready": True,
        "task_dir": task_dir,
        "start_frame": int(start_frame),
        "num_frames": int(num_frames),
        "missing_pose_policy": missing_pose_policy,
        "objects": objects,
        "overlay_state": {
            "start_frame": int(start_frame),
            "overlay_frames": {},
            "entries": overlay_entries,
            "objects_by_name": {e["object_name"]: e for e in overlay_entries},
        },
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
