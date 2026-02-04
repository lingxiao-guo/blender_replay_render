"""
Replay object poses in world frame using object-in-camera poses and camera trajectories.

Usage (from command line):
  blender --python replay_object.py -- <task_name>

Or open in Blender > Scripting > Run Script (task_name still required).
"""
import argparse
import bisect
import csv
import json
import os
import sys

import bpy
from mathutils import Matrix, Quaternion

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise RuntimeError(
        "pyarrow is required. Install into Blender's Python, e.g.\n"
        "  /path/to/blender/5.0/python/bin/python3 -m pip install pyarrow"
    ) from exc

try:
    import numpy as np
except Exception:
    np = None

# ----------------------
# User configuration
# ----------------------
OBJECT_COLLECTION_NAME = "ImportedObjects"

START_FRAME = 1
FPS = 60

# Add to image_id from ob_in_cam filename when matching.
CAMERA_ID_OFFSET = 0

CAMERA_CSV_BY_NAME = {
    "head": "head_camera_pose.csv",
    "left_wrist": "left_wrist_camera_pose.csv",
    "right_wrist": "right_wrist_camera_pose.csv",
}

CONTACT_SEGMENTS_KEY = "contact rich segments"
SOURCE_DEMO_DIRNAME = "source_demo"

# Proprioception indices for R1Pro (observation.state)
GRIPPER_LEFT_SLICE = slice(193, 195)
GRIPPER_RIGHT_SLICE = slice(232, 234)
GRIPPER_VEL_EPS = 1e-3

# Optional default for Blender Text Editor runs (leave empty to require args).
DEFAULT_TASK_NAME = "picking_up_trash"


# Global state for contact-rich overlay handling.
_OVERLAY_STATE = None


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(
        description="Replay object poses for all objects under a task directory."
    )
    parser.add_argument(
        "task_name",
        help="Task folder name under this script directory (e.g. picking_up_trash).",
    )
    if not argv and DEFAULT_TASK_NAME:
        return argparse.Namespace(task_name=DEFAULT_TASK_NAME)
    return parser.parse_args(argv)


def resolve_task_dir(task_name, base_dir):
    if os.path.isabs(task_name):
        if os.path.isdir(task_name):
            return task_name
        raise FileNotFoundError(f"Task directory not found: {task_name}")

    candidates = []
    if base_dir and base_dir not in (os.sep, ""):
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
        "Task directory not found: "
        f"{task_name} (tried: {', '.join(tried)})"
    )


def find_parquet(task_dir):
    parquet_paths = []
    for root, _dirs, files in os.walk(task_dir):
        for name in files:
            if name.endswith(".parquet"):
                parquet_paths.append(os.path.join(root, name))
    if not parquet_paths:
        raise FileNotFoundError(f"No .parquet files found under: {task_dir}")
    parquet_paths.sort()
    if len(parquet_paths) > 1:
        print(
            "Warning: multiple parquet files found. Using the first one:",
            parquet_paths[0],
        )
    return parquet_paths[0]


def load_parquet_state(path):
    pf = pq.ParquetFile(path)
    state_col = "observation.state"
    if state_col not in pf.schema_arrow.names:
        state_col = "observation/state"
    if state_col not in pf.schema_arrow.names:
        raise ValueError("Parquet missing observation.state column.")
    table = pq.read_table(path, columns=[state_col])
    return table.column(0).to_pylist()


def compute_close_segments(widths, vel_eps):
    n = len(widths)
    if n < 2:
        return []
    closing = [False] * n
    opening = [False] * n
    for i in range(1, n):
        v = widths[i] - widths[i - 1]
        if v < -vel_eps:
            closing[i] = True
        elif v > vel_eps:
            opening[i] = True

    close_moments = []
    i = 1
    while i < n:
        if closing[i]:
            j = i
            while j + 1 < n and closing[j + 1]:
                j += 1
            close_moments.append(j)
            i = j + 1
        else:
            i += 1

    segments = []
    for cm in close_moments:
        end = cm
        k = cm + 1
        while k < n and not opening[k]:
            end = k
            k += 1
        # Convert to 1-based timestep (use cm + 2 for robustness).
        segments.append((cm + 2, end + 1))
    return segments


def build_close_start_by_id(segments):
    mapping = {}
    for start, end in segments:
        for i in range(start, end + 1):
            mapping[i] = start
    return mapping


def ensure_collection(name):
    if name in bpy.data.collections:
        return bpy.data.collections[name]
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def get_new_objects(before_names):
    return [obj for name, obj in bpy.data.objects.items() if name not in before_names]


def move_objects_to_collection(objects, collection):
    for obj in objects:
        for coll in obj.users_collection:
            coll.objects.unlink(obj)
        collection.objects.link(obj)


def import_obj(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    before = set(bpy.data.objects.keys())
    bpy.ops.wm.obj_import(filepath=path)
    return get_new_objects(before)


def find_object_assets(task_dir):
    object_assets = []
    for root, _dirs, files in os.walk(task_dir):
        if "textured_mesh.obj" in files:
            object_assets.append(os.path.join(root, "textured_mesh.obj"))
    object_assets.sort()
    if not object_assets:
        raise FileNotFoundError(
            f"No textured_mesh.obj files found under task directory: {task_dir}"
        )
    return object_assets


def load_camera_poses(csv_path):
    poses = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = int(row["image_id"])
            pos = (float(row["x"]), float(row["y"]), float(row["z"]))
            quat = (float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"]))
            poses[image_id] = (pos, quat)
    return poses


def load_camera_name(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        name = f.read().strip()
    if name not in CAMERA_CSV_BY_NAME:
        raise ValueError(f"Unsupported camera name in {path}: {name}")
    return name


def load_ob_in_cam_files(path):
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".txt")]
        files.sort()
        return files
    if os.path.isfile(path):
        return [path]
    raise FileNotFoundError(path)


def load_ob_in_cam_matrix(path):
    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            values.append([float(p) for p in parts])
    if len(values) != 4 or any(len(row) != 4 for row in values):
        raise ValueError(f"Invalid 4x4 matrix in {path}")
    return Matrix(values)


def pose_matrix(position, quat):
    q = Quaternion(quat).normalized()
    return Matrix.Translation(position) @ q.to_matrix().to_4x4()


def get_or_import_object(obj_path, object_name, collection):
    if object_name in bpy.data.objects:
        obj = bpy.data.objects[object_name]
        obj.rotation_mode = "XYZ"
        return obj

    imported = import_obj(obj_path)
    if not imported:
        raise RuntimeError(f"No objects imported from {obj_path}")

    # If multiple objects were imported, keep the first mesh.
    obj = None
    for candidate in imported:
        if candidate.type == "MESH":
            obj = candidate
            break
    if obj is None:
        obj = imported[0]
    obj.name = object_name
    obj.rotation_mode = "XYZ"

    move_objects_to_collection([obj] + [o for o in imported if o != obj], collection)
    return obj


def parse_segment_list(raw_segments, path, label):
    parsed = []
    for seg in raw_segments:
        start_id = int(seg["start id"])
        end_id = int(seg["end id"])
        seg_type = seg.get("type", label)
        parsed.append(
            {
                "start_id": start_id,
                "end_id": end_id,
                "type": seg_type,
            }
        )
    return parsed


def load_object_metadata(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r") as f:
        data = json.load(f)
    objects = data.get("objects", [])
    if not objects:
        raise ValueError(f"No objects found in {path}")
    obj_data = objects[0]
    segments_raw = obj_data.get("segments", [])
    if not segments_raw:
        raise ValueError(f"No segments found in {path}")
    segments = parse_segment_list(segments_raw, path, "segments")
    contact_raw = obj_data.get(CONTACT_SEGMENTS_KEY, [])
    contact_segments = (
        parse_segment_list(contact_raw, path, CONTACT_SEGMENTS_KEY)
        if isinstance(contact_raw, list) and contact_raw
        else []
    )
    return segments, contact_segments


def segment_ids(segment):
    return range(segment["start_id"], segment["end_id"] + 1)


def collect_ids_for_type(segments, seg_type):
    ids = set()
    for seg in segments:
        if seg["type"] != seg_type:
            continue
        ids.update(segment_ids(seg))
    return ids


def collect_contact_ids(segments):
    ids = set()
    for seg in segments:
        ids.update(segment_ids(seg))
    return ids


def nearest_id(existing_ids_sorted, target_id):
    pos = bisect.bisect_left(existing_ids_sorted, target_id)
    if pos == 0:
        return existing_ids_sorted[0]
    if pos >= len(existing_ids_sorted):
        return existing_ids_sorted[-1]
    before = existing_ids_sorted[pos - 1]
    after = existing_ids_sorted[pos]
    return before if (target_id - before) <= (after - target_id) else after


def build_tracking_pose_map(pose_files, cam_poses, tracking_ids):
    obj_pose_by_id = {}
    for path in pose_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        if not stem.isdigit():
            continue
        image_id = int(stem) + CAMERA_ID_OFFSET
        if image_id not in tracking_ids:
            continue
        if image_id not in cam_poses:
            continue
        cam_pos, cam_quat = cam_poses[image_id]
        T_world_cam = pose_matrix(cam_pos, cam_quat)
        T_cam_obj = load_ob_in_cam_matrix(path)
        obj_pose_by_id[image_id] = T_world_cam @ T_cam_obj
    return obj_pose_by_id


def fill_segment_poses(
    segments,
    tracking_poses,
    cam_poses_by_name,
    tracking_cam_name,
    close_start_by_side,
):
    filled = dict(tracking_poses)
    tracking_ids_sorted = sorted(tracking_poses.keys())
    tracking_cam_poses = cam_poses_by_name.get(tracking_cam_name, {})
    tracking_cam_ids = [i for i in tracking_ids_sorted if i in tracking_cam_poses]
    for seg in segments:
        seg_type = seg["type"]
        if not tracking_ids_sorted:
            continue
        if seg_type == "tracking":
            # No fill-missing logic for tracking segments.
            # if not tracking_cam_ids:
            #     continue
            # if tracking_cam_name == "left_wrist":
            #     close_map = close_start_by_side.get("left", {})
            # elif tracking_cam_name == "right_wrist":
            #     close_map = close_start_by_side.get("right", {})
            # else:
            #     close_map = {}
            # for image_id in segment_ids(seg):
            #     if image_id in filled:
            #         continue
            #     if image_id not in tracking_cam_poses:
            #         continue
            #     ref_id = nearest_id(tracking_cam_ids, image_id)
            #     T_world_obj_ref = tracking_poses[ref_id]
            #     if image_id not in close_map:
            #         filled[image_id] = T_world_obj_ref
            #         continue
            #     close_id = close_map[image_id]
            #     if close_id not in tracking_cam_poses:
            #         filled[image_id] = T_world_obj_ref
            #         continue
            #     cam_pos_close, cam_quat_close = tracking_cam_poses[close_id]
            #     T_world_cam_close = pose_matrix(cam_pos_close, cam_quat_close)
            #     T_cam_obj_const = T_world_cam_close.inverted() @ T_world_obj_ref
            #     cam_pos, cam_quat = tracking_cam_poses[image_id]
            #     T_world_cam = pose_matrix(cam_pos, cam_quat)
            #     filled[image_id] = T_world_cam @ T_cam_obj_const
            continue
        if seg_type == "still_in_world":
            for image_id in segment_ids(seg):
                if image_id in filled:
                    continue
                # Fill missing object poses using nearest neighbor filling
                nearest = nearest_id(tracking_ids_sorted, image_id)
                filled[image_id] = tracking_poses[nearest]
            continue

        if seg_type in ("still_in_right_wrist", "still_in_left_wrist"):
            cam_name = "right_wrist" if seg_type == "still_in_right_wrist" else "left_wrist"
            cam_poses = cam_poses_by_name.get(cam_name)
            if not cam_poses:
                continue
            eligible_ids = [i for i in tracking_ids_sorted if i in cam_poses]
            if not eligible_ids:
                continue
            ref_id = nearest_id(eligible_ids, seg["start_id"])
            cam_pos_ref, cam_quat_ref = cam_poses[ref_id]
            T_world_cam_ref = pose_matrix(cam_pos_ref, cam_quat_ref)
            T_world_obj_ref = tracking_poses[ref_id]
            T_cam_obj_const = T_world_cam_ref.inverted() @ T_world_obj_ref
            for image_id in segment_ids(seg):
                if image_id in filled:
                    continue
                if image_id not in cam_poses:
                    continue
                cam_pos, cam_quat = cam_poses[image_id]
                T_world_cam = pose_matrix(cam_pos, cam_quat)
                filled[image_id] = T_world_cam @ T_cam_obj_const
            continue

        print(f"Warning: Unknown segment type '{seg_type}'.")

    return filled


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


def image_size_tuple(image):
    size = tuple(int(v) for v in image.size)
    if size[0] == 0 or size[1] == 0:
        try:
            image.reload()
        except Exception:
            pass
        size = tuple(int(v) for v in image.size)
    return size


def normalize_camera_name(name):
    if not name or not isinstance(name, str):
        return None
    cand_lower = name.lower()
    if cand_lower in CAMERA_CSV_BY_NAME:
        return cand_lower
    if "left" in cand_lower and "wrist" in cand_lower:
        return "left_wrist"
    if "right" in cand_lower and "wrist" in cand_lower:
        return "right_wrist"
    if "head" in cand_lower:
        return "head"
    return None


def resolve_render_camera_name(scene):
    for module_name in ("replay_4d", "__main__"):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        hint = getattr(module, "RENDER_CAMERA_NAME", None)
        normalized = normalize_camera_name(hint)
        if normalized:
            return normalized
    if scene is None:
        return None
    scene_hint = scene.get("render_camera_name") if hasattr(scene, "get") else None
    normalized = normalize_camera_name(scene_hint)
    if normalized:
        return normalized
    cam = scene.camera
    if cam is None:
        return None
    for prop_target in (cam, getattr(cam, "data", None)):
        if prop_target is None or not hasattr(prop_target, "get"):
            continue
        hint = prop_target.get("camera_name")
        normalized = normalize_camera_name(hint)
        if normalized:
            return normalized
    name_candidates = [cam.name]
    if getattr(cam, "data", None) is not None:
        name_candidates.append(cam.data.name)
    for cand in name_candidates:
        normalized = normalize_camera_name(cand)
        if normalized:
            return normalized
    return None


def overlay_pixels_list(
    render_pixels,
    render_channels,
    rgb_pixels,
    rgb_channels,
    mask_pixels,
    mask_channels,
    pixel_count,
):
    for i in range(pixel_count):
        mask_val = mask_pixels[i * mask_channels]
        if mask_val <= 0.5:
            continue
        render_base = i * render_channels
        rgb_base = i * rgb_channels
        r = rgb_pixels[rgb_base]
        g = rgb_pixels[rgb_base + 1] if rgb_channels > 1 else r
        b = rgb_pixels[rgb_base + 2] if rgb_channels > 2 else r
        render_pixels[render_base] = r
        if render_channels > 1:
            render_pixels[render_base + 1] = g
        if render_channels > 2:
            render_pixels[render_base + 2] = b
        if render_channels > 3:
            render_pixels[render_base + 3] = 1.0
    return render_pixels


def is_movie_format(file_format):
    return file_format in {"FFMPEG", "AVI_JPEG", "AVI_RAW", "FRAMESERVER"}


def remove_handlers_by_name(handler_list, name):
    for handler in list(handler_list):
        if getattr(handler, "__name__", "") == name:
            handler_list.remove(handler)


def configure_contact_overlay(entries):
    global _OVERLAY_STATE
    handler_map = {
        "render_pre": "_contact_overlay_render_pre",
        "frame_change_pre": "_contact_overlay_frame_change",
        "render_write": "_contact_overlay_render_write",
        "render_complete": "_contact_overlay_render_complete",
        "render_cancel": "_contact_overlay_render_cancel",
    }
    for handler_name, fn_name in handler_map.items():
        handler_list = getattr(bpy.app.handlers, handler_name, None)
        if handler_list is not None:
            remove_handlers_by_name(handler_list, fn_name)
    if not entries:
        _OVERLAY_STATE = None
        return
    overlay_objects = []
    for entry in entries:
        overlay_objects.append(
            {
                "object_name": entry["object_name"],
                "obj": entry["obj"],
                "camera_name": entry["camera_name"],
                "contact_ids": entry["contact_ids"],
                "demo_index": entry["demo_index"],
            }
        )
    _OVERLAY_STATE = {
        "objects": overlay_objects,
        "objects_by_name": {o["object_name"]: o for o in overlay_objects},
        "overlay_frames": {},
        "is_rendering": False,
        "start_frame": START_FRAME,
        "logged_first_overlay": False,
        "overlay_done": set(),
    }
    bpy.app.handlers.render_pre.append(_contact_overlay_render_pre)
    bpy.app.handlers.frame_change_pre.append(_contact_overlay_frame_change)
    bpy.app.handlers.render_write.append(_contact_overlay_render_write)
    bpy.app.handlers.render_complete.append(_contact_overlay_render_complete)
    bpy.app.handlers.render_cancel.append(_contact_overlay_render_cancel)


def _restore_overlay_visibility(state):
    for entry in state.get("objects", []):
        entry["obj"].hide_render = False


def _contact_overlay_render_pre(scene):
    state = _OVERLAY_STATE
    if not state:
        return
    if not state.get("is_rendering"):
        state["is_rendering"] = True
        state["overlay_frames"] = {}
        state["start_frame"] = scene.frame_start
        state["logged_first_overlay"] = False
        state["overlay_done"] = set()
    if resolve_render_camera_name(scene) is None:
        print(
            "Warning: could not infer render camera name; "
            "set scene['render_camera_name'] or camera['camera_name'] "
            "to enable contact-rich overlays."
        )


def _contact_overlay_frame_change(scene):
    state = _OVERLAY_STATE
    if not state or not state.get("is_rendering"):
        return
    start_frame = state.get("start_frame", START_FRAME)
    image_id = scene.frame_current - start_frame + 1
    if image_id < 1:
        return
    render_cam_name = resolve_render_camera_name(scene)
    for entry in state.get("objects", []):
        contact_ids = entry.get("contact_ids", set())
        demo_index = entry.get("demo_index") or {}
        has_demo = (
            image_id in demo_index.get("rgb", {})
            and image_id in demo_index.get("mask", {})
        )
        should_overlay = (
            render_cam_name == entry["camera_name"]
            and image_id in contact_ids
            and has_demo
        )
        entry["obj"].hide_render = should_overlay
        if should_overlay:
            frame_map = state["overlay_frames"].setdefault(image_id, set())
            frame_map.add(entry["object_name"])


def _apply_overlay_for_image_id(scene, state, image_id, object_names):
    file_format = scene.render.image_settings.file_format
    if is_movie_format(file_format):
        return False
    frame = state["start_frame"] + image_id - 1
    render_path = bpy.path.abspath(scene.render.frame_path(frame=frame))
    if not os.path.exists(render_path):
        print(f"Warning: missing render output for image_id {image_id}: {render_path}")
        return False
    render_img = bpy.data.images.load(render_path, check_existing=False)
    render_size = image_size_tuple(render_img)
    width, height = render_size
    render_channels = render_img.channels
    render_pixels = list(render_img.pixels)
    use_numpy = np is not None
    if use_numpy:
        render_arr = np.array(render_pixels, dtype=np.float32).reshape(
            (height, width, render_channels)
        )
    applied = False
    for object_name in sorted(object_names):
        entry = state["objects_by_name"].get(object_name)
        if entry is None:
            continue
        demo_index = entry.get("demo_index") or {}
        rgb_path = demo_index.get("rgb", {}).get(image_id)
        mask_path = demo_index.get("mask", {}).get(image_id)
        if not rgb_path or not mask_path:
            print(
                f"Warning: missing demo rgb/mask for {object_name} image_id {image_id}."
            )
            continue
        if not os.path.exists(rgb_path) or not os.path.exists(mask_path):
            print(
                f"Warning: demo files not found for {object_name} image_id {image_id}."
            )
            continue
        if not state.get("logged_first_overlay"):
            state["logged_first_overlay"] = True
            print(
                "Contact overlay: applying source_demo for "
                f"object={object_name} image_id={image_id} frame={frame}\n"
                f"  rgb={rgb_path}\n"
                f"  mask={mask_path}"
            )
        rgb_img = bpy.data.images.load(rgb_path, check_existing=False)
        mask_img = bpy.data.images.load(mask_path, check_existing=False)
        rgb_size = image_size_tuple(rgb_img)
        mask_size = image_size_tuple(mask_img)
        if rgb_size != render_size or mask_size != render_size:
            try:
                rgb_img.scale(width, height)
                mask_img.scale(width, height)
                rgb_size = image_size_tuple(rgb_img)
                mask_size = image_size_tuple(mask_img)
            except Exception:
                pass
        if rgb_size != render_size or mask_size != render_size:
            print(
                f"Warning: size mismatch for {object_name} image_id {image_id} "
                f"(render {render_size}, rgb {rgb_size}, mask {mask_size})."
            )
            bpy.data.images.remove(rgb_img)
            bpy.data.images.remove(mask_img)
            continue
        rgb_pixels = list(rgb_img.pixels)
        mask_pixels = list(mask_img.pixels)
        if use_numpy:
            rgb_arr = np.array(rgb_pixels, dtype=np.float32).reshape(
                (height, width, rgb_img.channels)
            )
            mask_arr = np.array(mask_pixels, dtype=np.float32).reshape(
                (height, width, mask_img.channels)
            )
            mask = mask_arr[:, :, 0] > 0.5
            if rgb_img.channels == 1:
                rgb_stack = np.repeat(rgb_arr, 3, axis=2)
            elif rgb_img.channels == 2:
                rgb_stack = np.concatenate([rgb_arr, rgb_arr[:, :, 1:2]], axis=2)
            else:
                rgb_stack = rgb_arr[:, :, :3]
            render_arr[mask, 0] = rgb_stack[mask, 0]
            if render_channels > 1:
                render_arr[mask, 1] = rgb_stack[mask, 1]
            if render_channels > 2:
                render_arr[mask, 2] = rgb_stack[mask, 2]
            if render_channels > 3:
                render_arr[mask, 3] = 1.0
        else:
            render_pixels = overlay_pixels_list(
                render_pixels,
                render_channels,
                rgb_pixels,
                rgb_img.channels,
                mask_pixels,
                mask_img.channels,
                width * height,
            )
        applied = True
        bpy.data.images.remove(rgb_img)
        bpy.data.images.remove(mask_img)
    if use_numpy:
        render_pixels = render_arr.ravel().tolist()
    if applied:
        render_img.pixels[:] = render_pixels
        render_img.filepath_raw = render_path
        render_img.file_format = file_format
        try:
            render_img.save()
        except Exception as exc:
            print(f"Warning: failed to save overlay frame {frame}: {exc}")
    bpy.data.images.remove(render_img)
    return applied


def _contact_overlay_render_write(scene, _depsgraph):
    state = _OVERLAY_STATE
    if not state or not state.get("is_rendering"):
        return
    start_frame = state.get("start_frame", START_FRAME)
    image_id = scene.frame_current - start_frame + 1
    if image_id < 1:
        return
    object_names = state.get("overlay_frames", {}).get(image_id)
    if not object_names:
        return
    if image_id in state.get("overlay_done", set()):
        return
    if _apply_overlay_for_image_id(scene, state, image_id, object_names):
        state["overlay_done"].add(image_id)


def _contact_overlay_render_complete(scene):
    state = _OVERLAY_STATE
    if not state:
        return
    state["is_rendering"] = False
    _restore_overlay_visibility(state)
    overlay_frames = state.get("overlay_frames") or {}
    if not overlay_frames:
        return
    file_format = scene.render.image_settings.file_format
    if is_movie_format(file_format):
        print("Warning: render output is a movie format; skipping contact-rich overlays.")
        return
    print(
        "Contact overlay: applying source_demo overlays for "
        f"{len(overlay_frames)} frame(s)."
    )
    for image_id in sorted(overlay_frames.keys()):
        if image_id in state.get("overlay_done", set()):
            continue
        _apply_overlay_for_image_id(scene, state, image_id, overlay_frames[image_id])


def _contact_overlay_render_cancel(scene):
    state = _OVERLAY_STATE
    if not state:
        return
    state["is_rendering"] = False
    _restore_overlay_visibility(state)


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = resolve_task_dir(args.task_name, base_dir)

    parquet_path = find_parquet(task_dir)
    state = load_parquet_state(parquet_path)
    left_width = [(row[0] + row[1]) * 0.5 for row in (row[GRIPPER_LEFT_SLICE] for row in state)]
    right_width = [(row[0] + row[1]) * 0.5 for row in (row[GRIPPER_RIGHT_SLICE] for row in state)]
    left_close_segments = compute_close_segments(left_width, GRIPPER_VEL_EPS)
    right_close_segments = compute_close_segments(right_width, GRIPPER_VEL_EPS)
    close_start_by_side = {
        "left": build_close_start_by_id(left_close_segments),
        "right": build_close_start_by_id(right_close_segments),
    }

    object_assets = find_object_assets(task_dir)
    object_collection = ensure_collection(OBJECT_COLLECTION_NAME)

    camera_pose_cache = {}
    global_ids = set()

    objects_to_animate = []
    for obj_path in object_assets:
        obj_dir = os.path.dirname(obj_path)
        object_name = os.path.basename(obj_dir)
        segments, contact_segments = load_object_metadata(
            os.path.join(obj_dir, "object.json")
        )
        contact_ids = collect_contact_ids(contact_segments) if contact_segments else set()
        demo_index = None
        if contact_ids:
            source_demo_dir = os.path.join(obj_dir, SOURCE_DEMO_DIRNAME)
            if os.path.isdir(source_demo_dir):
                demo_index = build_demo_index(source_demo_dir)
                if demo_index is None:
                    print(
                        f"Warning: source_demo missing rgb/masks for {object_name}; "
                        "contact overlays disabled."
                    )
            else:
                print(
                    f"Warning: contact-rich segments found but no {SOURCE_DEMO_DIRNAME} "
                    f"dir for {object_name}; rendering object as-is."
                )
        for seg in segments:
            global_ids.update(segment_ids(seg))
        camera_name_path = os.path.join(obj_dir, "camera_name.txt")
        camera_name = load_camera_name(camera_name_path)

        cam_poses_by_name = {}
        cam_names_needed = {camera_name}
        if any(seg["type"] == "still_in_left_wrist" for seg in segments):
            cam_names_needed.add("left_wrist")
        if any(seg["type"] == "still_in_right_wrist" for seg in segments):
            cam_names_needed.add("right_wrist")
        for cam_name in cam_names_needed:
            cam_csv = os.path.join(
                task_dir, "camera_poses", CAMERA_CSV_BY_NAME[cam_name]
            )
            if cam_csv not in camera_pose_cache:
                if not os.path.exists(cam_csv):
                    raise FileNotFoundError(cam_csv)
                camera_pose_cache[cam_csv] = load_camera_poses(cam_csv)
            cam_poses_by_name[cam_name] = camera_pose_cache[cam_csv]
        cam_poses_tracking = cam_poses_by_name[camera_name]
        if not cam_poses_tracking:
            raise RuntimeError(f"No camera poses for {camera_name}")

        pose_files = load_ob_in_cam_files(os.path.join(obj_dir, "ob_in_cam"))
        obj = get_or_import_object(obj_path, object_name, object_collection)
        objects_to_animate.append(
            {
                "obj": obj,
                "object_name": object_name,
                "pose_files": pose_files,
                "cam_poses_tracking": cam_poses_tracking,
                "cam_poses_by_name": cam_poses_by_name,
                "camera_name": camera_name,
                "segments": segments,
                "contact_ids": contact_ids,
                "demo_index": demo_index,
            }
        )

    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = START_FRAME
    sorted_global_ids = sorted(global_ids)
    if not sorted_global_ids:
        scene.frame_end = START_FRAME
    else:
        scene.frame_end = START_FRAME + sorted_global_ids[-1] - 1

    for entry in objects_to_animate:
        tracking_ids = collect_ids_for_type(entry["segments"], "tracking")
        obj_pose_by_id = build_tracking_pose_map(
            entry["pose_files"], entry["cam_poses_tracking"], tracking_ids
        )
        if not obj_pose_by_id:
            print(f"Warning: No matching ob_in_cam poses for object {entry['object_name']}")
        entry["filled_poses"] = fill_segment_poses(
            entry["segments"],
            obj_pose_by_id,
            entry["cam_poses_by_name"],
            entry["camera_name"],
            close_start_by_side,
        )

    overlay_entries = [
        entry
        for entry in objects_to_animate
        if entry.get("contact_ids") and entry.get("demo_index") is not None
    ]
    configure_contact_overlay(overlay_entries)

    total_keyframes = 0
    for image_id in sorted_global_ids:
        frame = START_FRAME + image_id - 1
        scene.frame_set(frame)
        for entry in objects_to_animate:
            poses = entry.get("filled_poses", {})
            if image_id not in poses:
                continue
            loc, rot, _scale = poses[image_id].decompose()
            obj = entry["obj"]
            obj.location = loc
            obj.rotation_euler = rot.to_euler("XYZ")
            obj.keyframe_insert(data_path="location")
            obj.keyframe_insert(data_path="rotation_euler")
            total_keyframes += 1

    print(
        f"Replayed object poses for {len(sorted_global_ids)} frames "
        f"({total_keyframes} total keyframes)."
    )


if __name__ == "__main__":
    main()
