"""
Replay R1Pro trajectory and object motion together.

Usage:
  blender --python replay_4d.py
"""

import importlib
import importlib.util
import os
import struct
import sys
import csv
import multiprocessing
import argparse

import bpy
from mathutils import Matrix, Quaternion, Euler
try:
    import cv2
    import numpy as np
except Exception:
    cv2 = None
    np = None


# ----------------------
# User configuration
# ----------------------
RUN_IMPORT_SCENE = False
RUN_IMPORT_URDF = True

DEFAULT_TASK_NAME = "picking_up_trash"
CAMERA_NAME = "head"  # "left_wrist", "right_wrist", or "head"
SAVE_CAMERA_RENDER = True
USE_OPENCV_CAMERA_POSE = True
CAMERA_CLIP_START = 0.05
CAMERA_CLIP_END = 1000.0
RENDER_CAMERA_ROT_OFFSET = (3.141592653589793, 0.0, 0.0)
REALTIME_PREVIEW = True  # Use frame-change handler for real-time viewport playback (no keyframes).
VIEWPORT_CAPTURE = True  # Capture viewport frames using the active camera (requires UI, no headless).
VIEWPORT_FORMAT = "PNG"  # One of the image formats supported by this build.
VIEWPORT_HIDE_OVERLAYS = True  # Hide axis/grid/origin overlays in viewport capture.
VIEWPORT_TRANSPARENT = True  # Write alpha so the foreground mask is in the PNG alpha channel.
VIEWPORT_COLOR_MODE = "RGBA"  # Must support alpha (e.g., PNG, OPEN_EXR).
OVERLAY_PARALLEL = True  # Apply overlays in parallel (requires OpenCV).
OVERLAY_WORKERS = 0  # 0 = auto; otherwise number of worker processes.
WORLD_BG_COLOR = (0.8666667, 0.7568628, 0.6196079, 1.0)  # Hex #DDC19EFF.

STATE = {}


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


def pose_matrix(position, quat):
    q = Quaternion(quat).normalized()
    return Matrix.Translation(position) @ q.to_matrix().to_4x4()


def load_cameras_bin(path):
    # COLMAP cameras.bin: num_cameras (uint64), then per camera:
    # int32 camera_id, int32 model_id, uint64 width, uint64 height, params (double[])
    with open(path, "rb") as f:
        num = struct.unpack("<Q", f.read(8))[0]
        if num < 1:
            raise ValueError("cameras.bin has no cameras.")
        camera_id = struct.unpack("<i", f.read(4))[0]
        model_id = struct.unpack("<i", f.read(4))[0]
        width = struct.unpack("<Q", f.read(8))[0]
        height = struct.unpack("<Q", f.read(8))[0]
        remaining = f.read()
        params = struct.unpack("<" + "d" * (len(remaining) // 8), remaining)
    if len(params) < 4:
        raise ValueError("Unexpected cameras.bin params length.")
    fx, fy, cx, cy = params[:4]
    return {
        "camera_id": camera_id,
        "model_id": model_id,
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }


def ensure_render_camera(name, fx, fy, cx, cy, width, height):
    cam_obj = bpy.data.objects.get(name)
    if cam_obj is None:
        cam_data = bpy.data.cameras.new(name)
        cam_obj = bpy.data.objects.new(name, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.parent = None
    cam_obj.matrix_parent_inverse = Matrix.Identity(4)
    cam_obj.location = (0.0, 0.0, 0.0)
    cam_obj.rotation_mode = "XYZ"
    cam_obj.rotation_euler = (0.0, 0.0, 0.0)
    cam_obj.scale = (1.0, 1.0, 1.0)

    cam = cam_obj.data
    cam.type = "PERSP"
    cam.sensor_fit = "HORIZONTAL"
    cam.sensor_width = 32.0
    cam.lens = fx * cam.sensor_width / width
    cam.shift_x = (cx - width * 0.5) / width
    cam.shift_y = (cy - height * 0.5) / height
    cam.clip_start = CAMERA_CLIP_START
    cam.clip_end = CAMERA_CLIP_END
    return cam_obj


def set_world_background_color(scene, rgba):
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        scene.world = world
    try:
        world.use_nodes = True
        bg_node = None
        for node in world.node_tree.nodes:
            if node.type == "BACKGROUND":
                bg_node = node
                break
        if bg_node is None:
            bg_node = world.node_tree.nodes.new(type="ShaderNodeBackground")
        bg_node.inputs[0].default_value = rgba
    except Exception:
        world.color = rgba[:3]


PROJECT_ROOT = os.path.abspath(os.getcwd())


def camera_name_to_render_name(name):
    parts = name.split("_")
    return "".join(p.capitalize() for p in parts) + "RenderCamera"


def build_camera_paths(name, task_name):
    base = os.path.join(PROJECT_ROOT, task_name, "camera_poses")
    csv_path = os.path.join(base, f"{name}_camera_pose.csv")
    cameras_bin_path = os.path.join(base, name, "cameras.bin")
    return csv_path, cameras_bin_path


def build_render_paths(task_name, camera_name):
    camera_csv_path, cameras_bin_path = build_camera_paths(camera_name, task_name)
    render_camera_name = camera_name_to_render_name(camera_name)
    render_output_path = os.path.join(PROJECT_ROOT, task_name, f"{camera_name}_renders")
    viewport_output_path = os.path.join(PROJECT_ROOT, task_name, f"{camera_name}_viewport")
    return camera_csv_path, cameras_bin_path, render_camera_name, render_output_path, viewport_output_path


def find_task_parquet(task_dir):
    candidates = []
    for entry in os.listdir(task_dir):
        if entry.endswith(".parquet"):
            candidates.append(os.path.join(task_dir, entry))
    candidates.sort()
    if not candidates:
        raise FileNotFoundError(f"No .parquet files found under: {task_dir}")
    return candidates[0]


def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    parser = argparse.ArgumentParser(description="Replay 4D fast viewport render.")
    parser.add_argument("--task_name", default=DEFAULT_TASK_NAME, help="Task folder name.")
    parser.add_argument(
        "--camera_name",
        default=CAMERA_NAME,
        choices=("left_wrist", "right_wrist", "head"),
        help="Camera name to render.",
    )
    return parser.parse_args(argv)


def replay_4d_fast_frame_handler(scene):
    try:
        state = STATE
        if not state.get("ready"):
            return

        frame = scene.frame_current
        image_id = (frame - state["start_frame"]) + 1
        cam_poses = state["cam_poses"]
        if image_id not in cam_poses:
            return

        render_cam = state.get("render_cam")
        if render_cam is not None:
            try:
                _ = render_cam.name
            except ReferenceError:
                render_cam = None
        if render_cam is None:
            render_cam_name = state.get("render_cam_name")
            if render_cam_name:
                render_cam = bpy.data.objects.get(render_cam_name)
                if render_cam is None:
                    return
                state["render_cam"] = render_cam
            else:
                return
        pos, quat = cam_poses[image_id]
        mat = pose_matrix(pos, quat)
        if USE_OPENCV_CAMERA_POSE:
            mat = mat @ Euler(RENDER_CAMERA_ROT_OFFSET, "XYZ").to_matrix().to_4x4()
        loc, rot, _scale = mat.decompose()
        render_cam.location = loc
        render_cam.rotation_euler = rot.to_euler("XYZ")
    except ReferenceError:
        return


def register_replay_handler():
    handlers = bpy.app.handlers.frame_change_pre
    for h in list(handlers):
        if getattr(h, "__name__", "") == "replay_4d_fast_frame_handler":
            handlers.remove(h)
    handlers.append(replay_4d_fast_frame_handler)


def _build_overlay_plan(overlay_module, scene):
    if overlay_module is None:
        return None
    state = getattr(overlay_module, "_OVERLAY_STATE", None)
    if not state:
        return None
    pre_fn = getattr(overlay_module, "_contact_overlay_render_pre", None)
    frame_fn = getattr(overlay_module, "_contact_overlay_frame_change", None)
    restore_fn = getattr(overlay_module, "_restore_overlay_visibility", None)
    if pre_fn:
        pre_fn(scene)
    if frame_fn:
        start = scene.frame_start
        end = scene.frame_end
        for frame in range(start, end + 1):
            scene.frame_set(frame)
            frame_fn(scene)
    overlay_frames = {
        image_id: set(object_names)
        for image_id, object_names in state.get("overlay_frames", {}).items()
    }
    overlay_paths = {}
    for image_id in overlay_frames:
        frame = scene.frame_start + image_id - 1
        overlay_paths[image_id] = bpy.path.abspath(scene.render.frame_path(frame=frame))
    if restore_fn and state:
        restore_fn(state)
    state["is_rendering"] = False
    overlay_object_names = [entry.get("object_name") for entry in state.get("objects", [])]
    return {
        "state": state,
        "frames": overlay_frames,
        "paths": overlay_paths,
        "object_names": [name for name in overlay_object_names if name],
    }


def _overlay_visibility_frame_handler(scene):
    overlay_plan = STATE.get("overlay_plan")
    if not overlay_plan:
        return
    start_frame = overlay_plan.get("start_frame", scene.frame_start)
    image_id = scene.frame_current - start_frame + 1
    if image_id < 1:
        return
    object_names = overlay_plan.get("frames", {}).get(image_id, set())
    for object_name in overlay_plan.get("object_names", []):
        if not object_name:
            continue
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            continue
        hide = object_name in object_names
        try:
            obj.hide_render = hide
            if hasattr(obj, "hide_viewport"):
                obj.hide_viewport = hide
            else:
                obj.hide_set(hide)
        except ReferenceError:
            continue


def _register_overlay_visibility_handler():
    handlers = bpy.app.handlers.frame_change_pre
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_overlay_visibility_frame_handler":
            handlers.remove(h)
    handlers.append(_overlay_visibility_frame_handler)


def _unregister_overlay_visibility_handler():
    handlers = bpy.app.handlers.frame_change_pre
    for h in list(handlers):
        if getattr(h, "__name__", "") == "_overlay_visibility_frame_handler":
            handlers.remove(h)


def _apply_overlay_after_viewport(overlay_module, scene, overlay_plan):
    if overlay_module is None or not overlay_plan:
        return False
    if cv2 is None or np is None:
        print("Overlay: OpenCV is not available; skipping overlay.")
        return False
    restore_fn = getattr(overlay_module, "_restore_overlay_visibility", None)
    tasks = _collect_overlay_tasks(overlay_plan)
    total = len(tasks)
    if total == 0:
        return False
    print(f"Overlay (OpenCV): applying source_demo to {total} frame(s)...")
    if OVERLAY_PARALLEL:
        workers = OVERLAY_WORKERS or max(1, (os.cpu_count() or 2) - 1)
        try:
            ctx = multiprocessing.get_context("fork") if hasattr(multiprocessing, "get_context") else multiprocessing
            with ctx.Pool(processes=workers) as pool:
                for idx, result in enumerate(pool.imap_unordered(_overlay_frame_cv2, tasks), start=1):
                    image_id, ok, err = result
                    msg = f"Overlay {idx}/{total}: image_id={image_id}"
                    if not ok and err:
                        msg += f" error={err}"
                    print(msg)
        except Exception as exc:
            print(f"Overlay: parallel OpenCV failed, falling back to serial. {exc}")
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
    if restore_fn:
        restore_fn(overlay_plan.get("state"))
    for object_name in overlay_plan.get("object_names", []):
        if not object_name:
            continue
        obj = bpy.data.objects.get(object_name)
        if obj is None:
            continue
        try:
            obj.hide_render = False
            if hasattr(obj, "hide_viewport"):
                obj.hide_viewport = False
            else:
                obj.hide_set(False)
        except ReferenceError:
            continue
    return True


def _collect_overlay_tasks(overlay_plan):
    state = overlay_plan.get("state") or {}
    frames = overlay_plan.get("frames", {}) or {}
    paths = overlay_plan.get("paths", {}) or {}
    objects_by_name = state.get("objects_by_name")
    if objects_by_name is None:
        objects_by_name = {o["object_name"]: o for o in state.get("objects", [])}
    tasks = []
    for image_id, object_names in frames.items():
        render_path = paths.get(image_id)
        if not render_path:
            continue
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
                alpha = 255 * np.ones((rgb.shape[0], rgb.shape[1], 1), dtype=rgb.dtype)
                rgb = np.concatenate([rgb, alpha], axis=2)
            elif rgb.shape[2] != 4:
                rgb = rgb[:, :, :4]
        else:
            if rgb.ndim == 2:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            elif rgb.shape[2] == 4:
                rgb = rgb[:, :, :3]
        try:
            render[mask_bool] = rgb[mask_bool]
            if render_has_alpha:
                render[mask_bool, 3] = 255
        except Exception:
            return image_id, False, "blend_failed"
    ok = cv2.imwrite(render_path, render)
    return image_id, bool(ok), None


def viewport_render_animation(output_path, file_format, overlay_module=None):
    if bpy.app.background:
        print("Viewport render is not available in background mode.")
        return False

    scene = bpy.context.scene
    prev_film_transparent = scene.render.film_transparent
    prev_color_mode = scene.render.image_settings.color_mode
    scene.render.image_settings.file_format = file_format
    scene.render.image_settings.color_mode = VIEWPORT_COLOR_MODE
    scene.render.film_transparent = VIEWPORT_TRANSPARENT
    frames_dir = output_path
    if os.path.splitext(output_path)[1]:
        frames_dir = os.path.splitext(output_path)[0]
    os.makedirs(frames_dir, exist_ok=True)
    scene.render.filepath = os.path.join(frames_dir, "frame_")

    overlay_plan = None
    if overlay_module is not None:
        plan = _build_overlay_plan(overlay_module, scene)
        if plan:
            overlay_plan = {
                "state": plan["state"],
                "frames": plan["frames"],
                "paths": plan["paths"],
                "object_names": plan.get("object_names", []),
                "start_frame": scene.frame_start,
            }
            STATE["overlay_plan"] = overlay_plan
            _register_overlay_visibility_handler()
            scene.frame_set(scene.frame_start)
            _overlay_visibility_frame_handler(scene)

    for window in bpy.context.window_manager.windows:
        screen = window.screen
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            space = next((s for s in area.spaces if s.type == "VIEW_3D"), None)
            if region is None or space is None:
                continue
            overlay = space.overlay
            prev_overlay = {}
            if VIEWPORT_HIDE_OVERLAYS and overlay:
                prev_overlay["show_overlays"] = getattr(overlay, "show_overlays", None)
                prev_overlay["show_floor"] = getattr(overlay, "show_floor", None)
                prev_overlay["show_axis_x"] = getattr(overlay, "show_axis_x", None)
                prev_overlay["show_axis_y"] = getattr(overlay, "show_axis_y", None)
                prev_overlay["show_axis_z"] = getattr(overlay, "show_axis_z", None)
                prev_overlay["show_cursor"] = getattr(overlay, "show_cursor", None)
                prev_overlay["show_extras"] = getattr(overlay, "show_extras", None)
                if hasattr(overlay, "show_overlays"):
                    overlay.show_overlays = False
                if hasattr(overlay, "show_floor"):
                    overlay.show_floor = False
                if hasattr(overlay, "show_axis_x"):
                    overlay.show_axis_x = False
                if hasattr(overlay, "show_axis_y"):
                    overlay.show_axis_y = False
                if hasattr(overlay, "show_axis_z"):
                    overlay.show_axis_z = False
                if hasattr(overlay, "show_cursor"):
                    overlay.show_cursor = False
                if hasattr(overlay, "show_extras"):
                    overlay.show_extras = False
            prev_gizmo = getattr(space, "show_gizmo", None)
            if VIEWPORT_HIDE_OVERLAYS and hasattr(space, "show_gizmo"):
                space.show_gizmo = False
            space.region_3d.view_perspective = "CAMERA"
            space.shading.type = "RENDERED"
            with bpy.context.temp_override(window=window, screen=screen, area=area, region=region):
                bpy.ops.render.opengl(animation=True)
            _unregister_overlay_visibility_handler()
            _apply_overlay_after_viewport(overlay_module, scene, overlay_plan)
            if VIEWPORT_HIDE_OVERLAYS and overlay:
                for key, value in prev_overlay.items():
                    if value is None:
                        continue
                    if hasattr(overlay, key):
                        setattr(overlay, key, value)
            if VIEWPORT_HIDE_OVERLAYS and hasattr(space, "show_gizmo"):
                space.show_gizmo = prev_gizmo
            scene.render.film_transparent = prev_film_transparent
            scene.render.image_settings.color_mode = prev_color_mode
            return True

    print("No VIEW_3D area found for viewport render.")
    scene.render.film_transparent = prev_film_transparent
    scene.render.image_settings.color_mode = prev_color_mode
    _unregister_overlay_visibility_handler()
    return False


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd_dir = os.getcwd()
    for p in (script_dir, cwd_dir):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    args = parse_args()
    task_name = args.task_name
    camera_name = args.camera_name
    task_dir = os.path.join(PROJECT_ROOT, task_name)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")
    task_parquet = find_task_parquet(task_dir)
    head_camera_csv = os.path.join(task_dir, "camera_poses", "head_camera_pose.csv")
    global DEFAULT_TASK_NAME, CAMERA_NAME
    DEFAULT_TASK_NAME = task_name
    CAMERA_NAME = camera_name
    (
        camera_csv_path,
        cameras_bin_path,
        render_camera_name,
        render_output_path,
        viewport_output_path,
    ) = build_render_paths(task_name, camera_name)

    def load_local_module(name, filename):
        candidate_dirs = [
            script_dir,
            cwd_dir,
            os.path.join(cwd_dir, "blender_replay_render/4d_replay"),
        ]
        candidates = []
        for base in candidate_dirs:
            if base:
                candidates.append(os.path.join(base, filename))
        path = None
        for c in candidates:
            if os.path.exists(c):
                path = c
                break
        if path is None:
            searched = ", ".join(d for d in candidate_dirs if d)
            raise FileNotFoundError(
                f"Could not find {filename} in {searched}"
            )
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module {name} from {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    if RUN_IMPORT_SCENE:
        import_scene_objects = load_local_module("import_scene_objects", "import_scene_objects.py")
        import_scene_objects.main()

    if RUN_IMPORT_URDF:
        import_urdf = load_local_module("import_urdf", "import_urdf.py")
        import_urdf.import_urdf(import_urdf.URDF_PATH)

    # Replay robot trajectory
    replay_r1pro_trajectory = load_local_module("replay_r1pro_trajectory", "replay_r1pro_trajectory.py")
    replay_r1pro_trajectory.PARQUET_PATH = task_parquet
    replay_r1pro_trajectory.HEAD_CAMERA_CSV_PATH = head_camera_csv
    if REALTIME_PREVIEW:
        replay_r1pro_trajectory.REALTIME_PREVIEW = True
    replay_r1pro_trajectory.main()
    scene = bpy.context.scene
    start_a = scene.frame_start
    end_a = scene.frame_end

    # Replay object trajectory
    replay_object = load_local_module("replay_object", "replay_object.py")
    if REALTIME_PREVIEW:
        replay_object.REALTIME_PREVIEW = True
    replay_object.DEFAULT_TASK_NAME = task_name
    _argv_backup = sys.argv[:]
    try:
        sys.argv = [sys.argv[0]]
        replay_object.main()
    finally:
        sys.argv = _argv_backup
    start_b = scene.frame_start
    end_b = scene.frame_end

    scene.frame_start = min(start_a, start_b)
    scene.frame_end = max(end_a, end_b)

    if not os.path.exists(camera_csv_path):
        raise FileNotFoundError(camera_csv_path)
    if not os.path.exists(cameras_bin_path):
        raise FileNotFoundError(cameras_bin_path)
    cam_info = load_cameras_bin(cameras_bin_path)
    render_cam = ensure_render_camera(
            render_camera_name,
            cam_info["fx"],
            cam_info["fy"],
            cam_info["cx"],
            cam_info["cy"],
            cam_info["width"],
            cam_info["height"],
        )
    cam_poses = load_camera_poses(camera_csv_path)
    scene.camera = render_cam
    scene["render_camera_name"] = camera_name
    scene.render.resolution_x = cam_info["width"]
    scene.render.resolution_y = cam_info["height"]
    scene.render.resolution_percentage = 100
    set_world_background_color(scene, WORLD_BG_COLOR)

    if REALTIME_PREVIEW:
        STATE.clear()
        STATE.update(
            {
                "ready": True,
                "start_frame": scene.frame_start,
                "render_cam": render_cam,
                "render_cam_name": render_cam.name,
                "cam_poses": cam_poses,
            }
        )
        register_replay_handler()
        scene.frame_set(scene.frame_start)
        if VIEWPORT_CAPTURE:
            print("Replay 4D fast handler registered. Capturing viewport animation...")
            viewport_render_animation(viewport_output_path, VIEWPORT_FORMAT, replay_object)
        else:
            print("Replay 4D fast handler registered. Press Play for real-time viewport playback.")
        return

    for image_id in sorted(cam_poses.keys()):
        frame = scene.frame_start + image_id - 1
        scene.frame_set(frame)
        pos, quat = cam_poses[image_id]
        mat = pose_matrix(pos, quat)
        if USE_OPENCV_CAMERA_POSE:
            mat = mat @ Euler(RENDER_CAMERA_ROT_OFFSET, "XYZ").to_matrix().to_4x4()
        loc, rot, _scale = mat.decompose()
        render_cam.location = loc
        render_cam.rotation_euler = rot.to_euler("XYZ")
        render_cam.keyframe_insert(data_path="location")
        render_cam.keyframe_insert(data_path="rotation_euler")

    if SAVE_CAMERA_RENDER:
        scene.render.filepath = render_output_path
        bpy.ops.render.render(animation=True, use_viewport=False)

    print("Replay complete: robot + object.")


if __name__ == "__main__":
    main()
