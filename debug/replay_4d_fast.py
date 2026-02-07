"""
Replay robot + objects together and capture viewport frames quickly.

Usage:
  blender --factory-startup --python debug/replay_4d_fast.py -- \
    --task_name picking_up_trash --camera_name head
"""

import argparse
import importlib.util
import os
import re
import sys

import bpy
from mathutils import Matrix, Quaternion


DEFAULT_TASK_NAME = "picking_up_trash"
DEFAULT_CAMERA_NAME = "head"
CAMERA_CHOICES = ("left_wrist", "right_wrist", "head")
WORLD_BG_COLOR = (1.0, 1.0, 1.0, 1.0)  # White
STATE = {}


def _parse_argv():
    argv = sys.argv
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return []


def parse_args():
    parser = argparse.ArgumentParser(description="Replay robot + object with viewport capture.")
    parser.add_argument(
        "--task_name",
        default=DEFAULT_TASK_NAME,
        help="Task folder name under project root, or absolute task directory path.",
    )
    parser.add_argument(
        "--camera_name",
        default=DEFAULT_CAMERA_NAME,
        choices=CAMERA_CHOICES,
        help="Camera name used for render trajectory and intrinsics.",
    )
    parser.add_argument(
        "--parquet_path",
        default=None,
        help="Optional explicit parquet path. If omitted, auto-picks latest episode parquet.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory for viewport frames.",
    )
    parser.add_argument(
        "--no_viewport_capture",
        action="store_true",
        help="Register handlers only (interactive playback); do not run viewport capture.",
    )
    argv = _parse_argv()
    return parser.parse_args(argv)


def resolve_task_dir(task_name, project_root):
    if os.path.isabs(task_name):
        if os.path.isdir(task_name):
            return task_name
        raise FileNotFoundError(f"Task directory not found: {task_name}")

    candidate = os.path.join(project_root, task_name)
    if os.path.isdir(candidate):
        return candidate

    raise FileNotFoundError(f"Task directory not found: {candidate}")


def _episode_index_from_name(name):
    match = re.match(r"^episode_(\d+)\.parquet$", name)
    if not match:
        return None
    return int(match.group(1))


def find_latest_task_parquet(task_dir):
    parquet_paths = []
    for root, _dirs, files in os.walk(task_dir):
        for name in files:
            if name.endswith(".parquet"):
                parquet_paths.append(os.path.join(root, name))

    if not parquet_paths:
        raise FileNotFoundError(f"No .parquet files found under task directory: {task_dir}")

    episode_candidates = []
    for path in parquet_paths:
        idx = _episode_index_from_name(os.path.basename(path))
        if idx is not None:
            episode_candidates.append((idx, path))

    if episode_candidates:
        episode_candidates.sort(key=lambda x: x[0])
        return episode_candidates[-1][1]

    parquet_paths.sort()
    return parquet_paths[-1]


def resolve_parquet_path(parquet_arg, task_dir, project_root):
    if parquet_arg is None:
        return find_latest_task_parquet(task_dir)

    if os.path.isabs(parquet_arg):
        candidate_paths = [parquet_arg]
    else:
        candidate_paths = [
            os.path.join(task_dir, parquet_arg),
            os.path.join(project_root, parquet_arg),
            os.path.abspath(parquet_arg),
        ]

    for path in candidate_paths:
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        "Parquet file not found. Tried: " + ", ".join(candidate_paths)
    )


def load_local_module(module_name, module_path):
    if not os.path.isfile(module_path):
        raise FileNotFoundError(f"Module file not found: {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def pose_matrix(position, quat_wxyz):
    q = Quaternion(quat_wxyz).normalized()
    return Matrix.Translation(position) @ q.to_matrix().to_4x4()


def ensure_render_camera(camera_name, intrinsics):
    object_name = f"ReplayRenderCamera_{camera_name}"
    cam_obj = bpy.data.objects.get(object_name)

    if cam_obj is None:
        cam_data = bpy.data.cameras.new(name=f"{object_name}_data")
        cam_obj = bpy.data.objects.new(object_name, cam_data)
        bpy.context.scene.collection.objects.link(cam_obj)
    else:
        cam_data = cam_obj.data

    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    fx = float(intrinsics["fx"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])

    cam_data.type = "PERSP"
    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 32.0
    cam_data.lens = fx * cam_data.sensor_width / float(width)
    cam_data.shift_x = (cx - 0.5 * width) / float(width)
    cam_data.shift_y = (cy - 0.5 * height) / float(height)
    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0

    cam_obj.rotation_mode = "XYZ"
    cam_obj["camera_name"] = camera_name

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


def _apply_camera_pose(cam_obj, pose):
    pos, quat = pose
    mat = pose_matrix(pos, quat)
    loc, rot, _scale = mat.decompose()
    cam_obj.location = loc
    cam_obj.rotation_mode = "XYZ"
    cam_obj.rotation_euler = rot.to_euler("XYZ")


def replay_4d_fast_frame_handler(scene):
    state = STATE
    if not state.get("ready"):
        return

    image_id = scene.frame_current - state["start_frame"] + 1
    if image_id < 1 or image_id > state["num_frames"]:
        return

    pose = state["cam_poses"].get(image_id)
    if pose is None:
        return

    cam_obj = state["render_cam"]
    try:
        _apply_camera_pose(cam_obj, pose)
    except ReferenceError:
        return


def register_replay_handler():
    handlers = bpy.app.handlers.frame_change_pre
    for handler in list(handlers):
        if getattr(handler, "__name__", "") == "replay_4d_fast_frame_handler":
            handlers.remove(handler)
    handlers.append(replay_4d_fast_frame_handler)


def _find_view3d_context():
    wm = bpy.context.window_manager
    if wm is None:
        return None

    for window in wm.windows:
        screen = window.screen
        if screen is None:
            continue
        for area in screen.areas:
            if area.type != "VIEW_3D":
                continue
            region = next((r for r in area.regions if r.type == "WINDOW"), None)
            space = next((s for s in area.spaces if s.type == "VIEW_3D"), None)
            if region is not None and space is not None:
                return window, screen, area, region, space
    return None


def viewport_capture_animation(output_dir):
    if bpy.app.background:
        raise RuntimeError("Viewport capture requires UI mode (not background mode).")

    context_info = _find_view3d_context()
    if context_info is None:
        raise RuntimeError("No VIEW_3D area found. Viewport capture requires an open 3D View.")

    window, screen, area, region, space = context_info
    scene = bpy.context.scene

    os.makedirs(output_dir, exist_ok=True)

    prev_filepath = scene.render.filepath
    prev_format = scene.render.image_settings.file_format
    prev_color_mode = scene.render.image_settings.color_mode
    prev_film_transparent = scene.render.film_transparent
    prev_overlay = getattr(space.overlay, "show_overlays", None)
    prev_gizmo = getattr(space, "show_gizmo", None)
    prev_shading = space.shading.type
    prev_perspective = None
    if space.region_3d is not None:
        prev_perspective = space.region_3d.view_perspective

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.render.filepath = os.path.join(output_dir, "")
    render_paths_by_frame = {}
    for frame in range(scene.frame_start, scene.frame_end + 1):
        render_paths_by_frame[frame] = bpy.path.abspath(scene.render.frame_path(frame=frame))

    try:
        if prev_overlay is not None:
            space.overlay.show_overlays = False
        if prev_gizmo is not None:
            space.show_gizmo = False
        space.shading.type = "RENDERED"
        if space.region_3d is not None:
            space.region_3d.view_perspective = "CAMERA"

        with bpy.context.temp_override(
            window=window,
            screen=screen,
            area=area,
            region=region,
            space_data=space,
        ):
            bpy.ops.render.opengl(animation=True, view_context=True)
    finally:
        scene.render.filepath = prev_filepath
        scene.render.image_settings.file_format = prev_format
        scene.render.image_settings.color_mode = prev_color_mode
        scene.render.film_transparent = prev_film_transparent
        if prev_overlay is not None:
            space.overlay.show_overlays = prev_overlay
        if prev_gizmo is not None:
            space.show_gizmo = prev_gizmo
        space.shading.type = prev_shading
        if space.region_3d is not None and prev_perspective is not None:
            space.region_3d.view_perspective = prev_perspective

    return {"render_paths_by_frame": render_paths_by_frame}


def main():
    args = parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    task_dir = resolve_task_dir(args.task_name, project_root)
    parquet_path = resolve_parquet_path(args.parquet_path, task_dir, project_root)
    output_dir = args.output_dir or os.path.join(task_dir, f"{args.camera_name}_viewport")

    robot_module = load_local_module(
        "replay_r1pro_trajectory_debug_runtime",
        os.path.join(script_dir, "replay_r1pro_trajectory_debug.py"),
    )
    robot_module.PARQUET_PATH = parquet_path
    robot_module.REALTIME_PREVIEW = True
    robot_module.CLEAR_SCENE_BEFORE_IMPORT = True

    robot_bundle = robot_module.main()
    if not isinstance(robot_bundle, dict):
        raise RuntimeError("Robot replay did not return a replay bundle dict.")

    required_keys = ("cam_pose_dict", "camera_intrinsics", "num_frames", "start_frame")
    missing_keys = [k for k in required_keys if k not in robot_bundle]
    if missing_keys:
        raise RuntimeError(f"Robot replay bundle missing keys: {missing_keys}")

    cam_pose_dict = robot_bundle["cam_pose_dict"]
    camera_intrinsics = robot_bundle["camera_intrinsics"]
    start_frame = int(robot_bundle["start_frame"])
    num_frames = int(robot_bundle["num_frames"])

    if args.camera_name not in cam_pose_dict:
        available = sorted(cam_pose_dict.keys())
        raise RuntimeError(
            f"Missing URDF camera poses for '{args.camera_name}'. Available: {available}"
        )
    if args.camera_name not in camera_intrinsics:
        available = sorted(camera_intrinsics.keys())
        raise RuntimeError(
            f"Missing camera intrinsics for '{args.camera_name}'. Available: {available}"
        )

    scene = bpy.context.scene
    scene.frame_start = start_frame
    scene.frame_end = start_frame + num_frames - 1

    render_cam = ensure_render_camera(args.camera_name, camera_intrinsics[args.camera_name])
    scene.camera = render_cam
    scene["render_camera_name"] = args.camera_name
    scene.render.resolution_x = int(camera_intrinsics[args.camera_name]["width"])
    scene.render.resolution_y = int(camera_intrinsics[args.camera_name]["height"])
    scene.render.resolution_percentage = 100
    set_world_background_color(scene, WORLD_BG_COLOR)

    # Load and register object replay with the same timing range.
    object_module = load_local_module(
        "replay_object_runtime", os.path.join(script_dir, "replay_object.py")
    )
    object_state = object_module.setup_object_replay(
        task_dir=task_dir,
        cam_pose_dict=cam_pose_dict,
        start_frame=start_frame,
        num_frames=num_frames,
        missing_pose_policy="hold_last",
    )
    object_module.register_replay_handler(object_state)

    # Register camera replay handler.
    STATE.clear()
    STATE.update(
        {
            "ready": True,
            "start_frame": start_frame,
            "num_frames": num_frames,
            "cam_poses": cam_pose_dict[args.camera_name],
            "render_cam": render_cam,
        }
    )
    register_replay_handler()
    scene.frame_set(start_frame)

    if args.no_viewport_capture:
        print("Replay handlers registered. Press Play to preview robot + object replay.")
        return {
            "task_dir": task_dir,
            "parquet_path": parquet_path,
            "output_dir": output_dir,
        }

    print(
        f"Capturing viewport animation: task={os.path.basename(task_dir)} "
        f"camera={args.camera_name} parquet={os.path.basename(parquet_path)}"
    )
    if hasattr(object_module, "reset_contact_overlay_state"):
        object_module.reset_contact_overlay_state(object_state, start_frame)
    capture_info = viewport_capture_animation(output_dir)
    if hasattr(object_module, "apply_contact_overlays_after_render"):
        object_module.apply_contact_overlays_after_render(
            scene=scene,
            state_dict=object_state,
            render_paths_by_frame=capture_info.get("render_paths_by_frame"),
        )
    print(f"Viewport capture complete: {output_dir}")

    return {
        "task_dir": task_dir,
        "parquet_path": parquet_path,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    main()
