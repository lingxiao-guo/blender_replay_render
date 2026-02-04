"""
Unified replay for R1Pro + object, with modular steps for debugging.

Usage:
  blender --python replay_4d_plus.py
"""

import os
import csv

import bpy
import numpy as np
from mathutils import Matrix, Vector, Euler, Quaternion

# ----------------------
# User configuration
# ----------------------
URDF_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/urdf/r1pro.urdf"
PARQUET_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/episode_00000110.parquet"
HEAD_CAMERA_CSV_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/turning_on_radio/xyz_traj_task-0000_observation.images.rgb.head.csv"
HEAD_CAMERA_BASE_LOCAL_CSV_PATH = (
    "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/head_cam_rel_poses_all.csv"
)
OB_IN_CAM_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/ob_in_cam"
CAM_K_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/turning_on_radio/cam_K.txt"

IMPORT_URDF = False
ROBOT_ROOT_NAME = "r1_pro_with_gripper"
HEAD_CAMERA_LINK_NAME = "zed_link"

OBJECT_NAME = ""  # leave empty to pick first mesh in ImportedObjects collection
OBJECT_COLLECTION_NAME = "ImportedObjects"

FPS = 60
START_FRAME = 1
FRAME_LIMIT = None
DATA_START_INDEX = 0  # 0-based index into data arrays (e.g., 1928 for step 1929)

RENDER_VIDEO = True
RENDER_OUTPUT_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/head_camera.mp4"
RENDER_ENGINE = "BLENDER_EEVEE"
RENDER_CAMERA_NAME = "HeadRenderCamera"
RENDER_CAMERA_ROT_OFFSET = (3.141592653589793, 0.0, 0.0)  # OpenCV->Blender camera axis flip.
REALTIME_PREVIEW = True  # Use frame-change handler for real-time viewport playback (no keyframes/render).
VIEWPORT_CAPTURE = True  # Capture viewport frames using the active camera (requires UI, no headless).
VIEWPORT_OUTPUT_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/head_camera_viewport"
VIEWPORT_FORMAT = "PNG"  # One of the image formats supported by this build.
VIEWPORT_HIDE_OVERLAYS = True  # Hide axis/grid/origin overlays in viewport capture.
VIEWPORT_TRANSPARENT = True  # Write alpha so the foreground mask is in the PNG alpha channel.
VIEWPORT_COLOR_MODE = "RGBA"  # Must support alpha (e.g., PNG, OPEN_EXR).

USE_HEAD_CAMERA_TRAJECTORY = True
HEAD_USE_ORIENTATION = True
HEAD_TRAJ_APPLY_Z = True
USE_BASE_QPOS_OFFSET = True
BASE_Z = None
SYMMETRIZE_GRIPPERS = False  # Force both fingers to use the same opening value.
GRIPPER_SYMMETRY_MODE = "mean"  # "mean" or "sum_half"

# Optional fixed offset between zed_link and the real head camera center.
# Used only for solving base pose (does NOT move the render camera).
APPLY_HEAD_CAMERA_WORLD_OFFSET = True
HEAD_CAMERA_WORLD_OFFSET = (-0.00420626, -0.0445856, 0.04473541)
# Optional fixed rotation offset between zed_link frame and real head camera frame.
# Quaternion is (w, x, y, z). Used only for base solving (render camera unchanged).
APPLY_HEAD_CAMERA_ROT_OFFSET = True
HEAD_CAMERA_ROT_OFFSET_QUAT = (1.0, 0.0, 0.0, 0.0)

MATCH_BY_IMAGE_ID = True
CAMERA_ID_OFFSET = 0

DEBUG_PRINT = False
DEBUG_PRINT_EVERY = 100

STATE = {}

PROPRIOCEPTION_INDICES = {
    "R1Pro": {
        "arm_left_qpos": slice(158, 165),
        "arm_right_qpos": slice(197, 204),
        "base_qpos": slice(244, 247),
        "gripper_left_qpos": slice(193, 195),
        "gripper_right_qpos": slice(232, 234),
        "trunk_qpos": slice(236, 240),
    }
}

JOINT_ORDER = [
    "torso_joint1",
    "torso_joint2",
    "torso_joint3",
    "torso_joint4",
    "left_arm_joint1",
    "right_arm_joint1",
    "left_arm_joint2",
    "right_arm_joint2",
    "left_arm_joint3",
    "right_arm_joint3",
    "left_arm_joint4",
    "right_arm_joint4",
    "left_arm_joint5",
    "right_arm_joint5",
    "left_arm_joint6",
    "right_arm_joint6",
    "left_arm_joint7",
    "right_arm_joint7",
    "left_gripper_finger_joint1",
    "left_gripper_finger_joint2",
    "right_gripper_finger_joint1",
    "right_gripper_finger_joint2",
]


def load_parquet_proprio(path):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    if "observation.state" in pf.schema_arrow.names:
        column = "observation.state"
    elif "observation/state" in pf.schema_arrow.names:
        column = "observation/state"
    else:
        raise ValueError("Parquet is missing observation.state column.")

    table = pq.read_table(path, columns=[column])
    col = table.column(0)
    return np.array(col.to_pylist(), dtype=np.float32)


def load_head_camera_pose(path, require_quat=True):
    positions = []
    quats = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            positions.append((float(row["x"]), float(row["y"]), float(row["z"])))
            if require_quat:
                quats.append(
                    (
                        float(row["qw"]),
                        float(row["qx"]),
                        float(row["qy"]),
                        float(row["qz"]),
                    )
                )
    pos_arr = np.array(positions, dtype=np.float32)
    quat_arr = np.array(quats, dtype=np.float32) if require_quat else None
    return pos_arr, quat_arr


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


def load_camera_intrinsics(path):
    values = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            values.append([float(v) for v in line.split()])
    if len(values) != 3 or any(len(row) != 3 for row in values):
        raise ValueError(f"Invalid 3x3 intrinsics in {path}")
    fx = values[0][0]
    fy = values[1][1]
    cx = values[0][2]
    cy = values[1][2]
    return fx, fy, cx, cy


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


def load_pose_list(csv_path):
    poses = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pos = (float(row["x"]), float(row["y"]), float(row["z"]))
            quat = (float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"]))
            poses.append((pos, quat))
    return poses


def pose_matrix(position, quat):
    q = Quaternion(quat).normalized()
    return Matrix.Translation(position) @ q.to_matrix().to_4x4()


def get_object():
    if OBJECT_NAME:
        obj = bpy.data.objects.get(OBJECT_NAME)
        if obj is None:
            raise RuntimeError(f"Object not found: {OBJECT_NAME}")
        return obj

    collection = bpy.data.collections.get(OBJECT_COLLECTION_NAME)
    if collection is None:
        raise RuntimeError(f"Collection not found: {OBJECT_COLLECTION_NAME}")
    for obj in collection.objects:
        if obj.type == "MESH":
            return obj
    raise RuntimeError("No mesh object found in ImportedObjects collection.")


def ensure_render_camera(head_cam, fx, fy, cx, cy):
    cam_obj = bpy.data.objects.get(RENDER_CAMERA_NAME)
    if cam_obj is None:
        cam_data = bpy.data.cameras.new(RENDER_CAMERA_NAME)
        cam_obj = bpy.data.objects.new(RENDER_CAMERA_NAME, cam_data)
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
    width = int(round(cx * 2.0))
    height = int(round(cy * 2.0))
    cam.lens = fx * cam.sensor_width / width
    cam.shift_x = (cx - width * 0.5) / width
    cam.shift_y = (cy - height * 0.5) / height
    return cam_obj, width, height


def build_joint_values(trunk, left_arm, right_arm, left_grip, right_grip):
    joints = []
    joints.extend(trunk.tolist())
    for left, right in zip(left_arm, right_arm):
        joints.append(float(left))
        joints.append(float(right))
    joints.extend(left_grip.tolist())
    joints.extend(right_grip.tolist())
    return joints


def load_joint_defs(urdf_path):
    import xml.etree.ElementTree as ET

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    joints = {}
    for joint_elem in root.findall("joint"):
        name = joint_elem.get("name")
        jtype = joint_elem.get("type")
        parent_elem = joint_elem.find("parent")
        child_elem = joint_elem.find("child")
        if parent_elem is None or child_elem is None:
            continue
        parent_link = parent_elem.get("link")
        child_link = child_elem.get("link")
        origin_elem = joint_elem.find("origin")
        xyz = [0.0, 0.0, 0.0]
        rpy = [0.0, 0.0, 0.0]
        if origin_elem is not None:
            xyz = [float(v) for v in origin_elem.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin_elem.get("rpy", "0 0 0").split()]
        axis_elem = joint_elem.find("axis")
        axis = [1.0, 0.0, 0.0]
        if axis_elem is not None:
            axis = [float(v) for v in axis_elem.get("xyz", "1 0 0").split()]
        joints[name] = {
            "type": jtype,
            "parent": parent_link,
            "child": child_link,
            "origin": Matrix.Translation(Vector(xyz)) @ Euler(tuple(rpy), "XYZ").to_matrix().to_4x4(),
            "axis": axis,
        }
    return joints


def build_joint_anim_data(joint_defs):
    anim_joints = []
    for name in JOINT_ORDER:
        if name not in joint_defs:
            continue
        info = joint_defs[name]
        obj = bpy.data.objects.get(info["child"])
        if obj is None:
            continue
        axis = Vector(info["axis"])
        if axis.length == 0.0:
            axis = Vector((1.0, 0.0, 0.0))
        axis.normalize()
        obj.rotation_mode = "XYZ"
        anim_joints.append(
            {
                "name": name,
                "obj": obj,
                "type": info["type"],
                "origin": info["origin"],
                "axis": axis,
            }
        )
    return anim_joints


def apply_joint_pose(anim_joint, value):
    jtype = anim_joint["type"]
    axis = anim_joint["axis"]
    if jtype in ("revolute", "continuous"):
        motion = Matrix.Rotation(value, 4, axis)
    elif jtype == "prismatic":
        motion = Matrix.Translation(axis * value)
    else:
        motion = Matrix.Identity(4)
    mat = anim_joint["origin"] @ motion
    loc, rot, _scale = mat.decompose()
    obj = anim_joint["obj"]
    obj.location = loc
    obj.rotation_euler = rot.to_euler("XYZ")


def world_offset_to_local(world_offset, head_matrix_world):
    rot = head_matrix_world.to_3x3()
    return rot.inverted() @ Vector(world_offset)


def quat_to_matrix(quat_wxyz):
    q = Quaternion(quat_wxyz).normalized()
    return q.to_matrix().to_4x4()


def build_object_pose_map(pose_files, cam_poses):
    obj_pose_by_id = {}
    for path in pose_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        if not stem.isdigit():
            continue
        image_id = int(stem) + CAMERA_ID_OFFSET
        if image_id not in cam_poses:
            continue
        cam_pos, cam_quat = cam_poses[image_id]
        T_world_cam = pose_matrix(cam_pos, cam_quat)
        T_cam_obj = load_ob_in_cam_matrix(path)
        obj_pose_by_id[image_id] = T_world_cam @ T_cam_obj
    return obj_pose_by_id


def nearest_pose(existing_ids, obj_pose_by_id, image_id):
    import bisect

    pos = bisect.bisect_left(existing_ids, image_id)
    if pos == 0:
        nearest_id = existing_ids[0]
    elif pos >= len(existing_ids):
        nearest_id = existing_ids[-1]
    else:
        before = existing_ids[pos - 1]
        after = existing_ids[pos]
        nearest_id = before if (image_id - before) <= (after - image_id) else after
    return obj_pose_by_id[nearest_id]


def replay_4d_plus_frame_handler(scene):
    state = STATE
    if not state.get("ready"):
        return

    frame = scene.frame_current
    data_idx = state["data_start_index"] + (frame - state["start_frame"])
    if data_idx < state["data_start_index"] or data_idx >= state["data_start_index"] + state["num_frames"]:
        return

    robot_root = state["robot_root"]
    render_cam = state["render_cam"]
    obj = state["obj"]
    anim_joints = state["anim_joints"]
    joint_count = state["joint_count"]
    trunk = state["trunk"]
    left_arm = state["left_arm"]
    right_arm = state["right_arm"]
    left_grip = state["left_grip"]
    right_grip = state["right_grip"]
    base_poses = state["base_poses"]
    head_positions = state["head_positions"]
    head_quats = state["head_quats"]
    base_headlocal_poses = state["base_headlocal_poses"]
    obj_pose_by_id = state["obj_pose_by_id"]
    existing_ids = state["existing_ids"]
    cam_poses = state["cam_poses"]
    pose_files = state["pose_files"]

    base_z = robot_root.location.z if BASE_Z is None else BASE_Z
    left_vals = left_grip[data_idx]
    right_vals = right_grip[data_idx]
    if SYMMETRIZE_GRIPPERS:
        if GRIPPER_SYMMETRY_MODE == "sum_half":
            left_val = float(left_vals.sum()) * 0.5
            right_val = float(right_vals.sum()) * 0.5
        else:
            left_val = float(left_vals.mean())
            right_val = float(right_vals.mean())
        left_val = max(0.0, min(0.05, left_val))
        right_val = max(0.0, min(0.05, right_val))
        left_vals = np.array([left_val, left_val], dtype=np.float32)
        right_vals = np.array([right_val, right_val], dtype=np.float32)
    joint_vals = build_joint_values(
        trunk[data_idx], left_arm[data_idx], right_arm[data_idx], left_vals, right_vals
    )
    if len(joint_vals) != joint_count:
        raise ValueError("Joint count mismatch.")
    robot_root.location = (0.0, 0.0, 0.0)
    robot_root.rotation_euler = (0.0, 0.0, 0.0)
    for anim_joint, value in zip(anim_joints, joint_vals):
        apply_joint_pose(anim_joint, float(value))

    target_mat = pose_matrix(head_positions[data_idx], head_quats[data_idx])
    cam_world = target_mat @ Euler(RENDER_CAMERA_ROT_OFFSET, "XYZ").to_matrix().to_4x4()
    cam_loc, cam_rot, _cam_scale = cam_world.decompose()
    render_cam.location = cam_loc
    render_cam.rotation_euler = cam_rot.to_euler("XYZ")

    if USE_HEAD_CAMERA_TRAJECTORY and HEAD_USE_ORIENTATION:
        base_pos, base_quat = base_headlocal_poses[data_idx]
        base_headlocal_mat = pose_matrix(base_pos, base_quat)
        root_mat = target_mat @ base_headlocal_mat.inverted()
        loc, rot, _scale = root_mat.decompose()
        if not HEAD_TRAJ_APPLY_Z:
            loc.z = base_z
        robot_root.location = loc
        robot_root.rotation_euler = rot.to_euler("XYZ")
    else:
        x, y, yaw = base_poses[data_idx]
        robot_root.location = (x, y, base_z)
        robot_root.rotation_euler = (0.0, 0.0, yaw)

    image_id = data_idx + 1
    if MATCH_BY_IMAGE_ID and existing_ids:
        if image_id in obj_pose_by_id:
            T_world_obj = obj_pose_by_id[image_id]
        else:
            T_world_obj = nearest_pose(existing_ids, obj_pose_by_id, image_id)
    else:
        cam_pos, cam_quat = cam_poses[image_id]
        T_world_cam = pose_matrix(cam_pos, cam_quat)
        T_cam_obj = load_ob_in_cam_matrix(pose_files[data_idx])
        T_world_obj = T_world_cam @ T_cam_obj

    loc, rot, _scale = T_world_obj.decompose()
    obj.location = loc
    obj.rotation_euler = rot.to_euler("XYZ")

    if DEBUG_PRINT and (frame % DEBUG_PRINT_EVERY == 0):
        print(
            f'frame {frame} data_idx {data_idx} '
            f'cam {render_cam.location} obj {obj.location}'
        )


def register_replay_handler():
    handlers = bpy.app.handlers.frame_change_pre
    for h in list(handlers):
        if getattr(h, "__name__", "") == "replay_4d_plus_frame_handler":
            handlers.remove(h)
    handlers.append(replay_4d_plus_frame_handler)


def viewport_render_animation(output_path, file_format):
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
    return False


def main():
    if IMPORT_URDF:
        import import_urdf

        import_urdf.import_urdf(URDF_PATH)

    # Load robot data
    proprio = load_parquet_proprio(PARQUET_PATH)
    idx = PROPRIOCEPTION_INDICES["R1Pro"]
    trunk = proprio[:, idx["trunk_qpos"]]
    left_arm = proprio[:, idx["arm_left_qpos"]]
    right_arm = proprio[:, idx["arm_right_qpos"]]
    left_grip = proprio[:, idx["gripper_left_qpos"]]
    right_grip = proprio[:, idx["gripper_right_qpos"]]
    base_qpos_values = proprio[:, idx["base_qpos"]]

    if USE_BASE_QPOS_OFFSET:
        base_poses = base_qpos_values
    else:
        base_poses = base_qpos_values - base_qpos_values[0]

    # Load head camera poses (always include orientation for rendering)
    head_positions, head_quats = load_head_camera_pose(HEAD_CAMERA_CSV_PATH, True)
    base_headlocal_poses = load_pose_list(HEAD_CAMERA_BASE_LOCAL_CSV_PATH)

    # Build camera pose dict from head_positions/head_quats (image_id starts at 1)
    cam_poses = {}
    for i in range(len(head_positions)):
        image_id = i + 1
        cam_poses[image_id] = (head_positions[i], head_quats[i])
    pose_files = load_ob_in_cam_files(OB_IN_CAM_PATH)
    obj_pose_by_id = build_object_pose_map(pose_files, cam_poses) if MATCH_BY_IMAGE_ID else {}
    existing_ids = sorted(obj_pose_by_id.keys())

    # Get objects
    robot_root = bpy.data.objects.get(ROBOT_ROOT_NAME)
    if robot_root is None:
        raise RuntimeError(f"Robot root not found: {ROBOT_ROOT_NAME}")
    robot_root.rotation_mode = "XYZ"
    head_cam = bpy.data.objects.get(HEAD_CAMERA_LINK_NAME)
    if head_cam is None:
        raise RuntimeError(f"Head camera link not found: {HEAD_CAMERA_LINK_NAME}")
    fx, fy, cx, cy = load_camera_intrinsics(CAM_K_PATH)
    render_cam, render_w, render_h = ensure_render_camera(head_cam, fx, fy, cx, cy)
    obj = get_object()
    obj.rotation_mode = "XYZ"

    # Joint setup
    joint_defs = load_joint_defs(URDF_PATH)
    anim_joints = build_joint_anim_data(joint_defs)
    joint_count = len(anim_joints)

    # Frame range
    total_frames = min(len(proprio), len(head_positions), len(base_headlocal_poses))
    if DATA_START_INDEX < 0 or DATA_START_INDEX >= total_frames:
        raise ValueError(f"DATA_START_INDEX out of range: {DATA_START_INDEX}")
    num_frames = total_frames - DATA_START_INDEX
    if FRAME_LIMIT is not None:
        num_frames = min(num_frames, FRAME_LIMIT)

    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = START_FRAME
    scene.frame_end = START_FRAME + num_frames - 1
    scene.camera = render_cam
    scene.render.engine = RENDER_ENGINE
    scene.render.resolution_x = render_w
    scene.render.resolution_y = render_h
    scene.render.resolution_percentage = 100

    if REALTIME_PREVIEW:
        STATE.clear()
        STATE.update(
            {
                "ready": True,
                "start_frame": START_FRAME,
                "data_start_index": DATA_START_INDEX,
                "num_frames": num_frames,
                "robot_root": robot_root,
                "render_cam": render_cam,
                "obj": obj,
                "anim_joints": anim_joints,
                "joint_count": joint_count,
                "trunk": trunk,
                "left_arm": left_arm,
                "right_arm": right_arm,
                "left_grip": left_grip,
                "right_grip": right_grip,
                "base_poses": base_poses,
                "head_positions": head_positions,
                "head_quats": head_quats,
                "base_headlocal_poses": base_headlocal_poses,
                "obj_pose_by_id": obj_pose_by_id,
                "existing_ids": existing_ids,
                "cam_poses": cam_poses,
                "pose_files": pose_files,
            }
        )
        register_replay_handler()
        scene.frame_set(START_FRAME)
        if VIEWPORT_CAPTURE:
            print("Replay 4D+ handler registered. Capturing viewport animation...")
            viewport_render_animation(VIEWPORT_OUTPUT_PATH, VIEWPORT_FORMAT)
        else:
            print("Replay 4D+ handler registered. Press Play for real-time viewport playback.")
        return

    # Replay loop
    wm = bpy.context.window_manager
    wm.progress_begin(0, num_frames)
    for i in range(num_frames):
        data_idx = DATA_START_INDEX + i
        frame = START_FRAME + i
        scene.frame_set(frame)

        base_z = robot_root.location.z if BASE_Z is None else BASE_Z
        # Apply joint qpos
        left_vals = left_grip[data_idx]
        right_vals = right_grip[data_idx]
        if SYMMETRIZE_GRIPPERS:
            if GRIPPER_SYMMETRY_MODE == "sum_half":
                left_val = float(left_vals.sum()) * 0.5
                right_val = float(right_vals.sum()) * 0.5
            else:
                left_val = float(left_vals.mean())
                right_val = float(right_vals.mean())
            left_val = max(0.0, min(0.05, left_val))
            right_val = max(0.0, min(0.05, right_val))
            left_vals = np.array([left_val, left_val], dtype=np.float32)
            right_vals = np.array([right_val, right_val], dtype=np.float32)
        joint_vals = build_joint_values(
            trunk[data_idx], left_arm[data_idx], right_arm[data_idx], left_vals, right_vals
        )
        if len(joint_vals) != joint_count:
            raise ValueError("Joint count mismatch.")
        robot_root.location = (0.0, 0.0, 0.0)
        robot_root.rotation_euler = (0.0, 0.0, 0.0)
        for anim_joint, value in zip(anim_joints, joint_vals):
            apply_joint_pose(anim_joint, float(value))

        bpy.context.view_layer.update()

        target_mat = pose_matrix(head_positions[data_idx], head_quats[data_idx])
        cam_world = target_mat @ Euler(RENDER_CAMERA_ROT_OFFSET, "XYZ").to_matrix().to_4x4()
        cam_loc, cam_rot, _cam_scale = cam_world.decompose()
        render_cam.location = cam_loc
        render_cam.rotation_euler = cam_rot.to_euler("XYZ")

        # Solve robot root from head camera pose
        if USE_HEAD_CAMERA_TRAJECTORY and HEAD_USE_ORIENTATION:
            base_pos, base_quat = base_headlocal_poses[data_idx]
            base_headlocal_mat = pose_matrix(base_pos, base_quat)
            root_mat = target_mat @ base_headlocal_mat.inverted()
            loc, rot, _scale = root_mat.decompose()
            if not HEAD_TRAJ_APPLY_Z:
                loc.z = base_z
            robot_root.location = loc
            robot_root.rotation_euler = rot.to_euler("XYZ")
        else:
            x, y, yaw = base_poses[data_idx]
            robot_root.location = (x, y, base_z)
            robot_root.rotation_euler = (0.0, 0.0, yaw)

        # Object pose (world)
        image_id = data_idx + 1
        if MATCH_BY_IMAGE_ID and existing_ids:
            if image_id in obj_pose_by_id:
                T_world_obj = obj_pose_by_id[image_id]
            else:
                T_world_obj = nearest_pose(existing_ids, obj_pose_by_id, image_id)
        else:
            cam_pos, cam_quat = cam_poses[image_id]
            T_world_cam = pose_matrix(cam_pos, cam_quat)
            T_cam_obj = load_ob_in_cam_matrix(pose_files[data_idx])
            T_world_obj = T_world_cam @ T_cam_obj

        loc, rot, _scale = T_world_obj.decompose()
        obj.location = loc
        obj.rotation_euler = rot.to_euler("XYZ")

        # Keyframe both
        render_cam.keyframe_insert(data_path="location")
        render_cam.keyframe_insert(data_path="rotation_euler")
        robot_root.keyframe_insert(data_path="location")
        robot_root.keyframe_insert(data_path="rotation_euler")
        for anim_joint in anim_joints:
            anim_joint["obj"].keyframe_insert(data_path="location")
            anim_joint["obj"].keyframe_insert(data_path="rotation_euler")
        obj.keyframe_insert(data_path="location")
        obj.keyframe_insert(data_path="rotation_euler")

        if DEBUG_PRINT and (i % DEBUG_PRINT_EVERY == 0):
            print(
                f'frame {frame} data_idx {data_idx} base {robot_root.location} '
                f'head {head_cam.matrix_world.translation} obj {obj.location}'
            )
        wm.progress_update(i + 1)

    wm.progress_end()
    print("Replay 4D+ complete.")
    if RENDER_VIDEO:
        import time

        scene.frame_set(START_FRAME)
        t0 = time.perf_counter()
        try:
            scene.render.image_settings.file_format = "FFMPEG"
            scene.render.ffmpeg.format = "MPEG4"
            scene.render.ffmpeg.codec = "H264"
            scene.render.filepath = RENDER_OUTPUT_PATH
        except TypeError:
             # Fallback: render PNG sequence if ffmpeg is not available in this build.
            frames_dir = os.path.splitext(RENDER_OUTPUT_PATH)[0] + "_frames"
            os.makedirs(frames_dir, exist_ok=True)
            scene.render.image_settings.file_format = "PNG"
            scene.render.filepath = os.path.join(frames_dir, "frame_")
            print("FFMPEG output not available; rendering PNG sequence to", frames_dir)
        bpy.ops.render.render(animation=True)
        t1 = time.perf_counter()
        print(f"Render time: {t1 - t0:.2f} sec")


if __name__ == "__main__":
    main()
