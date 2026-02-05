"""
Replay R1_pro joint qpos + navigation velocity trajectories in Blender.

Usage (from command line):
  blender --python replay_r1pro_trajectory.py

Or open in Blender > Scripting > Run Script.
"""

import math
import os
import xml.etree.ElementTree as ET
import csv

import bpy
import numpy as np
from mathutils import Matrix, Vector, Euler, Quaternion

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise RuntimeError(
        "pyarrow is required. Install into Blender's Python, e.g.\n"
        "  /path/to/blender/5.0/python/bin/python3 -m pip install pyarrow"
    ) from exc


# ----------------------
# User configuration
# ----------------------
URDF_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/urdf/r1pro.urdf"
PARQUET_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/picking_up_trash/episode_00010030.parquet"
HEAD_CAMERA_CSV_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/picking_up_trash/camera_poses/head_camera_pose.csv"

IMPORT_URDF = False  # Set True to import URDF at script start.
ROBOT_ROOT_NAME = "r1_pro_with_gripper"
HEAD_CAMERA_LINK_NAME = "zed_link"
USE_HEAD_CAMERA_TRAJECTORY = True
HEAD_USE_ORIENTATION = True
HEAD_TRAJ_APPLY_Z = True  # False = keep BASE_Z or current Z.

# Base pose handling.
USE_BASE_QPOS_OFFSET = True  # True = use absolute base_qpos; False = make the first pose zero.
BASE_Z = None  # None = keep current Z; otherwise set a fixed Z height.

FPS = 60
DT = 1.0 / FPS
START_FRAME = 1
FRAME_LIMIT = None  # Set to an int to limit frames for a quick preview.
REALTIME_PREVIEW = False  # Use frame-change handler for real-time playback (no keyframes).
 
# Joint order matching eval_utils.PROPRIO_QPOS_INDICES for R1Pro
# (qpos[6:10]=torso, qpos[10:24] interleaves L/R arm joints, qpos[24:28]=grippers).
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

PROPRIOCEPTION_INDICES = {
    "R1Pro": {
        "arm_left_qpos": slice(158, 165),
        "arm_right_qpos": slice(197, 204),
        "base_qpos": slice(244, 247),
        "base_qvel": slice(253, 256),
        "gripper_left_qpos": slice(193, 195),
        "gripper_right_qpos": slice(232, 234),
        "trunk_qpos": slice(236, 240),
    }
}

STATE = {}


def parse_origin(origin_elem):
    xyz = [0.0, 0.0, 0.0]
    rpy = [0.0, 0.0, 0.0]
    if origin_elem is not None:
        xyz = [float(v) for v in origin_elem.get("xyz", "0 0 0").split()]
        rpy = [float(v) for v in origin_elem.get("rpy", "0 0 0").split()]
    return xyz, rpy


def create_transform_matrix(xyz, rpy):
    rot_matrix = Euler((rpy[0], rpy[1], rpy[2]), "XYZ").to_matrix().to_4x4()
    trans_matrix = Matrix.Translation(Vector(xyz))
    return trans_matrix @ rot_matrix


def load_joint_defs(urdf_path):
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
        xyz, rpy = parse_origin(origin_elem)
        axis_elem = joint_elem.find("axis")
        axis = [1.0, 0.0, 0.0]
        if axis_elem is not None:
            axis = [float(v) for v in axis_elem.get("xyz", "1 0 0").split()]
        joints[name] = {
            "type": jtype,
            "parent": parent_link,
            "child": child_link,
            "origin": create_transform_matrix(xyz, rpy),
            "axis": axis,
        }
    return joints


def load_parquet_proprio(path):
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


def load_cam_rel(parquet_path):
    pf = pq.ParquetFile(parquet_path)
    cam_col = "observation.cam_rel_poses"
    if cam_col not in pf.schema_arrow.names:
        cam_col = "observation/cam_rel_poses"
    if cam_col not in pf.schema_arrow.names:
        raise ValueError("Parquet missing observation.cam_rel_poses column.")
    return pq.read_table(parquet_path, columns=[cam_col]).column(0).to_pylist()


def pick_head_index(cam_rel_row):
    groups = [cam_rel_row[i * 7 : (i + 1) * 7] for i in range(len(cam_rel_row) // 7)]
    if not groups:
        raise ValueError("cam_rel_poses row is empty.")
    return max(range(len(groups)), key=lambda i: groups[i][2])


def load_head_cam_rel_poses(parquet_path):
    cam_rel = load_cam_rel(parquet_path)
    if not cam_rel:
        raise ValueError("cam_rel_poses is empty.")
    head_idx = pick_head_index(cam_rel[0])
    poses = []
    for row in cam_rel:
        g = row[head_idx * 7 : head_idx * 7 + 7]
        poses.append(
            (
                (g[0], g[1], g[2]),
                (g[3], g[5], g[6], g[4]),
            )
        )
    return poses


def build_joint_anim_data(joint_defs):
    anim_joints = []
    for name in JOINT_ORDER:
        if name not in joint_defs:
            print(f"Warning: Joint not found in URDF: {name}")
            continue
        info = joint_defs[name]
        obj = bpy.data.objects.get(info["child"])
        if obj is None:
            print(f"Warning: Link object not found: {info['child']} (joint {name})")
            continue
        axis = Vector(info["axis"])
        if axis.length == 0.0:
            axis = Vector((1.0, 0.0, 0.0))
        axis.normalize()
        obj.rotation_mode = "XYZ"
        anim_joints.append(
            {
                "name": name,
                "type": info["type"],
                "obj": obj,
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


def integrate_nav(nav_vel, dt, in_robot_frame=True, initial_pose=None):
    if initial_pose is None:
        x, y, yaw = 0.0, 0.0, 0.0
    else:
        x, y, yaw = initial_pose
    poses = []
    for vx, vy, wz in nav_vel:
        if in_robot_frame:
            c = math.cos(yaw)
            s = math.sin(yaw)
            dx = (vx * c - vy * s) * dt
            dy = (vx * s + vy * c) * dt
        else:
            dx = vx * dt
            dy = vy * dt
        x += dx
        y += dy
        yaw += wz * dt
        poses.append((x, y, yaw))
    return poses


def build_joint_values(trunk, left_arm, right_arm, left_grip, right_grip):
    joints = []
    joints.extend(trunk.tolist())
    for left, right in zip(left_arm, right_arm):
        joints.append(float(left))
        joints.append(float(right))
    joints.extend(left_grip.tolist())
    joints.extend(right_grip.tolist())
    return joints

def load_head_camera_pose(path, require_quat):
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


def pose_matrix(position, quat):
    q = Quaternion(quat).normalized()
    return Matrix.Translation(position) @ q.to_matrix().to_4x4()


def replay_r1pro_frame_handler(scene):
    state = STATE
    if not state.get("ready"):
        return

    frame = scene.frame_current
    data_idx = state["data_start_index"] + (frame - state["start_frame"])
    if data_idx < state["data_start_index"] or data_idx >= state["data_start_index"] + state["num_frames"]:
        return

    robot_root = state["robot_root"]
    anim_joints = state["anim_joints"]
    joint_count = state["joint_count"]
    trunk = state["trunk"]
    left_arm = state["left_arm"]
    right_arm = state["right_arm"]
    left_grip = state["left_grip"]
    right_grip = state["right_grip"]
    nav_poses = state["nav_poses"]
    head_positions = state["head_positions"]
    head_quats = state["head_quats"]
    base_headlocal_poses = state["base_headlocal_poses"]
    head_cam = state["head_cam"]

    base_z = robot_root.location.z if BASE_Z is None else BASE_Z
    joint_vals = build_joint_values(
        trunk[data_idx], left_arm[data_idx], right_arm[data_idx], left_grip[data_idx], right_grip[data_idx]
    )
    if len(joint_vals) != joint_count:
        raise ValueError(
            f"Joint count mismatch: {len(joint_vals)} values for {joint_count} joints."
        )
    if USE_HEAD_CAMERA_TRAJECTORY:
        robot_root.location = (0.0, 0.0, 0.0)
        robot_root.rotation_euler = (0.0, 0.0, 0.0)
    for anim_joint, value in zip(anim_joints, joint_vals):
        apply_joint_pose(anim_joint, float(value))

    if USE_HEAD_CAMERA_TRAJECTORY:
        if not HEAD_USE_ORIENTATION:
            bpy.context.view_layer.update()
        target_pos = Vector(head_positions[data_idx])
        if HEAD_USE_ORIENTATION:
            target_mat = pose_matrix(target_pos, head_quats[data_idx])
            base_pos, base_quat = base_headlocal_poses[data_idx]
            base_headlocal_mat = pose_matrix(base_pos, base_quat)
            root_mat = target_mat @ base_headlocal_mat.inverted()
            loc, rot, _scale = root_mat.decompose()
            if not HEAD_TRAJ_APPLY_Z:
                loc.z = base_z
            robot_root.location = loc
            robot_root.rotation_euler = rot.to_euler("XYZ")
        else:
            cam_pos = head_cam.matrix_world.translation
            delta = target_pos - cam_pos
            if HEAD_TRAJ_APPLY_Z:
                robot_root.location = (delta.x, delta.y, delta.z)
            else:
                robot_root.location = (delta.x, delta.y, base_z)
            x, y, yaw = nav_poses[data_idx]
            robot_root.rotation_euler = (0.0, 0.0, yaw)
    else:
        x, y, yaw = nav_poses[data_idx]
        robot_root.location = (x, y, base_z)
        robot_root.rotation_euler = (0.0, 0.0, yaw)


def register_replay_handler():
    handlers = bpy.app.handlers.frame_change_pre
    for h in list(handlers):
        if getattr(h, "__name__", "") == "replay_r1pro_frame_handler":
            handlers.remove(h)
    handlers.append(replay_r1pro_frame_handler)


def main():
    if IMPORT_URDF:
        from import_urdf import import_urdf  # local file next to this script
        import_urdf(URDF_PATH)

    joint_defs = load_joint_defs(URDF_PATH)
    anim_joints = build_joint_anim_data(joint_defs)

    proprio = load_parquet_proprio(PARQUET_PATH)
    if proprio.ndim != 2 or proprio.shape[1] < 256:
        raise ValueError("observation.state must be 2D with at least 256 values.")

    idx = PROPRIOCEPTION_INDICES["R1Pro"]
    trunk = proprio[:, idx["trunk_qpos"]]
    left_arm = proprio[:, idx["arm_left_qpos"]]
    right_arm = proprio[:, idx["arm_right_qpos"]]
    left_grip = proprio[:, idx["gripper_left_qpos"]]
    right_grip = proprio[:, idx["gripper_right_qpos"]]
    base_qpos_values = proprio[:, idx["base_qpos"]]

    joint_count = len(anim_joints)
    num_frames = len(proprio)
    if FRAME_LIMIT is not None:
        num_frames = min(num_frames, FRAME_LIMIT)

    if USE_BASE_QPOS_OFFSET:
        nav_poses = base_qpos_values[:num_frames]
    else:
        offset = base_qpos_values[0]
        nav_poses = base_qpos_values[:num_frames] - offset

    head_positions = None
    head_quats = None
    base_headlocal_poses = None
    if USE_HEAD_CAMERA_TRAJECTORY:
        head_positions, head_quats = load_head_camera_pose(
            HEAD_CAMERA_CSV_PATH, HEAD_USE_ORIENTATION
        )
        base_headlocal_poses = load_head_cam_rel_poses(PARQUET_PATH)
        if len(head_positions) < num_frames:
            num_frames = len(head_positions)
        if base_headlocal_poses is not None and len(base_headlocal_poses) < num_frames:
            num_frames = len(base_headlocal_poses)
    robot_root = bpy.data.objects.get(ROBOT_ROOT_NAME)
    if robot_root is None:
        raise RuntimeError(f"Robot root not found: {ROBOT_ROOT_NAME}")
    robot_root.rotation_mode = "XYZ"
    head_cam = None
    if USE_HEAD_CAMERA_TRAJECTORY:
        head_cam = bpy.data.objects.get(HEAD_CAMERA_LINK_NAME)
        if head_cam is None:
            raise RuntimeError(f"Head camera link not found: {HEAD_CAMERA_LINK_NAME}")

    scene = bpy.context.scene
    scene.render.fps = FPS
    scene.frame_start = START_FRAME
    scene.frame_end = START_FRAME + num_frames - 1

    if REALTIME_PREVIEW:
        STATE.clear()
        STATE.update(
            {
                "ready": True,
                "start_frame": START_FRAME,
                "data_start_index": 0,
                "num_frames": num_frames,
                "robot_root": robot_root,
                "anim_joints": anim_joints,
                "joint_count": joint_count,
                "trunk": trunk,
                "left_arm": left_arm,
                "right_arm": right_arm,
                "left_grip": left_grip,
                "right_grip": right_grip,
                "nav_poses": nav_poses,
                "head_positions": head_positions,
                "head_quats": head_quats,
                "base_headlocal_poses": base_headlocal_poses,
                "head_cam": head_cam,
            }
        )
        register_replay_handler()
        scene.frame_set(START_FRAME)
        print("Replay handler registered. Press Play for real-time playback.")
        return

    for i in range(num_frames):
        frame = START_FRAME + i
        scene.frame_set(frame)

        base_z = robot_root.location.z if BASE_Z is None else BASE_Z
        # Apply joint qpos from observation.state.
        joint_vals = build_joint_values(
            trunk[i], left_arm[i], right_arm[i], left_grip[i], right_grip[i]
        )
        if len(joint_vals) != joint_count:
            raise ValueError(
                f"Joint count mismatch: {len(joint_vals)} values for {joint_count} joints."
            )
        if USE_HEAD_CAMERA_TRAJECTORY:
            robot_root.location = (0.0, 0.0, 0.0)
            robot_root.rotation_euler = (0.0, 0.0, 0.0)
        for anim_joint, value in zip(anim_joints, joint_vals):
            apply_joint_pose(anim_joint, float(value))

        if USE_HEAD_CAMERA_TRAJECTORY:
            bpy.context.view_layer.update()
            target_pos = Vector(head_positions[i])
            if HEAD_USE_ORIENTATION:
                target_mat = pose_matrix(target_pos, head_quats[i])
                base_pos, base_quat = base_headlocal_poses[i]
                base_headlocal_mat = pose_matrix(base_pos, base_quat)
                root_mat = target_mat @ base_headlocal_mat.inverted()
                loc, rot, _scale = root_mat.decompose()
                if not HEAD_TRAJ_APPLY_Z:
                    loc.z = base_z
                robot_root.location = loc
                robot_root.rotation_euler = rot.to_euler("XYZ")
            else:
                cam_pos = head_cam.matrix_world.translation
                delta = target_pos - cam_pos
                if HEAD_TRAJ_APPLY_Z:
                    robot_root.location = (delta.x, delta.y, delta.z)
                else:
                    robot_root.location = (delta.x, delta.y, base_z)
                x, y, yaw = nav_poses[i]
                robot_root.rotation_euler = (0.0, 0.0, yaw)
        else:
            x, y, yaw = nav_poses[i]
            robot_root.location = (x, y, base_z)
            robot_root.rotation_euler = (0.0, 0.0, yaw)

        robot_root.keyframe_insert(data_path="location")
        robot_root.keyframe_insert(data_path="rotation_euler")
        for anim_joint in anim_joints:
            anim_joint["obj"].keyframe_insert(data_path="location")
            anim_joint["obj"].keyframe_insert(data_path="rotation_euler")

    print(f"Replay complete. Keyframed {num_frames} frames.")


if __name__ == "__main__":
    main()
