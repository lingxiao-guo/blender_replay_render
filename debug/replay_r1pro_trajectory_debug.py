"""
Import R1Pro URDF and replay base/joint qpos from debug.parquet in Blender.

Usage:
  blender --python /home/lingxiao/Downloads/blender-5.0.1-linux-x64/debug/replay_r1pro_trajectory_debug.py
"""

import importlib.util
import os
import struct
import xml.etree.ElementTree as ET

import bpy
import numpy as np
from mathutils import Matrix, Vector, Euler

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    raise RuntimeError(
        "pyarrow is required. Install into Blender's Python, e.g.\n"
        "  /home/lingxiao/Downloads/blender-5.0.1-linux-x64/5.0/python/bin/python3.11 -m pip install pyarrow"
    ) from exc


# ----------------------
# User configuration
# ----------------------
URDF_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/urdf/r1pro.urdf"
PARQUET_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/debug/debug.parquet"
IMPORT_URDF_SCRIPT_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/import_urdf.py"
URDF_CAMERA_CFG_PATH = os.path.join(
    os.path.dirname(URDF_PATH), "r1pro_source_cfg.yaml"
)

URDF_CAMERA_LINK_OFFSETS = {
    "zed_link": {
        "position": [0.06, 0.0, 0.01],
        "orientation": [-1.0, 0.0, 0.0, 0.0],
    },
    "left_realsense_link": {
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.7071, -0.7071, 0.0, 0.0],
    },
    "right_realsense_link": {
        "position": [0.0, 0.0, 0.0],
        "orientation": [0.7071, -0.7071, 0.0, 0.0],
    },
}

CAMERA_CLIP_START = {
    "head": 0.1,
    "left_wrist": 0.01,
    "right_wrist": 0.01,
}

CAMERA_CLIP_END = {
    "head": 100.0,
    "left_wrist": 100.0,
    "right_wrist": 100.0,
}

RENDER_CAMERA = "right_wrist"
CLEAR_SCENE_BEFORE_IMPORT = True
FPS = 60
START_FRAME = 1
FRAME_LIMIT = None
USE_BASE_QPOS_OFFSET = True
BASE_Z = None  # None keeps imported root Z.
REALTIME_PREVIEW = False  # Register frame handler and return replay bundle (no internal capture loop).

STATE = {}

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
        "gripper_left_qpos": slice(193, 195),
        "gripper_right_qpos": slice(232, 234),
        "trunk_qpos": slice(236, 240),
    }
}


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
        xyz, rpy = parse_origin(joint_elem.find("origin"))
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


def _parse_float_list(value_text):
    value_text = value_text.strip()
    if not (value_text.startswith("[") and value_text.endswith("]")):
        raise ValueError(f"Expected list syntax, got: {value_text}")
    items = [item.strip() for item in value_text[1:-1].split(",")]
    return [float(item) for item in items if item]


def load_camera_link_offsets(cfg_path, fallback_offsets):
    offsets = {
        link: {
            "position": np.asarray(values["position"], dtype=np.float32),
            "orientation": np.asarray(values["orientation"], dtype=np.float32),
        }
        for link, values in fallback_offsets.items()
    }
    if not cfg_path or not os.path.isfile(cfg_path):
        print(f"Warning: camera cfg not found: {cfg_path}; using fallback offsets.")
        return offsets

    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        in_camera_links = False
        current_link = None
        current_position = None
        current_orientation = None

        def commit_current():
            if (
                current_link is None
                or current_position is None
                or current_orientation is None
                or len(current_position) != 3
                or len(current_orientation) != 4
            ):
                return
            offsets[current_link] = {
                "position": np.asarray(current_position, dtype=np.float32),
                "orientation": np.asarray(current_orientation, dtype=np.float32),
            }

        for raw_line in lines:
            line = raw_line.split("#", 1)[0].rstrip()
            if not line.strip():
                continue

            stripped = line.strip()
            indent = len(line) - len(line.lstrip(" "))
            if not in_camera_links:
                if stripped.startswith("camera_links:"):
                    in_camera_links = True
                continue

            if indent == 0 and not stripped.startswith("-"):
                break

            if stripped.startswith("- link:"):
                commit_current()
                current_link = stripped.split(":", 1)[1].strip()
                current_position = None
                current_orientation = None
                continue

            if current_link is None:
                continue

            if stripped.startswith("position:"):
                value_text = stripped.split(":", 1)[1].strip()
                current_position = _parse_float_list(value_text)
            elif stripped.startswith("orientation:"):
                value_text = stripped.split(":", 1)[1].strip()
                current_orientation = _parse_float_list(value_text)

        commit_current()
    except Exception as exc:
        print(f"Warning: failed to parse camera cfg '{cfg_path}': {exc}; using fallback offsets.")

    return offsets


def quat_xyzw_to_rotmat(quat):
    q = np.asarray(quat, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-8:
        return np.eye(3, dtype=np.float32)
    x, y, z, w = q / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float32,
    )


def make_transform_from_pos_quat(position, orientation):
    tf = np.eye(4, dtype=np.float32)
    tf[0:3, 0:3] = quat_xyzw_to_rotmat(orientation)
    tf[0:3, 3] = np.asarray(position, dtype=np.float32)
    return tf


def load_parquet_proprio(path):
    pf = pq.ParquetFile(path)
    if "observation.state" in pf.schema_arrow.names:
        column = "observation.state"
    elif "observation/state" in pf.schema_arrow.names:
        column = "observation/state"
    else:
        raise ValueError("Parquet is missing observation.state column.")
    table = pq.read_table(path, columns=[column])
    return np.array(table.column(0).to_pylist(), dtype=np.float32)


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


def build_joint_values(trunk, left_arm, right_arm, left_grip, right_grip):
    joints = []
    joints.extend(trunk.tolist())
    for left, right in zip(left_arm, right_arm):
        joints.append(float(left))
        joints.append(float(right))
    joints.extend(left_grip.tolist())
    joints.extend(right_grip.tolist())
    return joints


def load_import_urdf_func(import_script_path):
    spec = importlib.util.spec_from_file_location("import_urdf_local", import_script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load import script: {import_script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "import_urdf"):
        raise RuntimeError(f"import_urdf() not found in: {import_script_path}")
    return module.import_urdf


def load_colmap_pinhole_intrinsics(cameras_bin_path):
    with open(cameras_bin_path, "rb") as f:
        data = f.read()

    if len(data) < 64:
        raise ValueError(f"Invalid cameras.bin (too small): {cameras_bin_path}")

    (num_cameras,) = struct.unpack_from("<Q", data, 0)
    if num_cameras < 1:
        raise ValueError(f"No cameras found in: {cameras_bin_path}")

    camera_id, model_id = struct.unpack_from("<ii", data, 8)
    width, height = struct.unpack_from("<QQ", data, 16)
    if model_id != 1:
        raise ValueError(
            f"Expected COLMAP PINHOLE model (id=1), got id={model_id} in {cameras_bin_path}"
        )
    fx, fy, cx, cy = struct.unpack_from("<dddd", data, 32)
    return {
        "camera_id": int(camera_id),
        "width": int(width),
        "height": int(height),
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
    }


def rotmat_to_quat_xyzw(rot):
    trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        qw = (rot[2, 1] - rot[1, 2]) / s
        qx = 0.25 * s
        qy = (rot[0, 1] + rot[1, 0]) / s
        qz = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        qw = (rot[0, 2] - rot[2, 0]) / s
        qx = (rot[0, 1] + rot[1, 0]) / s
        qy = 0.25 * s
        qz = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        qw = (rot[1, 0] - rot[0, 1]) / s
        qx = (rot[0, 2] + rot[2, 0]) / s
        qy = (rot[1, 2] + rot[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float32)
    quat /= np.clip(np.linalg.norm(quat), 1e-8, None)
    return quat


def cam_abs_poses_urdf_to_pose_dict(cam_abs_poses_urdf):
    pose_dict = {}
    for camera_name, poses in cam_abs_poses_urdf.items():
        per_camera = {}
        for image_id, pose in enumerate(poses, start=1):
            per_camera[image_id] = (
                (float(pose[0]), float(pose[1]), float(pose[2])),
                (float(pose[6]), float(pose[3]), float(pose[4]), float(pose[5])),
            )
        pose_dict[camera_name] = per_camera
    return pose_dict


def replay_r1pro_debug_frame_handler(scene):
    state = STATE
    if not state.get("ready"):
        return
    frame_idx = scene.frame_current - state["start_frame"]
    if frame_idx < 0 or frame_idx >= state["num_frames"]:
        return

    try:
        joint_values = state["joint_values_matrix"][frame_idx]
        anim_joints = state["anim_joints"]
        if len(joint_values) != len(anim_joints):
            return
        for anim_joint, value in zip(anim_joints, joint_values):
            apply_joint_pose(anim_joint, float(value))

        x, y, yaw = state["base_poses"][frame_idx]
        robot_root = state["robot_root"]
        robot_root.location = (float(x), float(y), float(state["base_z"]))
        robot_root.rotation_euler = (0.0, 0.0, float(yaw))
    except ReferenceError:
        return


def register_replay_handler():
    handlers = bpy.app.handlers.frame_change_pre
    for handler in list(handlers):
        if getattr(handler, "__name__", "") == "replay_r1pro_debug_frame_handler":
            handlers.remove(handler)
    handlers.append(replay_r1pro_debug_frame_handler)


def build_replay_state(robot_root, joint_defs, script_dir):
    proprio = load_parquet_proprio(PARQUET_PATH)
    if proprio.ndim != 2 or proprio.shape[1] < 256:
        raise ValueError("observation.state must be 2D with at least 256 values.")

    camera_intrinsics = {}
    camera_root = os.path.join(script_dir, "camera_poses")
    if not os.path.isdir(camera_root):
        camera_root = os.path.join(os.path.dirname(PARQUET_PATH), "camera_poses")
    for camera_name in ("head", "left_wrist", "right_wrist"):
        cameras_bin = os.path.join(camera_root, camera_name, "cameras.bin")
        camera_intrinsics[camera_name] = load_colmap_pinhole_intrinsics(cameras_bin)

    idx = PROPRIOCEPTION_INDICES["R1Pro"]
    trunk = proprio[:, idx["trunk_qpos"]]
    left_arm = proprio[:, idx["arm_left_qpos"]]
    right_arm = proprio[:, idx["arm_right_qpos"]]
    left_grip = proprio[:, idx["gripper_left_qpos"]]
    right_grip = proprio[:, idx["gripper_right_qpos"]]
    base_qpos_values = proprio[:, idx["base_qpos"]]

    num_frames = len(proprio)
    if FRAME_LIMIT is not None:
        num_frames = min(num_frames, FRAME_LIMIT)
    if num_frames <= 0:
        raise RuntimeError("No frames to replay.")

    if USE_BASE_QPOS_OFFSET:
        base_poses = base_qpos_values[:num_frames]
    else:
        base_poses = base_qpos_values[:num_frames] - base_qpos_values[0]

    base_pos = np.empty((num_frames, 3), dtype=np.float32)
    base_pos[:, 0:2] = base_poses[:, 0:2]
    base_pos[:, 2] = robot_root.location.z if BASE_Z is None else float(BASE_Z)

    yaw = base_poses[:, 2].astype(np.float32, copy=False)
    cos_yaw = np.cos(yaw, dtype=np.float32)
    sin_yaw = np.sin(yaw, dtype=np.float32)

    half_yaw = 0.5 * yaw
    base_q = np.empty((num_frames, 4), dtype=np.float32)
    base_q[:, 0] = 0.0
    base_q[:, 1] = 0.0
    base_q[:, 2] = np.sin(half_yaw, dtype=np.float32)
    base_q[:, 3] = np.cos(half_yaw, dtype=np.float32)

    camera_link_names = {
        "head": "zed_link",
        "left_wrist": "left_realsense_link",
        "right_wrist": "right_realsense_link",
    }
    camera_link_offsets = load_camera_link_offsets(
        URDF_CAMERA_CFG_PATH, URDF_CAMERA_LINK_OFFSETS
    )
    camera_link_offset_tfs = {}
    for camera_link in camera_link_names.values():
        if camera_link not in camera_link_offsets:
            raise ValueError(
                f"Missing camera offset for link '{camera_link}' in cfg/fallback."
            )
        offset = camera_link_offsets[camera_link]
        camera_link_offset_tfs[camera_link] = make_transform_from_pos_quat(
            offset["position"], offset["orientation"]
        )

    joint_values_matrix = np.empty((num_frames, len(JOINT_ORDER)), dtype=np.float32)
    joint_values_matrix[:, 0:4] = trunk[:num_frames]
    joint_values_matrix[:, 4:18:2] = left_arm[:num_frames]
    joint_values_matrix[:, 5:18:2] = right_arm[:num_frames]
    joint_values_matrix[:, 18:20] = left_grip[:num_frames]
    joint_values_matrix[:, 20:22] = right_grip[:num_frames]
    joint_values_by_name = {
        joint_name: joint_values_matrix[:, i] for i, joint_name in enumerate(JOINT_ORDER)
    }

    child_to_joint = {info["child"]: name for name, info in joint_defs.items()}
    joint_origins = {
        name: np.asarray(info["origin"], dtype=np.float32) for name, info in joint_defs.items()
    }
    joint_axes = {}
    for name, info in joint_defs.items():
        axis = np.asarray(info["axis"], dtype=np.float32)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 1e-8:
            axis /= axis_norm
        else:
            axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        joint_axes[name] = axis

    camera_chains = {}
    for camera_name, camera_link in camera_link_names.items():
        chain = []
        link_cursor = camera_link
        while link_cursor != "base_link":
            joint_name = child_to_joint.get(link_cursor)
            if joint_name is None:
                raise ValueError(
                    f"Cannot build URDF chain for camera '{camera_name}': missing parent joint for link '{link_cursor}'."
                )
            info = joint_defs[joint_name]
            chain.append(joint_name)
            link_cursor = info["parent"]
        chain.reverse()
        camera_chains[camera_name] = chain

    cam_abs_poses_urdf = {}
    eye4 = np.eye(4, dtype=np.float32)
    for camera_name, chain in camera_chains.items():
        camera_link = camera_link_names[camera_name]
        camera_offset_tf = camera_link_offset_tfs[camera_link]
        rel_t = np.empty((num_frames, 3), dtype=np.float32)
        rel_q = np.empty((num_frames, 4), dtype=np.float32)
        for i in range(num_frames):
            tf = eye4.copy()
            for joint_name in chain:
                info = joint_defs[joint_name]
                origin = joint_origins[joint_name]
                jtype = info["type"]
                if jtype in ("revolute", "continuous"):
                    angle = (
                        float(joint_values_by_name[joint_name][i])
                        if joint_name in joint_values_by_name
                        else 0.0
                    )
                    x, y, z = joint_axes[joint_name]
                    c = float(np.cos(angle))
                    s = float(np.sin(angle))
                    t = 1.0 - c
                    motion = np.array(
                        [
                            [x * x * t + c, x * y * t - z * s, x * z * t + y * s, 0.0],
                            [y * x * t + z * s, y * y * t + c, y * z * t - x * s, 0.0],
                            [z * x * t - y * s, z * y * t + x * s, z * z * t + c, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                        dtype=np.float32,
                    )
                elif jtype == "prismatic":
                    distance = (
                        float(joint_values_by_name[joint_name][i])
                        if joint_name in joint_values_by_name
                        else 0.0
                    )
                    motion = eye4.copy()
                    motion[0:3, 3] = joint_axes[joint_name] * distance
                else:
                    motion = eye4
                tf = tf @ origin @ motion
            tf_cam = tf @ camera_offset_tf
            rel_t[i] = tf_cam[0:3, 3]
            rel_q[i] = rotmat_to_quat_xyzw(tf_cam[0:3, 0:3])

        abs_t = np.empty_like(rel_t)
        abs_t[:, 0] = cos_yaw * rel_t[:, 0] - sin_yaw * rel_t[:, 1] + base_pos[:, 0]
        abs_t[:, 1] = sin_yaw * rel_t[:, 0] + cos_yaw * rel_t[:, 1] + base_pos[:, 1]
        abs_t[:, 2] = rel_t[:, 2] + base_pos[:, 2]

        bx, by, bz, bw = base_q[:, 0], base_q[:, 1], base_q[:, 2], base_q[:, 3]
        rx, ry, rz, rw = rel_q[:, 0], rel_q[:, 1], rel_q[:, 2], rel_q[:, 3]
        abs_q = np.empty_like(rel_q)
        abs_q[:, 0] = bw * rx + bx * rw + by * rz - bz * ry
        abs_q[:, 1] = bw * ry - bx * rz + by * rw + bz * rx
        abs_q[:, 2] = bw * rz + bx * ry - by * rx + bz * rw
        abs_q[:, 3] = bw * rw - bx * rx - by * ry - bz * rz
        abs_q /= np.clip(np.linalg.norm(abs_q, axis=1, keepdims=True), 1e-8, None)

        cam_abs_poses_urdf[camera_name] = np.concatenate((abs_t, abs_q), axis=1)

    return {
        "camera_intrinsics": camera_intrinsics,
        "trunk": trunk,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_grip": left_grip,
        "right_grip": right_grip,
        "base_poses": base_poses,
        "joint_values_matrix": joint_values_matrix,
        "num_frames": num_frames,
        "base_z": robot_root.location.z if BASE_Z is None else float(BASE_Z),
        "cam_abs_poses_urdf": cam_abs_poses_urdf,
    }


def main():
    import_urdf = load_import_urdf_func(IMPORT_URDF_SCRIPT_PATH)
    robot_root = import_urdf(URDF_PATH, clear_scene_first=CLEAR_SCENE_BEFORE_IMPORT)
    if robot_root is None:
        raise RuntimeError("URDF import returned no robot root object.")
    robot_root.rotation_mode = "XYZ"

    joint_defs = load_joint_defs(URDF_PATH)
    anim_joints = build_joint_anim_data(joint_defs)

    script_dir = ""
    if "__file__" in globals():
        candidate_script_dir = os.path.dirname(os.path.abspath(__file__))
        if candidate_script_dir and candidate_script_dir != os.path.sep:
            script_dir = candidate_script_dir

    replay_state = build_replay_state(robot_root, joint_defs, script_dir)
    camera_intrinsics = replay_state["camera_intrinsics"]
    cam_abs_poses_urdf = replay_state["cam_abs_poses_urdf"]
    num_frames = replay_state["num_frames"]
    base_poses = replay_state["base_poses"]
    joint_values_matrix = replay_state["joint_values_matrix"]
    base_z = replay_state["base_z"]

    cam_abs_pose_urdf_out_dir = (
        os.path.join(script_dir, "cam_abs_poses_urdf")
        if script_dir
        else os.path.join(os.path.dirname(os.path.abspath(PARQUET_PATH)), "cam_abs_poses_urdf")
    )
    os.makedirs(cam_abs_pose_urdf_out_dir, exist_ok=True)
    for camera_name in ("head", "left_wrist", "right_wrist"):
        np.savetxt(
            os.path.join(cam_abs_pose_urdf_out_dir, f"{camera_name}.csv"),
            cam_abs_poses_urdf[camera_name],
            delimiter=",",
            fmt="%.8f",
        )

    cam_pose_dict = cam_abs_poses_urdf_to_pose_dict(cam_abs_poses_urdf)
    replay_bundle = {
        "camera_intrinsics": camera_intrinsics,
        "cam_pose_dict": cam_pose_dict,
        "num_frames": num_frames,
        "start_frame": START_FRAME,
    }

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
                "num_frames": num_frames,
                "robot_root": robot_root,
                "anim_joints": anim_joints,
                "joint_values_matrix": joint_values_matrix,
                "base_poses": base_poses,
                "base_z": base_z,
            }
        )
        register_replay_handler()
        scene.frame_set(START_FRAME)
        print("Replay handler registered. Press Play for real-time robot playback.")
        return replay_bundle

    joint_count = len(anim_joints)
    if RENDER_CAMERA not in cam_abs_poses_urdf:
        raise ValueError(
            f"Invalid RENDER_CAMERA='{RENDER_CAMERA}'. Expected one of: {sorted(cam_abs_poses_urdf.keys())}"
        )

    render_intr = camera_intrinsics[RENDER_CAMERA]
    render_poses = cam_abs_poses_urdf[RENDER_CAMERA]
    render_out_dir = os.path.join(
        os.path.dirname(PARQUET_PATH), f"replay_viewport_{RENDER_CAMERA}"
    )
    os.makedirs(render_out_dir, exist_ok=True)

    render_cam_data = bpy.data.cameras.new(name=f"replay_{RENDER_CAMERA}_camera_data")
    render_cam_data.type = "PERSP"
    render_cam_data.sensor_fit = "HORIZONTAL"
    render_cam_data.sensor_width = float(render_intr["width"])
    render_cam_data.sensor_height = float(render_intr["height"])
    render_cam_data.lens = float(render_intr["fx"])
    render_cam_data.shift_x = (
        float(render_intr["cx"]) - 0.5 * float(render_intr["width"])
    ) / float(render_intr["width"])
    render_cam_data.shift_y = (
        0.5 * float(render_intr["height"]) - float(render_intr["cy"])
    ) / float(render_intr["height"])
    render_cam_data.clip_start = float(CAMERA_CLIP_START[RENDER_CAMERA])
    render_cam_data.clip_end = float(CAMERA_CLIP_END[RENDER_CAMERA])

    render_cam_obj = bpy.data.objects.new(
        name=f"replay_{RENDER_CAMERA}_camera", object_data=render_cam_data
    )
    bpy.context.collection.objects.link(render_cam_obj)
    render_cam_obj.rotation_mode = "QUATERNION"
    scene.camera = render_cam_obj

    scene.render.resolution_x = int(render_intr["width"])
    scene.render.resolution_y = int(render_intr["height"])
    scene.render.resolution_percentage = 100
    engine_items = scene.render.bl_rna.properties["engine"].enum_items.keys()
    if "BLENDER_EEVEE_NEXT" in engine_items:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    elif "BLENDER_EEVEE" in engine_items:
        scene.render.engine = "BLENDER_EEVEE"

    view_window = bpy.context.window
    view_screen = view_window.screen if view_window else None
    view_area = None
    view_region = None
    if view_screen is not None:
        for area in view_screen.areas:
            if area.type == "VIEW_3D":
                view_area = area
                break
    if view_area is not None:
        for region in view_area.regions:
            if region.type == "WINDOW":
                view_region = region
                break
    if view_area is None or view_region is None:
        raise RuntimeError("No VIEW_3D area found for viewport capture rendering.")
    view_space = view_area.spaces.active
    if view_space.type != "VIEW_3D":
        raise RuntimeError("Active VIEW_3D space not found for viewport capture rendering.")
    view_space.shading.type = "RENDERED"
    if hasattr(view_space.shading, "use_scene_lights_render"):
        view_space.shading.use_scene_lights_render = True
    if hasattr(view_space.shading, "use_scene_world_render"):
        view_space.shading.use_scene_world_render = True
    if hasattr(view_space.overlay, "show_overlays"):
        view_space.overlay.show_overlays = False
    view_space.camera = render_cam_obj
    if view_space.region_3d is not None:
        view_space.region_3d.view_perspective = "CAMERA"

    for i in range(num_frames):
        frame = START_FRAME + i
        scene.frame_set(frame)

        joint_vals = joint_values_matrix[i]
        if len(joint_vals) != joint_count:
            raise ValueError(
                f"Joint count mismatch: {len(joint_vals)} values for {joint_count} joints."
            )
        for anim_joint, value in zip(anim_joints, joint_vals):
            apply_joint_pose(anim_joint, float(value))

        x, y, yaw = base_poses[i]
        robot_root.location = (float(x), float(y), float(base_z))
        robot_root.rotation_euler = (0.0, 0.0, float(yaw))

        robot_root.keyframe_insert(data_path="location")
        robot_root.keyframe_insert(data_path="rotation_euler")
        for anim_joint in anim_joints:
            anim_joint["obj"].keyframe_insert(data_path="location")
            anim_joint["obj"].keyframe_insert(data_path="rotation_euler")

        cam_pose = render_poses[i]
        render_cam_obj.location = (float(cam_pose[0]), float(cam_pose[1]), float(cam_pose[2]))
        render_cam_obj.rotation_quaternion = (
            float(cam_pose[6]),
            float(cam_pose[3]),
            float(cam_pose[4]),
            float(cam_pose[5]),
        )
        scene.render.filepath = os.path.join(render_out_dir, f"{frame:06d}.png")
        bpy.context.view_layer.update()
        view_space.shading.type = "RENDERED"
        view_space.camera = render_cam_obj
        if view_space.region_3d is not None:
            view_space.region_3d.view_perspective = "CAMERA"
        with bpy.context.temp_override(
            window=view_window,
            screen=view_screen,
            area=view_area,
            region=view_region,
            space_data=view_space,
        ):
            bpy.ops.render.opengl(write_still=True, view_context=True)

    print(f"Replay complete. Keyframed {num_frames} frames from {PARQUET_PATH}")
    return replay_bundle


if __name__ == "__main__":
    main()
