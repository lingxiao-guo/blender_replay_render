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

import bpy
from mathutils import Matrix, Quaternion, Euler


# ----------------------
# User configuration
# ----------------------
RUN_IMPORT_SCENE = False
RUN_IMPORT_URDF = True

USE_LEFT_WRIST_RENDER_CAMERA = True
LEFT_WRIST_CAMERA_CSV_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/picking_up_trash/camera_poses/left_wrist_camera_pose.csv"
LEFT_WRIST_CAMERAS_BIN_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/picking_up_trash/camera_poses/left_wrist/cameras.bin"
RENDER_CAMERA_NAME = "LeftWristRenderCamera"
SAVE_CAMERA_RENDER = True
RENDER_OUTPUT_PATH = "/home/lingxiao/Downloads/blender-5.0.1-linux-x64/left_wrist_render/"
USE_OPENCV_CAMERA_POSE = True
CAMERA_CLIP_START = 0.05
CAMERA_CLIP_END = 1000.0
RENDER_CAMERA_ROT_OFFSET = (3.141592653589793, 0.0, 0.0)


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


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cwd_dir = os.getcwd()
    for p in (script_dir, cwd_dir):
        if p and p not in sys.path:
            sys.path.insert(0, p)

    def load_local_module(name, filename):
        candidate_dirs = [
            script_dir,
            cwd_dir,
            os.path.join(cwd_dir, "4d_replay"),
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
    replay_r1pro_trajectory.main()
    scene = bpy.context.scene
    start_a = scene.frame_start
    end_a = scene.frame_end

    # Replay object trajectory
    replay_object = load_local_module("replay_object", "replay_object.py")
    replay_object.main()
    start_b = scene.frame_start
    end_b = scene.frame_end

    scene.frame_start = min(start_a, start_b)
    scene.frame_end = max(end_a, end_b)

    if USE_LEFT_WRIST_RENDER_CAMERA:
        if not os.path.exists(LEFT_WRIST_CAMERA_CSV_PATH):
            raise FileNotFoundError(LEFT_WRIST_CAMERA_CSV_PATH)
        if not os.path.exists(LEFT_WRIST_CAMERAS_BIN_PATH):
            raise FileNotFoundError(LEFT_WRIST_CAMERAS_BIN_PATH)
        cam_info = load_cameras_bin(LEFT_WRIST_CAMERAS_BIN_PATH)
        render_cam = ensure_render_camera(
            RENDER_CAMERA_NAME,
            cam_info["fx"],
            cam_info["fy"],
            cam_info["cx"],
            cam_info["cy"],
            cam_info["width"],
            cam_info["height"],
        )
        cam_poses = load_camera_poses(LEFT_WRIST_CAMERA_CSV_PATH)
        scene.camera = render_cam
        scene.render.resolution_x = cam_info["width"]
        scene.render.resolution_y = cam_info["height"]
        scene.render.resolution_percentage = 100

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
            scene.render.filepath = RENDER_OUTPUT_PATH
            bpy.ops.render.render(animation=True, use_viewport=False)

    print("Replay complete: robot + object.")


if __name__ == "__main__":
    main()
