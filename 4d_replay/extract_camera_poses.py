"""
Extract head/left/right camera poses from a task images.bin and write CSVs.
Applies OpenCV camera-frame -> Blender camera-frame conversion.

Usage:
  python extract_camera_poses.py <task_name>
"""

import argparse
import csv
import math
import os
import struct

import numpy as np

try:
    import pyarrow.parquet as pq
except Exception:
    pq = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract camera poses from task images.bin and write CSVs."
    )
    parser.add_argument(
        "task_name",
        help="Task folder name under this script directory (e.g. turning_on_radio).",
    )
    return parser.parse_args()


def find_images_bin(task_dir, exclude_dir=None):
    bin_paths = []
    for root, _dirs, files in os.walk(task_dir):
        if exclude_dir and os.path.commonpath([exclude_dir, root]) == exclude_dir:
            continue
        for name in files:
            if name == "images.bin":
                bin_paths.append(os.path.join(root, name))
    if not bin_paths:
        raise FileNotFoundError(f"No images.bin found under: {task_dir}")
    bin_paths.sort()
    if len(bin_paths) > 1:
        print(
            "Warning: multiple images.bin files found. Using the first one:",
            bin_paths[0],
        )
    return bin_paths[0]


def find_camera_bins(task_dir):
    bins = {}
    for cam_type in ("head", "left_wrist", "right_wrist"):
        path = os.path.join(task_dir, "camera_poses", cam_type, "images.bin")
        if os.path.exists(path):
            bins[cam_type] = path
    return bins


def find_parquet(task_dir):
    parquet_paths = []
    for root, _dirs, files in os.walk(task_dir):
        for name in files:
            if name.endswith(".parquet"):
                parquet_paths.append(os.path.join(root, name))
    parquet_paths.sort()
    return parquet_paths[0] if parquet_paths else None


def load_cam_rel_len(parquet_path):
    if pq is None or parquet_path is None:
        return None
    pf = pq.ParquetFile(parquet_path)
    cam_col = "observation.cam_rel_poses"
    if cam_col not in pf.schema_arrow.names:
        cam_col = "observation/cam_rel_poses"
    if cam_col not in pf.schema_arrow.names:
        return None
    return pf.metadata.num_rows if pf.metadata is not None else None


def read_next_bytes(fid, num_bytes, fmt, endian="<"):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError("Unexpected EOF while reading images.bin")
    return struct.unpack(endian + fmt, data)


def read_images_bin_wide(path):
    images = []
    with open(path, "rb") as f:
        num_images = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(f, 8, "Q")[0]
            qvec = read_next_bytes(f, 32, "dddd")  # qw, qx, qy, qz (world->camera)
            tvec = read_next_bytes(f, 24, "ddd")   # world->camera translation
            camera_id = read_next_bytes(f, 8, "Q")[0]
            name_bytes = bytearray()
            while True:
                c = read_next_bytes(f, 1, "c")[0]
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            num_points2D = read_next_bytes(f, 8, "Q")[0]
            if num_points2D:
                f.seek((16 + 8) * num_points2D, 1)
            images.append(
                {
                    "image_id": image_id,
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name,
                }
            )
    return images


def read_images_bin_colmap(path):
    images = []
    with open(path, "rb") as f:
        num_images = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(f, 4, "i")[0]
            qvec = read_next_bytes(f, 32, "dddd")  # qw, qx, qy, qz (world->camera)
            tvec = read_next_bytes(f, 24, "ddd")   # world->camera translation
            camera_id = read_next_bytes(f, 4, "i")[0]
            name_bytes = bytearray()
            while True:
                c = read_next_bytes(f, 1, "c")[0]
                if c == b"\x00":
                    break
                name_bytes += c
            name = name_bytes.decode("utf-8")
            num_points2D = read_next_bytes(f, 8, "Q")[0]
            if num_points2D:
                f.seek((16 + 8) * num_points2D, 1)
            images.append(
                {
                    "image_id": image_id,
                    "qvec": qvec,
                    "tvec": tvec,
                    "camera_id": camera_id,
                    "name": name,
                }
            )
    return images


def quat_to_R_wxyz(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def R_to_quat_wxyz(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return np.array([w, x, y, z], dtype=np.float64)


def parse_frame_index(name):
    base = os.path.basename(name)
    stem = os.path.splitext(base)[0]
    try:
        return int(stem)
    except ValueError:
        return None


def classify_camera(name):
    if "observation.images.rgb.head" in name:
        return "head"
    if "observation.images.rgb.left_wrist" in name:
        return "left_wrist"
    if "observation.images.rgb.right_wrist" in name:
        return "right_wrist"
    return None


def transform_pose_opencv_to_blender(qvec_wc, tvec_wc):
    # COLMAP: qvec/tvec are world->camera (OpenCV camera frame).
    # Convert to camera center in world coords.
    R_wc = quat_to_R_wxyz(qvec_wc)
    t_wc = np.array(tvec_wc, dtype=np.float64)
    cam_center = -R_wc.T @ t_wc

    # Camera-to-world rotation in OpenCV frame.
    R_cw = R_wc.T

    # Convert OpenCV camera frame to Blender camera frame:
    # x right, y down, z forward  ->  x right, y up, z back
    M = np.diag([1.0, -1.0, -1.0])
    R_cw_bl = R_cw @ M

    q_bl = R_to_quat_wxyz(R_cw_bl)
    # Additional quat remap requested for OpenCV->Blender alignment:
    # (qw, qx, qy, qz) = (qx, -qw, -qz, qy)
    q_bl = np.array([q_bl[1], -q_bl[0], -q_bl[3], q_bl[2]], dtype=np.float64)
    n = np.linalg.norm(q_bl)
    if n > 0:
        q_bl = q_bl / n
    if q_bl[0] < 0:
        q_bl = -q_bl
    return cam_center, q_bl


def write_camera_csv(out_path, rows):
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "x", "y", "z", "qw", "qx", "qy", "qz", "name"])
        for row in rows:
            writer.writerow(row)


def diagnose_index_alignment(label, rows, cam_rel_len):
    if not rows:
        print(f"Index check ({label}): no rows")
        return
    ids = sorted(int(r[0]) for r in rows)
    n = len(ids)
    min_id = ids[0]
    max_id = ids[-1]
    contiguous = all(ids[i] == min_id + i for i in range(n))
    offset = min_id - 1
    print(
        f"Index check ({label}): n={n}, min_id={min_id}, max_id={max_id}, "
        f"contiguous={contiguous}, offset(min_id-1)={offset}"
    )
    if cam_rel_len is not None:
        expected_len = max_id - offset
        if expected_len != cam_rel_len:
            print(
                f"  cam_rel_len={cam_rel_len}, expected_len_from_ids={expected_len} "
                "(potential misalignment)"
            )
        else:
            print(f"  cam_rel_len={cam_rel_len} (matches expected_len_from_ids)")


def main():
    args = parse_args()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    task_dir = os.path.join(base_dir, args.task_name)
    if not os.path.isdir(task_dir):
        raise FileNotFoundError(f"Task directory not found: {task_dir}")

    out_dir = os.path.join(task_dir, "camera_poses")
    os.makedirs(out_dir, exist_ok=True)
    head_out = os.path.join(out_dir, "head_camera_pose.csv")
    left_out = os.path.join(out_dir, "left_wrist_camera_pose.csv")
    right_out = os.path.join(out_dir, "right_wrist_camera_pose.csv")

    parquet_path = find_parquet(task_dir)
    cam_rel_len = load_cam_rel_len(parquet_path)
    if parquet_path is None:
        print("Index check: no parquet found; skipping cam_rel alignment diagnostics.")
    elif cam_rel_len is None:
        print("Index check: cam_rel_poses not found; skipping alignment diagnostics.")

    camera_bins = find_camera_bins(task_dir)
    if camera_bins:
        outputs = {"head": head_out, "left_wrist": left_out, "right_wrist": right_out}
        rows_by_type = {}
        for cam_type, bin_path in camera_bins.items():
            images = read_images_bin_colmap(bin_path)
            if not images:
                raise ValueError(f"images.bin is empty: {bin_path}")
            rows = []
            for img in images:
                frame_idx = img["image_id"]
                pos, quat = transform_pose_opencv_to_blender(img["qvec"], img["tvec"])
                rows.append(
                    [
                        frame_idx,
                        pos[0],
                        pos[1],
                        pos[2],
                        quat[0],
                        quat[1],
                        quat[2],
                        quat[3],
                        img["name"],
                    ]
                )
            rows.sort(key=lambda r: r[0])
            rows_by_type[cam_type] = rows
            write_camera_csv(outputs[cam_type], rows)
        for cam_type, rows in rows_by_type.items():
            diagnose_index_alignment(cam_type, rows, cam_rel_len)
    else:
        images_bin = find_images_bin(task_dir, exclude_dir=out_dir)
        images = read_images_bin_wide(images_bin)
        if not images:
            raise ValueError("images.bin is empty.")

        grouped = {"head": [], "left_wrist": [], "right_wrist": []}
        for img in images:
            cam_type = classify_camera(img["name"])
            if cam_type is None:
                continue
            frame_idx = parse_frame_index(img["name"])
            if frame_idx is None:
                continue
            pos, quat = transform_pose_opencv_to_blender(img["qvec"], img["tvec"])
            grouped[cam_type].append(
                [
                    frame_idx,
                    pos[0],
                    pos[1],
                    pos[2],
                    quat[0],
                    quat[1],
                    quat[2],
                    quat[3],
                    img["name"],
                ]
            )

        for cam_type in grouped:
            grouped[cam_type].sort(key=lambda r: r[0])

        write_camera_csv(head_out, grouped["head"])
        write_camera_csv(left_out, grouped["left_wrist"])
        write_camera_csv(right_out, grouped["right_wrist"])
        for cam_type, rows in grouped.items():
            diagnose_index_alignment(cam_type, rows, cam_rel_len)

    print("Wrote:")
    print(f"  {head_out}")
    print(f"  {left_out}")
    print(f"  {right_out}")


if __name__ == "__main__":
    main()
