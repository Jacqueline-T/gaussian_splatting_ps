#!/usr/bin/env python3

import os
import csv
import json

import cv2
import numpy as np
import open3d as o3d

import argparse
from pathlib import Path

# ============================================================
# CONFIGURATION — Paths to validation_test.py's output (trajectory,
# intrinsics, trusted frames) and where to write the fused point cloud.
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="RGB-D TSDF fusion")
    parser.add_argument("--input-dir", type=str, required=True,
                         help="Directory containing validation_test.py's output "
                              "(trajectory CSV, camera_intrinsics.json, trusted frames)")
    parser.add_argument("--output-ply", type=str, default=None,
                         help="Path to write the fused point cloud "
                              "(default: <input-dir>/fused_pointcloud.ply)")
    return parser.parse_args()

args = parse_args()

INPUT_DIR = Path(args.input_dir)

TRAJECTORY_FILE = INPUT_DIR / "rgbd_trajectory_diagnostic.csv"
INTRINSICS_FILE = INPUT_DIR / "camera_intrinsics.json"
TRUSTED_DIR = INPUT_DIR / "odometry_trusted_frames"

OUTPUT_PLY = Path(args.output_ply) if args.output_ply else INPUT_DIR / "fused_pointcloud.ply"


# ============================================================
# TSDF FUSION PARAMETERS — Controls voxel resolution, surface
# truncation distance, and depth range/units for volumetric fusion,
# plus statistical outlier removal on the extracted point cloud.
# ============================================================
VOXEL_SIZE = 0.005   # 5 mm
SDF_TRUNC = 0.02     # 20 mm

# Depth values are stored as 16-bit PNGs in millimeters.
DEPTH_SCALE = 1000.0

# Ignore depth beyond this range during integration. The camera's
# observed depth range in this bag topped out around 2.7m, so
# 3.0m gives headroom without keeping unreliable far-range noise.
# (Tighter than MAX_DEPTH in validation_test.py (5.0m) — that's
# intentional, based on what this specific bag actually contained.)
DEPTH_TRUNC = 3.0    # meters

# Outlier removal after fusion.
REMOVE_OUTLIERS = True
OUTLIER_NEIGHBORS = 20
OUTLIER_STD_RATIO = 2.0


# ============================================================
# LOAD INTRINSICS — Reads camera_intrinsics.json and builds an
# Open3D PinholeCameraIntrinsic, failing clearly if width/height
# weren't recorded (e.g. no CameraInfo in the source bag).
# ============================================================
def load_intrinsics(path):
    with open(path, "r") as f:
        data = json.load(f)

    if data["width"] is None or data["height"] is None:
        raise RuntimeError(
            "camera_intrinsics.json is missing width/height. "
            "Cannot build a PinholeCameraIntrinsic without them."
        )

    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(data["width"]),
        height=int(data["height"]),
        fx=data["fx"],
        fy=data["fy"],
        cx=data["cx"],
        cy=data["cy"]
    )

    print()
    print("=" * 60)
    print("LOADED INTRINSICS")
    print("=" * 60)
    print(f"  width  = {data['width']}")
    print(f"  height = {data['height']}")
    print(f"  fx     = {data['fx']}")
    print(f"  fy     = {data['fy']}")
    print(f"  cx     = {data['cx']}")
    print(f"  cy     = {data['cy']}")

    return intrinsic


# ============================================================
# LOAD TRAJECTORY — Reads the trajectory CSV from validation_test.py
# and reconstructs each trusted frame's camera-to-world pose (full
# rotation matrix + position) for use during fusion.
# ============================================================
def load_trajectory(path):
    """
    Returns a list of dicts:
        {
            "frame": int,
            "timestamp_ns": int,
            "R_world_camera": 3x3 np.ndarray,
            "t_world_camera": 3, np.ndarray
        }

    R_world_camera / t_world_camera describe the camera's pose IN
    the world frame (camera-to-world), matching how validation_test.py
    computed global_R / global_t.
    """
    entries = []

    with open(path, newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            R = np.array([
                [float(row["r00"]), float(row["r01"]), float(row["r02"])],
                [float(row["r10"]), float(row["r11"]), float(row["r12"])],
                [float(row["r20"]), float(row["r21"]), float(row["r22"])]
            ], dtype=np.float64)

            t = np.array([
                float(row["x_m"]), float(row["y_m"]), float(row["z_m"])
            ], dtype=np.float64)

            entries.append({
                "frame": int(row["frame"]),
                "timestamp_ns": int(row["timestamp_ns"]),
                "R_world_camera": R,
                "t_world_camera": t
            })

    print()
    print("=" * 60)
    print("LOADED TRAJECTORY")
    print("=" * 60)
    print(f"  Trusted frames: {len(entries)}")

    return entries


# ============================================================
# POSE HELPERS — Inverts a camera-to-world pose into the world-to-
# camera extrinsic TSDF fusion expects, using the rotation-transpose
# shortcut rather than a general matrix inverse.
# ============================================================
def world_to_camera_extrinsic(R_world_camera, t_world_camera):
    """
    TSDF integration wants the extrinsic as the WORLD-TO-CAMERA
    transform (a standard camera extrinsic matrix), while our
    trajectory stores the CAMERA-TO-WORLD pose.

    So we invert:

        P_world = R_world_camera @ P_camera + t_world_camera
        =>
        P_camera = R_world_camera.T @ (P_world - t_world_camera)
        P_camera = R_world_camera.T @ P_world
                   - R_world_camera.T @ t_world_camera
    """
    R_inv = R_world_camera.T
    t_inv = -R_inv @ t_world_camera

    extrinsic = np.eye(4, dtype=np.float64)
    extrinsic[:3, :3] = R_inv
    extrinsic[:3, 3] = t_inv

    return extrinsic


# ============================================================
# LOAD ONE RGBD FRAME — Reads a trusted frame's RGB/depth PNG pair
# from disk and packages them into an Open3D RGBDImage, applying
# depth scale/truncation for fusion.
# ============================================================
def load_rgbd_frame(frame_index):
    rgb_path = os.path.join(TRUSTED_DIR, "rgb", f"frame_{frame_index:04d}.png")
    depth_path = os.path.join(TRUSTED_DIR, "depth", f"frame_{frame_index:04d}.png")

    bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Missing RGB frame: {rgb_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    depth_mm = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth_mm is None:
        raise FileNotFoundError(f"Missing depth frame: {depth_path}")

    color_o3d = o3d.geometry.Image(rgb)
    depth_o3d = o3d.geometry.Image(depth_mm)

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=DEPTH_SCALE,
        depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=False
    )

    return rgbd


# ============================================================
# FUSION — Integrates every trusted frame's RGB-D data into a shared
# TSDF volume, using each frame's world-to-camera extrinsic to place
# it correctly in the combined 3D reconstruction.
# ============================================================
def fuse(trajectory, intrinsic):
    print()
    print("=" * 60)
    print("FUSING TSDF VOLUME")
    print("=" * 60)
    print(f"  Voxel size:  {VOXEL_SIZE} m")
    print(f"  SDF trunc:   {SDF_TRUNC} m")
    print(f"  Depth trunc: {DEPTH_TRUNC} m")

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=VOXEL_SIZE,
        sdf_trunc=SDF_TRUNC,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )

    for count, entry in enumerate(trajectory, start=1):
        frame_index = entry["frame"]
        rgbd = load_rgbd_frame(frame_index)

        extrinsic = world_to_camera_extrinsic(
            entry["R_world_camera"], entry["t_world_camera"]
        )

        volume.integrate(rgbd, intrinsic, extrinsic)

        if count % 100 == 0 or count == len(trajectory):
            print(f"  Integrated {count}/{len(trajectory)} frames (frame index {frame_index})")

    return volume

# ============================================================
# EXTRACT + CLEAN POINT CLOUD — Converts the fused TSDF volume into
# an actual point cloud, then removes statistical outliers (likely
# noise) before the result is saved.
# ============================================================
def extract_point_cloud(volume):
    print()
    print("=" * 60)
    print("EXTRACTING POINT CLOUD")
    print("=" * 60)

    pcd = volume.extract_point_cloud()
    print(f"  Raw points: {len(pcd.points)}")

    if REMOVE_OUTLIERS:
        pcd, _ = pcd.remove_statistical_outlier(
            nb_neighbors=OUTLIER_NEIGHBORS,
            std_ratio=OUTLIER_STD_RATIO
        )
        print(f"  Points after outlier removal: {len(pcd.points)}")

    return pcd


# ============================================================
# MAIN — Runs fusion end to end: load intrinsics and trajectory,
# fuse trusted frames into a TSDF volume, extract and clean the
# point cloud, and save it to disk.
# ============================================================
def main():
    intrinsic = load_intrinsics(INTRINSICS_FILE)
    trajectory = load_trajectory(TRAJECTORY_FILE)

    if len(trajectory) == 0:
        raise RuntimeError("Trajectory is empty — nothing to fuse.")

    volume = fuse(trajectory, intrinsic)
    pcd = extract_point_cloud(volume)

    o3d.io.write_point_cloud(str(OUTPUT_PLY), pcd)

    print()
    print("=" * 60)
    print("FUSION COMPLETE")
    print("=" * 60)
    print(f"  Point cloud saved to: {OUTPUT_PLY}")
    print(f"  Final point count:    {len(pcd.points)}")


if __name__ == "__main__":
    main()