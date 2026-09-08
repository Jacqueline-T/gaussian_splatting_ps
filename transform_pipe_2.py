#!/usr/bin/env python3

import argparse
import csv
import json
import os
import numpy as np


# ============================================================
# CONFIGURATION — Paths to validation_test.py's output (intrinsics,
# trajectory, trusted RGB frames) and fuse_pointcloud.py's output
# (the point cloud), plus where to write transforms.json.
# ============================================================
def parse_paths_args():
    parser = argparse.ArgumentParser(
        description="Export transforms.json for Nerfstudio splatfacto training.",
        add_help=False  # combined with parse_args() below
    )
    parser.add_argument("--input-dir", type=str, required=True,
                         help="Directory containing validation_test.py's output "
                              "(trajectory CSV, camera_intrinsics.json, trusted frames) "
                              "and fuse_pointcloud.py's output (fused_pointcloud.ply)")
    return parser

# (Paths resolved inside main() once args are parsed — see MAIN section)

# ============================================================
# COORDINATE CONVENTION — Fixed OpenCV-to-OpenGL/NeRF axis flip
# (negates Y and Z), applied to each frame's rotation before it's
# written to transforms.json.
# ============================================================
# OpenCV -> OpenGL/NeRF convention flip: negate Y and Z axes.
# Applied to the 3x3 rotation block of a camera-to-world matrix.
CV_TO_GL = np.diag([1.0, -1.0, -1.0])

def opencv_to_opengl(c2w_cv):
    """
    Convert a camera-to-world matrix from OpenCV convention (+Y down,
    +Z forward) to OpenGL/NeRF convention (+Y up, +Z backward), as
    required by transforms.json.

    This flips the Y and Z columns of the rotation block. Translation is
    unaffected since we're not changing the world frame, only which way
    the camera's local axes point within it.
    """
    c2w_gl = c2w_cv.copy()
    c2w_gl[:3, :3] = c2w_cv[:3, :3] @ CV_TO_GL
    return c2w_gl


# ============================================================
# LOADERS — Reads camera_intrinsics.json (validating all required
# fields are present) and reconstructs each trusted frame's camera-
# to-world pose as a 4x4 matrix from the trajectory CSV.
# ============================================================
def load_intrinsics(path):
    """Read camera_intrinsics.json. Fails loudly if width/height are null."""
    with open(path, "r") as f:
        data = json.load(f)

    for key in ("width", "height", "fx", "fy", "cx", "cy"):
        if data.get(key) is None:
            raise ValueError(f"Intrinsics missing required field: {key}")

    return data


def load_trajectory(path):
    """
    Reconstruct per-frame camera-to-world poses from the VO CSV.
    Actual columns (confirmed from rgbd_trajectory_diagnostic.csv header):
        frame, timestamp_ns, x_m, y_m, z_m, r00..r22 (row-major rotation)
    """
    trajectory = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            R = np.array([
                [float(row["r00"]), float(row["r01"]), float(row["r02"])],
                [float(row["r10"]), float(row["r11"]), float(row["r12"])],
                [float(row["r20"]), float(row["r21"]), float(row["r22"])],
            ])
            t = np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])])

            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = t

            trajectory.append({"frame_index": int(row["frame"]), "c2w": c2w})

    return trajectory


# ============================================================
# ASSEMBLE TRANSFORMS — Locates each trusted frame's RGB image and
# builds the full transforms.json dict: shared camera intrinsics plus
# one entry per frame with its OpenGL-convention pose and image path.
# ============================================================
def rgb_frame_path(frame_index, rgb_dir):
    """
    Locate the RGB PNG for a given trusted frame index. Adjust the filename
    pattern here if your VO script wrote frames under a different naming
    convention than zero-padded 4-digit indices.
    """
    filename = f"frame_{frame_index:04d}.png"
    full_path = os.path.join(rgb_dir, filename)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"Expected RGB frame not found: {full_path}")
    return full_path


def build_transforms(intrinsics, trajectory, rgb_dir, ply_path):
    """Assemble the full transforms.json dict per the nerfstudio-data schema."""
    transforms = {
        "fl_x": intrinsics["fx"],
        "fl_y": intrinsics["fy"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "w": intrinsics["width"],
        "h": intrinsics["height"],
        # Assumes rectified/undistorted frames (matches load_intrinsics'
        # PinholeCameraIntrinsic usage upstream in fuse_pointcloud.py).
        # If your depth/RGB pairs are NOT pre-rectified, replace these
        # zeros with actual distortion coefficients before training.
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "camera_model": "OPENCV",
        "ply_file_path": ply_path,
        "frames": [],
    }

    for entry in trajectory:
        frame_index = entry["frame_index"]
        c2w_gl = opencv_to_opengl(entry["c2w"])
        image_path = rgb_frame_path(frame_index, rgb_dir)

        transforms["frames"].append({
            "file_path": image_path,
            "transform_matrix": c2w_gl.tolist(),
        })

    return transforms

# ============================================================
# MAIN — Parses CLI args, loads intrinsics/trajectory, optionally
# limits to a smoke-test subset, builds transforms.json, and prints
# the next Nerfstudio command to run.
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Export transforms.json for Nerfstudio splatfacto training."
    )
    parser.add_argument(
        "--input-dir", type=str, required=True,
        help="Directory containing validation_test.py's output "
             "(trajectory CSV, camera_intrinsics.json, trusted frames) "
             "and fuse_pointcloud.py's output (fused_pointcloud.ply)"
    )
    parser.add_argument(
        "--limit-frames", type=int, default=0, metavar="N",
        help=(
            "Smoke-test mode: only export the first N trusted frames "
            "(e.g. --limit-frames 20), instead of all frames. "
            "Use this to sanity-check camera poses/convention in the "
            "Nerfstudio viewer before committing to the full trajectory. "
            "Omit or set to 0 for the full run."
        ),
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for transforms.json (default: <input-dir>/transforms.json).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    intrinsics_path = input_dir / "camera_intrinsics.json"
    trajectory_path = input_dir / "rgbd_trajectory_diagnostic.csv"
    rgb_dir = input_dir / "odometry_trusted_frames" / "rgb"
    ply_path = input_dir / "fused_pointcloud.ply"
    output_path = Path(args.output) if args.output else input_dir / "transforms.json"

    intrinsics = load_intrinsics(intrinsics_path)
    trajectory = load_trajectory(trajectory_path)

    total_frames = len(trajectory)
    print(f"Loaded {total_frames} trusted frames from trajectory CSV.")

    if args.limit_frames > 0:
        trajectory = trajectory[: args.limit_frames]
        print(
            f"--limit-frames {args.limit_frames} set: "
            f"exporting only the first {len(trajectory)} frames (smoke test)."
        )

    transforms = build_transforms(intrinsics, trajectory, str(rgb_dir), str(ply_path))

    with open(output_path, "w") as f:
        json.dump(transforms, f, indent=2)

    print(f"Wrote {output_path} with {len(transforms['frames'])} frames.")
    print(f"ply_file_path set to: {ply_path}")

    if args.limit_frames > 0:
        print(
            "\nThis is a smoke-test export. Once poses/orientation look "
            f"correct in the viewer, re-run without --limit-frames for the "
            f"full {total_frames}-frame export."
        )

    print("\nNext step:")
    print(f"  ns-train splatfacto --data <dir containing {output_path.name}>")


if __name__ == "__main__":
    main()