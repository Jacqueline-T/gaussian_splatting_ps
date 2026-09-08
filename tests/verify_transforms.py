#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np
import open3d as o3d


# ============================================================
# LOAD TRANSFORMS — Reads transforms.json and verifies it contains
# at least one frame before returning.
# ============================================================
def load_transforms(path):
    with open(path, "r") as f:
        data = json.load(f)

    if "frames" not in data or len(data["frames"]) == 0:
        raise RuntimeError(f"No frames found in {path}")

    return data


# ============================================================
# GEOMETRY BUILDERS — Constructs the visual camera frustums and the
# connecting trajectory line drawn over the point cloud for inspection.
# ============================================================
def make_frustum(c2w, scale=0.05, color=(1.0, 0.0, 0.0)):
    """
    Build a small pyramid wireframe representing a camera frustum, in the
    camera's local frame, then transform it into world space via c2w.

    NeRF/OpenGL convention: camera looks down -Z, +Y is up, +X is right.
    So the frustum "tip" (the camera center) sits at the origin, and the
    base of the pyramid sits at -Z (in front of the camera, i.e. the
    direction it's actually looking).
    """
    # Camera-space corners of the frustum base, plus the apex at origin.
    apex = np.array([0.0, 0.0, 0.0])
    base = np.array(
        [
            [ 0.5,  0.375, -1.0],
            [ 0.5, -0.375, -1.0],
            [-0.5, -0.375, -1.0],
            [-0.5,  0.375, -1.0],
        ]
    ) * scale

    points_cam = np.vstack([apex, base])  # shape (5, 3)

    # Transform into world space.
    points_h = np.hstack([points_cam, np.ones((5, 1))])  # (5, 4)
    points_world = (c2w @ points_h.T).T[:, :3]  # (5, 3)

    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # apex to each base corner
        [1, 2], [2, 3], [3, 4], [4, 1],  # base rectangle
    ]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points_world),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


def make_trajectory_path(c2w_list, color=(0.0, 1.0, 0.0)):
    """Connects successive camera centers with a line, so you can see the
    path the camera actually traced through space."""
    centers = np.array([c2w[:3, 3] for c2w in c2w_list])

    lines = [[i, i + 1] for i in range(len(centers) - 1)]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(centers),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector([color] * len(lines))

    return line_set


# ============================================================
# OFFSCREEN RENDER — Renders the scene to a PNG instead of opening
# an interactive window. Use this if draw_geometries() fails to
# create an OpenGL window (e.g. under WSL/WSLg).
# ============================================================
def render_to_image(geometries, output_path, width=1280, height=800):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)

    for geometry in geometries:
        vis.add_geometry(geometry)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path, do_render=True)
    vis.destroy_window()

    print(f"Saved render to: {output_path}")


# ============================================================
# MAIN — Loads transforms.json and its point cloud, builds a frustum
# per pose plus the connecting trajectory path, and either opens an
# interactive viewer or renders a PNG (--headless) for verification.
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="Visually verify transforms.json poses against the point cloud."
    )
    parser.add_argument("--transforms", default="transforms.json")
    parser.add_argument("--ply", default=None, help="Override ply_file_path from transforms.json")
    parser.add_argument("--frustum-scale", type=float, default=0.05, help="Frustum size in meters")
    parser.add_argument("--every", type=int, default=1, help="Draw every Nth frustum (use >1 if cluttered)")
    parser.add_argument(
        "--headless", action="store_true",
        help="Render to a PNG file instead of opening an interactive window "
             "(use this if draw_geometries() fails, e.g. under WSL)."
    )
    parser.add_argument(
        "--screenshot-out", type=str, default="transforms_check.png",
        help="Output path for the rendered screenshot when --headless is set."
    )
    args = parser.parse_args()

    data = load_transforms(args.transforms)
    frames = data["frames"]

    ply_path = args.ply or data.get("ply_file_path")
    if ply_path is None:
        raise RuntimeError("No --ply given and no ply_file_path in transforms.json")

    if not os.path.exists(ply_path):
        raise FileNotFoundError(
            f"Point cloud not found: {ply_path}\n"
            f"(resolved relative to your current working directory -- "
            f"run this from the same directory as transform_pipe_2.py, "
            f"or pass --ply with the correct path)"
        )

    print(f"Loaded {len(frames)} frames from {args.transforms}")
    print(f"Loading point cloud: {ply_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    print(f"  Point cloud has {len(pcd.points)} points")

    c2w_list = [np.array(f["transform_matrix"]) for f in frames]

    geometries = [pcd]

    for i, c2w in enumerate(c2w_list):
        if i % args.every != 0:
            continue
        # First frame highlighted blue (start), rest red.
        color = (0.0, 0.0, 1.0) if i == 0 else (1.0, 0.0, 0.0)
        geometries.append(make_frustum(c2w, scale=args.frustum_scale, color=color))

    geometries.append(make_trajectory_path(c2w_list))

    # World-origin axes for orientation reference (red=X, green=Y, blue=Z).
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=args.frustum_scale * 2)
    geometries.append(axes)

    if args.headless:
        render_to_image(geometries, args.screenshot_out)
    else:
        print()
        print("Opening viewer.")
        print("  Blue frustum  = first frame (start of trajectory)")
        print("  Red frustums  = subsequent frames")
        print("  Green line    = camera path")
        print("  RGB axes      = world origin (red=X, green=Y, blue=Z)")
        print()
        print("What to check:")
        print("  - Frustums should sit OUTSIDE the point cloud, pointing INTO it")
        print("    (narrow tip = camera center, wide end = what it's looking at)")
        print("  - The green path should look like a plausible camera motion,")
        print("    not a random scatter or a straight line through the geometry")
        print("  - If frustums are embedded in the cloud or facing away from it,")
        print("    check opencv_to_opengl() in transform_pipe_2.py")

        o3d.visualization.draw_geometries(
            geometries,
            window_name="transforms.json verification",
            width=1280,
            height=800,
        )


if __name__ == "__main__":
    main()