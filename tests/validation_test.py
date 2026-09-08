#!/usr/bin/env python3

import os
import csv
import json
import math
import sqlite3

import cv2
import numpy as np

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message

import argparse
from pathlib import Path

# ============================================================
# CONFIGURATION — Bag path, ROS2 topics, output file paths,
# and frame processing limits for the odometry pipeline.
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="RGB-D odometry pipeline")
    parser.add_argument("--bag-path", type=str, required=True,
                         help="Path to the input rosbag directory")
    parser.add_argument("--output-dir", type=str, default="./output",
                         help="Directory to write trajectory/diagnostic/debug outputs to")
    parser.add_argument("--max-frames", type=int, default=None,
                         help="Number of frame transitions to process (default: full bag)")
    return parser.parse_args()

args = parse_args()

BAG_PATH = args.bag_path
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RGB_TOPIC = "/camera/rgb/image_raw"
DEPTH_TOPIC = "/camera/depth_registered/image_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

TRAJECTORY_FILE = OUTPUT_DIR / "rgbd_trajectory_diagnostic.csv"
DIAGNOSTIC_FILE = OUTPUT_DIR / "rgbd_odometry_diagnostics.csv"
DEBUG_DIR = OUTPUT_DIR / "odometry_debug_frames"

TRUSTED_DIR = OUTPUT_DIR / "odometry_trusted_frames"
INTRINSICS_FILE = OUTPUT_DIR / "camera_intrinsics.json"

# Number of frame transitions to process.
# None = entire bag.
MAX_FRAMES = args.max_frames


# ============================================================
# ORB (Oriented FAST and Rotated BRIEF) — feature detector/descriptor
# used for frame-to-frame keypoint matching in the odometry step
# ============================================================
ORB_FEATURES = 1500        # max keypoints to detect per frame
ORB_SCALE_FACTOR = 1.2     # pyramid scale factor between levels
ORB_LEVELS = 8              # number of pyramid levels


# ============================================================
# MATCHING — Thresholds for pairing ORB keypoints across frames,
# lifting matches to 3D, and validating the RANSAC-estimated
# camera motion before a pose is trusted.
# ============================================================
LOWE_RATIO = 0.75

# Minimum number of usable 3D correspondences.
MIN_3D_MATCHES = 20

# Minimum RANSAC inliers.
MIN_INLIERS = 50

# Minimum fraction of 3D correspondences that must agree.
MIN_INLIER_RATIO = 0.50

# Maximum 3D correspondence residual used by RANSAC.
RANSAC_THRESHOLD = 0.03  # meters

# Maximum physically plausible motion between adjacent frames.
MAX_TRANSLATION = 0.05  # meters

MAX_ROTATION_DEG = 5.0


# ============================================================
# DEPTH — Valid depth range (meters) a reading must fall in to be
# trusted; readings outside this window are discarded before they
# reach the matching/RANSAC stage.
# ============================================================
MIN_DEPTH = 0.40
MAX_DEPTH = 5.0


# ============================================================
# CAMERA INTRINSICS — Fallback focal length/principal point used to
# back-project 2D pixels + depth into 3D points; overridden by the
# bag's recorded CameraInfo when present.
# ============================================================
FX = 570.3422241210938
FY = 570.3422241210938
CX = 319.5
CY = 239.5

# Fallback image resolution, matching the resolution these
# intrinsics were derived at. Overridden by the bag's recorded
# CameraInfo when present.
WIDTH = 640
HEIGHT = 480


# ============================================================
# GLOBAL ACTIVE INTRINSICS — Working copies of the intrinsics above,
# initialized to the fallback values and overwritten at runtime if
# the bag's CameraInfo provides real ones.
# ============================================================
active_fx = FX
active_fy = FY
active_cx = CX
active_cy = CY

active_width = None
active_height = None


# ============================================================
# HELPERS — Low-level building blocks: image/depth decoding,
# pixel-to-3D back-projection, point transforms, and rotation-
# angle extraction, used by the matching/RANSAC pipeline above.
# ============================================================
def rotation_matrix_to_angle(R):
    """
    Return rotation angle in degrees.
    """
    trace = np.trace(R)

    value = (trace - 1.0) / 2.0
    value = np.clip(value, -1.0, 1.0)

    angle = math.acos(value)

    return math.degrees(angle)

def image_message_to_numpy(msg):
    """
    Convert ROS sensor_msgs/Image to NumPy.

    Supported encodings:
        rgb8
        bgr8
        32FC1
        16UC1
        mono8
        mono16
    """
    width = msg.width
    height = msg.height
    encoding = msg.encoding

    if encoding == "rgb8":
        dtype = np.uint8
        channels = 3
    elif encoding == "bgr8":
        dtype = np.uint8
        channels = 3
    elif encoding == "32FC1":
        dtype = np.float32
        channels = 1
    elif encoding == "16UC1":
        dtype = np.uint16
        channels = 1
    elif encoding == "mono8":
        dtype = np.uint8
        channels = 1
    elif encoding == "mono16":
        dtype = np.uint16
        channels = 1
    else:
        raise ValueError(f"Unsupported image encoding: {encoding}")

    array = np.frombuffer(msg.data, dtype=dtype)

    expected = height * width * channels

    if array.size < expected:
        raise ValueError(
            f"Image data too small: "
            f"{array.size} values, expected {expected}"
        )

    array = array[:expected]

    if channels == 3:
        array = array.reshape((height, width, 3))
    else:
        array = array.reshape((height, width))

    return array.copy()


def depth_to_meters(depth):
    """
    Convert depth image to meters.

    For this bag:
        32FC1 = already meters

    For completeness:
        16UC1 = millimeters
    """
    if depth.dtype == np.float32:
        return depth.astype(np.float32, copy=False)

    if depth.dtype == np.uint16:
        return (depth.astype(np.float32) / 1000.0)

    raise ValueError(f"Unsupported depth dtype: {depth.dtype}")

def get_depth(depth_m, u, v):
    """
    Get valid depth at pixel coordinate.
    """
    h, w = depth_m.shape

    u = int(round(u))
    v = int(round(v))

    if u < 0 or u >= w:
        return None

    if v < 0 or v >= h:
        return None

    d = float(depth_m[v, u])

    if not np.isfinite(d):
        return None

    if d < MIN_DEPTH or d > MAX_DEPTH:
        return None

    return d

def pixel_to_3d(u, v, depth):
    """
    Back-project pixel into camera coordinates.

    X = right
    Y = down
    Z = forward
    """
    X = ((u - active_cx) * depth / active_fx)

    Y = ((v - active_cy) * depth / active_fy)

    Z = depth

    return np.array([X, Y, Z], dtype=np.float64)

def transform_points(points, R, t):
    """
    Transform Nx3 points.
    """
    return ((R @ points.T).T + t.reshape(1, 3))


# ============================================================
# ROS BAG — Locates and reads the .db3 bag file, verifies the
# required RGB/depth topics are present, and extracts RGB frames,
# depth frames, and CameraInfo (if recorded) in timestamp order.
# ============================================================
def get_db3_file(bag_path):
    if os.path.isfile(bag_path):
        return bag_path

    if not os.path.isdir(bag_path):
        raise FileNotFoundError(f"Bag path does not exist:\n{bag_path}")

    db3_files = [
        f
        for f in os.listdir(bag_path)
        if f.endswith(".db3")
    ]

    if not db3_files:
        raise FileNotFoundError(f"No .db3 file found in:\n{bag_path}")

    return os.path.join(bag_path, sorted(db3_files)[0])


def read_bag(bag_path):
    db3_file = get_db3_file(bag_path)

    print()
    print("=" * 60)
    print("RGB-D ODOMETRY — DIAGNOSTIC")
    print("=" * 60)

    print()
    print("Bag:")
    print(bag_path)

    print()
    print("Database:")
    print(db3_file)

    conn = sqlite3.connect(db3_file)

    cursor = conn.cursor()

    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------
    cursor.execute(
        "SELECT id, name, type FROM topics"
    )

    topics = cursor.fetchall()

    topic_map = {}

    print()
    print("=" * 60)
    print("TOPICS FOUND IN BAG")
    print("=" * 60)

    for topic_id, name, msg_type in topics:

        topic_map[name] = (
            topic_id,
            msg_type
        )

        print(f"ID={topic_id:3d} | {name}")

        print(f"          Type: {msg_type}")

    # --------------------------------------------------------
    # Message counts
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("MESSAGE COUNTS FROM DATABASE")
    print("=" * 60)

    cursor.execute(
        """
        SELECT topic_id, COUNT(*)
        FROM messages
        GROUP BY topic_id
        ORDER BY topic_id
        """
    )

    topic_names_by_id = {
        topic_id: name
        for name, (topic_id, _) in topic_map.items()
    }

    for topic_id, count in cursor.fetchall():
        print(
            f"ID={topic_id:3d} | "
            f"{count:5d} messages | "
            f"{topic_names_by_id.get(topic_id, 'UNKNOWN')}"
        )

    # --------------------------------------------------------
    # Required topics
    # --------------------------------------------------------
    if RGB_TOPIC not in topic_map:
        raise RuntimeError(f"RGB topic not found: {RGB_TOPIC}")

    if DEPTH_TOPIC not in topic_map:
        raise RuntimeError(
            f"Depth topic not found: {DEPTH_TOPIC}"
        )

    rgb_topic_id, rgb_type = topic_map[RGB_TOPIC]

    depth_topic_id, depth_type = topic_map[DEPTH_TOPIC]

    camera_info_topic_id = None
    camera_info_type = None

    if CAMERA_INFO_TOPIC in topic_map:
        camera_info_topic_id, camera_info_type = (
            topic_map[CAMERA_INFO_TOPIC]
        )

    print()
    print("=" * 60)
    print("SELECTED TOPICS")
    print("=" * 60)

    print(f"RGB topic ID:   {rgb_topic_id}")

    print(f"RGB type:       {rgb_type}")

    print(f"Depth topic ID: {depth_topic_id}")

    print(f"Depth type:     {depth_type}")

    if camera_info_topic_id is not None:
        print(f"CameraInfo ID:  {camera_info_topic_id}")

        print(f"CameraInfo type:{camera_info_type}")

    else:

        print("CameraInfo:     NOT FOUND")

    # --------------------------------------------------------
    # ROS message classes
    # --------------------------------------------------------
    RGBMsg = get_message(rgb_type)

    DepthMsg = get_message(depth_type)

    CameraInfoMsg = None

    if camera_info_type is not None:
        CameraInfoMsg = get_message(camera_info_type)

    # --------------------------------------------------------
    # Read messages
    # --------------------------------------------------------
    print()
    print("Reading bag...")

    rgb_frames = []
    depth_frames = []

    camera_info = None

    cursor.execute(
        """
        SELECT topic_id, timestamp, data
        FROM messages
        ORDER BY timestamp
        """
    )

    for topic_id, timestamp, data in cursor:
        if topic_id == rgb_topic_id:
            msg = deserialize_message(data, RGBMsg)

            rgb_frames.append((timestamp, msg))
        elif topic_id == depth_topic_id:
            msg = deserialize_message(data, DepthMsg)

            depth_frames.append((timestamp, msg))

        elif (
            camera_info_topic_id is not None
            and topic_id == camera_info_topic_id
            and camera_info is None
        ):

            camera_info = deserialize_message(data, CameraInfoMsg)

    conn.close()

    print()
    print("Frames:")
    print(f"  RGB:        {len(rgb_frames)}")

    print(f"  Depth:      {len(depth_frames)}")

    print(
        f"  CameraInfo: "
        f"{1 if camera_info is not None else 0}"
    )

    return (
        rgb_frames,
        depth_frames,
        camera_info
    )


# ============================================================
# CAMERA INFO — Overrides the fallback intrinsics with the bag's
# recorded CameraInfo (if present and valid), then persists whichever
# values are active to camera_intrinsics.json for downstream stages.
# ============================================================
def configure_camera_info(camera_info, image_width=WIDTH, image_height=HEIGHT):
    global active_fx
    global active_fy
    global active_cx
    global active_cy
    global active_width
    global active_height

    print()
    print("=" * 60)
    print("CAMERA INTRINSICS")
    print("=" * 60)

    if camera_info is not None:
        # ROS CameraInfo K matrix:
        #
        # [fx  0 cx]
        # [ 0 fy cy]
        # [ 0  0  1]

        recorded_fx = float(camera_info.k[0])

        recorded_fy = float(camera_info.k[4])

        recorded_cx = float(camera_info.k[2])

        recorded_cy = float(camera_info.k[5])

        print("Using recorded CameraInfo.")

        print(f"Recorded fx = {recorded_fx}")

        print(f"Recorded fy = {recorded_fy}")

        print(f"Recorded cx = {recorded_cx}")

        print(f"Recorded cy = {recorded_cy}")

        if (
            recorded_fx > 0
            and recorded_fy > 0
        ):

            active_fx = recorded_fx
            active_fy = recorded_fy
            active_cx = recorded_cx
            active_cy = recorded_cy

        if camera_info.width:
            image_width = camera_info.width

        if camera_info.height:
            image_height = camera_info.height

    else:
        print("CameraInfo not found.")

        print("Using fallback intrinsics.")

    active_width = image_width
    active_height = image_height

    print()
    print(f"Active fx = {active_fx}")

    print(f"Active fy = {active_fy}")

    print(f"Active cx = {active_cx}")

    print(f"Active cy = {active_cy}")

    print(f"Active width  = {active_width}")

    print(f"Active height = {active_height}")

    # --------------------------------------------------------
    # Persist intrinsics for downstream export (transforms.json,
    # COLMAP cameras.txt, etc.) — not just printed to stdout.
    # --------------------------------------------------------
    with open(INTRINSICS_FILE, "w") as f:

        json.dump(
            {
                "fx": active_fx,
                "fy": active_fy,
                "cx": active_cx,
                "cy": active_cy,
                "width": active_width,
                "height": active_height
            },
            f, indent=2
        )

    print()
    print(f"Intrinsics saved to: {INTRINSICS_FILE}")


# ============================================================
# RGB-D SYNCHRONIZATION — Pairs each RGB frame with its closest-
# timestamped depth frame, discarding pairs whose gap exceeds
# max_difference_ns so downstream stages never see mismatched data.
# ============================================================
def synchronize_frames(rgb_frames, depth_frames):
    print()
    print("=" * 60)
    print("SYNCHRONIZING RGB-D")
    print("=" * 60)

    pairs = []

    depth_index = 0

    # Maximum allowed RGB/depth timestamp gap for a pair to be
    # trusted. Loosen this if your camera's RGB and depth streams
    # drift further apart than 20ms.
    max_difference_ns = 20_000_000

    for rgb_timestamp, rgb_msg in rgb_frames:
        while (
            depth_index + 1 < len(depth_frames)
            and depth_frames[
                depth_index + 1
            ][0] < rgb_timestamp
        ):
            depth_index += 1

        candidates = []

        if depth_index < len(depth_frames):
            candidates.append(depth_frames[depth_index])

        if (depth_index + 1 < len(depth_frames)):
            candidates.append(depth_frames[depth_index + 1])

        if not candidates:
            continue

        best = min(
            candidates,
            key=lambda x:
            abs(x[0] - rgb_timestamp)
        )

        depth_timestamp, depth_msg = best

        difference = abs(rgb_timestamp - depth_timestamp)

        if difference <= max_difference_ns:
            pairs.append((rgb_timestamp, rgb_msg, depth_timestamp, depth_msg))

    print()
    print(
        f"Matched RGB-D pairs: "
        f"{len(pairs)}"
    )

    if pairs:
        differences_ms = [
            abs(p[0] - p[2]) / 1e6
            for p in pairs
        ]

        print(
            f"Minimum timestamp difference: "
            f"{min(differences_ms):.3f} ms"
        )

        print(
            f"Maximum timestamp difference: "
            f"{max(differences_ms):.3f} ms"
        )

        print(
            f"Average timestamp difference: "
            f"{np.mean(differences_ms):.3f} ms"
        )

        print(
            f"Median timestamp difference: "
            f"{np.median(differences_ms):.3f} ms"
        )

    return pairs


# ============================================================
# ORB — Detects and matches keypoints between consecutive frames:
# builds the ORB detector, converts frames to grayscale, and applies
# Lowe's ratio test to reject ambiguous matches before 3D lifting.
# ============================================================
def create_orb():
    return cv2.ORB_create(nfeatures=ORB_FEATURES, scaleFactor=ORB_SCALE_FACTOR, nlevels=ORB_LEVELS)

def prepare_rgb(rgb):
    if rgb.ndim == 3:

        if rgb.shape[2] == 3:
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        return rgb[:, :, 0]

    return rgb

def match_features(prev_gray, curr_gray, orb):
    kp1, des1 = orb.detectAndCompute(prev_gray, None)

    kp2, des2 = orb.detectAndCompute(curr_gray, None)

    if des1 is None or des2 is None:
        return (kp1 if kp1 is not None else [], kp2 if kp2 is not None else [], [], 0)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    knn_matches = matcher.knnMatch(des1, des2, k=2)

    good_matches = []

    for pair in knn_matches:
        if len(pair) < 2:
            continue

        m, n = pair

        if (m.distance < LOWE_RATIO * n.distance):
            good_matches.append(m)

    return (kp1, kp2, good_matches, len(good_matches))


# ============================================================
# 3D CORRESPONDENCES — Converts matched 2D keypoint pairs into 3D
# point pairs via depth lookup and back-projection, dropping any
# match whose depth is invalid at either end.
# ============================================================
def build_3d_correspondences(kp1, kp2, matches, depth1, depth2):
    points1 = []
    points2 = []

    for match in matches:
        u1, v1 = kp1[match.queryIdx].pt
        u2, v2 = kp2[match.trainIdx].pt

        d1 = get_depth(depth1, u1, v1)
        d2 = get_depth(depth2, u2, v2)

        if d1 is None or d2 is None:
            continue

        points1.append(pixel_to_3d(u1, v1, d1))
        points2.append(pixel_to_3d(u2, v2, d2))

    if not points1:
        return (np.empty((0, 3)), np.empty((0, 3)))

    return (
        np.asarray(points1, dtype=np.float64),
        np.asarray(points2, dtype=np.float64)
    )


# ============================================================
# RIGID TRANSFORM — Computes the best-fit rotation and translation
# between two 3D point sets via SVD (Kabsch algorithm), correcting
# for the reflection case where the naive solution isn't a true rotation.
# ============================================================
def rigid_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t


# ============================================================
# RANSAC — Robustly estimates the rigid camera motion between two
# frames by voting across random 3-point samples, then validates
# the winning transform against inlier count, ratio, and residual.
# ============================================================
def estimate_transform(points1, points2):
    if len(points1) < MIN_3D_MATCHES:
        return None

    # Initial correspondence rejection — reject individual pairs that
    # moved further than is physically plausible before RANSAC runs.
    # 0.50m was empirically determined as the minimum safe threshold
    # for this camera; lower values started rejecting valid matches.
    max_correspondence_jump = 0.50  # meters
    distances = np.linalg.norm(points2 - points1, axis=1)
    valid = distances < max_correspondence_jump
    points1 = points1[valid]
    points2 = points2[valid]

    if len(points1) < MIN_3D_MATCHES:
        return None

    # RANSAC — fixed seed for reproducible runs.
    best_inliers = None
    best_count = 0
    iterations = 500
    rng = np.random.default_rng(42)

    for _ in range(iterations):
        if len(points1) < 3:
            break

        indices = rng.choice(len(points1), size=3, replace=False)
        A = points1[indices]
        B = points2[indices]

        R, t = rigid_transform(A, B)
        transformed = transform_points(points1, R, t)
        errors = np.linalg.norm(transformed - points2, axis=1)
        inliers = errors < RANSAC_THRESHOLD
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None:
        return {
            "success": False,
            "reason": "no_ransac_solution",
            "inliers": 0,
            "total": len(points1)
        }

    if best_count < MIN_INLIERS:
        return {
            "success": False,
            "reason": "too_few_inliers",
            "inliers": best_count,
            "total": len(points1)
        }

    ratio = best_count / len(points1)

    if ratio < MIN_INLIER_RATIO:
        return {
            "success": False,
            "reason": "low_inlier_ratio",
            "inliers": best_count,
            "total": len(points1)
        }

    # Final transform, re-estimated from all inliers together.
    R, t = rigid_transform(points1[best_inliers], points2[best_inliers])

    transformed = transform_points(points1[best_inliers], R, t)
    residuals = np.linalg.norm(transformed - points2[best_inliers], axis=1)
    mean_residual = float(np.mean(residuals))
    median_residual = float(np.median(residuals))

    return {
        "success": True,
        "R": R,
        "t": t,
        "inliers": best_count,
        "total": len(points1),
        "inlier_ratio": ratio,
        "mean_residual": mean_residual,
        "median_residual": median_residual
    }


# ============================================================
# ODOMETRY — Main per-frame loop: runs feature matching, 3D lifting,
# and RANSAC frame-to-frame, accumulates accepted poses into a global
# trajectory, and saves trusted/debug frames and diagnostics for each.
# ============================================================
def process_odometry(pairs):
    total_planned = (
        len(pairs) - 1 if MAX_FRAMES is None
        else min(MAX_FRAMES, len(pairs) - 1)
    )

    print()
    print("=" * 60)
    print(f"Processing {total_planned} frame transitions...")
    print("=" * 60)

    orb = create_orb()

    os.makedirs(DEBUG_DIR, exist_ok=True)
    os.makedirs(os.path.join(TRUSTED_DIR, "rgb"), exist_ok=True)
    os.makedirs(os.path.join(TRUSTED_DIR, "depth"), exist_ok=True)

    # --- Initial frame ---
    timestamp0, rgb_msg0, depth_timestamp0, depth_msg0 = pairs[0]
    rgb0 = image_message_to_numpy(rgb_msg0)
    depth0_raw = image_message_to_numpy(depth_msg0)

    print()
    print("Initial frame encodings:")
    print(f"  RGB:   {rgb_msg0.encoding}")
    print(f"  Depth: {depth_msg0.encoding}")
    print()
    print("Initial frame dimensions:")
    print(f"  RGB:   {rgb0.shape}")
    print(f"  Depth: {depth0_raw.shape}")

    depth0 = depth_to_meters(depth0_raw)
    valid0 = np.isfinite(depth0) & (depth0 >= MIN_DEPTH) & (depth0 <= MAX_DEPTH)
    valid_depth0 = depth0[valid0]

    print()
    print("Initial depth statistics:")
    if len(valid_depth0) > 0:
        print(f"  Min valid depth:  {np.min(valid_depth0):.4f} m")
        print(f"  Max valid depth:  {np.max(valid_depth0):.4f} m")
        print(f"  Mean valid depth: {np.mean(valid_depth0):.4f} m")
    else:
        print("  No valid depth pixels.")
    print(f"  Valid pixels: {np.sum(valid0)} / {depth0.size}")
    print(f"  Valid percentage: {100.0 * np.sum(valid0) / depth0.size:.2f}%")

    prev_gray = prepare_rgb(rgb0)
    prev_depth = depth0

    # Global camera pose:
    #   P_current = R P_previous + t
    #   R_world_camera, t_world_camera track absolute pose.
    global_R = np.eye(3, dtype=np.float64)
    global_t = np.zeros(3, dtype=np.float64)

    # Trajectory stores ONLY accepted (trusted) poses. Frame 0 is
    # always trusted by definition (identity pose) and is also
    # saved to TRUSTED_DIR so it has a matching image on disk like
    # every other trusted frame.
    trajectory = [(0, timestamp0, global_t.copy(), global_R.copy())]

    bgr0 = cv2.cvtColor(rgb0, cv2.COLOR_RGB2BGR) if rgb0.ndim == 3 else rgb0
    cv2.imwrite(os.path.join(TRUSTED_DIR, "rgb", "frame_0000.png"), bgr0)

    depth0_mm = np.clip(depth0 * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(TRUSTED_DIR, "depth", "frame_0000.png"), depth0_mm)

    diagnostics = []
    accepted_count = rejected_count = failed_count = 0
    num_transitions = total_planned

    # --- Frame loop ---
    for i in range(1, num_transitions + 1):
        curr_rgb_timestamp, curr_rgb_msg, curr_depth_timestamp, curr_depth_msg = pairs[i]

        curr_rgb = image_message_to_numpy(curr_rgb_msg)
        curr_gray = prepare_rgb(curr_rgb)

        curr_depth_raw = image_message_to_numpy(curr_depth_msg)
        curr_depth = depth_to_meters(curr_depth_raw)

        valid_depth = np.isfinite(curr_depth) & (curr_depth >= MIN_DEPTH) & (curr_depth <= MAX_DEPTH)
        valid_count = int(np.sum(valid_depth))
        valid_percentage = 100.0 * valid_count / curr_depth.size

        if valid_count > 0:
            min_valid_depth = float(np.min(curr_depth[valid_depth]))
            max_valid_depth = float(np.max(curr_depth[valid_depth]))
            mean_valid_depth = float(np.mean(curr_depth[valid_depth]))
        else:
            min_valid_depth = max_valid_depth = mean_valid_depth = 0.0

        gray_mean = float(np.mean(curr_gray))
        gray_std = float(np.std(curr_gray))
        dt_ms = abs(curr_rgb_timestamp - curr_depth_timestamp) / 1e6

        kp1, kp2, matches, raw_matches = match_features(prev_gray, curr_gray, orb)
        features_prev, features_curr = len(kp1), len(kp2)

        points1, points2 = build_3d_correspondences(kp1, kp2, matches, prev_depth, curr_depth)
        usable_3d = len(points1)

        accepted = False
        inliers = 0
        inlier_ratio = 0.0
        translation = np.zeros(3, dtype=np.float64)
        rotation_deg = 0.0
        mean_residual = median_residual = 0.0
        reason = ""

        result = estimate_transform(points1, points2)

        if result is None:
            reason = "estimation_failed"
            failed_count += 1

        elif not result["success"]:
            inliers = result["inliers"]
            total = result["total"]
            inlier_ratio = inliers / total if total > 0 else 0.0
            reason = result["reason"]
            rejected_count += 1

        else:
            R, t = result["R"], result["t"]
            inliers = result["inliers"]
            total = result["total"]
            inlier_ratio = inliers / total if total > 0 else 0.0
            translation = t.copy()
            rotation_deg = rotation_matrix_to_angle(R)
            translation_magnitude = np.linalg.norm(t)
            mean_residual = result["mean_residual"]
            median_residual = result["median_residual"]

            # Final physical validation.
            if inliers < MIN_INLIERS:
                reason = "too_few_inliers"
            elif inlier_ratio < MIN_INLIER_RATIO:
                reason = "low_inlier_ratio"
            elif translation_magnitude > MAX_TRANSLATION:
                reason = "translation_too_large"
            elif rotation_deg > MAX_ROTATION_DEG:
                reason = "rotation_too_large"
            else:
                accepted = True
                reason = "accepted"

            if accepted:
                # P_current = R * P_previous + t, so camera motion in
                # world coordinates is:
                #   R_world_current = R_world_previous * R^T
                #   t_world_current = t_world_previous - R_world_current * t
                # This assumes the world frame is the initial camera frame.
                global_R = global_R @ R.T
                global_t = global_t - global_R @ t
                accepted_count += 1
            else:
                rejected_count += 1

        # Only accepted poses are trusted. Rejected/failed frames are
        # dropped from the trajectory entirely rather than inheriting a
        # stale pose — a splatting pipeline should never see a false
        # position claim.
        if accepted:
            trajectory.append((i, curr_rgb_timestamp, global_t.copy(), global_R.copy()))

            bgr = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2BGR) if curr_rgb.ndim == 3 else curr_rgb
            cv2.imwrite(os.path.join(TRUSTED_DIR, "rgb", f"frame_{i:04d}.png"), bgr)

            depth_mm = np.clip(curr_depth * 1000.0, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(TRUSTED_DIR, "depth", f"frame_{i:04d}.png"), depth_mm)

        # Debug frame when tracking becomes weak.
        if not accepted or usable_3d < MIN_3D_MATCHES:
            debug_path = os.path.join(DEBUG_DIR, f"frame_{i:04d}.png")
            bgr = cv2.cvtColor(curr_rgb, cv2.COLOR_RGB2BGR) if curr_rgb.ndim == 3 else curr_rgb
            cv2.imwrite(debug_path, bgr)

        diagnostics.append({
            "frame": i,
            "timestamp_ns": curr_rgb_timestamp,
            "rgb_depth_dt_ms": dt_ms,
            "gray_mean": gray_mean,
            "gray_std": gray_std,
            "features_previous": features_prev,
            "features_current": features_curr,
            "raw_matches": raw_matches,
            "usable_3d_matches": usable_3d,
            "inliers": inliers,
            "inlier_ratio": inlier_ratio,
            "translation_x": translation[0],
            "translation_y": translation[1],
            "translation_z": translation[2],
            "translation_magnitude": np.linalg.norm(translation),
            "rotation_deg": rotation_deg,
            "mean_residual_m": mean_residual,
            "median_residual_m": median_residual,
            "accepted": accepted,
            "reason": reason,
            "pose_x": global_t[0],
            "pose_y": global_t[1],
            "pose_z": global_t[2],
            "depth_valid_percentage": valid_percentage,
            "depth_min_m": min_valid_depth,
            "depth_max_m": max_valid_depth,
            "depth_mean_m": mean_valid_depth
        })

        if accepted:
            print(
                f"Frame {i:04d}: ACCEPTED "
                f"inliers={inliers:3d} ratio={inlier_ratio:.2f} "
                f"features={features_prev:4d}->{features_curr:4d} "
                f"3D={usable_3d:4d} dt={dt_ms:5.2f}ms "
                f"translation=({translation[0]:+.4f}, {translation[1]:+.4f}, {translation[2]:+.4f}) "
                f"rot={rotation_deg:.2f}° residual={median_residual*1000:.1f}mm "
                f"position=({global_t[0]:+.4f}, {global_t[1]:+.4f}, {global_t[2]:+.4f})"
            )
        elif reason == "estimation_failed":
            print(
                f"Frame {i:04d}: FAILED "
                f"features={features_prev:4d}->{features_curr:4d} "
                f"matches={raw_matches:4d} 3D={usable_3d:4d}"
            )
        else:
            print(
                f"Frame {i:04d}: REJECTED {reason} "
                f"inliers={inliers:3d} ratio={inlier_ratio:.2f} "
                f"features={features_prev:4d}->{features_curr:4d} "
                f"matches={raw_matches:4d} 3D={usable_3d:4d} "
                f"position=({global_t[0]:+.4f}, {global_t[1]:+.4f}, {global_t[2]:+.4f})"
            )

        # Advance reference frame — even if the current pose was
        # rejected. Confirmed via bag testing: freezing the reference
        # on rejection causes tracking to lock up permanently once one
        # frame is rejected, since it then compares against an
        # increasingly distant frame instead of the adjacent one.
        # Always advancing lets tracking recover once conditions improve.
        prev_gray = curr_gray
        prev_depth = curr_depth

    return trajectory, diagnostics, (accepted_count, rejected_count, failed_count)

# ============================================================
# SAVE TRAJECTORY — Writes the accepted-poses trajectory (position +
# full 3x3 rotation matrix per frame) to CSV for transform_pipe_2.py
# to consume.
# ============================================================
def save_trajectory(trajectory, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame", "timestamp_ns", "x_m", "y_m", "z_m",
            "r00", "r01", "r02",
            "r10", "r11", "r12",
            "r20", "r21", "r22"
        ])

        for frame, timestamp, position, rotation in trajectory:
            writer.writerow([
                frame, timestamp,
                position[0], position[1], position[2],
                rotation[0, 0], rotation[0, 1], rotation[0, 2],
                rotation[1, 0], rotation[1, 1], rotation[1, 2],
                rotation[2, 0], rotation[2, 1], rotation[2, 2]
            ])

    print()
    print(f"Trajectory saved to: {filename}")
    print(f"Trusted frames (with matching RGB/depth on disk): {len(trajectory)}")


# ============================================================
# SAVE DIAGNOSTICS — Writes the full per-frame diagnostic log (every
# attempted frame, not just accepted ones) to CSV for debugging and
# tuning the acceptance thresholds above.
# ============================================================
def save_diagnostics(diagnostics, filename):
    if not diagnostics:
        return

    # Keep in sync with the dict keys built in process_odometry().
    fields = [
        "frame", "timestamp_ns", "rgb_depth_dt_ms", "gray_mean", "gray_std",
        "features_previous", "features_current", "raw_matches",
        "usable_3d_matches", "inliers", "inlier_ratio",
        "translation_x", "translation_y", "translation_z",
        "translation_magnitude", "rotation_deg",
        "mean_residual_m", "median_residual_m",
        "accepted", "reason",
        "pose_x", "pose_y", "pose_z",
        "depth_valid_percentage", "depth_min_m", "depth_max_m", "depth_mean_m"
    ]

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in diagnostics:
            writer.writerow(row)

    print(f"Diagnostics saved to: {filename}")


# ============================================================
# SUMMARY — Prints a human-readable run summary: accept/reject/fail
# counts, tracking-quality averages, rejection-reason breakdown, and
# the final trusted camera position.
# ============================================================
def print_summary(trajectory, diagnostics, counts):
    accepted_count, rejected_count, failed_count = counts

    print()
    print("=" * 60)
    print("ODOMETRY COMPLETE")
    print("=" * 60)
    print()
    print(f"Accepted poses:       {accepted_count}")
    print(f"Rejected poses:       {rejected_count}")
    print(f"Failed transitions:   {failed_count}")
    print(f"Total transitions:    {len(diagnostics)}")
    print(f"Trusted frames saved: {len(trajectory)}")

    if diagnostics:
        usable = [d["usable_3d_matches"] for d in diagnostics]
        inliers = [d["inliers"] for d in diagnostics if d["inliers"] > 0]
        accepted_translations = [
            d["translation_magnitude"] for d in diagnostics if d["accepted"]
        ]

        print()
        print("Tracking statistics:")
        print(f"  Average usable 3D matches: {np.mean(usable):.1f}")

        if inliers:
            print(f"  Average RANSAC inliers: {np.mean(inliers):.1f}")
            print(f"  Maximum RANSAC inliers: {np.max(inliers)}")

        if accepted_translations:
            print(f"  Average accepted motion: {np.mean(accepted_translations):.5f} m")
            print(f"  Maximum accepted motion: {np.max(accepted_translations):.5f} m")

        print()
        print("Reasons:")
        reasons = {}
        for d in diagnostics:
            reasons[d["reason"]] = reasons.get(d["reason"], 0) + 1

        for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  {reason:25s}: {count}")

    if trajectory:
        final_position = trajectory[-1][2]
        print()
        print("Final trusted position:")
        print(f"  X = {final_position[0]:+.4f} m")
        print(f"  Y = {final_position[1]:+.4f} m")
        print(f"  Z = {final_position[2]:+.4f} m")


# ============================================================
# MAIN — Runs the full odometry pipeline end to end: read bag,
# resolve intrinsics, sync RGB-D, estimate trajectory, and save
# all outputs.
# ============================================================
def main():
    rclpy.init()

    try:
        rgb_frames, depth_frames, camera_info = read_bag(BAG_PATH)
        configure_camera_info(camera_info)

        pairs = synchronize_frames(rgb_frames, depth_frames)

        if len(pairs) < 2:
            raise RuntimeError("Not enough synchronized RGB-D frames.")

        trajectory, diagnostics, counts = process_odometry(pairs)

        save_trajectory(trajectory, TRAJECTORY_FILE)
        save_diagnostics(diagnostics, DIAGNOSTIC_FILE)
        print_summary(trajectory, diagnostics, counts)

    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()