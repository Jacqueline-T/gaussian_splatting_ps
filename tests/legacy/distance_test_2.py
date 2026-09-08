#!/usr/bin/env python3

import os
import csv
import math
import sqlite3
import struct

import cv2
import numpy as np

import rclpy
from rclpy.serialization import deserialize_message

from rosidl_runtime_py.utilities import get_message


# ============================================================
# CONFIGURATION
# ============================================================

BAG_PATH = (
    "/home/regality/ros2_ws/src/prime_sense/xtion_pkg/"
    "xtion_pkg/3d_construct/rosbag2_2026_08_27-15_11_02"
)

RGB_TOPIC = "/camera/rgb/image_raw"
DEPTH_TOPIC = "/camera/depth_registered/image_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

TRAJECTORY_FILE = "rgbd_trajectory_diagnostic.csv"
DIAGNOSTIC_FILE = "rgbd_odometry_diagnostics.csv"

# Number of frame transitions to process.
# Set to None to process the entire bag.
MAX_FRAMES = 100

# ------------------------------------------------------------
# ORB
# ------------------------------------------------------------

ORB_FEATURES = 1500

ORB_SCALE_FACTOR = 1.2
ORB_LEVELS = 8

# ------------------------------------------------------------
# Matching
# ------------------------------------------------------------

# Lowe ratio test.
LOWE_RATIO = 0.75

# Minimum number of 3D correspondences before attempting pose.
MIN_3D_MATCHES = 20

# Minimum RANSAC inliers required to accept motion.
MIN_INLIERS = 50

# Minimum inlier ratio.
MIN_INLIER_RATIO = 0.20

# Maximum physically plausible translation between consecutive
# frames. This is deliberately conservative for the diagnostic run.
MAX_TRANSLATION = 0.15  # meters

# Maximum physically plausible rotation between frames.
MAX_ROTATION_DEG = 20.0

# Depth validity range.
MIN_DEPTH = 0.40
MAX_DEPTH = 5.0


# ============================================================
# CAMERA INTRINSICS
# ============================================================

FX = 570.3422241210938
FY = 570.3422241210938
CX = 319.5
CY = 239.5


# ============================================================
# HELPERS
# ============================================================

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """
    Convert quaternion to 3x3 rotation matrix.
    """

    xx = qx * qx
    yy = qy * qy
    zz = qz * qz

    xy = qx * qy
    xz = qx * qz
    yz = qy * qz

    wx = qw * qx
    wy = qw * qy
    wz = qw * qz

    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)]
    ], dtype=np.float64)

    return R


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
    Convert ROS Image message to NumPy array.
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
        raise ValueError(
            f"Unsupported image encoding: {encoding}"
        )

    array = np.frombuffer(msg.data, dtype=dtype)

    if channels == 3:
        array = array.reshape((height, width, 3))
    else:
        array = array.reshape((height, width))

    return array.copy()


def depth_to_meters(depth):
    """
    Convert ROS depth image into meters.

    Your bag is currently confirmed as 32FC1,
    which is already meters.

    16UC1 is also supported for completeness.
    """

    if depth.dtype == np.float32:
        return depth.astype(np.float32)

    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / 1000.0

    raise ValueError(
        f"Unsupported depth dtype: {depth.dtype}"
    )


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
    Back-project a pixel into camera 3D coordinates.

    Camera convention:

        X = right
        Y = down
        Z = forward
    """

    X = (u - CX) * depth / FX
    Y = (v - CY) * depth / FY
    Z = depth

    return np.array(
        [X, Y, Z],
        dtype=np.float64
    )


def transform_points(points, R, t):
    """
    Transform Nx3 points.
    """

    return (R @ points.T).T + t.reshape(1, 3)


# ============================================================
# ROS BAG READING
# ============================================================

def get_db3_file(bag_path):

    if os.path.isfile(bag_path):
        return bag_path

    if not os.path.isdir(bag_path):
        raise FileNotFoundError(
            f"Bag path does not exist:\n{bag_path}"
        )

    db3_files = [
        f for f in os.listdir(bag_path)
        if f.endswith(".db3")
    ]

    if not db3_files:
        raise FileNotFoundError(
            f"No .db3 file found in:\n{bag_path}"
        )

    return os.path.join(
        bag_path,
        sorted(db3_files)[0]
    )


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
    # Read topic definitions FIRST
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

        print(
            f"ID={topic_id:3d} | "
            f"{name}"
        )

        print(
            f"          Type: {msg_type}"
        )

    # --------------------------------------------------------
    # Now print message counts
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

    for topic_id, count in cursor.fetchall():

        topic_name = "UNKNOWN"

        for name, (tid, _) in topic_map.items():

            if tid == topic_id:

                topic_name = name

                break

        print(
            f"ID={topic_id:3d} | "
            f"{count:5d} messages | "
            f"{topic_name}"
        )

    # --------------------------------------------------------
    # Verify required topics
    # --------------------------------------------------------

    print()
    print("=" * 60)
    print("SELECTED TOPICS")
    print("=" * 60)

    if RGB_TOPIC not in topic_map:

        raise RuntimeError(
            f"RGB topic not found: {RGB_TOPIC}"
        )

    if DEPTH_TOPIC not in topic_map:

        raise RuntimeError(
            f"Depth topic not found: {DEPTH_TOPIC}"
        )

    rgb_topic_id, rgb_type = topic_map[RGB_TOPIC]

    depth_topic_id, depth_type = topic_map[DEPTH_TOPIC]

    print(
        f"RGB topic ID:   {rgb_topic_id}"
    )

    print(
        f"RGB type:       {rgb_type}"
    )

    print(
        f"Depth topic ID: {depth_topic_id}"
    )

    print(
        f"Depth type:     {depth_type}"
    )

    camera_info_topic_id = None

    if CAMERA_INFO_TOPIC in topic_map:

        camera_info_topic_id = topic_map[
            CAMERA_INFO_TOPIC
        ][0]

    cursor.execute(
        """
        SELECT topic_id, COUNT(*)
        FROM messages
        GROUP BY topic_id
        ORDER BY topic_id
        """
    )

    for topic_id, count in cursor.fetchall():

        topic_name = "UNKNOWN"

        for name, (tid, _) in topic_map.items():

            if tid == topic_id:
                topic_name = name
                break

        print(
            f"ID={topic_id:3d} | "
            f"{count:5d} messages | "
            f"{topic_name}"
        )

    print()
    print("Reading bag...")

    cursor.execute(
        "SELECT id, name, type FROM topics"
    )

    topics = cursor.fetchall()

    print()
    print("=" * 60)
    print("SELECTED TOPICS")
    print("=" * 60)

    if RGB_TOPIC not in topic_map:
        raise RuntimeError(
            f"RGB topic not found: {RGB_TOPIC}"
        )

    if DEPTH_TOPIC not in topic_map:
        raise RuntimeError(
            f"Depth topic not found: {DEPTH_TOPIC}"
        )

    rgb_topic_id, rgb_type = topic_map[RGB_TOPIC]
    depth_topic_id, depth_type = topic_map[DEPTH_TOPIC]

    print(
        f"RGB topic ID:   {rgb_topic_id}"
    )

    print(
        f"RGB type:       {rgb_type}"
    )

    print(
        f"Depth topic ID: {depth_topic_id}"
    )

    print(
        f"Depth type:     {depth_type}"
    )

    camera_info_topic_id = None

    if CAMERA_INFO_TOPIC in topic_map:
        camera_info_topic_id = topic_map[
            CAMERA_INFO_TOPIC
        ][0]

    RGBMsg = get_message(rgb_type)
    DepthMsg = get_message(depth_type)

    CameraInfoMsg = None

    if camera_info_topic_id is not None:

        camera_info_type = topic_map[
            CAMERA_INFO_TOPIC
        ][1]

        CameraInfoMsg = get_message(
            camera_info_type
        )

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

            msg = deserialize_message(
                data,
                RGBMsg
            )

            rgb_frames.append(
                (
                    timestamp,
                    msg
                )
            )

        elif topic_id == depth_topic_id:

            msg = deserialize_message(
                data,
                DepthMsg
            )

            depth_frames.append(
                (
                    timestamp,
                    msg
                )
            )

        elif (
            camera_info_topic_id is not None
            and topic_id == camera_info_topic_id
        ):

            # CameraInfo is essentially static for this camera.
            # We only need one copy.
            if camera_info is None:

                camera_info = deserialize_message(
                    data,
                    CameraInfoMsg
                )

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
# RGB-D SYNCHRONIZATION
# ============================================================

def synchronize_frames(rgb_frames, depth_frames):

    print()
    print("=" * 60)
    print("SYNCHRONIZING RGB-D")
    print("=" * 60)

    pairs = []

    depth_index = 0

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
            candidates.append(
                depth_frames[depth_index]
            )

        if depth_index + 1 < len(depth_frames):
            candidates.append(
                depth_frames[depth_index + 1]
            )

        if not candidates:
            continue

        best = min(
            candidates,
            key=lambda x: abs(
                x[0] - rgb_timestamp
            )
        )

        depth_timestamp, depth_msg = best

        difference = abs(
            rgb_timestamp - depth_timestamp
        )

        if difference <= max_difference_ns:

            pairs.append(
                (
                    rgb_timestamp,
                    rgb_msg,
                    depth_timestamp,
                    depth_msg
                )
            )

    print()
    print(
        f"Matched RGB-D pairs: {len(pairs)}"
    )

    if pairs:

        differences_ms = [
            abs(
                p[0] - p[2]
            ) / 1e6
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
# ORB MATCHING
# ============================================================

def create_orb():

    return cv2.ORB_create(
        nfeatures=ORB_FEATURES,
        scaleFactor=ORB_SCALE_FACTOR,
        nlevels=ORB_LEVELS
    )


def prepare_rgb(rgb):

    if rgb.ndim == 3:

        if rgb.shape[2] == 3:

            gray = cv2.cvtColor(
                rgb,
                cv2.COLOR_RGB2GRAY
            )

        else:

            gray = rgb[:, :, 0]

    else:

        gray = rgb

    return gray


def match_features(
    prev_gray,
    curr_gray,
    orb
):

    kp1, des1 = orb.detectAndCompute(
        prev_gray,
        None
    )

    kp2, des2 = orb.detectAndCompute(
        curr_gray,
        None
    )

    if des1 is None or des2 is None:

        return (
            kp1 or [],
            kp2 or [],
            [],
            0
        )

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=False
    )

    knn_matches = matcher.knnMatch(
        des1,
        des2,
        k=2
    )

    good_matches = []

    for pair in knn_matches:

        if len(pair) < 2:
            continue

        m, n = pair

        if m.distance < LOWE_RATIO * n.distance:

            good_matches.append(m)

    return (
        kp1,
        kp2,
        good_matches,
        len(good_matches)
    )


# ============================================================
# 3D CORRESPONDENCE GENERATION
# ============================================================

def build_3d_correspondences(
    kp1,
    kp2,
    matches,
    depth1,
    depth2
):

    points1 = []
    points2 = []

    for match in matches:

        u1, v1 = kp1[
            match.queryIdx
        ].pt

        u2, v2 = kp2[
            match.trainIdx
        ].pt

        d1 = get_depth(
            depth1,
            u1,
            v1
        )

        d2 = get_depth(
            depth2,
            u2,
            v2
        )

        if d1 is None or d2 is None:
            continue

        p1 = pixel_to_3d(
            u1,
            v1,
            d1
        )

        p2 = pixel_to_3d(
            u2,
            v2,
            d2
        )

        points1.append(p1)
        points2.append(p2)

    if not points1:

        return (
            np.empty((0, 3)),
            np.empty((0, 3))
        )

    return (
        np.asarray(points1),
        np.asarray(points2)
    )


# ============================================================
# RIGID TRANSFORM ESTIMATION
# ============================================================

def estimate_transform(
    points1,
    points2
):

    if len(points1) < MIN_3D_MATCHES:

        return None

    # --------------------------------------------------------
    # Initial correspondence rejection
    # --------------------------------------------------------

    distances = np.linalg.norm(
        points2 - points1,
        axis=1
    )

    # Reject obviously impossible individual matches.
    valid = distances < 0.50

    points1 = points1[valid]
    points2 = points2[valid]

    if len(points1) < MIN_3D_MATCHES:

        return None

    # --------------------------------------------------------
    # RANSAC
    # --------------------------------------------------------

    best_inliers = None
    best_count = 0

    iterations = 500

    threshold = 0.03

    rng = np.random.default_rng(42)

    for _ in range(iterations):

        if len(points1) < 3:
            break

        indices = rng.choice(
            len(points1),
            size=3,
            replace=False
        )

        A = points1[indices]
        B = points2[indices]

        R, t = rigid_transform(
            A,
            B
        )

        transformed = transform_points(
            points1,
            R,
            t
        )

        errors = np.linalg.norm(
            transformed - points2,
            axis=1
        )

        inliers = errors < threshold

        count = int(
            np.sum(inliers)
        )

        if count > best_count:

            best_count = count
            best_inliers = inliers

    if best_inliers is None:

        return None

    if best_count < MIN_INLIERS:

        return {
            "success": False,
            "reason": "too_few_inliers",
            "inliers": best_count,
            "total": len(points1)
        }

    # Re-estimate using all RANSAC inliers.

    R, t = rigid_transform(
        points1[best_inliers],
        points2[best_inliers]
    )

    return {
        "success": True,
        "R": R,
        "t": t,
        "inliers": best_count,
        "total": len(points1)
    }


def rigid_transform(A, B):

    centroid_A = np.mean(
        A,
        axis=0
    )

    centroid_B = np.mean(
        B,
        axis=0
    )

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:

        Vt[-1, :] *= -1

        R = Vt.T @ U.T

    t = (
        centroid_B
        - R @ centroid_A
    )

    return R, t


# ============================================================
# ODOMETRY
# ============================================================

def process_odometry(pairs):

    print()
    print("=" * 60)
    print(
        f"Processing first "
        f"{min(MAX_FRAMES, len(pairs) - 1)} "
        f"frame transitions..."
    )
    print("=" * 60)

    orb = create_orb()

    # --------------------------------------------------------
    # Initial frame
    # --------------------------------------------------------

    timestamp0, rgb_msg0, depth_timestamp0, depth_msg0 = pairs[0]

    rgb0 = image_message_to_numpy(
        rgb_msg0
    )

    depth0 = image_message_to_numpy(
        depth_msg0
    )

    # New!
    print()
    print("=" * 60)
    print("DEPTH DATA CHECK")
    print("=" * 60)

    print(f"ROS encoding: {depth_msg0.encoding}")
    print(f"Width:        {depth_msg0.width}")
    print(f"Height:       {depth_msg0.height}")
    print(f"NumPy dtype:  {depth0.dtype}")
    print(f"Min raw:      {np.min(depth0)}")
    print(f"Max raw:      {np.max(depth0)}")
    print(f"Mean raw:     {np.mean(depth0)}")

    if depth0.dtype == np.float32:

        valid = np.isfinite(depth0) & (depth0 > 0)

    else:

        valid = depth0 > 0

    print(
        f"Valid pixels: {np.sum(valid)} / {depth0.size}"
    )

    print(
        f"Valid percentage: "
        f"{100.0 * np.sum(valid) / depth0.size:.2f}%"
    )

    depth0 = depth_to_meters(
        depth0
    )

    prev_gray = prepare_rgb(rgb0)

    prev_depth = depth0

    # --------------------------------------------------------
    # Global pose
    # --------------------------------------------------------

    global_R = np.eye(
        3,
        dtype=np.float64
    )

    global_t = np.zeros(
        3,
        dtype=np.float64
    )

    trajectory = [
        (
            0,
            timestamp0,
            global_t.copy()
        )
    ]

    diagnostics = []

    accepted_count = 0
    rejected_count = 0
    failed_count = 0

    # --------------------------------------------------------
    # Frame loop
    # --------------------------------------------------------

    num_transitions = min(
        MAX_FRAMES,
        len(pairs) - 1
    )

    for i in range(
        1,
        num_transitions + 1
    ):

        (
            curr_rgb_timestamp,
            curr_rgb_msg,
            curr_depth_timestamp,
            curr_depth_msg
        ) = pairs[i]

        curr_rgb = image_message_to_numpy(
            curr_rgb_msg
        )

        curr_depth = image_message_to_numpy(
            curr_depth_msg
        )

        curr_depth = depth_to_meters(
            curr_depth
        )

        curr_gray = prepare_rgb(
            curr_rgb
        )

        # ----------------------------------------------------
        # Timestamp difference
        # ----------------------------------------------------

        dt_ms = abs(
            curr_rgb_timestamp
            - curr_depth_timestamp
        ) / 1e6

        # ----------------------------------------------------
        # Feature detection
        # ----------------------------------------------------

        kp1, kp2, matches, raw_matches = match_features(
            prev_gray,
            curr_gray,
            orb
        )

        features_prev = len(kp1)
        features_curr = len(kp2)

        # ----------------------------------------------------
        # 3D correspondences
        # ----------------------------------------------------

        points1, points2 = build_3d_correspondences(
            kp1,
            kp2,
            matches,
            prev_depth,
            curr_depth
        )

        usable_3d = len(points1)

        # ----------------------------------------------------
        # Estimate transform
        # ----------------------------------------------------

        result = estimate_transform(
            points1,
            points2
        )

        accepted = False

        inliers = 0
        inlier_ratio = 0.0

        translation = np.zeros(
            3,
            dtype=np.float64
        )

        rotation_deg = 0.0

        reason = ""

        # ----------------------------------------------------
        # Transform validation
        # ----------------------------------------------------

        if result is None:

            reason = "estimation_failed"

            failed_count += 1

        elif not result["success"]:

            inliers = result[
                "inliers"
            ]

            if result["total"] > 0:

                inlier_ratio = (
                    inliers
                    / result["total"]
                )

            reason = result[
                "reason"
            ]

            rejected_count += 1

        else:

            R = result["R"]
            t = result["t"]

            inliers = result[
                "inliers"
            ]

            total = result[
                "total"
            ]

            if total > 0:

                inlier_ratio = (
                    inliers
                    / total
                )

            translation = t.copy()

            rotation_deg = rotation_matrix_to_angle(
                R
            )

            translation_magnitude = np.linalg.norm(
                t
            )

            # ------------------------------------------------
            # Validation rules
            # ------------------------------------------------

            if inliers < MIN_INLIERS:

                reason = (
                    f"too_few_inliers"
                )

            elif inlier_ratio < MIN_INLIER_RATIO:

                reason = (
                    f"low_inlier_ratio"
                )

            elif (
                translation_magnitude
                > MAX_TRANSLATION
            ):

                reason = (
                    f"translation_too_large"
                )

            elif (
                rotation_deg
                > MAX_ROTATION_DEG
            ):

                reason = (
                    f"rotation_too_large"
                )

            else:

                accepted = True
                reason = "accepted"

            # ------------------------------------------------
            # Accept/reject
            # ------------------------------------------------

            if accepted:

                # The transform maps previous-camera points
                # into current-camera coordinates.
                #
                # For trajectory estimation, we update the
                # camera pose in the world frame.

                global_R = (
                    global_R
                    @ R.T
                )

                global_t = (
                    global_t
                    - global_R @ t
                )

                accepted_count += 1

            else:

                rejected_count += 1

        # ----------------------------------------------------
        # Save trajectory regardless of acceptance.
        #
        # Rejected frames retain the last trusted pose.
        # ----------------------------------------------------

        trajectory.append(
            (
                i,
                curr_rgb_timestamp,
                global_t.copy()
            )
        )

        # ----------------------------------------------------
        # Diagnostics
        # ----------------------------------------------------

        diagnostics.append(
            {
                "frame": i,
                "timestamp_ns": curr_rgb_timestamp,
                "rgb_depth_dt_ms": dt_ms,
                "features_previous": features_prev,
                "features_current": features_curr,
                "raw_matches": raw_matches,
                "usable_3d_matches": usable_3d,
                "inliers": inliers,
                "inlier_ratio": inlier_ratio,
                "translation_x": translation[0],
                "translation_y": translation[1],
                "translation_z": translation[2],
                "translation_magnitude": np.linalg.norm(
                    translation
                ),
                "rotation_deg": rotation_deg,
                "accepted": accepted,
                "reason": reason,
                "pose_x": global_t[0],
                "pose_y": global_t[1],
                "pose_z": global_t[2],
            }
        )

        # ----------------------------------------------------
        # Console output
        # ----------------------------------------------------

        if accepted:

            print(
                f"Frame {i:04d}: "
                f"ACCEPTED "
                f"inliers={inliers:3d} "
                f"ratio={inlier_ratio:.2f} "
                f"features={features_prev:4d}"
                f"->{features_curr:4d} "
                f"3D={usable_3d:4d} "
                f"dt={dt_ms:5.2f}ms "
                f"translation="
                f"({translation[0]:+.4f}, "
                f"{translation[1]:+.4f}, "
                f"{translation[2]:+.4f}) "
                f"rot={rotation_deg:.2f}° "
                f"position="
                f"({global_t[0]:+.4f}, "
                f"{global_t[1]:+.4f}, "
                f"{global_t[2]:+.4f})"
            )

        elif reason == "estimation_failed":

            print(
                f"Frame {i:04d}: "
                f"FAILED "
                f"features="
                f"{features_prev:4d}"
                f"->{features_curr:4d} "
                f"matches={raw_matches:4d} "
                f"3D={usable_3d:4d}"
            )

        else:

            print(
                f"Frame {i:04d}: "
                f"REJECTED "
                f"{reason} "
                f"inliers={inliers:3d} "
                f"ratio={inlier_ratio:.2f} "
                f"features="
                f"{features_prev:4d}"
                f"->{features_curr:4d} "
                f"matches={raw_matches:4d} "
                f"3D={usable_3d:4d} "
                f"position="
                f"({global_t[0]:+.4f}, "
                f"{global_t[1]:+.4f}, "
                f"{global_t[2]:+.4f})"
            )

        # ----------------------------------------------------
        # Advance previous frame
        #
        # IMPORTANT:
        # Even if a pose is rejected, we use the current
        # image/depth as the next frame for tracking.
        # This allows the system to recover after a temporary
        # tracking failure.
        # ----------------------------------------------------

        prev_gray = curr_gray
        prev_depth = curr_depth

    return trajectory, diagnostics, (
        accepted_count,
        rejected_count,
        failed_count
    )


# ============================================================
# SAVE TRAJECTORY
# ============================================================

def save_trajectory(
    trajectory,
    filename
):

    with open(
        filename,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "frame",
                "timestamp_ns",
                "x_m",
                "y_m",
                "z_m"
            ]
        )

        for frame, timestamp, position in trajectory:

            writer.writerow(
                [
                    frame,
                    timestamp,
                    position[0],
                    position[1],
                    position[2]
                ]
            )

    print()
    print(
        f"Trajectory saved to: "
        f"{filename}"
    )


# ============================================================
# SAVE DIAGNOSTICS
# ============================================================

def save_diagnostics(
    diagnostics,
    filename
):

    if not diagnostics:
        return

    fields = [
        "frame",
        "timestamp_ns",
        "rgb_depth_dt_ms",
        "features_previous",
        "features_current",
        "raw_matches",
        "usable_3d_matches",
        "inliers",
        "inlier_ratio",
        "translation_x",
        "translation_y",
        "translation_z",
        "translation_magnitude",
        "rotation_deg",
        "accepted",
        "reason",
        "pose_x",
        "pose_y",
        "pose_z"
    ]

    with open(
        filename,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for row in diagnostics:

            writer.writerow(row)

    print(
        f"Diagnostics saved to: "
        f"{filename}"
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    trajectory,
    diagnostics,
    counts
):

    accepted_count, rejected_count, failed_count = counts

    print()
    print("=" * 60)
    print("ODOMETRY COMPLETE")
    print("=" * 60)

    print()
    print(
        f"Accepted poses:       "
        f"{accepted_count}"
    )

    print(
        f"Rejected poses:       "
        f"{rejected_count}"
    )

    print(
        f"Failed transitions:   "
        f"{failed_count}"
    )

    print(
        f"Total transitions:    "
        f"{len(diagnostics)}"
    )

    if diagnostics:

        usable = [
            d["usable_3d_matches"]
            for d in diagnostics
        ]

        inliers = [
            d["inliers"]
            for d in diagnostics
            if d["inliers"] > 0
        ]

        print()
        print("Tracking statistics:")

        print(
            f"  Average usable 3D matches: "
            f"{np.mean(usable):.1f}"
        )

        if inliers:

            print(
                f"  Average RANSAC inliers: "
                f"{np.mean(inliers):.1f}"
            )

            print(
                f"  Maximum RANSAC inliers: "
                f"{np.max(inliers)}"
            )

        reasons = {}

        for d in diagnostics:

            reason = d["reason"]

            reasons[reason] = (
                reasons.get(reason, 0)
                + 1
            )

        print()
        print("Reasons:")

        for reason, count in sorted(
            reasons.items(),
            key=lambda x: -x[1]
        ):

            print(
                f"  {reason:25s}: "
                f"{count}"
            )

    if trajectory:

        final_position = trajectory[-1][2]

        print()
        print("Final trusted position:")

        print(
            f"  X = "
            f"{final_position[0]:+.4f} m"
        )

        print(
            f"  Y = "
            f"{final_position[1]:+.4f} m"
        )

        print(
            f"  Z = "
            f"{final_position[2]:+.4f} m"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    rclpy.init()

    try:

        rgb_frames, depth_frames, camera_info = read_bag(
            BAG_PATH
        )

        pairs = synchronize_frames(
            rgb_frames,
            depth_frames
        )

        if len(pairs) < 2:

            raise RuntimeError(
                "Not enough synchronized RGB-D frames."
            )

        print()
        print("=" * 60)
        print("CAMERA INTRINSICS")
        print("=" * 60)

        print(
            f"fx = {FX}"
        )

        print(
            f"fy = {FY}"
        )

        print(
            f"cx = {CX}"
        )

        print(
            f"cy = {CY}"
        )

        trajectory, diagnostics, counts = process_odometry(
            pairs
        )

        save_trajectory(
            trajectory,
            TRAJECTORY_FILE
        )

        save_diagnostics(
            diagnostics,
            DIAGNOSTIC_FILE
        )

        print_summary(
            trajectory,
            diagnostics,
            counts
        )

    finally:

        rclpy.shutdown()


if __name__ == "__main__":
    main()