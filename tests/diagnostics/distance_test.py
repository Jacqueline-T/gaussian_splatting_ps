#!/usr/bin/env python3

import sys
import os
import bisect

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message

from sensor_msgs.msg import Image, CameraInfo


# ============================================================
# RGB-D ROSBAG DEPTH VALIDATION
# ============================================================

BAG_PATH = (
    "/home/regality/ros2_ws/src/prime_sense/"
    "xtion_pkg/xtion_pkg/3d_construct/"
    "tests/rosbag2_2026_08_27-15_46_20"
)

RGB_TOPIC = "/camera/rgb/image_raw"
DEPTH_TOPIC = "/camera/depth_registered/image_raw"
CAMERA_INFO_TOPIC = "/camera/rgb/camera_info"

# Maximum RGB/depth timestamp difference
MAX_TIME_DIFF_NS = 20_000_000


# ============================================================
# IMAGE → NUMPY
# ============================================================

def image_to_numpy(msg):

    height = msg.height
    width = msg.width

    if msg.encoding == "rgb8":

        array = np.frombuffer(
            msg.data,
            dtype=np.uint8
        )

        return array.reshape(
            (height, width, 3)
        )

    elif msg.encoding == "16UC1":

        array = np.frombuffer(
            msg.data,
            dtype=np.uint16
        )

        return array.reshape(
            (height, width)
        )

    elif msg.encoding == "32FC1":

        array = np.frombuffer(
            msg.data,
            dtype=np.float32
        )

        return array.reshape(
            (height, width)
        )

    else:

        raise ValueError(
            f"Unsupported encoding: "
            f"{msg.encoding}"
        )


# ============================================================
# DEPTH ANALYSIS
# ============================================================

def analyze_depth(depth_msg, frame_number):

    depth = image_to_numpy(depth_msg)

    # --------------------------------------------------------
    # Valid pixels
    # --------------------------------------------------------

    valid_mask = (
        np.isfinite(depth)
        &
        (depth > 0)
    )

    valid_values = depth[valid_mask]

    print("\n")
    print("========================================")
    print(f"DEPTH FRAME ANALYSIS #{frame_number}")
    print("========================================")

    print(
        f"Encoding:        {depth_msg.encoding}"
    )

    print(
        f"Resolution:      "
        f"{depth_msg.width}x{depth_msg.height}"
    )

    print(
        f"NumPy dtype:     {depth.dtype}"
    )

    print(
        f"Total pixels:    {depth.size}"
    )

    print(
        f"Valid pixels:    {len(valid_values)}"
    )

    print(
        f"Valid percentage:"
        f" {100.0 * len(valid_values) / depth.size:.2f}%"
    )

    if len(valid_values) == 0:

        print("\nNO VALID DEPTH DATA")

        return


    # --------------------------------------------------------
    # Basic statistics
    # --------------------------------------------------------

    print("\nDepth statistics:")

    print(
        f"  Minimum: {np.min(valid_values):.6f} m"
    )

    print(
        f"  Maximum: {np.max(valid_values):.6f} m"
    )

    print(
        f"  Mean:    {np.mean(valid_values):.6f} m"
    )

    print(
        f"  Median:  {np.median(valid_values):.6f} m"
    )


    # --------------------------------------------------------
    # Percentiles
    # --------------------------------------------------------

    percentiles = np.percentile(
        valid_values,
        [10, 25, 50, 75, 90]
    )

    print("\nDepth percentiles:")

    print(
        f"  10%: {percentiles[0]:.6f} m"
    )

    print(
        f"  25%: {percentiles[1]:.6f} m"
    )

    print(
        f"  50%: {percentiles[2]:.6f} m"
    )

    print(
        f"  75%: {percentiles[3]:.6f} m"
    )

    print(
        f"  90%: {percentiles[4]:.6f} m"
    )


    # --------------------------------------------------------
    # Closest valid pixels
    # --------------------------------------------------------

    min_depth = np.min(valid_values)

    closest_mask = (
        valid_mask
        &
        (depth <= min_depth + 0.001)
    )

    closest_y, closest_x = np.where(
        closest_mask
    )

    print("\nClosest valid depth:")

    print(
        f"  Distance: {min_depth:.6f} m"
    )

    print(
        f"  Number of pixels within "
        f"1 mm: {len(closest_x)}"
    )

    if len(closest_x) > 0:

        # Show up to 10 locations
        print("  Locations:")

        for x, y in zip(
            closest_x[:10],
            closest_y[:10]
        ):

            print(
                f"    ({x}, {y})"
            )


    # --------------------------------------------------------
    # Bounding box of all valid pixels
    # --------------------------------------------------------

    valid_y, valid_x = np.where(
        valid_mask
    )

    print("\nValid-depth bounding box:")

    print(
        f"  X: {valid_x.min()} "
        f"to {valid_x.max()}"
    )

    print(
        f"  Y: {valid_y.min()} "
        f"to {valid_y.max()}"
    )


    # --------------------------------------------------------
    # Center pixel
    # --------------------------------------------------------

    center_x = depth.shape[1] // 2
    center_y = depth.shape[0] // 2

    center_value = depth[
        center_y,
        center_x
    ]

    print("\nCenter pixel:")

    print(
        f"  ({center_x}, {center_y})"
    )

    print(
        f"  Value: {center_value}"
    )

    if np.isfinite(center_value):

        print(
            f"  Distance: "
            f"{center_value:.6f} m"
        )

    else:

        print(
            "  INVALID (NaN/Inf)"
        )


    # --------------------------------------------------------
    # 100x100 center ROI
    # --------------------------------------------------------

    half_size = 50

    x1 = max(
        0,
        center_x - half_size
    )

    x2 = min(
        depth.shape[1],
        center_x + half_size
    )

    y1 = max(
        0,
        center_y - half_size
    )

    y2 = min(
        depth.shape[0],
        center_y + half_size
    )

    roi = depth[
        y1:y2,
        x1:x2
    ]

    roi_valid = roi[
        np.isfinite(roi)
        &
        (roi > 0)
    ]

    print("\n100x100 center ROI:")

    print(
        f"  Coordinates: "
        f"x={x1}:{x2}, "
        f"y={y1}:{y2}"
    )

    print(
        f"  Valid pixels: "
        f"{len(roi_valid)}"
    )

    if len(roi_valid) > 0:

        print(
            f"  Mean:   "
            f"{np.mean(roi_valid):.6f} m"
        )

        print(
            f"  Median: "
            f"{np.median(roi_valid):.6f} m"
        )

        print(
            f"  Min:    "
            f"{np.min(roi_valid):.6f} m"
        )

        print(
            f"  Max:    "
            f"{np.max(roi_valid):.6f} m"
        )

    else:

        print(
            "  No valid depth in center ROI."
        )


# ============================================================
# READ BAG
# ============================================================

def read_bag(bag_path):

    storage_options = rosbag2_py.StorageOptions(
        uri=bag_path,
        storage_id="sqlite3"
    )

    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr"
    )

    reader = rosbag2_py.SequentialReader()

    reader.open(
        storage_options,
        converter_options
    )


    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------

    topic_types = (
        reader.get_all_topics_and_types()
    )

    type_map = {
        topic.name: topic.type
        for topic in topic_types
    }


    print("\n========================================")
    print("RGB-D ROSBAG DEPTH VALIDATION")
    print("========================================")

    print("\nBag:")
    print(bag_path)

    print("\nRequired topics:")

    for topic in [
        RGB_TOPIC,
        DEPTH_TOPIC,
        CAMERA_INFO_TOPIC
    ]:

        if topic in type_map:

            print(
                f"  [FOUND] {topic}"
            )

            print(
                f"          {type_map[topic]}"
            )

        else:

            print(
                f"  [MISSING] {topic}"
            )


    # --------------------------------------------------------
    # Message storage
    # --------------------------------------------------------

    rgb_frames = []

    depth_frames = []

    camera_info_frames = []


    # --------------------------------------------------------
    # Read messages
    # --------------------------------------------------------

    print("\nReading bag...")

    while reader.has_next():

        topic, data, timestamp = (
            reader.read_next()
        )

        if topic == RGB_TOPIC:

            msg = deserialize_message(
                data,
                Image
            )

            rgb_frames.append(
                (timestamp, msg)
            )

        elif topic == DEPTH_TOPIC:

            msg = deserialize_message(
                data,
                Image
            )

            depth_frames.append(
                (timestamp, msg)
            )

        elif topic == CAMERA_INFO_TOPIC:

            msg = deserialize_message(
                data,
                CameraInfo
            )

            camera_info_frames.append(
                (timestamp, msg)
            )


    # ========================================================
    # BAG CONTENT
    # ========================================================

    print("\n========================================")
    print("BAG CONTENT")
    print("========================================")

    print(
        f"RGB frames:        "
        f"{len(rgb_frames)}"
    )

    print(
        f"Depth frames:      "
        f"{len(depth_frames)}"
    )

    print(
        f"CameraInfo frames: "
        f"{len(camera_info_frames)}"
    )


    # ========================================================
    # DEPTH FORMAT
    # ========================================================

    if depth_frames:

        depth_msg = depth_frames[0][1]

        print("\n========================================")
        print("DEPTH FORMAT")
        print("========================================")

        print(
            f"Encoding: "
            f"{depth_msg.encoding}"
        )

        print(
            f"Step: "
            f"{depth_msg.step} bytes"
        )

        print(
            f"Is big endian: "
            f"{depth_msg.is_bigendian}"
        )

        print(
            f"Width: "
            f"{depth_msg.width}"
        )

        print(
            f"Height: "
            f"{depth_msg.height}"
        )


    # ========================================================
    # MATCH RGB + DEPTH
    # ========================================================

    print("\nMatching RGB and depth frames...")

    depth_timestamps = [
        timestamp
        for timestamp, msg
        in depth_frames
    ]

    matches = []

    for rgb_timestamp, rgb_msg in rgb_frames:

        index = bisect.bisect_left(
            depth_timestamps,
            rgb_timestamp
        )

        candidates = []

        if index < len(depth_frames):

            candidates.append(
                depth_frames[index]
            )

        if index > 0:

            candidates.append(
                depth_frames[index - 1]
            )

        if not candidates:

            continue

        best_timestamp, best_depth = min(
            candidates,
            key=lambda item:
                abs(
                    item[0]
                    -
                    rgb_timestamp
                )
        )

        difference = abs(
            best_timestamp
            -
            rgb_timestamp
        )

        if difference <= MAX_TIME_DIFF_NS:

            matches.append(
                (
                    rgb_timestamp,
                    rgb_msg,
                    best_timestamp,
                    best_depth
                )
            )


    # ========================================================
    # SYNCHRONIZATION
    # ========================================================

    print("\n========================================")
    print("SYNCHRONIZATION RESULTS")
    print("========================================")

    print(
        f"Matched RGB-D frames: "
        f"{len(matches)}"
    )

    if matches:

        differences_ms = [

            abs(
                depth_timestamp
                -
                rgb_timestamp
            )
            /
            1_000_000.0

            for
            rgb_timestamp,
            rgb_msg,
            depth_timestamp,
            depth_msg
            in matches
        ]

        print(
            f"Minimum: "
            f"{min(differences_ms):.3f} ms"
        )

        print(
            f"Maximum: "
            f"{max(differences_ms):.3f} ms"
        )

        print(
            f"Average: "
            f"{np.mean(differences_ms):.3f} ms"
        )

        print(
            f"Median: "
            f"{np.median(differences_ms):.3f} ms"
        )


    # ========================================================
    # CAMERA INTRINSICS
    # ========================================================

    if camera_info_frames:

        info = (
            camera_info_frames[0][1]
        )

        print("\n========================================")
        print("CAMERA INTRINSICS")
        print("========================================")

        print(
            f"Width:  {info.width}"
        )

        print(
            f"Height: {info.height}"
        )

        print(
            f"fx: {info.k[0]}"
        )

        print(
            f"fy: {info.k[4]}"
        )

        print(
            f"cx: {info.k[2]}"
        )

        print(
            f"cy: {info.k[5]}"
        )


    # ========================================================
    # ANALYZE MULTIPLE DEPTH FRAMES
    # ========================================================

    if matches:

        print("\n")
        print("========================================")
        print("MULTI-FRAME DEPTH CHECK")
        print("========================================")

        # Analyze several frames throughout the bag.
        indices = np.linspace(
            0,
            len(matches) - 1,
            5,
            dtype=int
        )

        for i, index in enumerate(indices):

            depth_msg = matches[index][3]

            analyze_depth(
                depth_msg,
                index
            )


    print("\n========================================")
    print("VALIDATION COMPLETE")
    print("========================================")


# ============================================================
# MAIN
# ============================================================

def main():

    if not os.path.exists(
        BAG_PATH
    ):

        print(
            "\nERROR: Bag directory does not exist:"
        )

        print(
            BAG_PATH
        )

        sys.exit(1)


    read_bag(
        BAG_PATH
    )


if __name__ == "__main__":

    main()
