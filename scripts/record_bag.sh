#!/bin/bash
# record_bag.sh — records the topics the reconstruction pipeline depends on
# Usage: ./record_bag.sh [bag_name]   (run from the 3d_construct/ root or scripts/ dir)

BAG_NAME=${1:-rosbag2_$(date +%Y_%m_%d-%H_%M_%S)}

ros2 bag record -o "$BAG_NAME" \
  /tf_static \
  /camera/rgb/image_raw \
  /camera/rgb/camera_info \
  /camera/depth_registered/image_raw \
  /camera/depth_registered/points
