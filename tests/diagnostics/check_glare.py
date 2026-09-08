#!/usr/bin/env python3

import csv

with open("rgbd_odometry_diagnostics.csv", newline="") as f:
    rows = list(csv.DictReader(f))

print(f"{'frame':>5}  {'gray_mean':>10}  {'gray_std':>9}  {'raw_matches':>11}  {'reason':<20}")

for row in rows:
    frame = int(row["frame"])
    if 850 <= frame <= 1060:
        print(
            f"{frame:5d}  "
            f"{float(row['gray_mean']):10.1f}  "
            f"{float(row['gray_std']):9.1f}  "
            f"{row['raw_matches']:>11}  "
            f"{row['reason']:<20}"
        )
