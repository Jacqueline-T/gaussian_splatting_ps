#!/usr/bin/env python3

import csv

DIAGNOSTIC_FILE = "rgbd_odometry_diagnostics.csv"

with open(DIAGNOSTIC_FILE, newline="") as f:
    rows = list(csv.DictReader(f))

# Walk the rows and group consecutive non-accepted frames into gaps.
gaps = []
current_gap = None

for row in rows:
    accepted = row["accepted"] == "True"
    frame = int(row["frame"])
    reason = row["reason"]

    if not accepted:
        if current_gap is None:
            current_gap = {
                "start": frame,
                "end": frame,
                "reasons": {reason: 1}
            }
        else:
            current_gap["end"] = frame
            current_gap["reasons"][reason] = (
                current_gap["reasons"].get(reason, 0) + 1
            )
    else:
        if current_gap is not None:
            gaps.append(current_gap)
            current_gap = None

if current_gap is not None:
    gaps.append(current_gap)

print(f"Total gaps (contiguous non-accepted stretches): {len(gaps)}\n")

for g in sorted(gaps, key=lambda x: -(x["end"] - x["start"])):
    length = g["end"] - g["start"] + 1
    print(
        f"Frames {g['start']:4d}-{g['end']:4d}  "
        f"(length {length:3d})  "
        f"reasons={g['reasons']}"
    )
