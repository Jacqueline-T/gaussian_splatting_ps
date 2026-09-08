# gaussian_splatting_ps

RGB-D 3D reconstruction pipeline (ROS2 + ASUS Xtion) — visual odometry via
ORB+RANSAC, Gaussian splatting via Nerfstudio/gsplat.

## Pipeline overview

```
ROS2 bag (.db3)
      │
      ▼
tests/validation_test.py      (visual odometry: ORB + RANSAC)
      │
      ├──────────────┬────────────────┐
      ▼               ▼
fusion_pipe_1.py   transform_pipe_2.py
(TSDF fusion)      (pose conversion)
      │               │
      ▼               ▼
fused_pointcloud.ply  transforms.json
      │               │
      └───────┬────────┘
              ▼
   tests/verify_transforms.py   (optional visual sanity check)
              │
              ▼
         Nerfstudio (ns-train splatfacto)
              │
              ▼
        outputs/ (trained model, viewer)
```

`fusion_pipe_1.py` and `transform_pipe_2.py` are independent branches that
both read `validation_test.py`'s output — neither depends on the other.
Both outputs (`fused_pointcloud.ply` and `transforms.json`) are needed
together by Nerfstudio.

## Requirements

- **ROS2** (sourced environment) — `validation_test.py` depends on `rclpy`
  and `rosidl_runtime_py`, which are not pip-installable; they must come
  from a sourced ROS2 install.
- **Python packages**: `opencv-python`, `numpy`, `open3d`. Install via:
  ```bash
  pip install opencv-python numpy open3d
  ```
- **Nerfstudio**, installed in its own conda environment (referred to
  below as `nerfstudio`) — see the
  [official Nerfstudio install guide](https://docs.nerf.studio/) for
  setup.
- An ASUS Xtion / PrimeSense-class RGB-D camera, recording to a ROS2 bag
  with the following topics:
  - `/camera/rgb/image_raw`
  - `/camera/depth_registered/image_raw`
  - `/camera/rgb/camera_info` (optional — falls back to hardcoded
    intrinsics if absent, see `validation_test.py`)

## Recording a bag

```bash
./scripts/record_bag.sh
```

Records the required topics to a timestamped `rosbag2_*/` folder.

## Running the pipeline

All three scripts below take a shared `--input-dir` / `--output-dir`
so their outputs land in one place. From `tests/`:

**1. Visual odometry**
```bash
python3 validation_test.py --bag-path /path/to/rosbag2_folder --output-dir ./output
```
Produces, inside `./output/`:
- `rgbd_trajectory_diagnostic.csv` — accepted camera poses
- `rgbd_odometry_diagnostics.csv` — per-frame diagnostic log
- `camera_intrinsics.json` — active camera calibration
- `odometry_trusted_frames/` — RGB/depth PNGs for accepted frames
- `odometry_debug_frames/` — RGB PNGs for rejected/weak-tracking frames

Optional: `--max-frames N` to process only the first N frame transitions
(useful for a quick test run).

**2. Point cloud fusion** (from the repo root)
```bash
python3 fusion_pipe_1.py --input-dir tests/output
```
Produces `tests/output/fused_pointcloud.ply`. Viewable in MeshLab, Open3D,
or CloudCompare as a first sanity check.

**3. Transform export for Nerfstudio** (from the repo root)
```bash
python3 transform_pipe_2.py --input-dir tests/output
```
Produces `tests/output/transforms.json`. Supports `--limit-frames N` for
a smoke-test export of just the first N frames.

**4. (Optional) Visual pose verification**
```bash
cd tests
python3 verify_transforms.py --transforms output/transforms.json --ply output/fused_pointcloud.ply
```
Opens an Open3D viewer showing camera frustums and the trajectory path
over the point cloud. If `draw_geometries()` fails to open a window
(e.g. under WSL/WSLg), add `--headless` to render a PNG screenshot
instead:
```bash
python3 verify_transforms.py --transforms output/transforms.json --ply output/fused_pointcloud.ply --headless
```
This step is optional — Nerfstudio's own viewer is the authoritative
check of whether the reconstruction is correct. Use this script only as
a quick pre-flight sanity check before committing to a full training run.

**5. Train with Nerfstudio**
```bash
conda activate nerfstudio
ns-train splatfacto --data tests/output
```

## Repository structure

```
3d_construct/
├── scripts/
│   └── record_bag.sh          # records the required ROS2 topics to a bag
├── fusion_pipe_1.py            # TSDF fusion → fused_pointcloud.ply
├── transform_pipe_2.py         # trajectory + intrinsics → transforms.json
├── tests/
│   ├── validation_test.py      # visual odometry (ORB + RANSAC) — active
│   ├── verify_transforms.py    # visual pose sanity check — optional
│   ├── legacy/                 # superseded odometry drafts, kept for reference
│   └── diagnostics/            # ad-hoc one-off tools (depth accuracy checks,
│                                # frame-failure investigation) — not part of
│                                # the pipeline, used occasionally when
│                                # debugging tracking issues
└── outputs/                    # Nerfstudio training output (gitignored)
```

Generated files (bags, CSVs, intrinsics JSON, trusted/debug frame folders,
point clouds, transforms.json, Nerfstudio outputs) are excluded via
`.gitignore` and are not tracked in this repo.

## Known limitations

- `validation_test.py`'s fallback camera intrinsics (`FX`, `FY`, `CX`,
  `CY`, `WIDTH`, `HEIGHT`) are tuned to this project's specific
  ASUS Xtion camera. If you're using a different sensor and your bag
  doesn't include a `CameraInfo` topic, update these values to match
  your camera's calibration before running.
- `validation_test.py`'s depth-encoding assumption (`32FC1` = already in
  meters) matches this project's driver output. If reusing this with a
  different depth driver, verify your bag's actual depth encoding first.
- `fusion_pipe_1.py`'s TSDF parameters (`VOXEL_SIZE`, `SDF_TRUNC`,
  `DEPTH_TRUNC`) are tuned to this project's scene scale (camera travel
  on the order of tens of cm, objects at 0.4–2.7m depth). Retune for a
  larger or smaller scene.
- `validation_test.py` expects a ROS2 bag stored in SQLite (`.db3`)
  format. Bags using MCAP storage are not currently supported.
