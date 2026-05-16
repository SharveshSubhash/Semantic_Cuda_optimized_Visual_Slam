# vslam_semantic — cuVSLAM + NanoOWL + Pixhawk + Redis on a Jetson with a D415

This package is specifically built for Pixhawk based UAV Perception Layer Pipeline with a VLM enriched semantic tagging to the Visual Inertial Odometry based Simultaneous Localization and Mapping Layer output for the Intelligent Robotics Systems Project at National University of Singapore by Sharvesh Subhash


End-to-end ROS 2 package: stereo VIO from a RealSense **D415**, open-vocabulary
detection from **NanoOWL**, optional IMU fusion from a **Pixhawk** flight
controller via uXRCE-DDS, and a Redis-backed JSON semantic graph for the global
planner. Built for **Jetson Orin** (Nano / NX / AGX).

## Nodes in this package

| Node | Role | Subscribes | Publishes |
|---|---|---|---|
| `px4_imu_bridge_node` | PX4 IMU → ROS Imu | `/fmu/out/sensor_combined` (px4_msgs) | `/imu/data_raw` (sensor_msgs/Imu) |
| `nanoowl_inference_node` | Open-vocab detection | `/camera/infra1/image_rect_raw` (or `/camera/color/image_raw`), `/nanoowl/input_query` (std_msgs/String) | `/nanoowl/detections` (vision_msgs/Detection2DArray), `/nanoowl/output_image` (sensor_msgs/Image, debug) |
| `semantic_graph_combiner_node` | Fuse pose + detections | `/visual_slam/tracking/vo_pose_covariance`, `/visual_slam/tracking/odometry`, `/nanoowl/detections`, `/camera/infra1/camera_info` | `/semantic_graph` (std_msgs/String, JSON) |
| `redis_writer_node` | Snapshot → Redis | `/semantic_graph` | (Redis KV + pub/sub) |

External nodes used (not in this package): `realsense2_camera`, `isaac_ros_visual_slam`, `MicroXRCEAgent`.

## Why this exists (and what it replaces)

| Generic VSLAM (D435i) | This Unique package (D415) |
|---|---|
| `enable_gyro: True, enable_accel: True` | Removed — D415 has no IMU |
| `unite_imu_method: 2`, `gyro_fps: 200`, `accel_fps: 200` | Removed |
| `enable_imu_fusion: True` in cuVSLAM | False by default; opt-in via `use_px4_imu` |
| cuVSLAM reads `/camera/imu` | If IMU fusion is on, reads `/imu/data_raw` from PX4 bridge |
| Pose used by MAVROS only | Pose also feeds semantic graph + Redis |
| No semantic perception | Adds NanoOWL inference + spatial landmark graph |

## Topology

```
realsense2_camera (D415, IR-only by default, emitter OFF)
    │  /camera/infra1/image_rect_raw   (60 Hz)
    │  /camera/infra1/camera_info
    │  /camera/infra2/image_rect_raw
    │  /camera/infra2/camera_info
    │  /camera/color/image_raw         (only if enable_color:=true)
    │
    │   ┌──────────────────────────────────────────┐
    │   │  MicroXRCEAgent on Jetson                │
    │   │  /fmu/out/sensor_combined (200 Hz)       │
    │   └────────────────┬─────────────────────────┘
    │                    │
    │       ┌────────────▼────────────┐
    │       │  px4_imu_bridge_node    │ (FRD→FLU, px4_msgs→sensor_msgs/Imu)
    │       │  /imu/data_raw          │
    │       └────────────┬────────────┘
    │                    │ (only when use_px4_imu:=true)
    │                    ▼
    │      ┌─────────────────────────────────────┐
    │      │  isaac_ros_visual_slam              │
    │      │  /visual_slam/tracking/vo_pose_covariance │
    │      │  /visual_slam/tracking/odometry     │
    │      │  /tf  (odom→base→camera_*)          │
    │      └────────────┬────────────────────────┘
    │                   │
    │  ┌────────────────┴─┐
    │  │                  │
    ▼  ▼                  ▼
┌────────────────────┐    │   /nanoowl/input_query (std_msgs/String)
│ nanoowl_inference_ │◄───┘   (dynamic prompt update)
│ node               │
│ /nanoowl/detections│
└─────────┬──────────┘
          │
          ▼
┌─────────────────────────────────────┐
│  semantic_graph_combiner_node       │
│  /semantic_graph (JSON String)      │
└─────────────┬───────────────────────┘
              ▼
       ┌────────────────────┐
       │ redis_writer_node  │
       └─────────┬──────────┘
                 ▼
           ┌──────────┐
           │  Redis   │  semantic_graph:latest, :node:<id>, :robot_pose, ...
           └──────────┘
```

## D415 RealSense topics — what we actually use

Only **four** RealSense topics are inputs to cuVSLAM (plus optionally the color
stream for NanoOWL):

| Topic | cuVSLAM | NanoOWL | Combiner |
|---|:---:|:---:|:---:|
| `/camera/infra1/image_rect_raw` | ✔ left input | ✔ default | — |
| `/camera/infra1/camera_info` | ✔ left intrinsics | — | ✔ back-projection |
| `/camera/infra2/image_rect_raw` | ✔ right input | — | — |
| `/camera/infra2/camera_info` | ✔ right intrinsics | — | — |
| `/camera/color/image_raw` | — | ✔ if `enable_color:=true` | — |
| `/camera/color/camera_info` | — | — | ✔ if `enable_color:=true` |
| `/camera/depth/*`, `/camera/aligned_depth_to_color/*`, `/camera/extrinsics/*` | — | — | — |

**Critical:** the IR projector must be off (`depth_module.emitter_enabled: 0`).
With it on, cuVSLAM tracks the projected dots and drifts. The launch file
already enforces this.

## Pixhawk side: get the IMU out of the FCU

In QGroundControl set:

| Parameter | Value | Notes |
|---|---|---|
| `UXRCE_DDS_CFG` | `TELEM2` | Or whichever UART We wired to the Jetson |
| `SER_TEL2_BAUD` | `921600` | Maximum recommended on TELEM2 |
| `MAV_1_CONFIG` | `Disabled` | Free TELEM2 from MAVLink |
| `IMU_INTEG_RATE` | `200` | Match cuVSLAM's `calibration_frequency` |
| `UXRCE_DDS_SYNCT` | `1` | Enable timestamp sync |

Then on the Jetson:

```bash
MicroXRCEAgent serial --dev /dev/ttyTHS1 -b 921600
# or UDP for ethernet: MicroXRCEAgent udp4 -p 8888

# Verify
ros2 topic hz /fmu/out/sensor_combined   # expected ~200 Hz
```

## tf We must publish

cuVSLAM publishes `odom → base_frame` automatically. You must add the static
transforms between the robot links, the camera, and (if fusing) the IMU.

```bash
# camera_link is the cuVSLAM base_frame in the launch file.
ros2 run tf2_ros static_transform_publisher \
    --x 0.10 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 \
    --frame-id base_link --child-frame-id camera_link

# Pixhawk IMU - measure this on the airframe.
ros2 run tf2_ros static_transform_publisher \
    --x -0.05 --y 0 --z 0 --yaw 0 --pitch 0 --roll 0 \
    --frame-id base_link --child-frame-id imu_link
```

`realsense2_camera` publishes the `camera_link → camera_*_optical_frame` chain
itself; We do not need to add those.

---

# Step-by-step installation on Jetson

This is the complete, in-order install for a fresh JetPack 6.x SD card / SSD.
Estimated time: **45–90 minutes** end-to-end, of which most is the one-time
NanoOWL engine build.

## Step 1 — Verify the JetPack and CUDA stack

```bash
# Confirm JetPack version (need 5.x or 6.x for cuVSLAM)
cat /etc/nv_tegra_release
sudo apt-cache show nvidia-jetpack | head -5

# CUDA, cuDNN, TensorRT come pre-installed with JetPack. Confirm:
nvcc --version                          # CUDA toolkit
dpkg -l | grep TensorRT | head -5       # TensorRT
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

If `import tensorrt` fails with `ModuleNotFoundError`:

```bash
sudo apt-get install -y python3-libnvinfer-dev
# If still missing, the bindings live in dist-packages, not site-packages:
echo 'export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc
```

## Step 2 — Add swap (Orin Nano only, skip on AGX)

The NanoOWL engine build consumes ~6 GB peak RAM. The 8 GB Orin Nano needs
swap, otherwise the build hangs forever ("maxes out RAM and swap memory"
is a known failure mode — see nanoowl issue #33).

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
free -h    # should now show 8 G of swap
```

## Step 3 — Install the Isaac ROS dev environment

Follow NVIDIA's official setup for the Isaac ROS Docker container:
<https://nvidia-isaac-ros.github.io/getting_started/dev_env_setup.html>.

In short:

```bash
mkdir -p ~/workspaces/isaac_ros-dev/src
echo 'export ISAAC_ROS_WS=$HOME/workspaces/isaac_ros-dev' >> ~/.bashrc
source ~/.bashrc

cd $ISAAC_ROS_WS/src
git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_common.git
```

## Step 4 — Clone all required repositories into the workspace

```bash
cd $ISAAC_ROS_WS/src

# Required Isaac ROS bits
git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_visual_slam.git
git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_image_pipeline.git
git clone --depth 1 https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nitros.git

# NanoOWL + dependencies
git clone --depth 1 https://github.com/NVIDIA-AI-IOT/nanoowl.git
git clone --depth 1 https://github.com/NVIDIA-AI-IOT/torch2trt.git

# px4_msgs for the IMU bridge
git clone --depth 1 https://github.com/PX4/px4_msgs.git

# This package
git clone <the-fork-or-tarball> vslam_semantic
# (or just: cp -r /path/to/vslam_semantic $ISAAC_ROS_WS/src/)
```

## Step 5 — Enter the Isaac ROS container

```bash
cd $ISAAC_ROS_WS/src/isaac_ros_common
./scripts/run_dev.sh
```

Everything from Step 6 onward runs **inside the container**.

## Step 6 — Install system + Python dependencies

```bash
# Inside the container
sudo apt-get update
sudo apt-get install -y \
    ros-humble-realsense2-camera \
    ros-humble-vision-msgs \
    ros-humble-cv-bridge \
    ros-humble-topic-tools \
    python3-opencv \
    redis-server \
    redis-tools \
    jq

pip3 install --upgrade \
    redis \
    "transformers>=4.36.0,<5" \
    "Pillow>=10.0.0" \
    matplotlib \
    aiohttp \
    onnx
```

## Step 7 — Install torch2trt and NanoOWL (Python packages)

Easiest path: use the helper script shipped with this package. It is idempotent
and verifies each step.

```bash
chmod +x $ISAAC_ROS_WS/src/vslam_semantic/scripts/build_nanoowl_engine.sh
$ISAAC_ROS_WS/src/vslam_semantic/scripts/build_nanoowl_engine.sh
```

That script does **everything** in Steps 7 + 8 (it installs torch2trt, installs
nanoowl, downloads the OWL-ViT weights, exports to ONNX, builds the TensorRT
engine, and runs a smoke-test inference). If We prefer to do it manually:

```bash
# Inside the container
sudo mkdir -p /opt
sudo chown $USER /opt

# torch2trt
git clone --depth 1 https://github.com/NVIDIA-AI-IOT/torch2trt /opt/torch2trt
cd /opt/torch2trt
sudo python3 setup.py develop

# NanoOWL package
git clone --depth 1 https://github.com/NVIDIA-AI-IOT/nanoowl /opt/nanoowl
cd /opt/nanoowl
sudo python3 setup.py develop
sudo mkdir -p /opt/nanoowl/data && sudo chmod a+rwx /opt/nanoowl/data
```

## Step 8 — Build the NanoOWL TensorRT engine (model weights)

NanoOWL doesn't ship "weights" as files We can wget. Instead the build step
downloads the HuggingFace `google/owlvit-base-patch32` checkpoint (~600 MB) on
first run, exports it to ONNX, and compiles a `.engine` for the Jetson's
exact CUDA/TensorRT versions.

**This is a one-time operation**: the resulting engine is portable across
reboots of the same Jetson but **not** across Jetson SKUs or TensorRT versions
(rebuild after a JetPack upgrade).

```bash
# Inside the container (skip if We ran build_nanoowl_engine.sh above)
cd /opt/nanoowl
python3 -m nanoowl.build_image_encoder_engine \
    /opt/nanoowl/data/owl_image_encoder_patch32.engine \
    --model_name google/owlvit-base-patch32
```

What happens:

1. `transformers` downloads `google/owlvit-base-patch32` weights from
   HuggingFace into `~/.cache/huggingface/` (~600 MB). You can pre-seed this
   by `huggingface-cli download google/owlvit-base-patch32` if the Jetson is
   behind a firewall.
2. The PyTorch model is traced to ONNX.
3. TensorRT (via `trtexec` or the Python API) builds the engine in FP16. This
   is the slow step — 5–15 minutes on Orin Nano, 1–3 min on AGX Orin.

Expected output:

```
[TensorRT] [I] Total Activation Memory: ...
[TensorRT] [I] Engine generation completed in ... seconds.
```

And the file:

```bash
ls -lh /opt/nanoowl/data/owl_image_encoder_patch32.engine
# ~120 MB on Orin Nano (FP16)
```

### Verify NanoOWL outside ROS first

Always confirm raw NanoOWL works before bringing ROS into the picture:

```bash
cd /opt/nanoowl/examples
python3 owl_predict.py \
    --prompt='[an owl, a glove]' \
    --threshold=0.1 \
    --image_encoder_engine=/opt/nanoowl/data/owl_image_encoder_patch32.engine
ls /opt/nanoowl/data/owl_predict_out.jpg   # should now exist
```

If this fails, **do not move on**. The error here will be much clearer than
debugging the same problem through a ROS launch file.

### Picking a different model

The default `google/owlvit-base-patch32` runs at ~25 FPS on AGX Orin (95 FPS in
the NanoOWL README's table). If accuracy matters more than throughput, try the
B/16 variant:

```bash
python3 -m nanoowl.build_image_encoder_engine \
    /opt/nanoowl/data/owl_image_encoder_patch16.engine \
    --model_name google/owlvit-base-patch16
```

Then pass `nanoowl_engine:=/opt/nanoowl/data/owl_image_encoder_patch16.engine`
when We launch.

| Model | Image size | AGX Orin FPS | mAP | Engine size |
|---|---|---|---|---|
| OWL-ViT B/32 (default) | 768 | ~95 | 28 | ~120 MB |
| OWL-ViT B/16 | 768 | ~25 | 31.7 | ~125 MB |

## Step 9 — Build the workspace

```bash
cd $ISAAC_ROS_WS

# Build px4_msgs first (the IMU bridge depends on it)
colcon build --packages-select px4_msgs --symlink-install
source install/setup.bash

# Then the rest
colcon build --symlink-install \
    --packages-up-to vslam_semantic isaac_ros_visual_slam
source install/setup.bash
```

## Step 10 — Start Redis

```bash
sudo systemctl enable --now redis-server
redis-cli ping     # PONG
```

If the container doesn't have systemd, start the daemon manually:

```bash
redis-server --daemonize yes
```

## Step 11 — Launch

**Default (pure stereo cuVSLAM, NanoOWL on IR, Redis on localhost):**

```bash
ros2 launch vslam_semantic cuvslam_d415.launch.py
```

**With Pixhawk IMU fusion:**

```bash
ros2 launch vslam_semantic cuvslam_d415.launch.py use_px4_imu:=true
```

**With color stream feeding NanoOWL** (slightly better detection quality, costs
USB3 bandwidth):

```bash
ros2 launch vslam_semantic cuvslam_d415.launch.py enable_color:=true
```

**Override the OWL prompt at launch:**

```bash
ros2 launch vslam_semantic cuvslam_d415.launch.py \
    nanoowl_prompt:='a person, a forklift, a pallet, a ladder, a fire extinguisher'
```

**Change the prompt at runtime** (works while running):

```bash
ros2 topic pub --once /nanoowl/input_query std_msgs/String \
    "data: 'a forklift, a pallet, a person'"
```

## Step 12 — Smoke tests (run in order)

```bash
# 1. Camera at 60 Hz
ros2 topic hz /camera/infra1/image_rect_raw      # ~60 Hz

# 2. cuVSLAM publishing
ros2 topic hz /visual_slam/tracking/vo_pose_covariance   # ~60 Hz

# 3. (only if use_px4_imu:=true) IMU bridge alive
ros2 topic hz /imu/data_raw                       # ~200 Hz

# 4. NanoOWL inference
ros2 topic hz /nanoowl/detections                 # ~10 Hz (matches inference_period_sec)
ros2 topic echo /nanoowl/detections --once | head -40

# 5. Visual debug (open in RViz or Foxglove)
#    Topic: /nanoowl/output_image  - shows boxes + labels overlaid

# 6. Combiner publishing graph
ros2 topic hz /semantic_graph                     # ~5 Hz
ros2 topic echo /semantic_graph --once | python3 -m json.tool | head -40

# 7. Redis has the snapshot
redis-cli GET semantic_graph:latest | python3 -m json.tool | head -40
redis-cli GET semantic_graph:robot_pose
redis-cli ZRANGE semantic_graph:nodes_index 0 -1 WITHSCORES
redis-cli SUBSCRIBE semantic_graph                # Ctrl-C to exit; live JSON pushes
```

---

## Python API

The package is importable as a regular Python module and exposes:

```python
import vslam_semantic
vslam_semantic.__version__       # '0.1.0'

# Pure helpers (numpy-only, no ROS deps - safe to import anywhere)
from vslam_semantic import quat_to_rotmat, triangulate_bearing_rays, first_attr

# Node entry points (each lazily imports rclpy + its own deps on first call)
from vslam_semantic import (
    px4_imu_bridge_main,
    nanoowl_inference_main,
    semantic_graph_combiner_main,
    redis_writer_main,
)
```

Imports are **lazy**: `import vslam_semantic` does NOT pull in `rclpy`,
`nanoowl`, `redis`, or `px4_msgs`. Each node main only loads its dependencies
when called. This lets We:

  * Run unit tests on the math helpers without a ROS environment
  * Run the IMU bridge on a machine that doesn't have NanoOWL installed
  * Import `vslam_semantic.__version__` from a deployment script

The three pure helpers in `vslam_semantic._helpers` are useful even outside this
package:

  * `quat_to_rotmat(qx, qy, qz, qw)` → 3×3 numpy rotation matrix. Identity quat
    maps to I, handles degenerate (near-zero norm) inputs as identity to avoid
    NaNs from cuVSLAM's first-frame zero-quaternion.
  * `triangulate_bearing_rays(rays, fallback_range)` → closed-form least-squares
    intersection of a set of `(origin, unit_direction)` rays. Falls back
    gracefully on parallel rays or single-ray input.
  * `first_attr(obj, names)` → first non-None attribute, used to read fields
    from objects whose names drift across library versions.

---

## Why this NanoOWL node and not NVIDIA's ROS2-NanoOWL?

NVIDIA's `ROS2-NanoOWL` is a fine reference, but our combiner needs three things
it doesn't provide:

1. **Label-as-string class_id.** ROS2-NanoOWL stamps each detection's
   `class_id` as the *index* of the prompt in the prompt list. Our combiner
   merges landmarks by label, so we need the actual text. This node puts the
   text in `class_id`.
2. **Mono/IR input handling.** ROS2-NanoOWL expects RGB. The D415 publishes
   IR as `mono8` (or `mono16` if requested). This node accepts mono8, mono16,
   rgb8, and bgr8 transparently — so we can feed it the same IR stream that
   cuVSLAM is using, getting perfect time-alignment for free.
3. **Dynamic prompt updates that survive re-encoding.** This node caches the
   text encodings and only re-runs them when the prompt actually changes,
   instead of paying ~500 ms per frame.

If We'd rather use NVIDIA's node, We can — just point the combiner at
`/output_detections` instead of `/nanoowl/detections`, and add a small
relay/remapper to convert the integer class_id to a label string.

## Common failure modes and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `import tensorrt` ModuleNotFound | Bindings in dist-packages | `export PYTHONPATH=/usr/lib/python3.10/dist-packages:$PYTHONPATH` |
| Engine build hangs on Orin Nano | Out of RAM | Add 8 GB swap (Step 2) |
| NanoOWL node logs "Cannot import NanoOWL" | Step 7 didn't install into the right Python | Run the helper script inside the container, not on host |
| `/nanoowl/detections` always empty | Prompt isn't set | `ros2 topic echo /nanoowl/input_query` — if no publisher, set `initial_prompt` |
| Detections never join the graph | Pose-vs-detection time skew > slop | Increase combiner's `sync_slop_sec`, check clocks |
| cuVSLAM pose drifts at ~constant velocity | IR emitter is on | Confirm `depth_module.emitter_enabled: 0` |
| `/fmu/out/sensor_combined` empty | Wrong QoS or agent not running | Bridge already uses BEST_EFFORT; restart `MicroXRCEAgent` |
| IMU bridge runs at 60 Hz, not 200 | `IMU_INTEG_RATE` too low on FCU | Set to 200 in QGC |
| cuVSLAM with IMU never initializes | `imu_link` tf missing or FRD vs FLU | Publish static tf; bridge already handles FRD→FLU |
| `class_id` is an integer not a label | Using NVIDIA's ROS2-NanoOWL instead of this | Use this package's `nanoowl_inference_node` |
| Redis writes silently fail | Redis server down | Writer logs every 10 s; `redis-cli ping` |

## File map

```
vslam_semantic/
├── README.md                            # this file
├── package.xml                          # ROS 2 package manifest
├── setup.py                             # ament_python build + entry points
├── setup.cfg                            # ament_python install-path config
├── resource/
│   └── vslam_semantic                   # empty ament marker
├── launch/
│   └── cuvslam_d415.launch.py           # top-level launch
├── scripts/
│   └── build_nanoowl_engine.sh          # one-shot NanoOWL installer
└── vslam_semantic/                      # Python package
    ├── __init__.py                      # lazy-import package init + public API
    ├── _helpers.py                      # ROS-free math (quat, triangulation, first_attr)
    ├── nanoowl_inference_node.py        # OWL-ViT TensorRT inference -> Detection2DArray
    ├── px4_imu_bridge_node.py           # PX4 SensorCombined -> sensor_msgs/Imu (FRD->FLU)
    ├── semantic_graph_combiner_node.py  # pose + detections -> graph
    └── redis_writer_node.py             # graph -> Redis
```
