"""Shared configuration for Tron2 deployment."""

from __future__ import annotations

import os

# ── Robot WebSocket ───────────────────────────────────────────────────────────
ROBOT_IP = "10.192.1.2"
ROBOT_WS_PORT = 5000
ROBOT_WS_URL = f"ws://{ROBOT_IP}:{ROBOT_WS_PORT}"

# ── Motion timing ────────────────────────────────────────────────────────────
MOVE_TIME = 0.05           # request_movej.time field, seconds
SEND_INTERVAL = 0.05       # pause between successive movej sends, seconds
MAX_JOINT_STEP = 0.1       # max per-send delta on any joint (safety)
WARMUP_HOLD_SECONDS = 3.0  # hold duration at each warmup waypoint

# After a chunk's last movej, wait this long so the robot physically settles
# AND the ROS topics (joint_states, decoded camera frames) catch up. A short
# sleep here is the main cause of "rebound": the next observation reflects the
# arm mid-motion, so the next chunk replans from a stale state.
SETTLE_SECONDS = float(os.environ.get("TRON2_SETTLE_SECONDS", "0.25"))

# ── Observation freshness ────────────────────────────────────────────────────
# Each (state, images) pair handed to the policy must:
#   1. Have joint state and every image younger than MAX_OBS_AGE / MAX_IMG_AGE.
#   2. Have all stamps within MAX_STAMP_SPREAD of each other so the model sees
#      a temporally coherent snapshot, not "fresh state + 200 ms-old image".
MAX_OBS_AGE = float(os.environ.get("TRON2_MAX_OBS_AGE", "0.05"))
MAX_IMG_AGE = float(os.environ.get("TRON2_MAX_IMG_AGE", "0.05"))
MAX_STAMP_SPREAD = float(os.environ.get("TRON2_MAX_STAMP_SPREAD", "0.05"))

# ── Server-Client TCP ────────────────────────────────────────────────────────
SERVER_HOST = os.environ.get("TRON2_SERVER_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("TRON2_SERVER_PORT", "5555"))

# ── ROS 2 topics ─────────────────────────────────────────────────────────────
JOINT_TOPIC = "/joint_states"
GRIPPER_TOPIC = "/gripper_state"
CAM_LEFT = "/camera/left/color/image_rect_raw/compressed"
CAM_HIGH = "/camera/top/color/image_raw/compressed"
CAM_RIGHT = "/camera/right/color/image_rect_raw/compressed"

# ── Model ────────────────────────────────────────────────────────────────────
# Selects which policy backend `inference.build_policy` constructs.
# Add new backends in inference.py and dispatch them in build_policy().
MODEL_BACKEND = os.environ.get("TRON2_MODEL_BACKEND", "gr00t")

DEVICE = os.environ.get("TRON2_DEVICE", "cuda:0")
TASK_TEXT = os.environ.get("TRON2_TASK_TEXT", "pick_up_stones_and_place_them_into_the_container")
CONSUME_STEPS = int(os.environ.get("TRON2_CONSUME_STEPS", "16"))

# ── GR00T backend ────────────────────────────────────────────────────────────
# Either an HF repo id (e.g. "trantor2nd/tron2_gr00t_pick_step6k") or a local
# snapshot directory. Repo ids resolve via huggingface_hub.snapshot_download
# and reuse the HF cache (HF_HOME) when present.
GR00T_CHECKPOINT = os.environ.get(
    "TRON2_CKPT",
    "trantor2nd/tron2_gr00t_pick_step6k",
)
# Same dual-form policy: repo id (e.g. "nvidia/GR00T-N1.5-3B") or local path.
# If unset, GR00TRunner auto-discovers a snapshot in the HF cache and falls
# back to downloading nvidia/GR00T-N1.5-3B.
GR00T_BASE_MODEL_PATH = os.environ.get("BASE_MODEL_PATH")

# ── Joint schema ─────────────────────────────────────────────────────────────
# Left-arm joint indices whose sign convention differs between the dataset
# and the robot's request_movej protocol.
LEFT_FLIP_IDX = [0, 1, 2, 3, 5, 6, 8, 9, 13]

# Three-waypoint safe path between [0]*14 and the inference-ready pose.
WARMUP_WAYPOINT_1 = [0.0, 0.23, 1.35, 0.0, 0.0, 0.0, 0.0,
                     0.0, -0.23, -1.35, 0.0, 0.0, 0.0, 0.0]
WARMUP_WAYPOINT_2 = [0.0, 0.23, 1.35, -1.6, 0.0, 0.0, 0.0,
                     0.0, -0.23, -1.35, -1.6, 0.0, 0.0, 0.0]
WARMUP_WAYPOINT_3 = [0.0, 0.23, 0.0, -1.6, 0.23, 0.0, 0.0,
                     0.0, -0.23, 0.0, -1.6, -0.23, 0.0, 0.0]
