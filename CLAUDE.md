# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

ROS 2 deployment system for the Tron2 humanoid robot using GR00T (Nvidia GR00T-N1.5-3B) policy. The system runs the robot arm through learned pick-and-place tasks.

## Architecture

Two separate processes in two separate conda environments communicate over TCP:

- **Server** (`server.py`, conda `py310`): ROS 2 subscriber + robot WebSocket controller
- **Client** (`client.py`, conda `lerobot_py310`): GR00T model inference only

They communicate using a 4-step handshake per cycle (see `protocol.py`):
1. Client → Server: `{"cmd": "get_action"}`
2. Server → Client: `Observation(task_text, state16, images)`
3. Client → Server: `ActionChunk(chunk)` or `{"cmd": "skip"}`
4. Server executes the chunk on the robot, then loops

Wire format: 4-byte big-endian uint32 length prefix + pickle payload.

## Run order

```bash
make start      # warmup: arm [0]*14 → WP3 (~26s, once per power-on)
make server     # start server in py310 env (terminal 1)
make client     # start client in lerobot_py310 env (terminal 2)
# Ctrl+C both when done
make shutdown   # park arm: WP3 → [0]*14
```

Each `make` target activates the right conda env and sources ROS 2 (`/opt/ros/humble/setup.bash`). Targets also disable ufw and set `NO_PROXY=10.192.1.2,localhost,127.0.0.1` so the robot WebSocket is reachable.

## Offline inference test (lerobot_py310 env)

```bash
python test_infer_on_dataset.py --indices 100,500,2000,10000 --episode 0
```

Tests the load → preprocess → model → postprocess pipeline against training data without a live robot. Computes MAE vs. ground-truth actions.

## Module map

| File | Role |
|------|------|
| `config.py` | All shared constants — edit this first for any tuning |
| `protocol.py` | TCP message protocol (size-prefixed pickle), `Observation`/`ActionChunk` dataclasses |
| `tron2_ws.py` | Robot WebSocket client: wire protocol, motion interpolation, warmup/shutdown sequences |
| `observer.py` | ROS 2 node buffering joint state + 3 camera topics; `build_state_reorder` aligns live joint names to dataset schema |
| `inference.py` | `Policy` protocol + `build_policy(backend)` factory; `GR00TRunner` is the default backend |
| `server.py` | Combines ROS observer + WebSocket sender; accepts one TCP client at a time |
| `client.py` | TCP loop: request obs → infer → send chunk |
| `start.py` / `shutdown.py` | Warmup / park entry points |
| `controller.py` | Standalone OOP WebSocket controller (scripting use) |
| `test.py` | Interactive joint controller |
| `test_infer_on_dataset.py` | Offline inference smoke test |
| `subscribe_and_viz.py` | ROS 2 topic visualization |

## Data flow

```
Robot sensors (ROS 2 DDS)
    ↓
Tron2Observer (observer.py)   ← joint_states + /gripper_state + 3 CompressedImage topics
    ↓
server.py  ──TCP──  client.py
    ↓                   ↓
WebSocket movej     GR00T inference (lerobot_py310)
    ↓
Robot at ws://10.192.1.2:5000
```

## State and action dimensions

All state/action vectors are **16-dimensional**: 14 arm joints (radians) + 2 gripper values.

- **Robot → model** (observer to server): gripper is 0–100, divided by 100 before sending to client
- **Model → robot** (client chunk to server): gripper is 0–1, multiplied by 100 before `send_gripper()`
- **`CONSUME_STEPS`** (default 30): server only executes the first N rows of the K-step chunk the model returns

## Critical sign/ordering details

- **Left-arm sign flip**: indices `[0,1,2,3,5,6,8,9,13]` in `LEFT_FLIP_IDX` are negated in `_send_step()` before sending to the robot (`request_movej` uses a different sign convention than the dataset)
- **Joint ordering**: `build_state_reorder` maps live ROS joint names to the training-dataset column order; logged once at server startup — check the log if predictions look wrong
- **Camera slots**: `left_wrist` ← `/camera/left/...`, `cam_high` ← `/camera/top/...`, `right_wrist` ← `/camera/right/...`

## Environment variables (overrides for `config.py` defaults)

| Variable | Default | Purpose |
|---|---|---|
| `TRON2_SERVER_HOST` | `127.0.0.1` | TCP bind/connect address |
| `TRON2_SERVER_PORT` | `5555` | TCP port |
| `TRON2_MODEL_BACKEND` | `gr00t` | Policy backend selected by `inference.build_policy` |
| `TRON2_DEVICE` | `cuda:0` | Torch device for inference |
| `TRON2_TASK_TEXT` | `pick_up_stones_and_place_them_into_the_container` | Language condition (server fills `Observation.task_text`) |
| `TRON2_CONSUME_STEPS` | `30` | Rows of action chunk to execute |
| `TRON2_CKPT` | `trantor2nd/tron2_gr00t_pick_step6k` | GR00T checkpoint — HF repo id **or** local snapshot dir (`GR00T_CHECKPOINT`) |
| `BASE_MODEL_PATH` | auto-discover, else download `nvidia/GR00T-N1.5-3B` | Base model — HF repo id or local path (`GR00T_BASE_MODEL_PATH`) |

`TRON2_CKPT` and `BASE_MODEL_PATH` accept either a local path or an HF repo id. Repo ids are resolved via `huggingface_hub.snapshot_download`, which reuses `HF_HOME` cache when present (default `/home/data/hf`). To switch checkpoints, just set `TRON2_CKPT` to a different repo id — no path edits needed. `HF_HUB_OFFLINE=1` (set by `client.py`) means cache misses fail fast; unset it once when introducing a new repo to let it download.

`client.py` also hard-codes these env vars at startup to suppress flash attention and use offline HF cache:
```
HF_HUB_OFFLINE=1  ATTENTION_IMPLEMENTATION=eager
HF_USE_FLASH_ATTENTION_2=0  FLASH_ATTENTION_SKIP_CUDA_BUILD=1
```

## Adding a new policy backend

`client.py` only knows about `inference.build_policy(backend)`; it never names a specific model. To add a new backend:

1. **Implement a class in `inference.py`** with an `infer(left_wrist_bgr, cam_high_bgr, right_wrist_bgr, state16, task_text) -> (K, 16) np.ndarray` method (matches the `Policy` protocol).
2. **Add backend-specific knobs to `config.py`** with a name prefix (e.g. `MYMODEL_CHECKPOINT`, `MYMODEL_DEVICE`).
3. **Add an `elif backend == "mymodel":` branch in `build_policy`** that reads those config values and constructs the class.
4. **Set `TRON2_MODEL_BACKEND=mymodel`** (or change the default in `config.py`).

The `Observation` message already carries `task_text` so different backends can share the same observation pipeline without protocol changes.
