"""Tron2 WebSocket protocol and motion layer.

Provides the shared WebSocket connection to the robot, wire-protocol helpers,
motion interpolation, and warmup/shutdown waypoint sequences.

Usage:
    import tron2_ws

    def task():
        if not tron2_ws.wait_for_accid(15.0):
            return
        tron2_ws.warmup_sequence()
        tron2_ws.close()

    if __name__ == "__main__":
        tron2_ws.run(on_ready=task)
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Callable, Optional, Sequence

import numpy as np
import websocket

from config import (
    MOVE_TIME, SEND_INTERVAL, MAX_JOINT_STEP, WARMUP_HOLD_SECONDS,
    ROBOT_WS_URL,
    WARMUP_WAYPOINT_1, WARMUP_WAYPOINT_2, WARMUP_WAYPOINT_3,
)

# ── Runtime state (module globals match test.py) ─────────────────────────────

ACCID: Optional[str] = None
ws_client: Optional[websocket.WebSocketApp] = None
should_exit = False

joint_values = [0.0] * 14
gripper_values = [0.0, 0.0]

_accid_event = threading.Event()


# ── Wire protocol ────────────────────────────────────────────────────────────

def send_request(title: str, data: Optional[dict] = None) -> None:
    """Send a JSON frame matching the robot's expected shape."""
    if data is None:
        data = {}
    message = {
        "accid": ACCID,
        "title": title,
        "timestamp": int(time.time() * 1000),
        "guid": str(uuid.uuid4()),
        "data": data,
    }
    msg_str = json.dumps(message)
    print(f"\n[Send] {msg_str}")
    if ws_client is not None:
        ws_client.send(msg_str)
    else:
        print("[Error] ws_client is None")


def send_movej() -> None:
    send_request("request_movej", {
        "joint": joint_values,
        "time": MOVE_TIME,
    })


def send_gripper() -> None:
    send_request("request_set_limx_2fclaw_cmd", {
        "left_opening":  float(gripper_values[0]),
        "left_speed":    50.0,
        "left_force":    50.0,
        "right_opening": float(gripper_values[1]),
        "right_speed":   50.0,
        "right_force":   50.0,
    })


# ── Motion helpers ───────────────────────────────────────────────────────────

def interp_send(from_pose: Sequence[float], to_pose: Sequence[float], label: str) -> None:
    """Step joint_values from ``from_pose`` to ``to_pose`` in ≤MAX_JOINT_STEP deltas."""
    global joint_values
    diff = [t - f for f, t in zip(from_pose, to_pose)]
    max_abs = max(abs(d) for d in diff)
    if max_abs < 1e-6:
        print(f"[{label}] from == to, skipping")
        return
    N = max(1, int(np.ceil(max_abs / MAX_JOINT_STEP)))
    print(f"[{label}] interp in {N} steps "
          f"(max|Δ|={max_abs:.3f} rad, ~{N * SEND_INTERVAL:.1f}s)")

    for k in range(1, N + 1):
        if should_exit:
            return
        alpha = k / N
        joint_values = [f + alpha * d for f, d in zip(from_pose, diff)]
        js = "[" + ",".join(f"{x:+.3f}" for x in joint_values) + "]"
        print(f"[{label}][{k:2d}/{N}] alpha={alpha:.3f} joint={js}")
        send_movej()
        time.sleep(SEND_INTERVAL)


def hold(seconds: float, label: str) -> None:
    """Sit on the last commanded pose for ``seconds`` so the arm can arrive."""
    if should_exit or seconds <= 0:
        return
    print(f"[{label}] holding commanded pose for {seconds:.1f}s")
    end = time.monotonic() + seconds
    while time.monotonic() < end and not should_exit:
        time.sleep(0.1)


def step_toward(target: Sequence[float], max_step: float) -> bool:
    """Advance joint_values toward target by at most max_step on any joint.

    Returns True when the target was reached this call.
    """
    global joint_values
    diff = [t - j for t, j in zip(target, joint_values)]
    max_abs = max(abs(d) for d in diff) if diff else 0.0
    if max_abs <= max_step:
        joint_values = [float(t) for t in target]
        return True
    scale = max_step / max_abs
    joint_values = [j + scale * d for j, d in zip(joint_values, diff)]
    return False


# ── Warmup / shutdown sequences ──────────────────────────────────────────────

def warmup_sequence() -> None:
    """Bring the arm from [0]*14 to the inference pose via WP1 → WP2 → WP3."""
    global joint_values, gripper_values
    joint_values = [0.0] * 14
    gripper_values = [0.0, 0.0]
    print("[warmup] anchor commanded state at [0]*14")
    send_movej()
    time.sleep(SEND_INTERVAL)

    interp_send([0.0] * 14, WARMUP_WAYPOINT_1, "warmup-A")
    hold(WARMUP_HOLD_SECONDS, "warmup-A")
    interp_send(WARMUP_WAYPOINT_1, WARMUP_WAYPOINT_2, "warmup-B")
    hold(WARMUP_HOLD_SECONDS, "warmup-B")
    interp_send(WARMUP_WAYPOINT_2, WARMUP_WAYPOINT_3, "warmup-C")
    hold(WARMUP_HOLD_SECONDS, "warmup-C")
    gripper_values = [0.97, 0.0]
    print("[warmup] opening left gripper to 0.97")
    send_gripper()
    time.sleep(SEND_INTERVAL)
    print("[warmup] done — arm is at WP3")


def shutdown_sequence() -> None:
    """Reverse of warmup: WP3 → WP2 → WP1 → [0]*14."""
    global joint_values, gripper_values
    joint_values = list(WARMUP_WAYPOINT_3)
    print("[shutdown] anchor commanded state at WP3")
    send_movej()
    time.sleep(SEND_INTERVAL)

    interp_send(WARMUP_WAYPOINT_3, WARMUP_WAYPOINT_2, "shutdown-A")
    hold(WARMUP_HOLD_SECONDS, "shutdown-A")
    interp_send(WARMUP_WAYPOINT_2, WARMUP_WAYPOINT_1, "shutdown-B")
    hold(WARMUP_HOLD_SECONDS, "shutdown-B")
    interp_send(WARMUP_WAYPOINT_1, [0.0] * 14, "shutdown-C")
    hold(WARMUP_HOLD_SECONDS, "shutdown-C")
    print("[shutdown] done — arm parked at [0]*14")


# ── WebSocket lifecycle ──────────────────────────────────────────────────────

def _on_message(_ws, message: str) -> None:
    global ACCID
    try:
        root = json.loads(message)
        title = root.get("title", "")
        recv_accid = root.get("accid", None)
        if recv_accid is not None and ACCID is None:
            ACCID = recv_accid
            _accid_event.set()
        if title != "notify_robot_info":
            print(f"\n[Recv] {message}")
    except Exception as e:
        print(f"[Error] on_message parse failed: {e}")


def _on_error(_ws, error) -> None:
    print(f"[WebSocket Error] {error}")


def _on_close(_ws, _code, _msg) -> None:
    print("Connection closed.")


def wait_for_accid(timeout: float = 15.0) -> bool:
    """Block until the first inbound frame populates ACCID, or timeout."""
    return _accid_event.wait(timeout)


def close() -> None:
    """End the session."""
    global should_exit
    should_exit = True
    if ws_client is not None:
        try:
            ws_client.close()
        except Exception:
            pass


def run(on_ready: Callable[[], None], url: Optional[str] = None) -> None:
    """Connect, spawn ``on_ready`` in a daemon thread, block on run_forever."""
    global ws_client
    actual_url = url or ROBOT_WS_URL

    def _on_open(_ws):
        print("Connected!")
        threading.Thread(target=on_ready, daemon=True).start()

    ws_client = websocket.WebSocketApp(
        actual_url,
        on_open=_on_open,
        on_message=_on_message,
        on_error=_on_error,
        on_close=_on_close,
    )
    print(f"Connecting to {actual_url}  (Ctrl+C to quit)")
    ws_client.run_forever()
