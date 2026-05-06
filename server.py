#!/usr/bin/env python3
"""Tron2 deployment server.

Handles ROS 2 observation and robot WebSocket control. Communicates with a
separate client process (running GR00T inference) over TCP.

Protocol flow:
    1. Wait for client to send {"cmd": "get_action"}
    2. Capture fresh observation (joint state + 3 cameras)
    3. Send Observation to client
    4. Wait for client to send back ActionChunk
    5. Execute the action chunk on the robot
    6. Go to 1

Run order:
    python start.py     # once per power-on
    make server         # this file
    make client         # on same or different machine
    python shutdown.py  # park the arm
"""

from __future__ import annotations

import logging
import signal
import socket
import threading
import time

import numpy as np

import tron2_ws
from config import (
    SERVER_HOST, SERVER_PORT, SEND_INTERVAL, LEFT_FLIP_IDX,
    WARMUP_WAYPOINT_3, JOINT_TOPIC, GRIPPER_TOPIC,
    CAM_LEFT, CAM_HIGH, CAM_RIGHT, TASK_TEXT,
    SETTLE_SECONDS, MAX_OBS_AGE, MAX_IMG_AGE, MAX_STAMP_SPREAD,
)
from observer import (
    Tron2Observer, build_state_reorder, log_reorder_once,
    wait_for_fresh_observation,
)
from protocol import Observation, ActionChunk, send_msg, recv_msg

log = logging.getLogger("server")


def _send_step(cmd: np.ndarray) -> None:
    """Apply one (16,) action row: sign-flip left arm, then send movej + gripper."""
    joints = [float(x) for x in cmd[:14]]
    for i in LEFT_FLIP_IDX:
        joints[i] = -joints[i]
    tron2_ws.joint_values = joints
    tron2_ws.send_movej()
    if cmd.shape[0] >= 16:
        tron2_ws.gripper_values = [float(cmd[14]) * 100.0, float(cmd[15]) * 100.0]
        tron2_ws.send_gripper()


def _execute_chunk(chunk: np.ndarray, cycle: int) -> None:
    """Send each row of chunk at SEND_INTERVAL, then settle so the next observation
    reflects the chunk-end pose (not a mid-motion snapshot)."""
    next_tick = time.monotonic()
    for k, cmd in enumerate(chunk):
        if tron2_ws.should_exit:
            break
        sleep = next_tick - time.monotonic()
        if sleep > 0:
            time.sleep(sleep)
        _send_step(cmd)
        joints_str = "[" + ",".join(f"{x:+.4f}" for x in tron2_ws.joint_values) + "]"
        log.info(
            f"[cycle {cycle}][{k+1:2d}/{len(chunk)}] "
            f"joint={joints_str} "
            f"grip=L{tron2_ws.gripper_values[0]:.1f},R{tron2_ws.gripper_values[1]:.1f}"
        )
        next_tick += SEND_INTERVAL
    time.sleep(SETTLE_SECONDS)


def _handle_client(
    conn: socket.socket,
    addr,
    observer: Tron2Observer,
    reorder_idx: np.ndarray,
    stop_event: threading.Event,
) -> None:
    """Handle one client connection in a loop until disconnect or shutdown."""
    log.info(f"client connected: {addr}")
    cycle = 0
    try:
        while not stop_event.is_set() and not tron2_ws.should_exit:
            # Wait for client to request an action.
            try:
                msg = recv_msg(conn)
            except ConnectionError:
                log.info(f"client disconnected: {addr}")
                break
            if not isinstance(msg, dict) or msg.get("cmd") != "get_action":
                log.warning(f"unexpected message from client: {msg}")
                continue

            # Capture fresh observation. State + every image must be young AND
            # within MAX_STAMP_SPREAD of each other so the policy sees a
            # temporally coherent snapshot.
            obs = wait_for_fresh_observation(
                observer, log, stop_event,
                max_obs_age=MAX_OBS_AGE,
                max_img_age=MAX_IMG_AGE,
                max_stamp_spread=MAX_STAMP_SPREAD,
            )
            if obs is None:
                log.warning("observation unavailable, sending skip")
                send_msg(conn, {"cmd": "skip"})
                continue

            names, state, frames, _ = obs
            state16_raw = state.astype(np.float32)[reorder_idx][:16]
            state_for_model = state16_raw.copy()
            state_for_model[14:16] /= 100.0  # gripper: robot 0-100 → model 0-1

            arm_str = "[" + ",".join(f"{x:+.4f}" for x in state16_raw[:14]) + "]"
            log.info(f"[cycle {cycle}] STATE arm={arm_str} "
                     f"grip=L{state16_raw[14]:.1f},R{state16_raw[15]:.1f}")

            # Send observation to client.
            obs_msg = Observation(
                task_text=TASK_TEXT,
                state16=state_for_model,
                images={
                    "left_wrist": frames["left_wrist"][0],
                    "cam_high": frames["cam_high"][0],
                    "right_wrist": frames["right_wrist"][0],
                },
            )
            send_msg(conn, obs_msg)

            # Wait for client to send back action chunk.
            try:
                result = recv_msg(conn)
            except ConnectionError:
                log.info(f"client disconnected during inference: {addr}")
                break

            if isinstance(result, ActionChunk):
                cycle += 1
                chunk = result.chunk
                log.info(f"[cycle {cycle}] received chunk K={len(chunk)}, executing")
                _execute_chunk(chunk, cycle)
            elif isinstance(result, dict) and result.get("cmd") == "skip":
                log.info("[cycle] client skipped, no action executed")
            else:
                log.warning(f"unexpected result from client: {type(result)}")
    except Exception as e:
        log.error(f"client handler error: {e}", exc_info=True)
    finally:
        conn.close()
        log.info(f"client handler exited: {addr}")


def server_task(stop_event: threading.Event) -> None:
    """Main server logic: init ROS, wait for obs, accept clients."""
    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    # 1. Wait for WebSocket handshake.
    if not tron2_ws.wait_for_accid(timeout=15.0):
        log.error("timeout waiting for accid — aborting")
        tron2_ws.close()
        return
    log.info(f"ACCID = {tron2_ws.ACCID}")

    # 2. Anchor the rate-limiter at WP3.
    tron2_ws.joint_values = list(WARMUP_WAYPOINT_3)
    tron2_ws.gripper_values = [0.97 * 100.0, 0.0]
    tron2_ws.send_movej()
    time.sleep(SEND_INTERVAL)
    tron2_ws.send_gripper()
    time.sleep(SEND_INTERVAL)

    # 3. Start ROS observer.
    rclpy.init()
    observer = Tron2Observer(
        joint_topic=JOINT_TOPIC,
        gripper_topic=GRIPPER_TOPIC,
        cam_topics={
            "left_wrist": CAM_LEFT,
            "cam_high": CAM_HIGH,
            "right_wrist": CAM_RIGHT,
        },
    )
    executor = MultiThreadedExecutor()
    executor.add_node(observer)
    threading.Thread(target=executor.spin, daemon=True).start()
    log.info("ROS observer spinning")

    # 4. Wait for first observation and resolve joint-name ordering.
    log.info("waiting for first observation…")
    obs = wait_for_fresh_observation(observer, log, stop_event)
    if obs is None:
        return
    names, _, _, _ = obs
    reorder_idx = build_state_reorder(names)
    log_reorder_once(log, names, reorder_idx)
    if reorder_idx is None:
        log.error("joint name schema mismatch — aborting")
        return

    # 5. Start TCP server.
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.settimeout(1.0)  # so we can check stop_event periodically
    srv.bind((SERVER_HOST, SERVER_PORT))
    srv.listen(1)
    log.info(f"TCP server listening on {SERVER_HOST}:{SERVER_PORT}")

    try:
        while not stop_event.is_set() and not tron2_ws.should_exit:
            try:
                conn, addr = srv.accept()
            except socket.timeout:
                continue
            _handle_client(conn, addr, observer, reorder_idx, stop_event)
    finally:
        srv.close()
        executor.shutdown()
        observer.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        log.info("server shutdown complete")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    stop_event = threading.Event()

    def _on_sigint(signum, frame):
        log.info("SIGINT received, shutting down…")
        stop_event.set()
        tron2_ws.close()

    signal.signal(signal.SIGINT, _on_sigint)
    tron2_ws.run(on_ready=lambda: server_task(stop_event))


if __name__ == "__main__":
    main()
