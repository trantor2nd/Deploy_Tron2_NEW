#!/usr/bin/env python3
"""Tron2 robot control layer (WebSocket protocol).

OOP alternative to tron2_ws module-level globals. Not used by server.py
(which uses tron2_ws directly); provided for standalone scripting.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from typing import Optional, Sequence
from urllib.parse import urlparse

import numpy as np
import websocket


class WSController:
    """Tron2 websocket controller: accid handshake + synchronous send helpers."""

    def __init__(
        self,
        url: str,
        move_time: float = 0.5,
        log_notify_seconds: float = 5.0,
    ) -> None:
        self.url = url
        self.move_time = move_time
        self.log = logging.getLogger("ws")

        self._accid: Optional[str] = None
        self._accid_event = threading.Event()
        self._ws: Optional[websocket.WebSocketApp] = None
        self._send_lock = threading.Lock()
        self._running = False
        self._ws_thread: Optional[threading.Thread] = None

        self._notify_log_until: float = 0.0
        self._log_notify_seconds = log_notify_seconds

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self._running = True
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
                  "all_proxy", "ALL_PROXY"):
            stripped = os.environ.pop(k, None)
            if stripped:
                self.log.info(f"unset proxy env {k}={stripped}")
        host = urlparse(self.url).hostname or ""
        if host:
            os.environ["NO_PROXY"] = (os.environ.get("NO_PROXY", "") + "," + host).lstrip(",")

        self._ws = websocket.WebSocketApp(
            self.url,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        self._ws_thread = threading.Thread(target=self._ws.run_forever, daemon=True)
        self._ws_thread.start()

    def stop(self) -> None:
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return self._accid_event.wait(timeout)

    # ── WS callbacks ─────────────────────────────────────────────────────────

    def _on_open(self, _ws):
        self.log.info(f"connected to {self.url}")

    def _on_message(self, _ws, message: str):
        try:
            payload = json.loads(message)
        except Exception:
            return
        accid = payload.get("accid")
        if accid and self._accid is None:
            self._accid = accid
            self._accid_event.set()
            self.log.info(f"acquired accid={accid}")
            self._notify_log_until = time.monotonic() + self._log_notify_seconds

        title = payload.get("title", "")
        if title == "notify_robot_info":
            if time.monotonic() < self._notify_log_until:
                self.log.info(
                    f"[notify] {json.dumps(payload.get('data'), ensure_ascii=False)[:400]}"
                )
        else:
            self.log.info(
                f"[recv] {title}: {json.dumps(payload.get('data'), ensure_ascii=False)[:300]}"
            )

    def _on_error(self, _ws, error):
        self.log.warning(f"ws error: {error}")

    def _on_close(self, _ws, code, msg):
        self.log.info(f"ws closed code={code} msg={msg}")

    # ── Send path ────────────────────────────────────────────────────────────

    def _send(self, title: str, data: dict) -> None:
        if self._accid is None or self._ws is None:
            return
        frame = {
            "accid": self._accid,
            "title": title,
            "timestamp": int(time.time() * 1000),
            "guid": str(uuid.uuid4()),
            "data": data,
        }
        with self._send_lock:
            try:
                self._ws.send(json.dumps(frame))
            except Exception as exc:
                self.log.warning(f"send failed: {exc}")

    def send_movej(self, joint14: Sequence[float]) -> None:
        self._send("request_movej", {"joint": list(map(float, joint14)), "time": self.move_time})

    def warmup_hold(
        self,
        joint14: Sequence[float],
        repeats: int = 3,
        interval: float = 0.3,
    ) -> None:
        if self._accid is None:
            self.log.warning("warmup_hold: no accid yet, skipping")
            return
        js = "[" + ",".join(f"{float(x):+.4f}" for x in joint14) + "]"
        self.log.info(
            f"warmup: sending {repeats}x movej(hold) joint={js} time={self.move_time}s"
        )
        for i in range(repeats):
            self._send("request_movej", {"joint": list(map(float, joint14)),
                                         "time": self.move_time})
            time.sleep(interval)

    def send_gripper(
        self,
        left_opening: float,
        right_opening: float,
        speed: float = 50.0,
        force: float = 50.0,
    ) -> None:
        clamp = lambda v: max(0.0, min(100.0, float(v)))
        self._send("request_set_limx_2fclaw_cmd", {
            "left_opening":  clamp(left_opening),
            "left_speed":    clamp(speed),
            "left_force":    clamp(force),
            "right_opening": clamp(right_opening),
            "right_speed":   clamp(speed),
            "right_force":   clamp(force),
        })

    # ── Chunk playback ───────────────────────────────────────────────────────

    def play_chunk(
        self,
        chunk: np.ndarray,
        rate_hz: float,
        stop_event: Optional[threading.Event] = None,
    ) -> int:
        if chunk is None or len(chunk) == 0:
            return 0
        dt = 1.0 / max(rate_hz, 1e-3)
        sent = 0
        next_tick = time.monotonic()
        for k, cmd in enumerate(chunk):
            if stop_event is not None and stop_event.is_set():
                break

            joint14 = [float(x) for x in cmd[:14]]
            self.send_movej(joint14)
            joint_str = "[" + ",".join(f"{x:+.4f}" for x in joint14) + "]"

            grip_str = ""
            if cmd.shape[0] >= 16:
                left_grip = float(cmd[14]) * 100.0
                right_grip = float(cmd[15]) * 100.0
                self.send_gripper(left_grip, right_grip)
                grip_str = f" | grip L={left_grip:5.1f} R={right_grip:5.1f}"

            self.log.info(
                f"step {k+1:2d}/{len(chunk)} movej t={self.move_time:.2f}s "
                f"joint={joint_str}{grip_str}"
            )

            sent = k + 1
            next_tick += dt
            sleep = next_tick - time.monotonic()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.monotonic()
        return sent
