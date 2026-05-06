#!/usr/bin/env python3
"""Tron2 deployment client.

Connects to the server, receives observations, runs GR00T inference, and sends
back action chunks. Runs in the lerobot_py310 conda environment (separate from
the server which runs in py310).

Protocol flow:
    1. Send {"cmd": "get_action"} to server
    2. Receive Observation (task text, images, state)
    3. Run GR00T inference → action chunk (K, 16)
    4. Send ActionChunk back to server
    5. Go to 1
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import time

from config import SERVER_HOST, SERVER_PORT, MODEL_BACKEND, CONSUME_STEPS
from protocol import Observation, ActionChunk, send_msg, recv_msg

log = logging.getLogger("client")


def main() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_USE_FLASH_ATTENTION_2", "0")
    os.environ.setdefault("USE_FLASH_ATTENTION", "0")
    os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "1")
    os.environ.setdefault("ATTENTION_IMPLEMENTATION", "eager")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    from inference import build_policy

    runner = build_policy(MODEL_BACKEND)
    log.info(f"policy loaded: backend={MODEL_BACKEND}")

    # Connect to server.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    log.info(f"connecting to server at {SERVER_HOST}:{SERVER_PORT}…")
    sock.connect((SERVER_HOST, SERVER_PORT))
    log.info("connected to server")

    should_exit = False

    def _on_sigint(signum, frame):
        nonlocal should_exit
        log.info("SIGINT received, shutting down…")
        should_exit = True

    signal.signal(signal.SIGINT, _on_sigint)

    cycle = 0
    try:
        while not should_exit:
            # Request an action from the server.
            send_msg(sock, {"cmd": "get_action"})

            # Receive observation.
            obs = recv_msg(sock)
            if isinstance(obs, dict) and obs.get("cmd") == "skip":
                log.info("server sent skip, waiting…")
                time.sleep(0.1)
                continue
            if not isinstance(obs, Observation):
                log.warning(f"unexpected message from server: {type(obs)}")
                continue

            cycle += 1
            log.info(f"[cycle {cycle}] received observation, running inference…")

            # Run inference.
            t0 = time.monotonic()
            try:
                chunk = runner.infer(
                    left_wrist_bgr=obs.images["left_wrist"],
                    cam_high_bgr=obs.images["cam_high"],
                    right_wrist_bgr=obs.images["right_wrist"],
                    state16=obs.state16,
                    task_text=obs.task_text,
                )
            except Exception as exc:
                log.error(f"inference failed: {exc}")
                send_msg(sock, {"cmd": "skip"})
                continue

            infer_ms = 1000 * (time.monotonic() - t0)
            log.info(f"[cycle {cycle}] inference done: K={len(chunk)}, {infer_ms:.0f}ms")

            # Send action chunk to server.
            send_msg(sock, ActionChunk(chunk=chunk[:CONSUME_STEPS]))

    except ConnectionError:
        log.info("server disconnected")
    except KeyboardInterrupt:
        pass
    finally:
        sock.close()
        log.info("client shutdown")


if __name__ == "__main__":
    main()
