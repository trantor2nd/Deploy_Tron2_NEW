#!/usr/bin/env python3
"""Bring Tron2 from the fresh [0]*14 boot state to the inference pose (WP3).

Run once per power-on, before server/client:
    python start.py    # arm walks from [0]*14 to WP3, ~26 s
"""

import tron2_ws


def task():
    if not tron2_ws.wait_for_accid(timeout=15.0):
        print("[start] timeout waiting for accid — aborting")
        tron2_ws.close()
        return
    print(f"[start] ACCID acquired = {tron2_ws.ACCID}")
    tron2_ws.warmup_sequence()
    print("[start] arm at WP3 — closing connection")
    tron2_ws.close()


if __name__ == "__main__":
    tron2_ws.run(on_ready=task)
