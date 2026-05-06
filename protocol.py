"""Server-client TCP protocol for Tron2 deployment.

Uses size-prefixed pickle messages over TCP.

Protocol flow:
    1. Client sends {"cmd": "get_action"}
    2. Server replies with Observation(...) message
    3. Client sends ActionChunk(...) or {"cmd": "skip"}
    4. Server executes the chunk, then loops back to step 1
"""

from __future__ import annotations

import pickle
import socket
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

_HEADER = struct.Struct("!I")  # 4-byte big-endian uint32 length prefix


def send_msg(sock: socket.socket, msg: Any) -> None:
    """Send a pickle-serialized message with a 4-byte length prefix."""
    data = pickle.dumps(msg, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(_HEADER.pack(len(data)) + data)


def recv_msg(sock: socket.socket) -> Any:
    """Receive one length-prefixed pickle message. Raises ConnectionError on EOF."""
    hdr = _recvall(sock, _HEADER.size)
    if not hdr:
        raise ConnectionError("connection closed")
    (length,) = _HEADER.unpack(hdr)
    data = _recvall(sock, length)
    if not data:
        raise ConnectionError("connection closed mid-message")
    return pickle.loads(data)


def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    """Read exactly n bytes, or return None on EOF."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


@dataclass
class Observation:
    """Observation data sent from server to client."""
    task_text: str
    state16: np.ndarray          # (16,) float32, arm in rad, gripper in 0-1
    images: Dict[str, np.ndarray]  # slot -> BGR HWC uint8 (left_wrist, cam_high, right_wrist)


@dataclass
class ActionChunk:
    """Action chunk sent from client to server."""
    chunk: np.ndarray  # (K, 16) float32, arm in rad, gripper in 0-1
