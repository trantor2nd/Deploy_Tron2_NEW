"""Policy backends and a synthetic stub for control-path verification.

A backend is any object with an ``infer(...)`` method matching the ``Policy``
protocol below. ``build_policy(backend_name)`` is the single entry point used
by client.py — to add a new backend, implement a class here and add a dispatch
branch in ``build_policy``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Protocol, Tuple

import cv2
import numpy as np
import torch


class Policy(Protocol):
    """Backend-agnostic interface every policy must satisfy.

    Inputs are robot-frame BGR images and a 16-dim state (14 arm joints in rad
    + 2 grippers in 0–1). Output is a (K, 16) action chunk in dataset units.
    """

    def infer(
        self,
        left_wrist_bgr: np.ndarray,
        cam_high_bgr: np.ndarray,
        right_wrist_bgr: np.ndarray,
        state16: np.ndarray,
        task_text: str,
    ) -> np.ndarray: ...


def _import_groot() -> dict:
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.policies.groot.configuration_groot import GrootConfig
    from lerobot.policies.groot.modeling_groot import GrootPolicy
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition
    return {
        "FeatureType": FeatureType,
        "NormalizationMode": NormalizationMode,
        "PolicyFeature": PolicyFeature,
        "GrootConfig": GrootConfig,
        "GrootPolicy": GrootPolicy,
        "PolicyProcessorPipeline": PolicyProcessorPipeline,
        "policy_action_to_transition": policy_action_to_transition,
    }


class GR00TRunner:
    """Loads the GR00T policy + pre/post pipelines and runs a single-step forward pass."""

    def __init__(
        self,
        checkpoint: str,
        device: str,
        base_model_path: Optional[str],
    ) -> None:
        # Either a local snapshot dir or an HF repo id ("owner/name").
        self.checkpoint = checkpoint
        self.device = torch.device(
            device if device.startswith("cuda") and torch.cuda.is_available() else "cpu"
        )
        self.base_model_path = base_model_path
        self.log = logging.getLogger("groot")

        mods = _import_groot()
        self._FeatureType = mods["FeatureType"]
        self._NormalizationMode = mods["NormalizationMode"]
        self._PolicyFeature = mods["PolicyFeature"]
        self._GrootConfig = mods["GrootConfig"]
        self._GrootPolicy = mods["GrootPolicy"]
        self._PolicyProcessorPipeline = mods["PolicyProcessorPipeline"]
        self._policy_action_to_transition = mods["policy_action_to_transition"]

        torch.set_grad_enabled(False)
        self.cfg, self.pre, self.post, self.model = self._load()
        self.model.eval().to(self.device)
        self.log.info(f"policy ready on {self.device}: {self.checkpoint}")

    @staticmethod
    def _resolve(spec: str) -> Path:
        """Return a local snapshot dir for ``spec``.

        ``spec`` may be a local path or an HF repo id ("owner/name"). Repo ids
        are routed through ``snapshot_download``, which hits the HF cache when
        present and downloads otherwise.
        """
        p = Path(spec)
        if p.exists():
            return p
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=str(spec)))

    def _resolve_base_model(self) -> str:
        if self.base_model_path:
            return str(self._resolve(self.base_model_path))
        # No override: prefer existing cache; fall back to downloading the default.
        cache = Path.home() / ".cache/huggingface/hub/models--nvidia--GR00T-N1.5-3B/snapshots"
        if cache.exists():
            snaps = sorted(cache.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
            if snaps:
                return str(snaps[0])
        return str(self._resolve("nvidia/GR00T-N1.5-3B"))

    def _load(self):
        ckpt = self._resolve(self.checkpoint)
        self.log.info(f"checkpoint resolved: {self.checkpoint!r} -> {ckpt}")
        if (ckpt / "pretrained_model").is_dir():
            ckpt = ckpt / "pretrained_model"

        cfg_path = ckpt / "config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(cfg_path)
        cfg_dict = json.loads(cfg_path.read_text())
        cfg_dict.pop("type", None)

        def _to_feature(d):
            return self._PolicyFeature(
                type=self._FeatureType[d["type"]],
                shape=tuple(d.get("shape") or ()),
            )

        if "input_features" in cfg_dict:
            cfg_dict["input_features"] = {k: _to_feature(v) for k, v in cfg_dict["input_features"].items()}
        if "output_features" in cfg_dict:
            cfg_dict["output_features"] = {k: _to_feature(v) for k, v in cfg_dict["output_features"].items()}
        if "normalization_mapping" in cfg_dict:
            cfg_dict["normalization_mapping"] = {
                k: self._NormalizationMode[v] for k, v in cfg_dict["normalization_mapping"].items()
            }

        cfg_dict["base_model_path"] = self._resolve_base_model()
        cfg = self._GrootConfig(**cfg_dict)
        cfg.device = str(self.device)

        overrides = {"device_processor": {"device": str(self.device)}}
        pre = self._PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt,
            config_filename="policy_preprocessor.json",
            overrides=overrides,
        )
        post = self._PolicyProcessorPipeline.from_pretrained(
            pretrained_model_name_or_path=ckpt,
            config_filename="policy_postprocessor.json",
            overrides=overrides,
            to_transition=self._policy_action_to_transition,
        )
        policy = self._GrootPolicy.from_pretrained(pretrained_name_or_path=ckpt, config=cfg)
        return cfg, pre, post, policy

    def _to_img_tensor(self, bgr: np.ndarray, out_hw: Tuple[int, int] = (480, 640)) -> torch.Tensor:
        """BGR HWC uint8 → (1, 3, H, W) float32 in [0, 1] on device."""
        out_h, out_w = out_hw
        if bgr.shape[0] != out_h or bgr.shape[1] != out_w:
            bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        return t.unsqueeze(0).to(self.device)

    def infer(
        self,
        left_wrist_bgr: np.ndarray,
        cam_high_bgr: np.ndarray,
        right_wrist_bgr: np.ndarray,
        state16: np.ndarray,
        task_text: str,
    ) -> np.ndarray:
        """Run one forward pass and return a (K, 16) action chunk in dataset units."""
        batch = {
            "observation.images.cam_left_wrist":  self._to_img_tensor(left_wrist_bgr),
            "observation.images.cam_high":         self._to_img_tensor(cam_high_bgr),
            "observation.images.cam_right_wrist": self._to_img_tensor(right_wrist_bgr),
            "observation.state": torch.from_numpy(state16.astype(np.float32)).unsqueeze(0).to(self.device),
            "task": task_text,
        }
        model_in = self.pre(batch)
        model_in = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in model_in.items()}
        with torch.no_grad():
            raw = self.model.predict_action_chunk(model_in)   # (1, K, action_dim)
        if raw.dim() != 3:
            raise RuntimeError(f"Unexpected action chunk shape: {tuple(raw.shape)}")

        steps = []
        for k in range(raw.shape[1]):
            out_k = self.post(raw[:, k, :])["action"]
            steps.append(out_k.reshape(-1))
        action = torch.stack(steps, dim=0)                     # (K, 16)
        return action.detach().cpu().numpy().astype(np.float32)


def build_stub_chunk(
    base: np.ndarray,
    chunk_length: int,
    joint_idx: int,
    amplitude: float,
    period: float,
    t_now: float,
    step_dt: float,
) -> np.ndarray:
    """Synthetic (K, 16) chunk for testing the control path without a real model."""
    chunk = np.tile(base, (chunk_length, 1)).astype(np.float32)
    for k in range(chunk_length):
        t = t_now + k * step_dt
        chunk[k, joint_idx] = base[joint_idx] + amplitude * float(np.sin(2 * np.pi * t / period))
    return chunk


def build_policy(backend: str) -> Policy:
    """Construct the configured policy backend.

    Reads backend-specific knobs from ``config`` and returns an object that
    implements the ``Policy`` protocol. ``client.py`` calls this once at startup.

    To add a new backend:
      1. Implement a class with the ``Policy.infer`` signature in this module.
      2. Add an ``elif`` branch below that pulls its knobs from ``config`` and
         constructs it.
      3. Set ``TRON2_MODEL_BACKEND`` (or edit ``config.MODEL_BACKEND``) to its
         registered name.
    """
    if backend == "gr00t":
        import config
        return GR00TRunner(
            checkpoint=config.GR00T_CHECKPOINT,
            device=config.DEVICE,
            base_model_path=config.GR00T_BASE_MODEL_PATH,
        )
    raise ValueError(
        f"unknown policy backend: {backend!r}. "
        f"Add a dispatch branch in inference.build_policy()."
    )
