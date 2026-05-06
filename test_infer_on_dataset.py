#!/usr/bin/env python3
"""Smoke-test GR00T inference against the LeRobot training dataset.

Pulls N samples directly from the training dataset, runs them through the same
load / preprocess / model / postprocess path that client.py uses, and compares
the predicted K-step action chunk against the ground-truth action[t : t+K].

Run inside the lerobot_py310 conda env:
    python test_infer_on_dataset.py --indices 100,500,2000,10000 --episode 0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def run_one(pre, post, model, sample, task_text: str, device: torch.device, K: int, verbose: bool = False):
    batch = {
        "observation.images.cam_left_wrist": sample["observation.images.cam_left_wrist"].unsqueeze(0).to(device),
        "observation.images.cam_high":       sample["observation.images.cam_high"].unsqueeze(0).to(device),
        "observation.images.cam_right_wrist":sample["observation.images.cam_right_wrist"].unsqueeze(0).to(device),
        "observation.state":                 sample["observation.state"].unsqueeze(0).to(device),
        "task": task_text,
    }
    model_in = pre(batch)
    model_in = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in model_in.items()}
    with torch.no_grad():
        raw = model.predict_action_chunk(model_in)

    bulk_out = post(raw)["action"]

    perstep = None
    if raw.dim() == 3 and raw.shape[1] >= 1:
        steps = []
        Ksteps = raw.shape[1]
        for k in range(Ksteps):
            single = raw[:, k, :]
            out = post(single)["action"]
            steps.append(out.reshape(-1))
        perstep = torch.stack(steps, dim=0)

    if bulk_out.dim() == 3:
        bulk_pred = bulk_out[0]
    elif bulk_out.dim() == 2:
        bulk_pred = bulk_out
    elif bulk_out.dim() == 1:
        bulk_pred = bulk_out.unsqueeze(0)
    else:
        raise RuntimeError(f"unexpected post shape: {tuple(bulk_out.shape)}")

    if verbose:
        print(f"    raw shape      : {tuple(raw.shape)}")
        print(f"    post(raw) shape: {tuple(bulk_out.shape)}   -> bulk_pred {tuple(bulk_pred.shape)}")
        if perstep is not None:
            print(f"    per-step shape : {tuple(perstep.shape)}")

    return {
        "raw_shape": tuple(raw.shape),
        "bulk_shape": tuple(bulk_out.shape),
        "bulk_pred": bulk_pred.detach().cpu().numpy().astype(np.float32),
        "perstep_pred": None if perstep is None else perstep.detach().cpu().numpy().astype(np.float32),
    }


def _mae_report(label: str, pred: np.ndarray, gt: np.ndarray, valid: np.ndarray, act_range: np.ndarray):
    k_pred = pred.shape[0]
    k_gt = gt.shape[0]
    k = min(k_pred, k_gt, int(valid.sum()))
    if k == 0:
        print(f"  [{label}] no comparable steps")
        return
    v = valid[:min(k_pred, k_gt)]
    p = pred[:min(k_pred, k_gt)][v]
    g = gt[:min(k_pred, k_gt)][v]
    err = p - g
    mae_per_joint = np.abs(err).mean(axis=0)
    nmae = mae_per_joint / np.maximum(act_range, 1e-6)
    print(f"  [{label}] compared {p.shape[0]} steps | overall MAE = {float(np.abs(err).mean()):.5f}")
    print(f"  [{label}] MAE/joint: {np.array2string(mae_per_joint, precision=4, separator=',')}")
    print(f"  [{label}] MAE/range: {np.array2string(nmae, precision=3, separator=',')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path,
                   default=Path(os.environ.get(
                       "TRON2_CKPT",
                       "/home/data/hf/hub/models--trantor2nd--tron2_gr00t_pick_step6k/"
                       "snapshots/64108047dc22892c31f84856b507009578be5e79",
                   )))
    p.add_argument("--dataset-root", type=Path, default=Path("/home/hsb/TRON2_data/pick_stones"))
    p.add_argument("--dataset-repo-id", type=str, default="/home/hsb/TRON2_data/pick_stones")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--indices", type=str, default="30,200,600,1200")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--task-text", type=str, default="pick_up_stones_and_place_them_into_the_container")
    p.add_argument("--base-model-path", type=str, default=os.environ.get("BASE_MODEL_PATH"))
    p.add_argument("--chunk-size", type=int, default=50)
    args = p.parse_args()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from inference import GR00TRunner

    device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    torch.set_grad_enabled(False)

    fps = 30.0
    K = args.chunk_size
    delta_ts = {"action": [i / fps for i in range(K)]}

    print(f"[dataset] loading episode {args.episode} from {args.dataset_root} ...")
    ds = LeRobotDataset(
        repo_id=args.dataset_repo_id,
        root=args.dataset_root,
        episodes=[args.episode],
        delta_timestamps=delta_ts,
        download_videos=False,
    )
    print(f"[dataset] len={len(ds)} frames, fps={ds.fps}")

    print(f"[policy] loading checkpoint {args.checkpoint} ...")
    runner = GR00TRunner(
        checkpoint=args.checkpoint,
        device=str(device),
        base_model_path=args.base_model_path,
    )
    cfg, pre, post, model = runner.cfg, runner.pre, runner.post, runner.model
    print(f"[policy] chunk_size={cfg.chunk_size}, device={device}")

    if cfg.chunk_size != K:
        print(f"[warn] cfg.chunk_size={cfg.chunk_size} but --chunk-size={K}; continuing with {K}")

    stats_path = args.dataset_root / "meta" / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    act_range = np.asarray(stats.get("action", {}).get("max", [0] * 16), dtype=np.float32) - \
                np.asarray(stats.get("action", {}).get("min", [0] * 16), dtype=np.float32)

    indices = [int(s) for s in args.indices.split(",") if s.strip()]

    for i, t in enumerate(indices):
        if t >= len(ds):
            print(f"[skip] frame {t} >= episode length {len(ds)}")
            continue

        sample = ds[t]
        gt = sample["action"].cpu().numpy().astype(np.float32)
        pad_flag = sample.get("action_is_pad")
        if pad_flag is not None and bool(pad_flag.any()):
            n_pad = int(pad_flag.sum())
            print(f"[note] t={t}: {n_pad}/{K} GT steps are padded; metrics on valid only")
            valid = (~pad_flag.bool().cpu().numpy())
        else:
            valid = np.ones(K, dtype=bool)

        result = run_one(pre, post, model, sample, args.task_text, device, K, verbose=(i == 0))

        print(f"\n=== frame t={t} (episode {args.episode}, {int(valid.sum())}/{K} valid steps in GT) ===")
        _mae_report("bulk   ", result["bulk_pred"], gt, valid, act_range)
        if result["perstep_pred"] is not None:
            _mae_report("perstep", result["perstep_pred"], gt, valid, act_range)


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_USE_FLASH_ATTENTION_2", "0")
    os.environ.setdefault("USE_FLASH_ATTENTION", "0")
    os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "1")
    os.environ.setdefault("ATTENTION_IMPLEMENTATION", "eager")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
