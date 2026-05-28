#!/usr/bin/env python3
import argparse
import csv
import html
import json
import os
import pathlib
import sys
from typing import Iterable

import dill
import hydra
import numpy as np
import torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DP3_ROOT = REPO_ROOT / "3D-Diffusion-Policy"
sys.path.insert(0, str(DP3_ROOT))

from train import TrainDP3Workspace  # noqa: E402


def parse_indices(value: str | None, length: int, num_samples: int) -> list[int]:
    if value:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        if num_samples <= 1:
            indices = [0]
        else:
            stop = max(length - 1, 0)
            indices = np.linspace(0, stop, num=min(num_samples, length), dtype=int).tolist()
    return [idx for idx in indices if 0 <= idx < length]


def to_device_obs(obs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.unsqueeze(0).to(device=device, dtype=torch.float32)
        for key, value in obs.items()
    }


def scalar_range(values: Iterable[float]) -> tuple[float, float]:
    values = list(values)
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    if np.isclose(low, high):
        pad = 1.0 if np.isclose(low, 0.0) else abs(low) * 0.1
        return low - pad, high + pad
    pad = (high - low) * 0.08
    return low - pad, high + pad


def polyline(points: np.ndarray, x_min: float, x_max: float, y_min: float, y_max: float,
             width: int, height: int, pad: int) -> str:
    coords = []
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    for x, y in points:
        sx = pad + (x - x_min) / x_span * (width - 2 * pad)
        sy = height - pad - (y - y_min) / y_span * (height - 2 * pad)
        coords.append(f"{sx:.2f},{sy:.2f}")
    return " ".join(coords)


def save_action_html(path: pathlib.Path, pred: np.ndarray, gt: np.ndarray, mse: float) -> None:
    steps = np.arange(pred.shape[0], dtype=np.float32)
    dim_count = pred.shape[1]
    plots = []
    width = 360
    height = 180
    pad = 28
    x_min, x_max = scalar_range(steps)
    for dim in range(dim_count):
        y_min, y_max = scalar_range([*pred[:, dim].tolist(), *gt[:, dim].tolist()])
        pred_points = np.stack([steps, pred[:, dim]], axis=1)
        gt_points = np.stack([steps, gt[:, dim]], axis=1)
        pred_line = polyline(pred_points, x_min, x_max, y_min, y_max, width, height, pad)
        gt_line = polyline(gt_points, x_min, x_max, y_min, y_max, width, height, pad)
        plots.append(
            f"""
            <section class="plot">
              <h2>action dim {dim}</h2>
              <svg viewBox="0 0 {width} {height}" role="img">
                <rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>
                <line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#999"/>
                <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#999"/>
                <polyline points="{gt_line}" fill="none" stroke="#2f6fdd" stroke-width="2"/>
                <polyline points="{pred_line}" fill="none" stroke="#d94b3d" stroke-width="2"/>
              </svg>
            </section>
            """
        )

    body = "\n".join(plots)
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>UltraDexGrasp Action Prediction</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    .meta {{ margin-bottom: 16px; }}
    .legend span {{ display: inline-block; margin-right: 16px; }}
    .gt {{ color: #2f6fdd; }}
    .pred {{ color: #d94b3d; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 14px; }}
    .plot {{ border: 1px solid #ddd; border-radius: 6px; padding: 10px; }}
    h1 {{ font-size: 20px; margin: 0 0 8px; }}
    h2 {{ font-size: 13px; margin: 0 0 8px; }}
    svg {{ width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Action prediction vs dataset action</h1>
  <div class="meta">MSE: {mse:.8f}</div>
  <div class="legend"><span class="gt">blue: dataset action</span><span class="pred">red: predicted action</span></div>
  <div class="grid">{body}</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def save_action_csv(path: pathlib.Path, pred: np.ndarray, gt: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["step"]
        for dim in range(pred.shape[1]):
            header += [f"gt_{dim}", f"pred_{dim}", f"err_{dim}"]
        writer.writerow(header)
        for step in range(pred.shape[0]):
            row = [step]
            for dim in range(pred.shape[1]):
                row += [gt[step, dim], pred[step, dim], pred[step, dim] - gt[step, dim]]
            writer.writerow(row)


def project_points(points: np.ndarray, axes: tuple[int, int], width: int, height: int, pad: int) -> list[tuple[float, float]]:
    xy = points[:, axes]
    mins = xy.min(axis=0)
    maxs = xy.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-9)
    scaled = (xy - mins) / spans
    sx = pad + scaled[:, 0] * (width - 2 * pad)
    sy = height - pad - scaled[:, 1] * (height - 2 * pad)
    return list(zip(sx.tolist(), sy.tolist()))


def save_pointcloud_html(path: pathlib.Path, point_cloud: np.ndarray) -> None:
    point_cloud = point_cloud[:, :3]
    # Keep the file responsive even for dense clouds.
    max_points = 2048
    if point_cloud.shape[0] > max_points:
        choice = np.linspace(0, point_cloud.shape[0] - 1, max_points, dtype=int)
        point_cloud = point_cloud[choice]

    views = [("XY", (0, 1)), ("XZ", (0, 2)), ("YZ", (1, 2))]
    width = 420
    height = 420
    pad = 24
    panels = []
    for title, axes in views:
        projected = project_points(point_cloud, axes, width, height, pad)
        circles = "\n".join(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.5" fill="#1f6f78" opacity="0.75"/>'
            for x, y in projected
        )
        panels.append(
            f"""
            <section class="panel">
              <h2>{html.escape(title)}</h2>
              <svg viewBox="0 0 {width} {height}">
                <rect x="0" y="0" width="{width}" height="{height}" fill="#fff"/>
                {circles}
              </svg>
            </section>
            """
        )

    mins = point_cloud.min(axis=0).tolist()
    maxs = point_cloud.max(axis=0).tolist()
    path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>UltraDexGrasp Point Cloud</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #202124; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 14px; }}
    .panel {{ border: 1px solid #ddd; border-radius: 6px; padding: 10px; }}
    h1 {{ font-size: 20px; margin: 0 0 8px; }}
    h2 {{ font-size: 14px; margin: 0 0 8px; }}
    svg {{ width: 100%; height: auto; }}
  </style>
</head>
<body>
  <h1>Point cloud projections</h1>
  <p>points: {point_cloud.shape[0]}, min xyz: {mins}, max xyz: {maxs}</p>
  <div class="grid">{''.join(panels)}</div>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize an UltraDexGrasp DP3 checkpoint on offline zarr samples."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "data/outputs/ultradexgrasp_single_left-dp3-ultradex_seed0/checkpoints/latest.ckpt"),
        help="Path to latest.ckpt or an epoch checkpoint.",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for HTML/CSV/JSON outputs.")
    parser.add_argument("--device", default="cuda:0", help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--indices", default=None, help="Comma-separated dataset indices, e.g. 0,10,20.")
    args = parser.parse_args()

    checkpoint = pathlib.Path(args.checkpoint).expanduser().resolve()
    if args.output_dir is None:
        output_dir = checkpoint.parent.parent / "visualizations" / checkpoint.stem
    else:
        output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    os.chdir(REPO_ROOT)
    payload = torch.load(checkpoint.open("rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]
    workspace = TrainDP3Workspace(cfg)
    workspace.load_payload(payload)

    dataset = hydra.utils.instantiate(cfg.task.dataset)
    if args.split == "val":
        val_dataset = dataset.get_validation_dataset()
        if len(val_dataset) > 0:
            dataset = val_dataset
        else:
            print("Validation split is empty; falling back to train split.")

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()

    indices = parse_indices(args.indices, len(dataset), args.num_samples)
    if not indices:
        raise ValueError(f"No valid indices selected for dataset length {len(dataset)}")

    summary = {
        "checkpoint": str(checkpoint),
        "dataset_length": len(dataset),
        "split": args.split,
        "device": str(device),
        "samples": [],
    }

    for idx in indices:
        sample = dataset[idx]
        obs = to_device_obs(sample["obs"], device)
        with torch.no_grad():
            result = policy.predict_action(obs)

        pred = result["action"].squeeze(0).detach().cpu().numpy()
        start = int(cfg.n_obs_steps) - 1
        gt = sample["action"][start:start + pred.shape[0]].detach().cpu().numpy()
        horizon = min(pred.shape[0], gt.shape[0])
        pred = pred[:horizon]
        gt = gt[:horizon]
        mse = float(np.mean((pred - gt) ** 2))

        prefix = f"sample_{idx:06d}"
        action_html = output_dir / f"{prefix}_action.html"
        action_csv = output_dir / f"{prefix}_action.csv"
        pointcloud_html = output_dir / f"{prefix}_pointcloud.html"
        save_action_html(action_html, pred, gt, mse)
        save_action_csv(action_csv, pred, gt)

        point_step = min(start, sample["obs"]["point_cloud"].shape[0] - 1)
        point_cloud = sample["obs"]["point_cloud"][point_step].detach().cpu().numpy()
        save_pointcloud_html(pointcloud_html, point_cloud)

        summary["samples"].append({
            "index": idx,
            "mse": mse,
            "action_html": str(action_html),
            "action_csv": str(action_csv),
            "pointcloud_html": str(pointcloud_html),
        })
        print(f"[{idx}] mse={mse:.8f}")
        print(f"  action: {action_html}")
        print(f"  point cloud: {pointcloud_html}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
