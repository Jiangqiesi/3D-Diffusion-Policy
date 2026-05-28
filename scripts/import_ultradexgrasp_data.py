#!/usr/bin/env python3
"""Import UltraDexGrasp episode npz files into DP3 zarr datasets."""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np


sys.modules.setdefault("numpy._core", np.core)
if not hasattr(np, "product"):
    np.product = np.prod

import zarr


MODE_TO_GROUP = {
    0: "single_left",
    1: "single_right",
    2: "bimanual",
    3: "single_left",
    4: "single_right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-npz-root",
        default="/home/ym/fzb/UltraDexGrasp/outputs/batch_run",
        help="Root directory scanned recursively for episode_*.npz",
    )
    parser.add_argument(
        "--output-root",
        default="/home/ym/fzb/dp3-workspace/data/ultradexgrasp",
        help="Output directory for converted zarr datasets",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=1024,
        help="Downsample each frame to this many points; <=0 keeps original size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for point subsampling",
    )
    return parser.parse_args()


def load_episode(path: str) -> Tuple[Dict[str, np.ndarray], Dict]:
    npz = np.load(path, allow_pickle=True)
    meta = json.loads(str(npz["meta"].item()))
    arrays = {key: npz[key] for key in npz.files if key != "meta"}
    return arrays, meta


def maybe_downsample_points(
    points: np.ndarray,
    masks: np.ndarray,
    num_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_points <= 0 or points.shape[1] == num_points:
        return points, masks

    n_frames, n_points, n_channels = points.shape
    mask_channels = masks.shape[-1]

    if n_points < num_points:
        padded_points = np.zeros((n_frames, num_points, n_channels), dtype=points.dtype)
        padded_masks = np.zeros((n_frames, num_points, mask_channels), dtype=masks.dtype)
        padded_points[:, :n_points] = points
        padded_masks[:, :n_points] = masks
        return padded_points, padded_masks

    sampled_points = np.empty((n_frames, num_points, n_channels), dtype=points.dtype)
    sampled_masks = np.empty((n_frames, num_points, mask_channels), dtype=masks.dtype)
    for frame_idx in range(n_frames):
        indices = rng.choice(n_points, size=num_points, replace=False)
        sampled_points[frame_idx] = points[frame_idx, indices]
        sampled_masks[frame_idx] = masks[frame_idx, indices]
    return sampled_points, sampled_masks


def write_group(out_path: str, episodes: List[Tuple[Dict[str, np.ndarray], Dict]]) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    root = zarr.open(out_path, mode="w")
    data_group = root.create_group("data")
    meta_group = root.create_group("meta")

    data_keys = ["point_cloud", "point_cloud_mask", "agent_pos", "action", "object_pose"]
    concatenated = {
        key: np.concatenate([episode_arrays[key] for episode_arrays, _ in episodes], axis=0)
        for key in data_keys
    }
    episode_ends = np.cumsum([episode_arrays["action"].shape[0] for episode_arrays, _ in episodes]).astype(
        np.int64
    )

    for key, value in concatenated.items():
        chunks = (min(64, value.shape[0]),) + value.shape[1:]
        data_group.create_dataset(key, data=value, chunks=chunks, dtype=value.dtype, overwrite=True)

    meta_group.create_dataset("episode_ends", data=episode_ends, overwrite=True)
    meta_group.create_dataset(
        "bodex_mode",
        data=np.array([meta["bodex_mode"] for _, meta in episodes], dtype=np.int8),
        overwrite=True,
    )
    meta_group.create_dataset(
        "object_name",
        data=np.array([meta["object_name"] for _, meta in episodes], dtype=object),
        object_codec=zarr.codecs.VLenUTF8(),
        overwrite=True,
    )
    meta_group.create_dataset(
        "object_scale",
        data=np.array([meta["object_scale"] for _, meta in episodes], dtype=np.float32),
        overwrite=True,
    )
    meta_group.create_dataset(
        "num_steps",
        data=np.array([meta["num_steps"] for _, meta in episodes], dtype=np.int32),
        overwrite=True,
    )


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    paths = sorted(glob.glob(os.path.join(args.source_npz_root, "**", "episode_*.npz"), recursive=True))
    if not paths:
        raise SystemExit(f"No episode_*.npz files found under {args.source_npz_root}")

    grouped: Dict[str, List[Tuple[Dict[str, np.ndarray], Dict]]] = {
        "single_left": [],
        "single_right": [],
        "bimanual": [],
    }

    for path in paths:
        arrays, meta = load_episode(path)
        if not meta.get("success", False):
            continue

        if "point_cloud" not in arrays or "agent_pos" not in arrays or "action" not in arrays:
            continue

        point_cloud = arrays["point_cloud"].astype(np.float32)
        point_cloud_mask = arrays["point_cloud_mask"].astype(np.uint8)
        point_cloud, point_cloud_mask = maybe_downsample_points(
            point_cloud, point_cloud_mask, args.num_points, rng
        )

        converted = {
            "point_cloud": point_cloud,
            "point_cloud_mask": point_cloud_mask,
            "agent_pos": arrays["agent_pos"].astype(np.float32),
            "action": arrays["action"].astype(np.float32),
            "object_pose": arrays["object_pose"].astype(np.float32),
        }
        grouped[MODE_TO_GROUP[meta["bodex_mode"]]].append((converted, meta))

    for group_name, episodes in grouped.items():
        if not episodes:
            print(f"[skip] {group_name}: no successful episodes")
            continue
        out_path = os.path.join(args.output_root, f"{group_name}.zarr")
        print(f"[write] {group_name}: {len(episodes)} episodes -> {out_path}")
        write_group(out_path, episodes)

    print("done")


if __name__ == "__main__":
    main()
