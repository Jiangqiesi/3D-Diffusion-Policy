#!/usr/bin/env python3
import argparse
import json
import os
import pathlib
import sys
from collections import deque

import dill
import numpy as np
import torch
import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
DP3_ROOT = REPO_ROOT / "3D-Diffusion-Policy"
DEFAULT_ULTRADEX_ROOT = pathlib.Path("/home/ym/fzb/UltraDexGrasp")

sys.path.insert(0, str(DP3_ROOT))

from train import TrainDP3Workspace  # noqa: E402


def add_ultradex_to_path(root: pathlib.Path) -> None:
    root = root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    # The DP3 venv has model dependencies, while the UltraDexGrasp venv has
    # Sapien/cuRobo simulation dependencies. Append UltraDexGrasp's packages so
    # missing sim modules such as `sapien` can be imported without shadowing DP3.
    for site_packages in (
        root / ".venv/lib/python3.10/site-packages",
        root / ".venv/lib/python3.10/site-packages/cmeel.prefix/lib/python3.10/site-packages",
        root / "third_party/pytorch3d",
        root / "third_party/curobo/src",
        root / "third_party/BODex_api/src",
    ):
        if site_packages.is_dir() and str(site_packages) not in sys.path:
            sys.path.append(str(site_packages))


def load_ultradex_config(root: pathlib.Path) -> dict:
    config_path = root / "env/config/env.yaml"
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["asset_path"] = str(root / config["asset_path"])
    config["object_mesh_path"] = str(root / config["object_mesh_path"])
    return config


def infer_hand(task_name: str) -> int:
    if task_name.endswith("single_left"):
        return 0
    if task_name.endswith("single_right"):
        return 1
    if task_name.endswith("bimanual"):
        return 2
    raise ValueError(f"Cannot infer hand from task name: {task_name}")


def build_state(obs_t: dict, hand: int) -> np.ndarray:
    if hand == 0:
        return np.concatenate([obs_t["robot_0"]["qpos"], obs_t["robot_0"]["ee_pose"]]).astype(np.float32)
    if hand == 1:
        return np.concatenate([obs_t["robot_1"]["qpos"], obs_t["robot_1"]["ee_pose"]]).astype(np.float32)
    if hand == 2:
        return np.concatenate([
            obs_t["robot_0"]["qpos"],
            obs_t["robot_0"]["ee_pose"],
            obs_t["robot_1"]["qpos"],
            obs_t["robot_1"]["ee_pose"],
        ]).astype(np.float32)
    raise ValueError(f"Unsupported hand: {hand}")


def resize_point_cloud(point_cloud: np.ndarray, num_points: int) -> np.ndarray:
    point_cloud = point_cloud[:, :3].astype(np.float32)
    if point_cloud.shape[0] == num_points:
        return point_cloud
    if point_cloud.shape[0] > num_points:
        idx = np.linspace(0, point_cloud.shape[0] - 1, num_points, dtype=np.int64)
        return point_cloud[idx]

    padded = np.zeros((num_points, point_cloud.shape[1]), dtype=np.float32)
    padded[: point_cloud.shape[0]] = point_cloud
    return padded


def obs_to_policy_frame(obs_t: dict, hand: int, num_points: int) -> dict[str, np.ndarray]:
    return {
        "point_cloud": resize_point_cloud(obs_t["point_cloud"], num_points),
        "agent_pos": build_state(obs_t, hand),
    }


def stack_obs(history: deque[dict[str, np.ndarray]], device: torch.device) -> dict[str, torch.Tensor]:
    point_cloud = np.stack([item["point_cloud"] for item in history], axis=0)
    agent_pos = np.stack([item["agent_pos"] for item in history], axis=0)
    return {
        "point_cloud": torch.from_numpy(point_cloud).unsqueeze(0).to(device=device, dtype=torch.float32),
        "agent_pos": torch.from_numpy(agent_pos).unsqueeze(0).to(device=device, dtype=torch.float32),
    }


def expand_action(action: np.ndarray, hand: int, init_qpos: list[np.ndarray]) -> np.ndarray:
    if hand == 0:
        return np.concatenate([action.astype(np.float32), init_qpos[1].astype(np.float32)])
    if hand == 1:
        return np.concatenate([init_qpos[0].astype(np.float32), action.astype(np.float32)])
    if hand == 2:
        return action.astype(np.float32)
    raise ValueError(f"Unsupported hand: {hand}")


def resolve_mesh_path(root: pathlib.Path, config: dict, object_path: str | None) -> pathlib.Path:
    if object_path is None:
        object_path = config["object_mesh_path"]
    path = pathlib.Path(object_path)
    if not path.is_absolute():
        path = root / path
    if path.is_dir():
        path = path / "mesh/simplified.obj"
    return path.resolve()


def resolve_checkpoint_path(checkpoint: str) -> pathlib.Path:
    path = pathlib.Path(checkpoint).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Roll out a trained DP3 UltraDexGrasp policy in Sapien and save video.")
    parser.add_argument(
        "--checkpoint",
        default=str(REPO_ROOT / "data/outputs/ultradexgrasp_single_left-dp3-ultradex_seed0/checkpoints/latest.ckpt"),
    )
    parser.add_argument("--ultradex-root", default=str(DEFAULT_ULTRADEX_ROOT))
    parser.add_argument("--object-path", default=None, help="Object directory or mesh/simplified.obj path.")
    parser.add_argument("--object-scale", type=float, default=0.08)
    parser.add_argument("--episode-idx", type=int, default=0)
    parser.add_argument("--hand", type=int, default=None, choices=[0, 1, 2])
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--replan-every", type=int, default=None, help="Defaults to policy n_action_steps.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "data/rollouts/ultradexgrasp_policy"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ultradex_root = pathlib.Path(args.ultradex_root).resolve()
    add_ultradex_to_path(ultradex_root)
    os.chdir(ultradex_root)

    from env.base_env import BaseEnv  # noqa: E402
    from util.util import save_rgb_images_to_video  # noqa: E402

    checkpoint = resolve_checkpoint_path(args.checkpoint)
    payload = torch.load(checkpoint.open("rb"), pickle_module=dill, map_location="cpu")
    workspace = TrainDP3Workspace(payload["cfg"])
    workspace.load_payload(payload)
    cfg = workspace.cfg

    hand = args.hand if args.hand is not None else infer_hand(cfg.task_name)
    device_name = args.device
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required here because UltraDexGrasp BaseEnv.get_obs() uses CUDA tensors.")
    device = torch.device(device_name)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model
    policy.to(device)
    policy.eval()
    policy.reset()

    config = load_ultradex_config(ultradex_root)
    mesh_path = resolve_mesh_path(ultradex_root, config, args.object_path)
    if not mesh_path.is_file():
        raise FileNotFoundError(f"Object mesh not found: {mesh_path}")

    env = BaseEnv(config)
    env.set_object_path_and_scale_and_hand(str(mesh_path), args.object_scale, hand, config["xy_step_str"])

    obs = env.reset(args.episode_idx)
    while not env.is_object_in_boundary(env.get_object_pose()):
        obs = env.reset(args.episode_idx)

    num_points = int(cfg.shape_meta.obs.point_cloud.shape[0])
    n_obs_steps = int(cfg.n_obs_steps)
    replan_every = args.replan_every or int(cfg.n_action_steps)
    obs_history = deque(maxlen=n_obs_steps)
    first_frame = obs_to_policy_frame(obs, hand, num_points)
    for _ in range(n_obs_steps):
        obs_history.append(first_frame)

    images = []
    step_count = 0
    success = False

    while step_count < args.max_steps:
        obs_dict = stack_obs(obs_history, device)
        with torch.no_grad():
            action_seq = policy.predict_action(obs_dict)["action"].squeeze(0).detach().cpu().numpy()

        for action in action_seq[:replan_every]:
            images.append(obs["Primary_0"]["color_image"])
            full_action = expand_action(action, hand, env.init_qpos)
            obs = env.step(full_action)
            obs_history.append(obs_to_policy_frame(obs, hand, num_points))
            step_count += 1
            success = bool(obs.get("success", False))
            if success or step_count >= args.max_steps:
                break
        if success:
            break

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"dp3_rollout_ep{args.episode_idx}_success-{success}.mp4"
    save_rgb_images_to_video(images, str(video_path))

    result = {
        "checkpoint": str(checkpoint),
        "video": str(video_path),
        "success": success,
        "steps": step_count,
        "hand": hand,
        "object_mesh": str(mesh_path),
        "object_scale": args.object_scale,
        "episode_idx": args.episode_idx,
    }
    result_path = output_dir / f"dp3_rollout_ep{args.episode_idx}_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
