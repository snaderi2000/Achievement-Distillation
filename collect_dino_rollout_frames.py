import argparse
import json
import os
from functools import partial
from typing import Dict, List

import imageio.v3 as imageio
import numpy as np
import torch as th

from collect_value_map import load_model, observation_to_uint8_hwc, set_seed


def collect_rollout_frames(
    model,
    device: th.device,
    num_episodes: int,
    num_envs: int,
    eval_seed: int,
    frame_stride: int,
    max_frames: int | None,
) -> List[Dict]:
    from crafter.env import Env
    from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv

    from achievement_distillation.wrapper import VecPyTorch

    if frame_stride <= 0:
        raise ValueError("--frame_stride must be positive.")
    if num_envs <= 0:
        raise ValueError("--num_envs must be positive.")

    seeds = [eval_seed + env_idx for env_idx in range(num_envs)]
    env_fns = [partial(Env, seed=seed) for seed in seeds]
    venv = SubprocVecEnv(env_fns)
    venv = VecPyTorch(venv, device=device)

    records: List[Dict] = []
    global_frame = 0
    obs = venv.reset()

    next_episode_id = 0
    active_episode_ids = [-1 for _ in range(num_envs)]
    step_ids = [0 for _ in range(num_envs)]
    episode_counts = [0 for _ in range(num_envs)]

    for env_idx in range(num_envs):
        if next_episode_id < num_episodes:
            active_episode_ids[env_idx] = next_episode_id
            next_episode_id += 1

    while any(episode_id >= 0 for episode_id in active_episode_ids):
        with th.no_grad():
            outputs = model.act(obs)

        actions = outputs["actions"].clone()
        for env_idx, episode_id in enumerate(active_episode_ids):
            if episode_id < 0:
                actions[env_idx, 0] = 0

        next_obs, reward, done_tensor, infos = venv.step(actions)

        for env_idx, episode_id in enumerate(active_episode_ids):
            if episode_id < 0:
                continue

            step_idx = step_ids[env_idx]
            if step_idx % frame_stride == 0:
                records.append(
                    {
                        "observation": observation_to_uint8_hwc(obs[env_idx].cpu()),
                        "episode_id": episode_id,
                        "step_id": step_idx,
                        "env_index": env_idx,
                        "env_episode_index": episode_counts[env_idx],
                        "eval_seed": seeds[env_idx],
                        "action": int(outputs["actions"][env_idx].item()),
                        "reward": float(reward[env_idx].item()),
                        "done": bool(done_tensor[env_idx].item()),
                        "value": float(outputs["vpreds"][env_idx].item()),
                        "achievements": infos["achievements"][env_idx].cpu().tolist(),
                    }
                )
                global_frame += 1
                if max_frames is not None and global_frame >= max_frames:
                    venv.close()
                    return records

            step_ids[env_idx] += 1
            if bool(done_tensor[env_idx].item()):
                episode_counts[env_idx] += 1
                step_ids[env_idx] = 0
                if next_episode_id < num_episodes:
                    active_episode_ids[env_idx] = next_episode_id
                    next_episode_id += 1
                else:
                    active_episode_ids[env_idx] = -1

        obs = next_obs

    venv.close()
    return records


def save_records(records: List[Dict], output_dir: str, manifest_name: str, run_info: Dict):
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, manifest_name)

    with open(manifest_path, "w", encoding="utf-8") as manifest:
        manifest.write(json.dumps({"run_info": run_info}) + "\n")
        for idx, record in enumerate(records):
            filename = f"frame-{idx:07d}.png"
            image_path = os.path.join(images_dir, filename)
            imageio.imwrite(image_path, record["observation"])

            metadata = {
                "image_path": os.path.join("images", filename),
                "episode_id": record["episode_id"],
                "step_id": record["step_id"],
                "env_index": record["env_index"],
                "env_episode_index": record["env_episode_index"],
                "eval_seed": record["eval_seed"],
                "action": record["action"],
                "reward": record["reward"],
                "done": record["done"],
                "value": record["value"],
                "achievements": record["achievements"],
            }
            manifest.write(json.dumps(metadata) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Roll out a trained Crafter agent and save visited frames as an image dataset for DINO fine-tuning."
    )
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--frame_stride", type=int, default=8)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="dino_rollout_frames")
    parser.add_argument("--manifest_name", type=str, default="metadata.jsonl")
    args = parser.parse_args()

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    th.set_num_threads(1)
    set_seed(args.eval_seed)

    model, _, ckpt_path = load_model(
        args.exp_name,
        args.timestamp,
        args.train_seed,
        args.ckpt_epoch,
        device,
    )
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Using device: {device}")

    records = collect_rollout_frames(
        model=model,
        device=device,
        num_episodes=args.num_episodes,
        num_envs=args.num_envs,
        eval_seed=args.eval_seed,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
    )

    run_info = {
        "exp_name": args.exp_name,
        "timestamp": args.timestamp,
        "train_seed": args.train_seed,
        "ckpt_epoch": args.ckpt_epoch,
        "eval_seed_start": args.eval_seed,
        "num_episodes": args.num_episodes,
        "num_envs": args.num_envs,
        "frame_stride": args.frame_stride,
        "max_frames": args.max_frames,
        "num_saved_frames": len(records),
    }
    save_records(records, output_dir=args.output_dir, manifest_name=args.manifest_name, run_info=run_info)
    print(f"Saved {len(records)} frames to {args.output_dir}")
    print(f"Manifest: {os.path.join(args.output_dir, args.manifest_name)}")


if __name__ == "__main__":
    main()
