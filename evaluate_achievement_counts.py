import argparse
import json
import os
from typing import List

import numpy as np
import torch as th

from collect_value_map import load_model, set_seed
from achievement_distillation.constant import TASKS


def obs_to_tensor(obs: np.ndarray, device: th.device) -> th.Tensor:
    obs = th.from_numpy(np.transpose(obs, (2, 0, 1))).unsqueeze(0).to(device)
    return obs.float() / 255.0


def rollout_achievement_counts(
    model,
    config,
    eval_seed: int,
    num_episodes: int,
    device: th.device,
) -> List[int]:
    from crafter.env import Env

    counts: List[int] = []
    hidsize = int(config.get("model_kwargs", {}).get("hidsize", 512))

    for episode_idx in range(num_episodes):
        env = Env(seed=eval_seed + episode_idx)
        obs = env.reset()
        states = th.zeros(1, hidsize, device=device)
        done = False

        while not done:
            obs_tensor = obs_to_tensor(obs, device)
            with th.no_grad():
                outputs = model.act(obs_tensor, states=states)
                action = int(outputs["actions"].item())
                if "next_states" in outputs:
                    states = outputs["next_states"]
            obs, _, done, info = env.step(action)

        achievement_count = int(sum(int(v > 0) for v in info["achievements"].values()))
        counts.append(achievement_count)
        env.close()

    return counts


def main():
    parser = argparse.ArgumentParser(description="Roll out episodes and report unlocked-achievement counts per episode.")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--ckpt_epoch", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--output_json_path", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.eval_seed)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, config, ckpt_path = load_model(
        args.exp_name,
        args.timestamp,
        args.train_seed,
        args.ckpt_epoch,
        device,
    )
    print(f"Loaded checkpoint: {ckpt_path}")

    counts = rollout_achievement_counts(
        model=model,
        config=config,
        eval_seed=args.eval_seed,
        num_episodes=args.num_episodes,
        device=device,
    )

    average = float(np.mean(counts)) if counts else 0.0
    total_tasks = len(TASKS)

    print("\nPer-episode achievement counts:")
    for idx, count in enumerate(counts, start=1):
        print(f"  episode {idx:02d}: {count}/{total_tasks}")
    print(f"\nAverage achievements per episode: {average:.3f}/{total_tasks}")

    if args.output_json_path:
        payload = {
            "checkpoint": ckpt_path,
            "eval_seed": args.eval_seed,
            "num_episodes": args.num_episodes,
            "achievement_counts": counts,
            "average_achievements": average,
            "total_tasks": total_tasks,
        }
        os.makedirs(os.path.dirname(args.output_json_path) or ".", exist_ok=True)
        with open(args.output_json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {args.output_json_path}")


if __name__ == "__main__":
    main()
