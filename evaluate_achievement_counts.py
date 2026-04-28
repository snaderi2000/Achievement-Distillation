import argparse
import json
import os
from typing import Dict, List

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
) -> Dict[str, List[float]]:
    from crafter.env import Env

    counts: List[int] = []
    rewards: List[float] = []
    lengths: List[int] = []
    hidsize = int(config.get("model_kwargs", {}).get("hidsize", 512))
    rnn_hidsize = config.get("model_kwargs", {}).get("rnn_hidsize")

    for episode_idx in range(num_episodes):
        env = Env(seed=eval_seed + episode_idx)
        obs = env.reset()
        states = th.zeros(1, hidsize, device=device)
        rnn_states = None
        if rnn_hidsize is not None:
            rnn_states = th.zeros(1, int(rnn_hidsize), device=device)
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            obs_tensor = obs_to_tensor(obs, device)
            act_kwargs = {"states": states}
            if rnn_states is not None:
                act_kwargs["rnn_states"] = rnn_states
            with th.no_grad():
                outputs = model.act(obs_tensor, **act_kwargs)
                action = int(outputs["actions"].item())
                if "next_states" in outputs:
                    states = outputs["next_states"]
                if "next_rnn_states" in outputs:
                    rnn_states = outputs["next_rnn_states"]
            obs, reward, done, info = env.step(action)
            episode_reward += float(reward)
            episode_length += 1

        achievement_count = int(sum(int(v > 0) for v in info["achievements"].values()))
        counts.append(achievement_count)
        rewards.append(episode_reward)
        lengths.append(episode_length)
        env.close()

    return {
        "achievement_counts": counts,
        "episode_rewards": rewards,
        "episode_lengths": lengths,
    }


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

    results = rollout_achievement_counts(
        model=model,
        config=config,
        eval_seed=args.eval_seed,
        num_episodes=args.num_episodes,
        device=device,
    )
    counts = results["achievement_counts"]
    rewards = results["episode_rewards"]
    lengths = results["episode_lengths"]

    average = float(np.mean(counts)) if counts else 0.0
    average_reward = float(np.mean(rewards)) if rewards else 0.0
    average_length = float(np.mean(lengths)) if lengths else 0.0
    total_tasks = len(TASKS)

    print("\nPer-episode results:")
    for idx, (count, reward, length) in enumerate(zip(counts, rewards, lengths), start=1):
        print(f"  episode {idx:02d}: achievements={count}/{total_tasks} reward={reward:.1f} length={length}")
    print(f"\nAverage achievements per episode: {average:.3f}/{total_tasks}")
    print(f"Average reward per episode: {average_reward:.3f}")
    print(f"Average episode length: {average_length:.1f}")

    if args.output_json_path:
        payload = {
            "checkpoint": ckpt_path,
            "eval_seed": args.eval_seed,
            "num_episodes": args.num_episodes,
            "achievement_counts": counts,
            "average_achievements": average,
            "episode_rewards": rewards,
            "average_reward": average_reward,
            "episode_lengths": lengths,
            "average_length": average_length,
            "total_tasks": total_tasks,
        }
        os.makedirs(os.path.dirname(args.output_json_path) or ".", exist_ok=True)
        with open(args.output_json_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved results to {args.output_json_path}")


if __name__ == "__main__":
    main()
