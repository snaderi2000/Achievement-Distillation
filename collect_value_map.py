import argparse
import importlib
import os
import random
from typing import Dict, List

import numpy as np
import torch as th
import yaml


def load_config(exp_name: str) -> Dict:
    config_path = f"configs/{exp_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def load_model(exp_name: str, timestamp: str, train_seed: int, ckpt_epoch: int, device: th.device):
    from crafter.env import Env
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

    from achievement_distillation.model import BaseModel
    import achievement_distillation.model as model_module
    from achievement_distillation.wrapper import VecPyTorch

    config = load_config(exp_name)

    temp_venv = VecPyTorch(DummyVecEnv([lambda: Env(seed=train_seed)]), device=device)
    try:
        model_cls = getattr(model_module, config["model_cls"])
        model: BaseModel = model_cls(
            observation_space=temp_venv.observation_space,
            action_space=temp_venv.action_space,
            **config["model_kwargs"],
        )
        model.to(device)
    finally:
        temp_venv.close()

    run_name = f"{exp_name}-{timestamp}-s{train_seed:02}"
    ckpt_path = os.path.join("models", run_name, f"agent-e{ckpt_epoch:03}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    state_dict = th.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, ckpt_path


def pca_2d(latents: np.ndarray) -> np.ndarray:
    latents = latents.astype(np.float64, copy=False)
    mean = latents.mean(axis=0, keepdims=True)
    centered = latents - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    return centered @ components


def collect_value_dataset(model, device: th.device, num_episodes: int, eval_seed: int) -> Dict[str, th.Tensor]:
    from crafter.env import Env
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

    from achievement_distillation.constant import TASKS
    from achievement_distillation.wrapper import VecPyTorch

    venv = DummyVecEnv([lambda: Env(seed=eval_seed)])
    venv = VecPyTorch(venv, device=device)

    observations: List[th.Tensor] = []
    latents: List[th.Tensor] = []
    values: List[th.Tensor] = []
    actions: List[th.Tensor] = []
    rewards: List[float] = []
    dones: List[bool] = []
    achievements: List[th.Tensor] = []
    success_flags: List[th.Tensor] = []
    episode_ids: List[int] = []
    step_ids: List[int] = []

    for episode_idx in range(num_episodes):
        try:
            venv.env_method("seed", eval_seed + episode_idx)
        except Exception:
            pass

        obs = venv.reset()
        done = False
        step_idx = 0
        while not done:
            with th.no_grad():
                outputs = model.act(obs)
                action = outputs["actions"]
                value = outputs["vpreds"]
                latent = outputs["latents"]

            next_obs, reward, done_tensor, infos = venv.step(action)

            observations.append(obs.squeeze(0).detach().cpu())
            latents.append(latent.squeeze(0).detach().cpu())
            values.append(value.squeeze(0).detach().cpu())
            actions.append(action.squeeze(0).detach().cpu())
            rewards.append(float(reward.item()))
            dones.append(bool(done_tensor.item()))
            achievements.append(infos["achievements"].squeeze(0).detach().cpu())
            success_flags.append(infos["successes"].squeeze(0).detach().cpu())
            episode_ids.append(episode_idx)
            step_ids.append(step_idx)

            obs = next_obs
            done = bool(done_tensor.item())
            step_idx += 1

    venv.close()

    dataset = {
        "observations": th.stack(observations),
        "latents": th.stack(latents),
        "values": th.stack(values).view(-1),
        "actions": th.stack(actions).view(-1),
        "rewards": th.tensor(rewards, dtype=th.float32),
        "dones": th.tensor(dones, dtype=th.bool),
        "achievements": th.stack(achievements),
        "successes": th.stack(success_flags),
        "episode_ids": th.tensor(episode_ids, dtype=th.long),
        "step_ids": th.tensor(step_ids, dtype=th.long),
        "task_names": TASKS,
    }
    return dataset


def save_value_map(dataset: Dict[str, th.Tensor], output_path: str, max_points: int = 5000):
    plt = importlib.import_module("matplotlib.pyplot")

    latents = dataset["latents"].cpu().numpy()
    values = dataset["values"].cpu().numpy()
    episode_ids = dataset["episode_ids"].cpu().numpy()

    if len(latents) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(latents), size=max_points, replace=False)
        latents = latents[idx]
        values = values[idx]
        episode_ids = episode_ids[idx]

    coords = pca_2d(latents)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        cmap="viridis",
        s=12,
        alpha=0.8,
        linewidths=0,
    )
    plt.colorbar(scatter, label="Predicted value")
    plt.xlabel("Latent PC 1")
    plt.ylabel("Latent PC 2")
    plt.title("State-value map from rollout states")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    base, ext = os.path.splitext(output_path)
    if ext.lower() not in {".png", ".jpg", ".jpeg", ".pdf"}:
        base = output_path
    np.savez_compressed(
        f"{base}_embedding.npz",
        coords=coords,
        values=values,
        episode_ids=episode_ids,
    )


def main():
    parser = argparse.ArgumentParser(description="Collect rollout states and export predicted values from a trained Crafter agent.")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--output_dataset_path", type=str, default="value_dataset.pt")
    parser.add_argument("--value_map_path", type=str, default=None)
    parser.add_argument("--value_map_max_points", type=int, default=5000)
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

    dataset = collect_value_dataset(
        model=model,
        device=device,
        num_episodes=args.num_episodes,
        eval_seed=args.eval_seed,
    )

    th.save(dataset, args.output_dataset_path)
    print(f"Saved dataset with {len(dataset['values'])} states to {args.output_dataset_path}")
    print(f"Latents shape: {tuple(dataset['latents'].shape)}")
    print(
        f"Value stats: mean={dataset['values'].mean().item():.4f}, "
        f"std={dataset['values'].std().item():.4f}, "
        f"min={dataset['values'].min().item():.4f}, "
        f"max={dataset['values'].max().item():.4f}"
    )

    if args.value_map_path:
        save_value_map(dataset, args.value_map_path, max_points=args.value_map_max_points)
        print(f"Saved value-map figure to {args.value_map_path}")
        print(f"Saved embedding data to {os.path.splitext(args.value_map_path)[0]}_embedding.npz")


if __name__ == "__main__":
    main()
