import argparse
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch as th
import yaml

from crafter.env import Env
from gym import spaces

from achievement_distillation.model import BaseModel
import achievement_distillation.model as model_module


ACTION_NAMES = [
    "noop",
    "left",
    "right",
    "up",
    "down",
    "grab_or_attack",
    "sleep",
    "place_table",
    "place_stone",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def load_config(exp_name: str) -> Dict:
    path = os.path.join("configs", f"{exp_name}.yaml")
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def obs_to_tensor(obs: np.ndarray, device: th.device) -> th.Tensor:
    obs_t = th.from_numpy(np.transpose(obs, (2, 0, 1))).float().unsqueeze(0)
    return obs_t.to(device) / 255.0


def obs_space_from_raw_obs(obs: np.ndarray) -> spaces.Box:
    height, width, channels = obs.shape
    return spaces.Box(low=0.0, high=1.0, shape=(channels, height, width), dtype=np.float32)


def build_model(config: Dict, observation_space: spaces.Box, action_space: spaces.Discrete, device: th.device) -> BaseModel:
    model_cls = getattr(model_module, config["model_cls"])
    model: BaseModel = model_cls(
        observation_space=observation_space,
        action_space=action_space,
        **config["model_kwargs"],
    )
    return model.to(device)


def inventory_vector(info: Dict, inventory_keys: List[str]) -> np.ndarray:
    inventory = info.get("inventory", {})
    return np.array([inventory.get(key, 0) for key in inventory_keys], dtype=np.float32)


def achievements_vector(info: Dict, achievement_keys: List[str]) -> np.ndarray:
    achievements = info.get("achievements", {})
    return np.array([achievements.get(key, 0) for key in achievement_keys], dtype=np.float32)


def maybe_position(info: Dict) -> np.ndarray:
    for key in ("player_pos", "position", "pos"):
        if key in info:
            value = np.asarray(info[key], dtype=np.float32).reshape(-1)
            if value.size >= 2:
                return value[:2]
    return np.full(2, np.nan, dtype=np.float32)


def semantic_value(info: Dict):
    for key in ("semantic_map", "semantic", "map"):
        if key in info:
            value = np.asarray(info[key])
            if value.ndim >= 2 and np.issubdtype(value.dtype, np.number):
                return value.copy()
    return None


def main(args):
    set_seed(args.seed)
    th.set_num_threads(1)
    device = th.device("cuda:0" if th.cuda.is_available() and not args.cpu else "cpu")

    config = load_config(args.exp_name)
    env = Env(seed=args.env_seed)
    obs = env.reset()
    observation_space = obs_space_from_raw_obs(obs)
    action_space = spaces.Discrete(len(ACTION_NAMES))

    model = build_model(config, observation_space, action_space, device)
    run_name = f"{args.exp_name}-{args.timestamp}-s{args.train_seed:02}"
    ckpt_path = os.path.join("models", run_name, f"agent-e{args.ckpt_epoch:03}.pt")
    state_dict = th.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    first_info = env.step(0)[3]
    env = Env(seed=args.env_seed)
    obs = env.reset()
    inventory_keys = sorted(first_info.get("inventory", {}).keys())
    achievement_keys = sorted(first_info.get("achievements", {}).keys())

    observations = []
    next_observations = []
    actions = []
    rewards = []
    dones = []
    episode_ids = []
    step_ids = []
    inventories = []
    next_inventories = []
    achievements = []
    next_achievements = []
    positions = []
    next_positions = []
    next_semantic_maps = []
    keep_semantic_maps = True

    episode_id = 0
    step_id = 0

    while len(actions) < args.num_transitions:
        obs_t = obs_to_tensor(obs, device)
        with th.no_grad():
            outputs = model.act(obs_t)
        action = int(outputs["actions"].item())

        prev_obs = obs
        prev_info = getattr(env, "_last_info", None) or {}
        next_obs, reward, done, info = env.step(action)

        observations.append(prev_obs.copy())
        next_observations.append(next_obs.copy())
        actions.append(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        episode_ids.append(episode_id)
        step_ids.append(step_id)
        inventories.append(inventory_vector(prev_info, inventory_keys))
        next_inventories.append(inventory_vector(info, inventory_keys))
        achievements.append(achievements_vector(prev_info, achievement_keys))
        next_achievements.append(achievements_vector(info, achievement_keys))
        positions.append(maybe_position(prev_info))
        next_positions.append(maybe_position(info))
        next_semantic = semantic_value(info)
        if next_semantic is not None and keep_semantic_maps:
            next_semantic_maps.append(next_semantic)
        else:
            keep_semantic_maps = False

        setattr(env, "_last_info", info)
        obs = next_obs
        step_id += 1
        if done:
            episode_id += 1
            step_id = 0
            obs = env.reset()
            setattr(env, "_last_info", {})

        if len(actions) % args.report_every == 0:
            print(f"collected {len(actions):,}/{args.num_transitions:,} transitions")

    dataset = {
        "observations": th.from_numpy(np.stack(observations)).to(th.uint8),
        "next_observations": th.from_numpy(np.stack(next_observations)).to(th.uint8),
        "actions": th.tensor(actions, dtype=th.long),
        "rewards": th.tensor(rewards, dtype=th.float32),
        "dones": th.tensor(dones, dtype=th.bool),
        "episode_ids": th.tensor(episode_ids, dtype=th.long),
        "step_ids": th.tensor(step_ids, dtype=th.long),
        "inventories": th.from_numpy(np.stack(inventories)).float(),
        "next_inventories": th.from_numpy(np.stack(next_inventories)).float(),
        "achievements": th.from_numpy(np.stack(achievements)).float(),
        "next_achievements": th.from_numpy(np.stack(next_achievements)).float(),
        "positions": th.from_numpy(np.stack(positions)).float(),
        "next_positions": th.from_numpy(np.stack(next_positions)).float(),
        "metadata": {
            "exp_name": args.exp_name,
            "timestamp": args.timestamp,
            "train_seed": args.train_seed,
            "ckpt_epoch": args.ckpt_epoch,
            "env_seed": args.env_seed,
            "checkpoint": ckpt_path,
            "action_names": ACTION_NAMES,
            "inventory_keys": inventory_keys,
            "achievement_keys": achievement_keys,
        },
    }
    if keep_semantic_maps and next_semantic_maps:
        dataset["next_semantic_maps"] = th.from_numpy(np.stack(next_semantic_maps))
        dataset["metadata"]["has_semantic_maps"] = True
    else:
        dataset["metadata"]["has_semantic_maps"] = False

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    th.save(dataset, args.output_path)
    sidecar = os.path.splitext(args.output_path)[0] + ".metadata.json"
    with open(sidecar, "w") as f:
        json.dump(dataset["metadata"], f, indent=2)
    print(f"saved {len(actions):,} transitions to {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default="ppo")
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--env_seed", type=int, default=123)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_transitions", type=int, default=100000)
    parser.add_argument("--output_path", type=str, default="affordance_transitions.pt")
    parser.add_argument("--report_every", type=int, default=10000)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
