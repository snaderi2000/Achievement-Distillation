import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, Iterable

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from crafter.env import Env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from achievement_distillation.model import *
from achievement_distillation.wrapper import VecPyTorch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def obs_to_tensor(obs: np.ndarray, device: th.device) -> th.Tensor:
    obs = th.from_numpy(np.transpose(obs, (2, 0, 1))).unsqueeze(0).to(device)
    return obs.float() / 255.0


def load_config(exp_name: str) -> Dict:
    config_path = f"configs/{exp_name}.yaml"
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def instantiate_model(exp_name: str, timestamp: str, train_seed: int, ckpt_epoch: int, device: th.device):
    config = load_config(exp_name)
    temp_venv = VecPyTorch(DummyVecEnv([lambda: Env()]), device=device)
    try:
        model_cls = getattr(sys.modules[__name__], config["model_cls"])
        model = model_cls(
            observation_space=temp_venv.observation_space,
            action_space=temp_venv.action_space,
            **config["model_kwargs"],
        ).to(device)
    finally:
        temp_venv.close()

    run_name = f"{exp_name}-{timestamp}-s{train_seed:02}"
    ckpt_path = os.path.join("models", run_name, f"agent-e{ckpt_epoch:03}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = th.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, ckpt_path


def infer_tree_ids(env: Env) -> set[int]:
    mat_ids = getattr(getattr(env, "_world", None), "_mat_ids", None)
    if not isinstance(mat_ids, dict):
        raise RuntimeError("Could not access Crafter material id mapping from env._world._mat_ids.")

    material_names = [name for name in mat_ids.keys() if isinstance(name, str)]
    exact = [name for name in material_names if name == "tree"]
    partial = [name for name in material_names if "tree" in name]
    fallback = [name for name in material_names if name == "wood"]
    names = exact or partial or fallback
    if not names:
        raise RuntimeError(
            "Could not infer a tree material id from Crafter materials: "
            f"{sorted(material_names)}"
        )
    return {int(mat_ids[name]) for name in names}


def current_health(env: Env) -> int:
    inventory = getattr(getattr(env, "_player", None), "inventory", None)
    if inventory is None or "health" not in inventory:
        raise RuntimeError("Could not read current health from env._player.inventory['health'].")
    return int(inventory["health"])


def current_tree_count(env: Env, tree_ids: set[int]) -> int:
    if not hasattr(env, "_sem_view"):
        raise RuntimeError("Could not access Crafter semantic view via env._sem_view.")
    if not hasattr(env, "_local_view"):
        raise RuntimeError("Could not access Crafter local view via env._local_view.")
    if not hasattr(env, "_player"):
        raise RuntimeError("Could not access Crafter player state via env._player.")

    semantic = np.asarray(env._sem_view(), dtype=np.int64)
    local_grid = np.asarray(env._local_view._grid, dtype=np.int64)
    offset = local_grid // 2
    center = np.asarray(env._player.pos, dtype=np.int64)

    visible = np.zeros(tuple(local_grid), dtype=semantic.dtype)
    for x in range(local_grid[0]):
        for y in range(local_grid[1]):
            pos = center + np.array([x, y], dtype=np.int64) - offset
            if 0 <= pos[0] < semantic.shape[0] and 0 <= pos[1] < semantic.shape[1]:
                visible[x, y] = semantic[pos[0], pos[1]]

    return int(np.isin(visible, list(tree_ids)).sum())


def collect_dataset(args, device: th.device) -> Dict[str, th.Tensor]:
    model, config, ckpt_path = instantiate_model(
        args.collect_exp_name,
        args.collect_timestamp,
        args.collect_train_seed,
        args.collect_ckpt_epoch,
        device,
    )
    print(f"Loaded collection checkpoint: {ckpt_path}")

    env = Env(seed=args.eval_seed)
    tree_ids = infer_tree_ids(env)
    print(f"Tree semantic ids used for counting: {sorted(tree_ids)}")

    hidsize = config.get("model_kwargs", {}).get("hidsize", 512)
    obs = env.reset()
    states = th.zeros(1, hidsize, device=device)

    observations = []
    health_labels = []
    tree_labels = []
    episode_ids = []
    step_ids = []

    total_steps = 0
    for episode_idx in range(args.num_episodes):
        if episode_idx > 0:
            obs = env.reset()
            states = th.zeros(1, hidsize, device=device)

        done = False
        step_idx = 0
        while True:
            observations.append(th.from_numpy(np.transpose(obs, (2, 0, 1))).to(th.uint8))
            health_labels.append(current_health(env))
            tree_labels.append(current_tree_count(env, tree_ids))
            episode_ids.append(episode_idx)
            step_ids.append(step_idx)

            if args.max_states and len(observations) >= args.max_states:
                done = True
                break

            obs_tensor = obs_to_tensor(obs, device)
            with th.no_grad():
                outputs = model.act(obs_tensor, states=states)
                action = int(outputs["actions"].item())
                if "next_states" in outputs:
                    states = outputs["next_states"]

            obs, _, done, _ = env.step(action)
            step_idx += 1
            total_steps += 1

            if done:
                break

        print(
            f"[collect] episode={episode_idx + 1}/{args.num_episodes} "
            f"states={len(observations)} steps={total_steps}"
        )
        if args.max_states and len(observations) >= args.max_states:
            break

    env.close()

    dataset = {
        "observations": th.stack(observations),
        "health": th.tensor(health_labels, dtype=th.long),
        "tree_count": th.tensor(tree_labels, dtype=th.long),
        "episode_ids": th.tensor(episode_ids, dtype=th.long),
        "step_ids": th.tensor(step_ids, dtype=th.long),
        "metadata": {
            "collect_exp_name": args.collect_exp_name,
            "collect_timestamp": args.collect_timestamp,
            "collect_train_seed": args.collect_train_seed,
            "collect_ckpt_epoch": args.collect_ckpt_epoch,
            "eval_seed": args.eval_seed,
            "num_episodes": args.num_episodes,
            "tree_ids": sorted(tree_ids),
        },
    }
    print(
        f"Collected {dataset['observations'].shape[0]} states. "
        f"Health range: {dataset['health'].min().item()}-{dataset['health'].max().item()}, "
        f"tree_count range: {dataset['tree_count'].min().item()}-{dataset['tree_count'].max().item()}"
    )
    return dataset


def maybe_cap_tree_counts(labels: th.Tensor, tree_count_cap: int | None) -> th.Tensor:
    if tree_count_cap is None:
        return labels
    return th.clamp(labels, max=tree_count_cap)


def build_splits(observations: th.Tensor, labels: th.Tensor, train_size: int, test_size: int, seed: int):
    total = observations.shape[0]
    if total == 0:
        raise ValueError("No states available for probing.")

    if train_size + test_size > total:
        test_fraction = 0.2 if total > 1 else 0.0
    else:
        subset_size = train_size + test_size
        g = th.Generator().manual_seed(seed)
        perm = th.randperm(total, generator=g)[:subset_size]
        observations = observations[perm]
        labels = labels[perm]
        test_fraction = test_size / subset_size

    stratify = labels.numpy()
    counts = Counter(stratify.tolist())
    if any(count < 2 for count in counts.values()):
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        observations,
        labels,
        test_size=test_fraction,
        random_state=seed,
        stratify=stratify,
    )
    return X_train, X_test, y_train, y_test


def remap_labels(y_train: th.Tensor, y_test: th.Tensor):
    unique_values = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    value_to_class = {value: idx for idx, value in enumerate(unique_values)}
    class_to_value = {idx: value for value, idx in value_to_class.items()}
    y_train = th.tensor([value_to_class[int(v)] for v in y_train.tolist()], dtype=th.long)
    y_test = th.tensor([value_to_class[int(v)] for v in y_test.tolist()], dtype=th.long)
    return y_train, y_test, value_to_class, class_to_value


def extract_latents(model, observations: th.Tensor, device: th.device, batch_size: int) -> th.Tensor:
    loader = DataLoader(TensorDataset(observations), batch_size=batch_size)
    latents = []
    model.eval()
    with th.no_grad():
        for (obs_batch,) in loader:
            obs_batch = obs_batch.to(device).float() / 255.0
            latents.append(model.encode(obs_batch).cpu())
    return th.cat(latents, dim=0)


def train_probe(
    X_train_latents: th.Tensor,
    y_train: th.Tensor,
    X_test_latents: th.Tensor,
    y_test: th.Tensor,
    num_classes: int,
    device: th.device,
    epochs: int,
    batch_size: int,
    seed: int,
):
    classifier = nn.Linear(X_train_latents.shape[1], num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    class_counts = np.bincount(y_train.numpy(), minlength=num_classes)
    class_weights = class_counts.max() / np.maximum(class_counts, 1)
    criterion = nn.CrossEntropyLoss(weight=th.tensor(class_weights, dtype=th.float32, device=device))

    g = th.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(X_train_latents, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    test_loader = DataLoader(TensorDataset(X_test_latents, y_test), batch_size=max(batch_size, 256))

    for epoch in range(epochs):
        classifier.train()
        for lat_batch, label_batch in train_loader:
            lat_batch = lat_batch.to(device)
            label_batch = label_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(classifier(lat_batch), label_batch)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % max(epochs // 5, 1) == 0 or epoch == 0:
            print(f"[probe] epoch={epoch + 1}/{epochs}")

    classifier.eval()
    preds = []
    labels = []
    with th.no_grad():
        for lat_batch, label_batch in test_loader:
            logits = classifier(lat_batch.to(device))
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(label_batch.tolist())

    return np.array(labels), np.array(preds)


def main():
    parser = argparse.ArgumentParser(description="Collect Crafter states and probe latent representations for simple attributes.")
    parser.add_argument("--collect_exp_name", type=str, help="Checkpoint used to collect the shared probe dataset.")
    parser.add_argument("--collect_timestamp", type=str, help="Timestamp of collection checkpoint.")
    parser.add_argument("--collect_train_seed", type=int, help="Train seed of collection checkpoint.")
    parser.add_argument("--collect_ckpt_epoch", type=int, default=250, help="Epoch of collection checkpoint.")
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--max_states", type=int, default=0, help="Optional hard cap on collected states.")
    parser.add_argument("--output_dataset_path", type=str, default=None)
    parser.add_argument("--load_dataset_path", type=str, default=None)

    parser.add_argument("--analysis_exp_name", type=str, help="Checkpoint whose latents will be probed.")
    parser.add_argument("--analysis_timestamp", type=str, help="Timestamp of analysis checkpoint.")
    parser.add_argument("--analysis_train_seed", type=int, help="Train seed of analysis checkpoint.")
    parser.add_argument("--analysis_ckpt_epoch", type=int, default=250)
    parser.add_argument("--target", type=str, choices=("health", "tree_count"), default="health")
    parser.add_argument("--tree_count_cap", type=int, default=None, help="Optionally cap tree counts into a final bucket.")
    parser.add_argument("--train_size", type=int, default=50000)
    parser.add_argument("--test_size", type=int, default=10000)
    parser.add_argument("--probe_epochs", type=int, default=200)
    parser.add_argument("--probe_batch_size", type=int, default=64)
    parser.add_argument("--extract_batch_size", type=int, default=256)
    parser.add_argument("--split_seed", type=int, default=22)
    parser.add_argument("--probe_seed", type=int, default=420)
    parser.add_argument("--results_json_path", type=str, default=None)
    args = parser.parse_args()

    if not args.load_dataset_path and not all(
        value is not None for value in [args.collect_exp_name, args.collect_timestamp, args.collect_train_seed]
    ):
        parser.error("Provide collection checkpoint args or --load_dataset_path.")
    if not all(
        value is not None for value in [args.analysis_exp_name, args.analysis_timestamp, args.analysis_train_seed]
    ):
        parser.error("Provide analysis checkpoint args.")

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    set_seed(args.eval_seed)

    if args.load_dataset_path:
        dataset = th.load(args.load_dataset_path)
        print(f"Loaded dataset from {args.load_dataset_path}")
    else:
        dataset = collect_dataset(args, device)
        if args.output_dataset_path:
            th.save(dataset, args.output_dataset_path)
            print(f"Saved dataset to {args.output_dataset_path}")

    observations = dataset["observations"]
    labels = dataset[args.target].clone()
    if args.target == "tree_count":
        labels = maybe_cap_tree_counts(labels, args.tree_count_cap)

    X_train, X_test, y_train_raw, y_test_raw = build_splits(
        observations,
        labels,
        train_size=args.train_size,
        test_size=args.test_size,
        seed=args.split_seed,
    )
    y_train, y_test, value_to_class, class_to_value = remap_labels(y_train_raw, y_test_raw)
    print(
        f"Target={args.target}, classes={len(class_to_value)}, "
        f"label_values={list(class_to_value.values())}"
    )

    model, _, ckpt_path = instantiate_model(
        args.analysis_exp_name,
        args.analysis_timestamp,
        args.analysis_train_seed,
        args.analysis_ckpt_epoch,
        device,
    )
    print(f"Loaded analysis checkpoint: {ckpt_path}")

    X_train_latents = extract_latents(model, X_train, device, args.extract_batch_size)
    X_test_latents = extract_latents(model, X_test, device, args.extract_batch_size)
    print(f"Latent shapes: train={tuple(X_train_latents.shape)}, test={tuple(X_test_latents.shape)}")

    y_true, y_pred = train_probe(
        X_train_latents,
        y_train,
        X_test_latents,
        y_test,
        num_classes=len(class_to_value),
        device=device,
        epochs=args.probe_epochs,
        batch_size=args.probe_batch_size,
        seed=args.probe_seed,
    )

    accuracy = float(accuracy_score(y_true, y_pred))
    print(f"Probe accuracy: {accuracy:.4f}")
    class_names = [str(class_to_value[idx]) for idx in range(len(class_to_value))]
    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    if args.results_json_path:
        results = {
            "target": args.target,
            "accuracy": accuracy,
            "class_to_value": class_to_value,
            "analysis_exp_name": args.analysis_exp_name,
            "analysis_timestamp": args.analysis_timestamp,
            "analysis_train_seed": args.analysis_train_seed,
            "analysis_ckpt_epoch": args.analysis_ckpt_epoch,
            "dataset_path": args.load_dataset_path or args.output_dataset_path,
        }
        with open(args.results_json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.results_json_path}")


if __name__ == "__main__":
    main()
