import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


class LinearProbe(nn.Module):
    def __init__(self, insize: int, outsize: int):
        super().__init__()
        self.linear = nn.Linear(insize, outsize)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(x)


class MLPProbe(nn.Module):
    def __init__(self, insize: int, outsize: int, hidden_size: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(insize, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, outsize),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


def standardize_features(
    x_train: th.Tensor,
    x_test: th.Tensor,
    eps: float = 1e-6,
) -> Tuple[th.Tensor, th.Tensor]:
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True, unbiased=False)
    std = th.clamp(std, min=eps)
    return (x_train - mean) / std, (x_test - mean) / std


def split_by_episode(
    features: th.Tensor,
    labels: th.Tensor,
    episode_ids: th.Tensor,
    test_size: float,
    seed: int,
) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
    unique_episodes = sorted(set(int(v) for v in episode_ids.tolist()))
    if len(unique_episodes) < 2:
        raise ValueError("Need at least 2 episodes for an episode-disjoint split.")

    train_episodes, test_episodes = train_test_split(
        unique_episodes,
        test_size=test_size,
        random_state=seed,
    )
    train_episode_set = set(train_episodes)
    test_episode_set = set(test_episodes)

    train_mask = th.tensor(
        [int(ep) in train_episode_set for ep in episode_ids.tolist()],
        dtype=th.bool,
    )
    test_mask = th.tensor(
        [int(ep) in test_episode_set for ep in episode_ids.tolist()],
        dtype=th.bool,
    )

    if not train_mask.any() or not test_mask.any():
        raise ValueError("Episode split produced an empty train or test set.")

    return (
        features[train_mask],
        features[test_mask],
        labels[train_mask],
        labels[test_mask],
    )


def build_probe(probe_type: str, insize: int, outsize: int, hidden_size: int) -> nn.Module:
    if probe_type == "linear":
        return LinearProbe(insize, outsize)
    if probe_type == "mlp":
        return MLPProbe(insize, outsize, hidden_size=hidden_size)
    raise ValueError(f"Unsupported probe_type: {probe_type}")


def train_multiclass_probe(
    x_train: th.Tensor,
    y_train: th.Tensor,
    x_test: th.Tensor,
    y_test: th.Tensor,
    probe_type: str,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: th.device,
    log_prefix: str = "",
) -> Dict:
    model = build_probe(probe_type, x_train.shape[1], int(y_train.max().item()) + 1, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    class_counts = np.bincount(y_train.cpu().numpy(), minlength=int(y_train.max().item()) + 1)
    class_weights = class_counts.max() / np.maximum(class_counts, 1)
    criterion = nn.CrossEntropyLoss(weight=th.tensor(class_weights, dtype=th.float32, device=device))

    generator = th.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=max(batch_size, 256))

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item())
            train_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            model.eval()
            eval_preds: List[int] = []
            eval_labels: List[int] = []
            eval_loss_total = 0.0
            eval_batches = 0
            with th.no_grad():
                for xb, yb in test_loader:
                    logits = model(xb.to(device))
                    loss = criterion(logits, yb.to(device))
                    eval_loss_total += float(loss.item())
                    eval_batches += 1
                    eval_preds.extend(logits.argmax(dim=1).cpu().tolist())
                    eval_labels.extend(yb.tolist())
            print(
                f"[probe-{probe_type}-multiclass{log_prefix}] "
                f"epoch={epoch + 1}/{epochs} "
                f"train_loss={train_loss_total / max(train_batches, 1):.4f} "
                f"test_loss={eval_loss_total / max(eval_batches, 1):.4f} "
                f"test_acc={accuracy_score(eval_labels, eval_preds):.4f}"
            )

    model.eval()
    preds: List[int] = []
    labels: List[int] = []
    with th.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            preds.extend(logits.argmax(dim=1).cpu().tolist())
            labels.extend(yb.tolist())

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "num_classes": int(y_train.max().item()) + 1,
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }


def train_binary_probe(
    x_train: th.Tensor,
    y_train: th.Tensor,
    x_test: th.Tensor,
    y_test: th.Tensor,
    probe_type: str,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: th.device,
    log_prefix: str = "",
) -> Dict:
    model = build_probe(probe_type, x_train.shape[1], 1, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    positives = float(y_train.sum().item())
    negatives = float(len(y_train) - positives)
    pos_weight = negatives / max(positives, 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=th.tensor([pos_weight], dtype=th.float32, device=device))

    generator = th.Generator().manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(x_train, y_train.float().unsqueeze(-1)),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    test_loader = DataLoader(
        TensorDataset(x_test, y_test.float().unsqueeze(-1)),
        batch_size=max(batch_size, 256),
    )

    for epoch in range(epochs):
        model.train()
        train_loss_total = 0.0
        train_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss.item())
            train_batches += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch + 1 == epochs:
            model.eval()
            eval_preds: List[int] = []
            eval_labels: List[int] = []
            eval_loss_total = 0.0
            eval_batches = 0
            with th.no_grad():
                for xb, yb in test_loader:
                    logits = model(xb.to(device))
                    loss = criterion(logits, yb.to(device))
                    eval_loss_total += float(loss.item())
                    eval_batches += 1
                    pred = (th.sigmoid(logits) >= 0.5).long().cpu().view(-1)
                    eval_preds.extend(pred.tolist())
                    eval_labels.extend(yb.long().view(-1).tolist())
            print(
                f"[probe-{probe_type}-binary{log_prefix}] "
                f"epoch={epoch + 1}/{epochs} "
                f"train_loss={train_loss_total / max(train_batches, 1):.4f} "
                f"test_loss={eval_loss_total / max(eval_batches, 1):.4f} "
                f"test_acc={accuracy_score(eval_labels, eval_preds):.4f}"
            )

    model.eval()
    preds: List[int] = []
    labels: List[int] = []
    with th.no_grad():
        for xb, yb in test_loader:
            logits = model(xb.to(device))
            pred = (th.sigmoid(logits) >= 0.5).long().cpu().view(-1)
            preds.extend(pred.tolist())
            labels.extend(yb.long().view(-1).tolist())

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "positive_rate_train": float(y_train.float().mean().item()),
        "positive_rate_test": float(y_test.float().mean().item()),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
    }


def get_health_labels(dataset: Dict[str, th.Tensor]) -> th.Tensor:
    if "vitals" in dataset:
        return dataset["vitals"][:, 3].long()
    raise KeyError("Dataset does not contain 'vitals', so health labels are unavailable.")


def get_achievement_targets(dataset: Dict[str, th.Tensor]) -> Tuple[th.Tensor, List[str]]:
    if "successes" not in dataset:
        raise KeyError("Dataset does not contain 'successes', so achievement labels are unavailable.")
    task_names = list(dataset.get("task_names", [f"task_{i}" for i in range(dataset["successes"].shape[1])]))
    return dataset["successes"].long(), task_names


def main():
    parser = argparse.ArgumentParser(description="Probe saved rollout features like rnn_states, latents, and critic features.")
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument(
        "--feature_key",
        type=str,
        choices=("latents", "states", "rnn_states", "memory_latents", "pi_latents", "vf_latents"),
        default="rnn_states",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=("health", "achievements"),
        default="health",
    )
    parser.add_argument("--probe_type", type=str, choices=("linear", "mlp"), default="linear")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--split_mode", type=str, choices=("state", "episode"), default="state")
    parser.add_argument("--split_seed", type=int, default=22)
    parser.add_argument("--probe_seed", type=int, default=420)
    parser.add_argument("--results_json_path", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.probe_seed)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    dataset = th.load(args.dataset_path, map_location="cpu")

    if args.feature_key not in dataset:
        raise KeyError(
            f"Feature '{args.feature_key}' not found in dataset. "
            f"Available keys include: {sorted(k for k in dataset.keys() if isinstance(dataset[k], th.Tensor))}"
        )

    features = dataset[args.feature_key].float()
    episode_ids = dataset.get("episode_ids")
    results = {
        "dataset_path": args.dataset_path,
        "feature_key": args.feature_key,
        "target": args.target,
        "probe_type": args.probe_type,
        "split_mode": args.split_mode,
        "num_states": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
    }

    if args.target == "health":
        labels = get_health_labels(dataset)
        if args.split_mode == "episode":
            if episode_ids is None:
                raise KeyError("Dataset does not contain 'episode_ids', required for split_mode=episode.")
            x_train, x_test, y_train, y_test = split_by_episode(
                features=features,
                labels=labels,
                episode_ids=episode_ids,
                test_size=args.test_size,
                seed=args.split_seed,
            )
        else:
            x_train, x_test, y_train, y_test = train_test_split(
                features,
                labels,
                test_size=args.test_size,
                random_state=args.split_seed,
                stratify=labels.numpy(),
            )
        x_train, x_test = standardize_features(x_train, x_test)
        probe_results = train_multiclass_probe(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            probe_type=args.probe_type,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.probe_seed,
            device=device,
            log_prefix=f" {args.target}",
        )
        results["health"] = {
            **probe_results,
            "label_values": sorted(set(labels.tolist())),
        }
        print(json.dumps(results["health"], indent=2))

    elif args.target == "achievements":
        achievement_targets, task_names = get_achievement_targets(dataset)
        per_task = {}
        macro_accuracy = []
        macro_balanced = []
        macro_f1 = []

        for task_idx, task_name in enumerate(task_names):
            labels = achievement_targets[:, task_idx]
            unique = sorted(set(labels.tolist()))
            if len(unique) < 2:
                per_task[task_name] = {
                    "skipped": True,
                    "reason": f"only one class present: {unique}",
                    "positive_rate": float(labels.float().mean().item()),
                }
                continue

            if args.split_mode == "episode":
                if episode_ids is None:
                    raise KeyError("Dataset does not contain 'episode_ids', required for split_mode=episode.")
                x_train, x_test, y_train, y_test = split_by_episode(
                    features=features,
                    labels=labels,
                    episode_ids=episode_ids,
                    test_size=args.test_size,
                    seed=args.split_seed,
                )
            else:
                x_train, x_test, y_train, y_test = train_test_split(
                    features,
                    labels,
                    test_size=args.test_size,
                    random_state=args.split_seed,
                    stratify=labels.numpy(),
                )
            x_train, x_test = standardize_features(x_train, x_test)
            task_result = train_binary_probe(
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                probe_type=args.probe_type,
                hidden_size=args.hidden_size,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.probe_seed,
                device=device,
                log_prefix=f" {task_name}",
            )
            per_task[task_name] = task_result
            macro_accuracy.append(task_result["accuracy"])
            macro_balanced.append(task_result["balanced_accuracy"])
            macro_f1.append(task_result["f1"])

        results["achievements"] = {
            "macro_accuracy": float(np.mean(macro_accuracy)) if macro_accuracy else None,
            "macro_balanced_accuracy": float(np.mean(macro_balanced)) if macro_balanced else None,
            "macro_f1": float(np.mean(macro_f1)) if macro_f1 else None,
            "per_task": per_task,
        }
        print(json.dumps({
            "macro_accuracy": results["achievements"]["macro_accuracy"],
            "macro_balanced_accuracy": results["achievements"]["macro_balanced_accuracy"],
            "macro_f1": results["achievements"]["macro_f1"],
        }, indent=2))

    if args.results_json_path:
        os.makedirs(os.path.dirname(args.results_json_path) or ".", exist_ok=True)
        with open(args.results_json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved results to {args.results_json_path}")


if __name__ == "__main__":
    main()
