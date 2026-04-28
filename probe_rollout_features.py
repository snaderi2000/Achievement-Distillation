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

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

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

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

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
    results = {
        "dataset_path": args.dataset_path,
        "feature_key": args.feature_key,
        "target": args.target,
        "probe_type": args.probe_type,
        "num_states": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
    }

    if args.target == "health":
        labels = get_health_labels(dataset)
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
