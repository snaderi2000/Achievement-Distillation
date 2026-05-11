import argparse
import copy
import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from achievement_distillation.constant import TASKS
from collect_value_map import load_model, set_seed
from counterfactual_env_editor import (
    apply_inventory_edits,
    apply_spawn_object,
    material_names,
    obs_to_tensor,
    parse_inventory_assignments,
    render_env,
    valid_spawn_objects,
    visible_world_cells,
)
from probe_material_value_preference import clear_visible_objects, make_visible_material


SPAWN_ALIASES = {
    "arrow": "arrow_down",
}


def parse_csv(text: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_counts(text: str) -> Tuple[int, ...]:
    counts = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not counts:
        raise ValueError("--counts must contain at least one count.")
    if min(counts) < 0:
        raise ValueError("--counts must be non-negative.")
    return counts


def memory_vector(task_names: Tuple[str, ...], device: th.device) -> th.Tensor:
    unknown = sorted(set(task_names) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown achievement memory tasks: {unknown}")
    values = [1.0 if task in task_names else 0.0 for task in TASKS]
    return th.tensor([values], dtype=th.float32, device=device)


def canonical_texture_name(texture: str) -> str:
    return SPAWN_ALIASES.get(texture, texture)


def available_target_cells(env) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    cells = []
    for cell_index, world_pos, _material, obj in visible_world_cells(env):
        if obj is env._player:
            continue
        cells.append((cell_index, world_pos))
    return cells


def place_count_texture(env, texture: str, count: int, rng: np.random.Generator) -> List[Tuple[int, int]]:
    texture = canonical_texture_name(texture)
    cells = available_target_cells(env)
    if count > len(cells):
        raise ValueError(f"Cannot place count={count}; only {len(cells)} visible non-player cells available.")
    chosen_indices = rng.choice(len(cells), size=count, replace=False)
    chosen = [cells[int(idx)] for idx in chosen_indices]

    valid_materials = set(material_names(env))
    valid_objects = set(valid_spawn_objects())
    if texture not in valid_materials and texture not in valid_objects:
        raise ValueError(
            f"Unknown texture '{texture}'. Valid materials: {sorted(valid_materials)}. "
            f"Valid objects: {sorted(valid_objects)}."
        )

    player_pos = np.asarray(env._player.pos, dtype=np.int64)
    placed_cells = []
    for cell_index, world_pos in chosen:
        pos = tuple(world_pos)
        placed_cells.append(tuple(int(v) for v in cell_index))
        if texture in valid_materials:
            _material, obj = env._world[pos]
            if obj is not None and obj is not env._player:
                env._world.remove(obj)
            env._world[pos] = texture
        else:
            env._world[pos] = "grass"
            rel = np.asarray(world_pos, dtype=np.int64) - player_pos
            edits_log = []
            apply_spawn_object(env, [(int(rel[0]), int(rel[1]), texture)], edits_log)
    return placed_cells


def prepare_base_env(eval_seed: int, background_material: str, inventory_updates: Dict[str, int]):
    from crafter.env import Env

    env = Env(seed=eval_seed)
    env.reset()
    clear_visible_objects(env)
    make_visible_material(env, background_material, keep_player_tile=True)
    env._world.daylight = 1.0
    edits_log = []
    apply_inventory_edits(env, inventory_updates, edits_log)
    return env


def extract_feature(
    model,
    obs: np.ndarray,
    device: th.device,
    feature_key: str,
    achievement_progress: th.Tensor | None,
) -> Tuple[np.ndarray, float]:
    obs_tensor = obs_to_tensor(obs, device)
    kwargs = {}
    if achievement_progress is not None:
        uses_achievement_progress = (
            getattr(model, "use_achievement_progress_input", False) or hasattr(model, "achievement_progress_dim")
        )
        if uses_achievement_progress:
            kwargs["achievement_progress"] = achievement_progress
    with th.no_grad():
        outputs = model.act(obs_tensor, **kwargs)
    if feature_key not in outputs:
        raise KeyError(f"Feature key '{feature_key}' not in model outputs: {sorted(outputs.keys())}")
    feature = outputs[feature_key].detach().cpu().view(-1).numpy().astype(np.float32)
    value = float(outputs["vpreds"].item())
    return feature, value


def score_value(
    model,
    obs: np.ndarray,
    device: th.device,
    achievement_progress: th.Tensor | None,
) -> float:
    obs_tensor = obs_to_tensor(obs, device)
    kwargs = {}
    if achievement_progress is not None:
        uses_achievement_progress = (
            getattr(model, "use_achievement_progress_input", False) or hasattr(model, "achievement_progress_dim")
        )
        if uses_achievement_progress:
            kwargs["achievement_progress"] = achievement_progress
    with th.no_grad():
        outputs = model.act(obs_tensor, **kwargs)
    return float(outputs["vpreds"].item())


def quantity_bin(count: int) -> int:
    if count == 0:
        return 0
    if count == 1:
        return 1
    return 2


def coarse_quantity_bin(count: int) -> int:
    if count == 0:
        return 0
    if count == 1:
        return 1
    if count <= 3:
        return 2
    return 3


def fit_ridge_count_probe(x_train: np.ndarray, y_train: np.ndarray, ridge: float):
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x = (x_train - mean) / std
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    reg = ridge * np.eye(x_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    weights = np.linalg.solve(x_aug.T @ x_aug + reg, x_aug.T @ y_train.astype(np.float64))
    return weights, mean, std


def predict_ridge_count(x: np.ndarray, weights: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x = (x - mean) / std
    x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=x.dtype)], axis=1)
    return x_aug @ weights


class CountMLP(th.nn.Module):
    def __init__(self, insize: int, hidden: int):
        super().__init__()
        self.net = th.nn.Sequential(
            th.nn.Linear(insize, hidden),
            th.nn.ReLU(),
            th.nn.Linear(hidden, hidden),
            th.nn.ReLU(),
            th.nn.Linear(hidden, 1),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x).squeeze(-1)


class CoarseMagnitudeMLP(th.nn.Module):
    def __init__(self, insize: int, hidden: int, nclasses: int):
        super().__init__()
        self.net = th.nn.Sequential(
            th.nn.Linear(insize, hidden),
            th.nn.ReLU(),
            th.nn.Linear(hidden, hidden),
            th.nn.ReLU(),
            th.nn.Linear(hidden, nclasses),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)


def fit_mlp_count_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_all: np.ndarray,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    progress_every: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_std = ((x_train - mean) / std).astype(np.float32)
    x_all_std = ((x_all - mean) / std).astype(np.float32)
    y_train = y_train.astype(np.float32)

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    generator = th.Generator(device="cpu")
    generator.manual_seed(seed)
    dataset = th.utils.data.TensorDataset(th.from_numpy(x_train_std), th.from_numpy(y_train))
    loader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = CountMLP(x_train.shape[1], hidden).to(device)
    opt = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = th.nn.functional.mse_loss(pred, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        if progress_every > 0 and ((epoch + 1) % progress_every == 0 or epoch == 0 or epoch + 1 == epochs):
            print(f"mlp epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.6f}", flush=True)

    model.eval()
    preds = []
    with th.no_grad():
        for start in range(0, len(x_all_std), 1024):
            xb = th.from_numpy(x_all_std[start : start + 1024]).to(device)
            preds.append(model(xb).detach().cpu().numpy())
    return np.concatenate(preds, axis=0), mean, std


def magnitude_label(count: int) -> int:
    if count == 1:
        return 0
    if count <= 3:
        return 1
    return 2


def binary_magnitude_label(count: int) -> int:
    if count in (1, 2):
        return 0
    if count in (5, 6):
        return 1
    raise ValueError("binary_mlp expects only counts 1,2,5,6.")


def magnitude_label_name(label: int) -> str:
    return ("low", "medium", "high")[int(label)]


def binary_magnitude_label_name(label: int) -> str:
    return ("low", "high")[int(label)]


def fit_coarse_magnitude_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_all: np.ndarray,
    hidden: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    nclasses: int = 3,
    progress_every: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True) + 1e-6
    x_train_std = ((x_train - mean) / std).astype(np.float32)
    x_all_std = ((x_all - mean) / std).astype(np.float32)
    y_train = y_train.astype(np.int64)

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    generator = th.Generator(device="cpu")
    generator.manual_seed(seed)
    dataset = th.utils.data.TensorDataset(th.from_numpy(x_train_std), th.from_numpy(y_train))
    loader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    model = CoarseMagnitudeMLP(x_train.shape[1], hidden, nclasses=nclasses).to(device)
    opt = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for epoch in range(epochs):
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = th.nn.functional.cross_entropy(logits, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu().item()))
        if progress_every > 0 and ((epoch + 1) % progress_every == 0 or epoch == 0 or epoch + 1 == epochs):
            print(f"mlp epoch {epoch + 1}/{epochs}: loss={np.mean(losses):.6f}", flush=True)

    model.eval()
    probs = []
    with th.no_grad():
        for start in range(0, len(x_all_std), 1024):
            xb = th.from_numpy(x_all_std[start : start + 1024]).to(device)
            probs.append(th.softmax(model(xb), dim=-1).detach().cpu().numpy())
    probs = np.concatenate(probs, axis=0)
    preds = probs.argmax(axis=1)
    return preds, probs, mean, std


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / denom) if denom > 0 else float("nan")
    rounded = np.clip(np.rint(y_pred), np.min(y_true), np.max(y_true))
    exact_acc = float(np.mean(rounded == y_true))
    within_1_acc = float(np.mean(np.abs(y_pred - y_true) <= 1.0))
    true_bins = np.array([quantity_bin(int(v)) for v in y_true])
    pred_bins = np.array([quantity_bin(int(v)) for v in rounded])
    bin_acc = float(np.mean(true_bins == pred_bins))
    true_coarse = np.array([coarse_quantity_bin(int(v)) for v in y_true])
    pred_coarse = np.array([coarse_quantity_bin(int(v)) for v in rounded])
    coarse_acc = float(np.mean(true_coarse == pred_coarse))
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        corr = float("nan")
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "pearson_corr": corr,
        "rounded_count_accuracy": exact_acc,
        "within_1_count_accuracy": within_1_acc,
        "zero_one_many_accuracy": bin_acc,
        "zero_one_low_high_accuracy": coarse_acc,
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "low_accuracy": float(np.mean(y_pred[y_true == 0] == 0)) if np.any(y_true == 0) else float("nan"),
        "medium_accuracy": float(np.mean(y_pred[y_true == 1] == 1)) if np.any(y_true == 1) else float("nan"),
        "high_accuracy": float(np.mean(y_pred[y_true == 2] == 2)) if np.any(y_true == 2) else float("nan"),
    }


def binary_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "low_accuracy": float(np.mean(y_pred[y_true == 0] == 0)) if np.any(y_true == 0) else float("nan"),
        "high_accuracy": float(np.mean(y_pred[y_true == 1] == 1)) if np.any(y_true == 1) else float("nan"),
    }


def per_texture_classification_rows(rows: Sequence[Dict], y_true: np.ndarray, y_pred: np.ndarray, binary: bool) -> List[Dict]:
    out = []
    textures = sorted(set(row["texture"] for row in rows))
    row_textures = np.array([row["texture"] for row in rows])
    for texture in textures:
        mask = row_textures == texture
        metrics = (
            binary_classification_metrics(y_true[mask], y_pred[mask])
            if binary
            else classification_metrics(y_true[mask], y_pred[mask])
        )
        out.append(
            {
                "texture": texture,
                "split": next(row["split"] for row in rows if row["texture"] == texture),
                "samples": int(mask.sum()),
                **metrics,
            }
        )
    return out


def centroid_rows(features: np.ndarray, rows: Sequence[Dict]) -> Tuple[List[Dict], np.ndarray]:
    groups = sorted({(row["texture"], int(row["count"])) for row in rows})
    centroids = []
    out_rows = []
    for texture, count in groups:
        idx = [i for i, row in enumerate(rows) if row["texture"] == texture and int(row["count"]) == count]
        centroid = features[idx].mean(axis=0)
        centroids.append(centroid)
        out_rows.append({"texture": texture, "count": count, "n": len(idx)})
    centroids = np.stack(centroids, axis=0)
    return out_rows, centroids


def centroid_distance_rows(features: np.ndarray, rows: Sequence[Dict]) -> List[Dict]:
    group_rows, centroids = centroid_rows(features, rows)
    out = []
    for i in range(len(group_rows)):
        for j in range(i + 1, len(group_rows)):
            left = group_rows[i]
            right = group_rows[j]
            out.append(
                {
                    "left_texture": left["texture"],
                    "left_count": left["count"],
                    "right_texture": right["texture"],
                    "right_count": right["count"],
                    "same_count": left["count"] == right["count"],
                    "same_texture": left["texture"] == right["texture"],
                    "distance": float(np.linalg.norm(centroids[i] - centroids[j])),
                }
            )
    return out


def distance_diagnostic(features: np.ndarray, counts: np.ndarray, types: np.ndarray, max_pairs: int, seed: int):
    rng = np.random.default_rng(seed)
    n = len(features)
    if n < 2:
        return {}
    sample_pairs = []
    total_pairs = n * (n - 1) // 2
    if total_pairs <= max_pairs:
        for i in range(n):
            for j in range(i + 1, n):
                sample_pairs.append((i, j))
    else:
        seen = set()
        while len(sample_pairs) < max_pairs:
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            if i > j:
                i, j = j, i
            if (i, j) in seen:
                continue
            seen.add((i, j))
            sample_pairs.append((int(i), int(j)))

    same_count_diff_type = []
    diff_count_same_type = []
    same_type_same_count = []
    diff_type_diff_count = []
    for i, j in sample_pairs:
        dist = float(np.linalg.norm(features[i] - features[j]))
        same_count = counts[i] == counts[j]
        same_type = types[i] == types[j]
        if same_count and not same_type:
            same_count_diff_type.append(dist)
        elif (not same_count) and same_type:
            diff_count_same_type.append(dist)
        elif same_count and same_type:
            same_type_same_count.append(dist)
        else:
            diff_type_diff_count.append(dist)

    def mean_or_nan(values):
        return float(np.mean(values)) if values else float("nan")

    return {
        "same_count_diff_type_mean_distance": mean_or_nan(same_count_diff_type),
        "diff_count_same_type_mean_distance": mean_or_nan(diff_count_same_type),
        "same_type_same_count_mean_distance": mean_or_nan(same_type_same_count),
        "diff_type_diff_count_mean_distance": mean_or_nan(diff_type_diff_count),
        "same_count_diff_type_pairs": len(same_count_diff_type),
        "diff_count_same_type_pairs": len(diff_count_same_type),
    }


def save_examples_montage(examples: Dict[Tuple[str, int], np.ndarray], textures: Sequence[str], count: int, output_path: str):
    available = [(texture, examples[(texture, count)]) for texture in textures if (texture, count) in examples]
    if not available:
        return
    fig, axes = plt.subplots(1, len(available), figsize=(3.0 * len(available), 3.2))
    axes = np.atleast_1d(axes)
    for ax, (texture, obs) in zip(axes, available):
        ax.imshow(obs)
        ax.set_title(f"{count} x {texture}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Example scenes with count={count}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_plots(
    rows: Sequence[Dict],
    y_pred: np.ndarray,
    probe_train_mask: np.ndarray,
    in_domain_holdout_mask: np.ndarray,
    texture_holdout_mask: np.ndarray,
    probe_type: str,
    output_path: str,
):
    types = sorted(set(row["texture"] for row in rows))
    counts = sorted(set(int(row["count"]) for row in rows))
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    for texture in types:
        means = []
        stds = []
        for count in counts:
            values = [row["value"] for row in rows if row["texture"] == texture and row["count"] == count]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
        axes[0].errorbar(counts, means, yerr=stds, marker="o", capsize=3, label=texture)
    axes[0].set_title("Value by count and texture")
    axes[0].set_xlabel("count")
    axes[0].set_ylabel("V(obs)")
    axes[0].legend(frameon=False, fontsize=8)

    y_true = np.array([row["count"] for row in rows], dtype=np.float64)
    axes[1].scatter(y_true[probe_train_mask], y_pred[probe_train_mask], alpha=0.35, label="probe train", s=18)
    axes[1].scatter(
        y_true[in_domain_holdout_mask],
        y_pred[in_domain_holdout_mask],
        alpha=0.65,
        label="same texture holdout",
        s=22,
    )
    axes[1].scatter(
        y_true[texture_holdout_mask],
        y_pred[texture_holdout_mask],
        alpha=0.75,
        label="held-out texture",
        s=22,
    )
    lo, hi = min(counts), max(counts)
    axes[1].plot([lo, hi], [lo, hi], color="black", linewidth=1, alpha=0.55)
    axes[1].set_title(f"{probe_type.upper()} count probe")
    axes[1].set_xlabel("true count")
    axes[1].set_ylabel("predicted count")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_value_plot(rows: Sequence[Dict], output_path: str):
    types = sorted(set(row["texture"] for row in rows))
    counts = sorted(set(int(row["count"]) for row in rows))
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 5.0))

    for texture in types:
        means = []
        stds = []
        for count in counts:
            values = [row["value"] for row in rows if row["texture"] == texture and row["count"] == count]
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
        ax.errorbar(counts, means, yerr=stds, marker="o", capsize=3, label=texture)

    ax.set_title("Value by count and texture")
    ax.set_xlabel("count")
    ax.set_ylabel("V(obs)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_coarse_accuracy_plot(per_texture_rows: Sequence[Dict], output_path: str, binary: bool = False):
    rows = list(per_texture_rows)
    textures = [row["texture"] for row in rows]
    accuracies = [row["accuracy"] for row in rows]
    colors = ["#8abf88" if row["split"] == "train" else "#df8f44" for row in rows]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.6))
    ax.bar(textures, accuracies, color=colors)
    chance = 0.5 if binary else 1.0 / 3.0
    ax.axhline(chance, color="black", linestyle="--", linewidth=1.2, alpha=0.65, label="chance")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("low / high accuracy" if binary else "low / medium / high accuracy")
    ax.set_title("Binary magnitude probe by texture" if binary else "Coarse magnitude probe by texture")
    ax.legend(frameon=False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
        tick.set_ha("right")
    for i, acc in enumerate(accuracies):
        ax.text(i, min(acc + 0.03, 0.97), f"{acc:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Test whether count is linearly decodable across held-out Crafter textures/objects."
    )
    parser.add_argument("--exp_name", type=str, default="ppo_achievement_memory_strong_v100_all")
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument("--train_textures", type=str, default="cow,tree,skeleton")
    parser.add_argument("--test_textures", type=str, default="stone,arrow,plant")
    parser.add_argument("--counts", type=str, default="0,1,2,3,4,5")
    parser.add_argument("--num_layouts", type=int, default=50)
    parser.add_argument("--feature_key", type=str, default="vf_latents")
    parser.add_argument("--value_only", action="store_true", help="Only make the value-by-count plot; skip probes.")
    parser.add_argument("--probe_type", choices=["ridge", "mlp", "coarse_mlp", "binary_mlp"], default="ridge")
    parser.add_argument("--ridge", type=float, default=10.0)
    parser.add_argument("--train_holdout_fraction", type=float, default=0.2)
    parser.add_argument("--mlp_hidden", type=int, default=256)
    parser.add_argument("--mlp_epochs", type=int, default=200)
    parser.add_argument("--mlp_batch_size", type=int, default=256)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_progress_every", type=int, default=25)
    parser.add_argument("--background_material", type=str, default="grass")
    parser.add_argument("--memory_tasks", type=str, default="")
    parser.add_argument(
        "--set_inventory",
        type=str,
        default="health=9,food=9,drink=9,energy=9,wood_pickaxe=0",
    )
    parser.add_argument("--max_distance_pairs", type=int, default=25000)
    parser.add_argument("--example_count", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="count_abstraction_probe")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, _config, ckpt_path = load_model(
        args.exp_name,
        args.timestamp,
        args.train_seed,
        args.ckpt_epoch,
        device,
    )

    train_textures = parse_csv(args.train_textures)
    test_textures = parse_csv(args.test_textures)
    textures = train_textures + test_textures
    counts = parse_counts(args.counts)
    memory_tasks = parse_csv(args.memory_tasks)
    achievement_progress = memory_vector(memory_tasks, device)
    inventory_updates = parse_inventory_assignments(args.set_inventory)

    base_env = prepare_base_env(args.eval_seed, args.background_material, inventory_updates)
    rng = np.random.default_rng(args.eval_seed)
    rows = []
    features = []
    obs_examples = {}

    for texture in textures:
        split = "train" if texture in train_textures else "test"
        for count in counts:
            for layout_id in range(args.num_layouts):
                env = copy.deepcopy(base_env)
                placed = place_count_texture(env, texture, count, rng)
                obs = render_env(env)
                if args.value_only:
                    feature = None
                    value = score_value(model, obs, device, achievement_progress)
                else:
                    feature, value = extract_feature(
                        model,
                        obs,
                        device,
                        feature_key=args.feature_key,
                        achievement_progress=achievement_progress,
                    )
                rows.append(
                    {
                        "texture": texture,
                        "canonical_texture": canonical_texture_name(texture),
                        "split": split,
                        "count": count,
                        "quantity_bin": quantity_bin(count),
                        "layout_id": layout_id,
                        "value": value,
                        "placed_cells": json.dumps(placed),
                    }
                )
                if feature is not None:
                    features.append(feature)
                if layout_id == 0:
                    obs_examples[(texture, count)] = obs
                env.close()

    base_env.close()
    stem = f"{args.exp_name}-s{args.train_seed:02}-e{args.ckpt_epoch:03}-value-count"
    if args.value_only:
        csv_path = os.path.join(args.output_dir, f"{stem}.csv")
        json_path = os.path.join(args.output_dir, f"{stem}.json")
        plot_path = os.path.join(args.output_dir, f"{stem}.png")
        examples_path = os.path.join(args.output_dir, f"{stem}-examples_count{args.example_count}.png")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        save_value_plot(rows, plot_path)
        save_examples_montage(obs_examples, textures, args.example_count, examples_path)
        summary = {
            "checkpoint": ckpt_path,
            "mode": "value_only",
            "train_textures": list(train_textures),
            "test_textures": list(test_textures),
            "counts": list(counts),
            "num_layouts": args.num_layouts,
            "memory_tasks": list(memory_tasks),
            "inventory": inventory_updates,
            "csv_path": csv_path,
            "plot_path": plot_path,
            "examples_path": examples_path,
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
        return

    features = np.stack(features, axis=0)
    y = np.array([row["count"] for row in rows], dtype=np.float64)
    types = np.array([row["texture"] for row in rows])
    train_mask = np.array([row["split"] == "train" for row in rows])
    if not 0.0 <= args.train_holdout_fraction < 1.0:
        raise ValueError("--train_holdout_fraction must be in [0, 1).")
    train_cutoff = int(round(args.num_layouts * (1.0 - args.train_holdout_fraction)))
    train_cutoff = min(max(train_cutoff, 1), args.num_layouts)
    layout_ids = np.array([row["layout_id"] for row in rows], dtype=np.int64)
    probe_train_mask = train_mask & (layout_ids < train_cutoff)
    in_domain_holdout_mask = train_mask & (layout_ids >= train_cutoff)
    texture_holdout_mask = ~train_mask

    if args.probe_type in ("coarse_mlp", "binary_mlp"):
        binary = args.probe_type == "binary_mlp"
        if binary:
            if set(counts) != {1, 2, 5, 6}:
                raise ValueError("--probe_type binary_mlp expects exactly --counts 1,2,5,6.")
            y_class = np.array([binary_magnitude_label(int(v)) for v in y], dtype=np.int64)
            nclasses = 2
            label_name = binary_magnitude_label_name
            magnitude_bins = {"low": [1, 2], "high": [5, 6]}
        else:
            if 0 in counts:
                raise ValueError("--probe_type coarse_mlp expects positive counts only, e.g. --counts 1,2,3,4,5,6.")
            y_class = np.array([magnitude_label(int(v)) for v in y], dtype=np.int64)
            nclasses = 3
            label_name = magnitude_label_name
            magnitude_bins = {"low": [1], "medium": [2, 3], "high": [4, 5, 6]}
        y_pred_class, y_prob, mean, std = fit_coarse_magnitude_probe(
            features[probe_train_mask],
            y_class[probe_train_mask],
            features,
            hidden=args.mlp_hidden,
            epochs=args.mlp_epochs,
            batch_size=args.mlp_batch_size,
            lr=args.mlp_lr,
            seed=args.eval_seed,
            nclasses=nclasses,
            progress_every=args.mlp_progress_every,
        )
        for row, pred, prob in zip(rows, y_pred_class, y_prob):
            true_label = binary_magnitude_label(int(row["count"])) if binary else magnitude_label(int(row["count"]))
            row["magnitude_label"] = label_name(true_label)
            row["predicted_magnitude_label"] = label_name(int(pred))
            row["prob_low"] = float(prob[0])
            if binary:
                row["prob_high"] = float(prob[1])
            else:
                row["prob_medium"] = float(prob[1])
                row["prob_high"] = float(prob[2])

        metrics_fn = binary_classification_metrics if binary else classification_metrics
        train_metrics = metrics_fn(y_class[probe_train_mask], y_pred_class[probe_train_mask])
        in_domain_holdout_metrics = (
            metrics_fn(y_class[in_domain_holdout_mask], y_pred_class[in_domain_holdout_mask])
            if np.any(in_domain_holdout_mask)
            else {}
        )
        test_metrics = metrics_fn(y_class[texture_holdout_mask], y_pred_class[texture_holdout_mask])
        standardized_features = (features - mean) / std
        distance_metrics = distance_diagnostic(
            standardized_features,
            y.astype(np.int64),
            types,
            max_pairs=args.max_distance_pairs,
            seed=args.eval_seed,
        )
        centroid_distances = centroid_distance_rows(standardized_features, rows)
        per_texture_rows = per_texture_classification_rows(rows, y_class, y_pred_class, binary=binary)

        suffix = "binary_magnitude" if binary else "coarse_magnitude"
        stem = f"{args.exp_name}-s{args.train_seed:02}-e{args.ckpt_epoch:03}-{args.feature_key}-{suffix}"
        csv_path = os.path.join(args.output_dir, f"{stem}.csv")
        texture_csv_path = os.path.join(args.output_dir, f"{stem}-texture_accuracy.csv")
        centroid_csv_path = os.path.join(args.output_dir, f"{stem}-centroid_distances.csv")
        json_path = os.path.join(args.output_dir, f"{stem}.json")
        plot_path = os.path.join(args.output_dir, f"{stem}.png")
        accuracy_plot_path = os.path.join(args.output_dir, f"{stem}-texture_accuracy.png")
        examples_path = os.path.join(args.output_dir, f"{stem}-examples_count{args.example_count}.png")
        npz_path = os.path.join(args.output_dir, f"{stem}.npz")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(texture_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_texture_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_texture_rows)
        with open(centroid_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(centroid_distances[0].keys()))
            writer.writeheader()
            writer.writerows(centroid_distances)
        np.savez_compressed(
            npz_path,
            features=features,
            counts=y,
            magnitude_labels=y_class,
            predicted_magnitude_labels=y_pred_class,
            magnitude_probs=y_prob,
            textures=types,
            probe_train_mask=probe_train_mask,
            in_domain_holdout_mask=in_domain_holdout_mask,
            texture_holdout_mask=texture_holdout_mask,
            feature_key=args.feature_key,
        )
        save_value_plot(rows, plot_path)
        save_coarse_accuracy_plot(per_texture_rows, accuracy_plot_path, binary=binary)
        save_examples_montage(obs_examples, textures, args.example_count, examples_path)

        summary = {
            "checkpoint": ckpt_path,
            "feature_key": args.feature_key,
            "train_textures": list(train_textures),
            "test_textures": list(test_textures),
            "counts": list(counts),
            "magnitude_bins": magnitude_bins,
            "num_layouts": args.num_layouts,
            "probe_type": args.probe_type,
            "train_holdout_fraction": args.train_holdout_fraction,
            "probe_train_samples": int(probe_train_mask.sum()),
            "in_domain_holdout_samples": int(in_domain_holdout_mask.sum()),
            "heldout_texture_samples": int(texture_holdout_mask.sum()),
            "memory_tasks": list(memory_tasks),
            "inventory": inventory_updates,
            "mlp_hidden": args.mlp_hidden,
            "mlp_epochs": args.mlp_epochs,
            "mlp_progress_every": args.mlp_progress_every,
            "train_metrics": train_metrics,
            "in_domain_holdout_metrics": in_domain_holdout_metrics,
            "heldout_texture_metrics": test_metrics,
            "per_texture_metrics": per_texture_rows,
            "distance_metrics": distance_metrics,
            "csv_path": csv_path,
            "texture_csv_path": texture_csv_path,
            "centroid_csv_path": centroid_csv_path,
            "npz_path": npz_path,
            "plot_path": plot_path,
            "accuracy_plot_path": accuracy_plot_path,
            "examples_path": examples_path,
        }
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
        return

    if args.probe_type == "ridge":
        weights, mean, std = fit_ridge_count_probe(features[probe_train_mask], y[probe_train_mask], args.ridge)
        y_pred = predict_ridge_count(features, weights, mean, std)
    else:
        y_pred, mean, std = fit_mlp_count_probe(
            features[probe_train_mask],
            y[probe_train_mask],
            features,
            hidden=args.mlp_hidden,
            epochs=args.mlp_epochs,
            batch_size=args.mlp_batch_size,
            lr=args.mlp_lr,
            seed=args.eval_seed,
            progress_every=args.mlp_progress_every,
        )

    for row, pred in zip(rows, y_pred):
        row["predicted_count"] = float(pred)
        rounded = int(np.clip(np.rint(pred), min(counts), max(counts)))
        row["predicted_count_rounded"] = int(rounded)
        row["predicted_quantity_bin"] = quantity_bin(rounded)
        row["predicted_coarse_quantity_bin"] = coarse_quantity_bin(rounded)

    train_metrics = regression_metrics(y[probe_train_mask], y_pred[probe_train_mask])
    in_domain_holdout_metrics = (
        regression_metrics(y[in_domain_holdout_mask], y_pred[in_domain_holdout_mask])
        if np.any(in_domain_holdout_mask)
        else {}
    )
    test_metrics = regression_metrics(y[texture_holdout_mask], y_pred[texture_holdout_mask])
    distance_metrics = distance_diagnostic(
        (features - mean) / std,
        y.astype(np.int64),
        types,
        max_pairs=args.max_distance_pairs,
        seed=args.eval_seed,
    )
    standardized_features = (features - mean) / std
    centroid_distances = centroid_distance_rows(standardized_features, rows)

    stem = f"{args.exp_name}-s{args.train_seed:02}-e{args.ckpt_epoch:03}-{args.feature_key}-{args.probe_type}-count"
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    centroid_csv_path = os.path.join(args.output_dir, f"{stem}-centroid_distances.csv")
    json_path = os.path.join(args.output_dir, f"{stem}.json")
    plot_path = os.path.join(args.output_dir, f"{stem}.png")
    examples_path = os.path.join(args.output_dir, f"{stem}-examples_count{args.example_count}.png")
    npz_path = os.path.join(args.output_dir, f"{stem}.npz")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(centroid_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(centroid_distances[0].keys()))
        writer.writeheader()
        writer.writerows(centroid_distances)
    np.savez_compressed(
        npz_path,
        features=features,
        counts=y,
        predicted_counts=y_pred,
        textures=types,
        probe_train_mask=probe_train_mask,
        in_domain_holdout_mask=in_domain_holdout_mask,
        texture_holdout_mask=texture_holdout_mask,
        feature_key=args.feature_key,
    )
    save_plots(rows, y_pred, probe_train_mask, in_domain_holdout_mask, texture_holdout_mask, args.probe_type, plot_path)
    save_examples_montage(obs_examples, textures, args.example_count, examples_path)

    summary = {
        "checkpoint": ckpt_path,
        "feature_key": args.feature_key,
        "train_textures": list(train_textures),
        "test_textures": list(test_textures),
        "counts": list(counts),
        "num_layouts": args.num_layouts,
        "probe_type": args.probe_type,
        "train_holdout_fraction": args.train_holdout_fraction,
        "probe_train_samples": int(probe_train_mask.sum()),
        "in_domain_holdout_samples": int(in_domain_holdout_mask.sum()),
        "heldout_texture_samples": int(texture_holdout_mask.sum()),
        "memory_tasks": list(memory_tasks),
        "inventory": inventory_updates,
        "ridge": args.ridge,
        "mlp_hidden": args.mlp_hidden if args.probe_type == "mlp" else None,
        "mlp_epochs": args.mlp_epochs if args.probe_type == "mlp" else None,
        "train_metrics": train_metrics,
        "in_domain_holdout_metrics": in_domain_holdout_metrics,
        "heldout_texture_metrics": test_metrics,
        "distance_metrics": distance_metrics,
        "csv_path": csv_path,
        "centroid_csv_path": centroid_csv_path,
        "npz_path": npz_path,
        "plot_path": plot_path,
        "examples_path": examples_path,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
