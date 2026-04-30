import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_affordance_inverse_model import InverseDynamicsCNN, TransitionDataset


CRAFT_REQUIREMENTS = {
    "make_wood_pickaxe": {"wood": 1},
    "make_stone_pickaxe": {"wood": 1, "stone": 1},
    "make_iron_pickaxe": {"wood": 1, "stone": 1, "iron": 1, "coal": 1},
    "make_wood_sword": {"wood": 1},
    "make_stone_sword": {"wood": 1, "stone": 1},
    "make_iron_sword": {"wood": 1, "stone": 1, "iron": 1, "coal": 1},
    "place_table": {"wood": 2},
    "place_stone": {"stone": 1},
    "place_furnace": {"stone": 4},
    "place_plant": {"sapling": 1},
}


def score_dataset(model, dataset, device, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_log_probs = []
    all_probs = []
    model.eval()
    with th.no_grad():
        for x, actions in loader:
            x = x.to(device)
            actions = actions.to(device)
            logits = model(x)
            log_probs = F.log_softmax(logits, dim=-1)
            true_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
            all_log_probs.append(true_log_probs.cpu())
            all_probs.append(true_log_probs.exp().cpu())
    return th.cat(all_log_probs).numpy(), th.cat(all_probs).numpy()


def inventory_column(inventory: np.ndarray, keys: List[str], name: str) -> np.ndarray:
    if name not in keys:
        return np.zeros(inventory.shape[0], dtype=np.float32)
    return inventory[:, keys.index(name)]


def group_stats(name: str, mask: np.ndarray, values: np.ndarray) -> Dict:
    count = int(mask.sum())
    if count == 0:
        return {"group": name, "count": 0, "mean": None, "std": None}
    selected = values[mask]
    return {
        "group": name,
        "count": count,
        "mean": float(selected.mean()),
        "std": float(selected.std()),
    }


def add_binary_split(rows: List[Dict], prefix: str, mask: np.ndarray, values: np.ndarray):
    rows.append(group_stats(f"{prefix}:yes", mask, values))
    rows.append(group_stats(f"{prefix}:no", ~mask, values))


def add_action_conditioned_split(
    rows: List[Dict],
    action_names: List[str],
    actions: np.ndarray,
    condition_name: str,
    condition_mask: np.ndarray,
    values: np.ndarray,
    min_count: int,
):
    for action_id, action_name in enumerate(action_names):
        action_mask = actions == action_id
        if int(action_mask.sum()) < min_count:
            continue
        rows.append(group_stats(f"action:{action_name}/{condition_name}:yes", action_mask & condition_mask, values))
        rows.append(group_stats(f"action:{action_name}/{condition_name}:no", action_mask & ~condition_mask, values))


def add_count_bins(rows: List[Dict], prefix: str, counts: np.ndarray, values: np.ndarray):
    max_count = int(counts.max()) if len(counts) else 0
    for count in range(max_count + 1):
        rows.append(group_stats(f"{prefix}:{count}", counts == count, values))


def is_valid_action(action_name: str, inventory_row: np.ndarray, inventory_keys: List[str]) -> bool:
    reqs = CRAFT_REQUIREMENTS.get(action_name)
    if reqs is None:
        return True
    for item, needed in reqs.items():
        if item not in inventory_keys:
            return False
        if inventory_row[inventory_keys.index(item)] < needed:
            return False
    return True


def write_group_csv(path: str, rows: List[Dict]):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "count", "mean", "std"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def maybe_plot(output_dir: str, step_ids: np.ndarray, achievement_count: np.ndarray, empowerment: np.ndarray, rows: List[Dict]):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    def binned_plot(x, y, xlabel, ylabel, path, bins=20):
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if len(x) == 0:
            return
        edges = np.linspace(x.min(), x.max(), bins + 1)
        centers = 0.5 * (edges[:-1] + edges[1:])
        means = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (x >= lo) & (x < hi)
            means.append(float(y[mask].mean()) if mask.any() else np.nan)
        plt.figure(figsize=(7, 4))
        plt.plot(centers, means)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    binned_plot(step_ids.astype(np.float32), empowerment, "episode step", "true-action probability", os.path.join(output_dir, "empowerment_vs_time.png"))
    binned_plot(achievement_count.astype(np.float32), empowerment, "achievement count", "true-action probability", os.path.join(output_dir, "empowerment_vs_achievement_count.png"), bins=max(1, int(achievement_count.max()) + 1))

    plot_rows = [row for row in rows if row["count"] > 0 and row["mean"] is not None]
    if plot_rows:
        labels = [row["group"] for row in plot_rows]
        means = [row["mean"] for row in plot_rows]
        plt.figure(figsize=(max(8, len(labels) * 0.45), 4))
        plt.bar(np.arange(len(labels)), means)
        plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
        plt.ylabel("true-action probability")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "empowerment_groups.png"))
        plt.close()


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() and not args.cpu else "cpu")

    data = th.load(args.dataset_path, map_location="cpu")
    ckpt = th.load(args.model_path, map_location=device)
    model = InverseDynamicsCNN(ckpt["in_channels"], ckpt["num_actions"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    dataset = TransitionDataset(data)
    log_empowerment, prob_empowerment = score_dataset(model, dataset, device, args.batch_size)

    inventory = data["inventories"].numpy()
    next_inventory = data["next_inventories"].numpy()
    achievements = data["achievements"].numpy()
    actions = data["actions"].numpy()
    step_ids = data["step_ids"].numpy()
    obs = data["observations"].numpy()
    next_obs = data["next_observations"].numpy()
    metadata = data.get("metadata", {})
    inventory_keys = metadata.get("inventory_keys", [])
    action_names = metadata.get("action_names", [str(i) for i in range(ckpt["num_actions"])])

    changed = (obs != next_obs).reshape(obs.shape[0], -1).any(axis=1)
    achievement_count = achievements.sum(axis=1)
    has_any_pickaxe = (
        inventory_column(inventory, inventory_keys, "wood_pickaxe")
        + inventory_column(inventory, inventory_keys, "stone_pickaxe")
        + inventory_column(inventory, inventory_keys, "iron_pickaxe")
    ) > 0
    has_any_sword = (
        inventory_column(inventory, inventory_keys, "wood_sword")
        + inventory_column(inventory, inventory_keys, "stone_sword")
        + inventory_column(inventory, inventory_keys, "iron_sword")
    ) > 0
    acquired_pickaxe = (
        inventory_column(next_inventory, inventory_keys, "wood_pickaxe")
        + inventory_column(next_inventory, inventory_keys, "stone_pickaxe")
        + inventory_column(next_inventory, inventory_keys, "iron_pickaxe")
        >
        inventory_column(inventory, inventory_keys, "wood_pickaxe")
        + inventory_column(inventory, inventory_keys, "stone_pickaxe")
        + inventory_column(inventory, inventory_keys, "iron_pickaxe")
    )

    valid = np.array(
        [is_valid_action(action_names[int(action)], inventory[idx], inventory_keys) for idx, action in enumerate(actions)],
        dtype=bool,
    )

    rows = [
        group_stats("all", np.ones_like(valid, dtype=bool), prob_empowerment),
        group_stats("obs_changed", changed, prob_empowerment),
        group_stats("obs_unchanged", ~changed, prob_empowerment),
        group_stats("valid_action", valid, prob_empowerment),
        group_stats("invalid_craft_or_place", ~valid, prob_empowerment),
        group_stats("has_pickaxe", has_any_pickaxe, prob_empowerment),
        group_stats("no_pickaxe", ~has_any_pickaxe, prob_empowerment),
        group_stats("has_sword", has_any_sword, prob_empowerment),
        group_stats("no_sword", ~has_any_sword, prob_empowerment),
        group_stats("pickaxe_acquisition_step", acquired_pickaxe, prob_empowerment),
    ]

    for action_id, action_name in enumerate(action_names):
        rows.append(group_stats(f"action:{action_name}", actions == action_id, prob_empowerment))
        rows.append(group_stats(f"action:{action_name}/valid", (actions == action_id) & valid, prob_empowerment))
        rows.append(group_stats(f"action:{action_name}/invalid", (actions == action_id) & ~valid, prob_empowerment))

    add_action_conditioned_split(
        rows,
        action_names,
        actions,
        "has_pickaxe",
        has_any_pickaxe,
        prob_empowerment,
        args.min_action_count,
    )
    add_action_conditioned_split(
        rows,
        action_names,
        actions,
        "has_sword",
        has_any_sword,
        prob_empowerment,
        args.min_action_count,
    )
    add_action_conditioned_split(
        rows,
        action_names,
        actions,
        "obs_changed",
        changed,
        prob_empowerment,
        args.min_action_count,
    )
    add_count_bins(rows, "achievement_count", achievement_count.astype(np.int64), prob_empowerment)

    write_group_csv(os.path.join(args.output_dir, "group_stats.csv"), rows)
    np.savez_compressed(
        os.path.join(args.output_dir, "empowerment_scores.npz"),
        log_empowerment=log_empowerment,
        prob_empowerment=prob_empowerment,
        valid_action=valid,
        obs_changed=changed,
        achievement_count=achievement_count,
        has_pickaxe=has_any_pickaxe,
        has_sword=has_any_sword,
        step_ids=step_ids,
        actions=actions,
    )

    summary = {
        "num_transitions": int(len(actions)),
        "mean_log_empowerment": float(log_empowerment.mean()),
        "mean_prob_empowerment": float(prob_empowerment.mean()),
        "random_action_prob": float(1.0 / ckpt["num_actions"]),
        "groups": rows,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    maybe_plot(args.output_dir, step_ids, achievement_count, prob_empowerment, rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="affordance_transitions.pt")
    parser.add_argument("--model_path", type=str, default="affordance_probe/inverse_model_best.pt")
    parser.add_argument("--output_dir", type=str, default="affordance_analysis")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--min_action_count", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
