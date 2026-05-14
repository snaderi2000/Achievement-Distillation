import argparse
import copy
import csv
import json
import os
import pickle
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from achievement_distillation.constant import TASKS
from collect_value_map import load_model, observation_to_uint8_hwc, set_seed
from counterfactual_env_editor import apply_inventory_edits


def normalize_path(path: str) -> str:
    if path.startswith("file://"):
        return path[len("file://") :]
    return path


def obs_batch_from_uint8(obs_hwc: np.ndarray, device: th.device) -> th.Tensor:
    obs = th.from_numpy(np.ascontiguousarray(obs_hwc)).permute(2, 0, 1).unsqueeze(0).to(device)
    return obs.float() / 255.0


def score(model, obs_hwc: np.ndarray, achievement_progress: th.Tensor, device: th.device) -> float:
    kwargs = {}
    if getattr(model, "use_achievement_progress_input", False) or hasattr(model, "achievement_progress_dim"):
        kwargs["achievement_progress"] = achievement_progress
    with th.no_grad():
        outputs = model.act(obs_batch_from_uint8(obs_hwc, device), **kwargs)
    return float(outputs["vpreds"].item())


def load_dataset_snapshot(dataset_path: str, episode_id: int, step_id: int, device: th.device) -> Dict:
    dataset = th.load(normalize_path(dataset_path), map_location="cpu")
    if "env_snapshots" not in dataset:
        raise KeyError("Dataset has no env_snapshots. Regenerate it with collect_value_map.py --save_env_snapshots.")
    episode_ids = dataset["episode_ids"].cpu()
    step_ids = dataset["step_ids"].cpu()
    matches = th.nonzero((episode_ids == episode_id) & (step_ids == step_id), as_tuple=False).view(-1)
    if matches.numel() == 0:
        raise ValueError(f"Dataset has no episode={episode_id}, step={step_id}.")
    idx = int(matches[0].item())
    env = pickle.loads(dataset["env_snapshots"][idx])
    obs = env.render()
    saved_obs = observation_to_uint8_hwc(dataset["observations"][idx])
    obs_l1 = float(np.mean(np.abs(saved_obs.astype(np.float32) - obs.astype(np.float32))))
    progress = dataset.get("achievement_progress_inputs")
    if progress is None:
        memory = th.zeros(1, len(TASKS), device=device)
    else:
        memory = progress[idx].float().view(1, -1).to(device)
    values = dataset.get("values")
    source_value = None if values is None else float(values[idx].item())
    return {
        "dataset_index": idx,
        "env": env,
        "obs_hwc": obs,
        "achievement_progress": memory,
        "dataset_value": source_value,
        "snapshot_obs_l1": obs_l1,
    }


def zero_all_inventory(env):
    for key in list(env._player.inventory.keys()):
        env._player.inventory[key] = 0
    env._player.health = 0


def render_with_inventory(base_env, inventory_updates: Dict[str, int]) -> np.ndarray:
    env = copy.deepcopy(base_env)
    zero_all_inventory(env)
    apply_inventory_edits(env, inventory_updates, [])
    obs = env.render()
    env.close()
    return obs


def scenario_specs() -> List[Dict]:
    vitals = {"health": 9, "food": 9, "drink": 9, "energy": 9}
    return [
        {
            "name": "empty_full_health",
            "title": "empty inventory\nfull vitals",
            "inventory": dict(vitals),
            "memory_tasks": [],
        },
        {
            "name": "wood",
            "title": "+ wood",
            "inventory": {**vitals, "wood": 1},
            "memory_tasks": ["collect_wood"],
        },
        {
            "name": "place_table",
            "title": "+ table memory",
            "inventory": {**vitals, "wood": 1},
            "memory_tasks": ["collect_wood", "place_table"],
        },
        {
            "name": "wood_pickaxe_wood",
            "title": "+ wood pickaxe",
            "inventory": {**vitals, "wood": 1, "wood_pickaxe": 1},
            "memory_tasks": ["collect_wood", "place_table", "make_wood_pickaxe"],
        },
        {
            "name": "wood_pickaxe_wood_stone",
            "title": "+ stone",
            "inventory": {**vitals, "wood": 1, "wood_pickaxe": 1, "stone": 1},
            "memory_tasks": ["collect_wood", "place_table", "make_wood_pickaxe", "collect_stone"],
        },
        {
            "name": "stone_stockpile",
            "title": "+ more stone",
            "inventory": {**vitals, "wood": 1, "wood_pickaxe": 1, "stone": 2},
            "memory_tasks": ["collect_wood", "place_table", "make_wood_pickaxe", "collect_stone"],
        },
        {
            "name": "stone_pickaxe_stone_remains",
            "title": "+ stone pickaxe\nstone remains",
            "inventory": {**vitals, "wood": 1, "wood_pickaxe": 1, "stone": 1, "stone_pickaxe": 1},
            "memory_tasks": ["collect_wood", "place_table", "make_wood_pickaxe", "collect_stone", "make_stone_pickaxe"],
        },
        {
            "name": "stone_sword",
            "title": "+ stone sword",
            "inventory": {**vitals, "wood": 1, "wood_pickaxe": 1, "stone": 1, "stone_pickaxe": 1, "stone_sword": 1},
            "memory_tasks": [
                "collect_wood",
                "place_table",
                "make_wood_pickaxe",
                "collect_stone",
                "make_stone_pickaxe",
                "make_stone_sword",
            ],
        },
    ]


def memory_from_tasks(task_names: List[str], device: th.device) -> th.Tensor:
    unknown = sorted(set(task_names) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown memory tasks: {unknown}")
    values = [1.0 if task in task_names else 0.0 for task in TASKS]
    return th.tensor([values], dtype=th.float32, device=device)


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path: str, rows: List[Dict], observations: List[np.ndarray], episode_id: int, step_id: int, memory_mode: str):
    n = len(rows)
    fig = plt.figure(figsize=(15.5, 6.7))
    gs = fig.add_gridspec(2, n, height_ratios=[1.0, 0.8])
    for idx, (row, obs) in enumerate(zip(rows, observations)):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(obs)
        ax.set_title(f"{idx + 1}. {row['title']}\nV={row['value']:.3f}", fontsize=9)
        ax.axis("off")

    ax_bar = fig.add_subplot(gs[1, :])
    labels = [f"{idx + 1}" for idx in range(n)]
    values = [float(row["value"]) for row in rows]
    bars = ax_bar.bar(labels, values, color="#4c78a8")
    ax_bar.set_title(f"Inventory progression counterfactuals, episode {episode_id}, step {step_id}, memory={memory_mode}")
    ax_bar.set_xlabel("scenario")
    ax_bar.set_ylabel("V(obs)")
    for bar, value in zip(bars, values):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="One-image inventory progression counterfactuals from a saved env snapshot.")
    parser.add_argument("--exp_name", type=str, default="ppo")
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--step_id", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument(
        "--memory_mode",
        choices=["zero", "progression", "snapshot"],
        default="zero",
        help="zero: all-zero memory; progression: scenario-specific memory; snapshot: saved memory for all panels.",
    )
    parser.add_argument("--output_dir", type=str, default="expirements/inventory_progression_counterfactuals")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, _config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)

    snapshot = load_dataset_snapshot(args.dataset_path, args.episode_id, args.step_id, device)
    scenarios = scenario_specs()
    observations = []
    rows = []
    for idx, spec in enumerate(scenarios, start=1):
        obs = render_with_inventory(snapshot["env"], spec["inventory"])
        if args.memory_mode == "zero":
            memory = th.zeros(1, len(TASKS), device=device)
        elif args.memory_mode == "snapshot":
            memory = snapshot["achievement_progress"]
        else:
            memory = memory_from_tasks(spec["memory_tasks"], device)
        value = score(model, obs, memory, device)
        observations.append(obs)
        rows.append(
            {
                "scenario_index": idx,
                "name": spec["name"],
                "title": spec["title"].replace("\n", " "),
                "value": value,
                "inventory": json.dumps(spec["inventory"], sort_keys=True),
                "memory_tasks": ",".join(spec["memory_tasks"]) if args.memory_mode == "progression" else args.memory_mode,
            }
        )

    stem = f"{args.exp_name}-s{args.train_seed:02d}-ep{args.episode_id:03d}-step{args.step_id:04d}-inventory-progression"
    fig_path = os.path.join(args.output_dir, f"{stem}.png")
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    json_path = os.path.join(args.output_dir, f"{stem}.json")
    save_figure(fig_path, rows, observations, args.episode_id, args.step_id, args.memory_mode)
    write_csv(csv_path, rows)
    summary = {
        "checkpoint": ckpt_path,
        "dataset_path": normalize_path(args.dataset_path),
        "episode_id": args.episode_id,
        "step_id": args.step_id,
        "dataset_index": snapshot["dataset_index"],
        "dataset_value": snapshot["dataset_value"],
        "snapshot_obs_l1": snapshot["snapshot_obs_l1"],
        "memory_mode": args.memory_mode,
        "rows": rows,
        "figure_path": fig_path,
        "csv_path": csv_path,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
