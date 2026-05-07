import argparse
import copy
import csv
import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from achievement_distillation.constant import TASKS
from collect_value_map import load_model, set_seed
from counterfactual_env_editor import (
    apply_inventory_edits,
    parse_inventory_assignments,
    parse_vec2,
    render_env,
    score_observation,
)
from probe_material_value_preference import (
    clear_visible_objects,
    make_visible_material,
    set_target_material,
)


DEFAULT_MEMORY_TASKS = ("collect_wood", "place_table", "make_wood_pickaxe")


def memory_vector(task_names: Tuple[str, ...], device: th.device) -> th.Tensor:
    unknown = sorted(set(task_names) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown achievement memory tasks: {unknown}")
    values = [1.0 if task in task_names else 0.0 for task in TASKS]
    return th.tensor([values], dtype=th.float32, device=device)


def prepare_inventory(
    env,
    keep_vitals_full: bool = True,
    inventory_overrides: Dict[str, int] | None = None,
) -> Dict[str, int]:
    inventory = getattr(env._player, "inventory", None)
    if inventory is None:
        raise RuntimeError("Crafter player inventory is unavailable.")
    updates = {name: 0 for name in inventory.keys()}
    updates["health"] = 9
    updates["wood_pickaxe"] = 1
    if keep_vitals_full:
        updates["food"] = 9
        updates["drink"] = 9
        updates["energy"] = 9
    if inventory_overrides:
        unknown = sorted(set(inventory_overrides) - set(inventory.keys()))
        if unknown:
            raise ValueError(f"Unknown inventory keys: {unknown}")
        updates.update(inventory_overrides)
    edits_log = []
    apply_inventory_edits(env, updates, edits_log)
    return updates


def prepare_scene(
    env,
    target_delta: Tuple[int, int],
    target_material: str,
    keep_vitals_full: bool,
    inventory_overrides: Dict[str, int] | None,
):
    clear_visible_objects(env)
    make_visible_material(env, "grass", keep_player_tile=True)
    prepare_inventory(
        env,
        keep_vitals_full=keep_vitals_full,
        inventory_overrides=inventory_overrides,
    )
    env._world.daylight = 1.0
    set_target_material(env, target_delta, target_material)
    return render_env(env)


def save_figure(
    baseline_obs: np.ndarray,
    target_obs: np.ndarray,
    baseline_value: float,
    target_value: float,
    target_material: str,
    memory_tasks: Tuple[str, ...],
    inventory_updates: Dict[str, int],
    target_delta: Tuple[int, int],
    output_path: str,
):
    delta = target_value - baseline_value
    shown_inventory = [
        f"{name}={inventory_updates[name]}"
        for name in ("health", "food", "drink", "energy", "wood_pickaxe")
        if name in inventory_updates and inventory_updates[name] > 0
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))

    panels = [
        (baseline_obs, f"Baseline: all grass\nV={baseline_value:.3f}"),
        (target_obs, f"Counterfactual: {target_material} in front\nV={target_value:.3f}  d={delta:+.3f}"),
    ]
    for ax, (obs, title) in zip(axes[:2], panels):
        ax.imshow(obs)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    axes[2].bar(["all grass", target_material], [baseline_value, target_value], color=["#7fbf7b", "#9a9a9a"])
    axes[2].axhline(baseline_value, color="black", linewidth=1, alpha=0.45)
    axes[2].set_title("Predicted value", fontsize=11)
    axes[2].set_ylabel("V")
    axes[2].text(
        0.02,
        0.02,
        "memory: " + (", ".join(memory_tasks) if memory_tasks else "empty")
        + "\ninv: "
        + ", ".join(shown_inventory)
        + "\ntarget delta: "
        + str(tuple(target_delta)),
        transform=axes[2].transAxes,
        fontsize=9,
        va="bottom",
    )

    fig.suptitle(f"Achievement-memory {target_material} counterfactual", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_sweep_figure(rows, target_material: str, output_path: str):
    epochs = [row["ckpt_epoch"] for row in rows]
    deltas = [row["delta_value"] for row in rows]
    baseline_values = [row["baseline_value"] for row in rows]
    target_values = [row["target_value"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[0].plot(epochs, deltas, marker="o", color="#444444")
    axes[0].fill_between(epochs, 0.0, deltas, where=np.array(deltas) >= 0.0, color="#d95f02", alpha=0.18)
    axes[0].fill_between(epochs, 0.0, deltas, where=np.array(deltas) < 0.0, color="#1b9e77", alpha=0.18)
    axes[0].set_title(f"{target_material} preference")
    axes[0].set_xlabel("checkpoint epoch")
    axes[0].set_ylabel(f"V({target_material}) - V(grass)")

    axes[1].plot(epochs, baseline_values, marker="o", label="grass", color="#7fbf7b")
    axes[1].plot(epochs, target_values, marker="o", label=target_material, color="#777777")
    axes[1].set_title("Predicted values")
    axes[1].set_xlabel("checkpoint epoch")
    axes[1].set_ylabel("V")
    axes[1].legend(frameon=False)

    fig.suptitle(f"Achievement-memory {target_material} preference across checkpoints")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def score_pair_for_checkpoint(
    exp_name: str,
    timestamp: str,
    train_seed: int,
    ckpt_epoch: int,
    device: th.device,
    baseline_obs: np.ndarray,
    target_obs: np.ndarray,
    achievement_progress: th.Tensor,
):
    model, _config, ckpt_path = load_model(
        exp_name,
        timestamp,
        train_seed,
        ckpt_epoch,
        device,
    )
    baseline_scores = score_observation(
        model,
        baseline_obs,
        device,
        achievement_progress=achievement_progress,
    )
    target_scores = score_observation(
        model,
        target_obs,
        device,
        achievement_progress=achievement_progress,
    )
    return {
        "checkpoint": ckpt_path,
        "ckpt_epoch": ckpt_epoch,
        "baseline_value": baseline_scores["value"],
        "target_value": target_scores["value"],
        "delta_value": target_scores["value"] - baseline_scores["value"],
    }


def parse_epochs(text: str) -> Tuple[int, ...]:
    epochs = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not epochs:
        raise ValueError("--ckpt_epochs did not contain any epochs.")
    return epochs


def main():
    parser = argparse.ArgumentParser(
        description="Compare an achievement-memory agent's value for all-grass vs material-in-front scenes."
    )
    parser.add_argument("--exp_name", type=str, default="ppo_achievement_memory_strong_v100_all")
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument(
        "--ckpt_epochs",
        type=str,
        default=None,
        help="Optional comma-separated checkpoint epochs to sweep, e.g. 50,100,150,200,250.",
    )
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument("--target_delta", type=str, default="0,1")
    parser.add_argument("--target_material", type=str, default="stone")
    parser.add_argument(
        "--memory_tasks",
        type=str,
        default=",".join(DEFAULT_MEMORY_TASKS),
        help="Comma-separated achievement bits supplied to the explicit memory input.",
    )
    parser.add_argument(
        "--empty_vitals",
        action="store_true",
        help="Only set health=9 and wood_pickaxe=1; leave food/drink/energy at 0.",
    )
    parser.add_argument(
        "--set_inventory",
        type=str,
        default=None,
        help="Comma-separated item=value overrides applied after the default inventory, e.g. stone=1.",
    )
    parser.add_argument("--output_dir", type=str, default="achievement_memory_stone_counterfactual")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    target_delta = parse_vec2(args.target_delta)
    memory_tasks = tuple(task.strip() for task in args.memory_tasks.split(",") if task.strip())
    achievement_progress = memory_vector(memory_tasks, device)
    inventory_overrides = parse_inventory_assignments(args.set_inventory)

    from crafter.env import Env

    env = Env(seed=args.eval_seed)
    env.reset()
    baseline_env = copy.deepcopy(env)
    target_env = copy.deepcopy(env)

    keep_vitals_full = not args.empty_vitals
    baseline_obs = prepare_scene(
        baseline_env,
        target_delta,
        "grass",
        keep_vitals_full,
        inventory_overrides,
    )
    target_obs = prepare_scene(
        target_env,
        target_delta,
        args.target_material,
        keep_vitals_full,
        inventory_overrides,
    )
    inventory_updates = dict(target_env._player.inventory)

    epochs = parse_epochs(args.ckpt_epochs) if args.ckpt_epochs else (args.ckpt_epoch,)
    rows = [
        score_pair_for_checkpoint(
            args.exp_name,
            args.timestamp,
            args.train_seed,
            epoch,
            device,
            baseline_obs,
            target_obs,
            achievement_progress,
        )
        for epoch in epochs
    ]

    if args.ckpt_epochs:
        stem = f"{args.exp_name}-s{args.train_seed:02}-{args.target_material}-sweep"
        figure_path = os.path.join(args.output_dir, f"{stem}.png")
        csv_path = os.path.join(args.output_dir, f"{stem}.csv")
        summary_path = os.path.join(args.output_dir, f"{stem}.json")
        save_sweep_figure(rows, args.target_material, figure_path)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["ckpt_epoch", "baseline_value", "target_value", "delta_value", "checkpoint"],
            )
            writer.writeheader()
            writer.writerows(rows)
        summary = {
            "eval_seed": args.eval_seed,
            "target_delta": list(target_delta),
            "target_material": args.target_material,
            "memory_tasks": list(memory_tasks),
            "inventory_overrides": inventory_overrides,
            "inventory": inventory_updates,
            "rows": rows,
            "figure_path": figure_path,
            "csv_path": csv_path,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
        env.close()
        baseline_env.close()
        target_env.close()
        return

    row = rows[0]
    stem = f"{args.exp_name}-s{args.train_seed:02}-e{args.ckpt_epoch:03}-{args.target_material}"
    figure_path = os.path.join(args.output_dir, f"{stem}.png")
    summary_path = os.path.join(args.output_dir, f"{stem}.json")
    save_figure(
        baseline_obs,
        target_obs,
        row["baseline_value"],
        row["target_value"],
        args.target_material,
        memory_tasks,
        inventory_updates,
        target_delta,
        figure_path,
    )

    summary = {
        "eval_seed": args.eval_seed,
        "target_delta": list(target_delta),
        "target_material": args.target_material,
        "memory_tasks": list(memory_tasks),
        "inventory_overrides": inventory_overrides,
        "inventory": inventory_updates,
        "checkpoint": row["checkpoint"],
        "baseline_value": row["baseline_value"],
        "target_value": row["target_value"],
        "delta_value": row["delta_value"],
        "figure_path": figure_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    env.close()
    baseline_env.close()
    target_env.close()


if __name__ == "__main__":
    main()
