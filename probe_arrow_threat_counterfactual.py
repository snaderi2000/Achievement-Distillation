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
    apply_spawn_object,
    parse_inventory_assignments,
    render_env,
    score_observation,
)
from probe_material_value_preference import clear_visible_objects, make_visible_material


ARROW_OBJECTS = {
    "left": "arrow_left",
    "right": "arrow_right",
    "up": "arrow_up",
    "down": "arrow_down",
}


def parse_csv(text: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def parse_vec2(text: str) -> Tuple[int, int]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected two comma-separated ints, got {text!r}")
    return int(parts[0]), int(parts[1])


def parse_distances(text: str) -> Tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values:
        raise ValueError("--distances must contain at least one integer.")
    if min(values) < 1:
        raise ValueError("--distances must be positive.")
    return values


def memory_vector(task_names: Tuple[str, ...], device: th.device) -> th.Tensor:
    unknown = sorted(set(task_names) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown achievement memory tasks: {unknown}")
    values = [1.0 if task in task_names else 0.0 for task in TASKS]
    return th.tensor([values], dtype=th.float32, device=device)


def prepare_scene(env, inventory_updates: Dict[str, int]):
    clear_visible_objects(env)
    make_visible_material(env, "grass", keep_player_tile=True)
    env._world.daylight = 1.0
    edits_log = []
    apply_inventory_edits(env, inventory_updates, edits_log)


def make_observation(eval_seed: int, inventory_updates: Dict[str, int], arrow_delta=None, arrow_direction="down"):
    from crafter.env import Env

    env = Env(seed=eval_seed)
    env.reset()
    prepare_scene(env, inventory_updates)
    if arrow_delta is not None:
        edits_log = []
        apply_spawn_object(env, [(arrow_delta[0], arrow_delta[1], ARROW_OBJECTS[arrow_direction])], edits_log)
    obs = render_env(env)
    inventory = dict(env._player.inventory)
    env.close()
    return obs, inventory


def save_figure(
    baseline_obs,
    arrow_obs,
    baseline_value: float,
    arrow_value: float,
    arrow_delta: Tuple[int, int],
    arrow_direction: str,
    memory_tasks: Tuple[str, ...],
    inventory_updates: Dict[str, int],
    output_path: str,
):
    delta = arrow_value - baseline_value
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    panels = [
        (baseline_obs, f"Baseline: all grass\nV={baseline_value:.3f}"),
        (
            arrow_obs,
            f"Arrow {arrow_direction} at delta {arrow_delta}\nV={arrow_value:.3f}  d={delta:+.3f}",
        ),
    ]
    for ax, (obs, title) in zip(axes[:2], panels):
        ax.imshow(obs)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    axes[2].bar(["grass", "arrow"], [baseline_value, arrow_value], color=["#7fbf7b", "#d55e00"])
    axes[2].axhline(baseline_value, color="black", linewidth=1, alpha=0.45)
    axes[2].set_title("Predicted value", fontsize=11)
    axes[2].set_ylabel("V")
    axes[2].text(
        0.02,
        0.02,
        "memory: "
        + (", ".join(memory_tasks) if memory_tasks else "empty")
        + "\ninv: "
        + ", ".join(f"{k}={v}" for k, v in inventory_updates.items() if v > 0),
        transform=axes[2].transAxes,
        fontsize=9,
        va="bottom",
    )
    fig.suptitle("Incoming-arrow value counterfactual", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_distance_sweep(rows, output_path: str):
    distances = [row["distance"] for row in rows]
    deltas = [row["delta_value"] for row in rows]
    values = [row["arrow_value"] for row in rows]
    baseline = rows[0]["baseline_value"]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[0].plot(distances, deltas, marker="o", color="#d55e00")
    axes[0].invert_xaxis()
    axes[0].set_title("Threat effect by arrow distance")
    axes[0].set_xlabel("cells above player")
    axes[0].set_ylabel("V(arrow) - V(grass)")

    axes[1].axhline(baseline, color="#7fbf7b", linewidth=1.5, alpha=0.8, label="grass")
    axes[1].plot(distances, values, marker="o", color="#d55e00", label="arrow")
    axes[1].invert_xaxis()
    axes[1].set_title("Predicted value")
    axes[1].set_xlabel("cells above player")
    axes[1].set_ylabel("V")
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_direction_sweep(
    baseline_obs,
    rows,
    baseline_value: float,
    arrow_delta: Tuple[int, int],
    memory_tasks: Tuple[str, ...],
    inventory_updates: Dict[str, int],
    output_path: str,
):
    fig = plt.figure(figsize=(14.0, 7.2))
    grid = fig.add_gridspec(2, 5, height_ratios=[1.0, 0.9], width_ratios=[1, 1, 1, 1, 1])

    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(baseline_obs)
    ax.set_title(f"Baseline\nV={baseline_value:.3f}", fontsize=11)
    ax.axis("off")

    for idx, row in enumerate(rows):
        ax = fig.add_subplot(grid[0, idx + 1])
        ax.imshow(row["obs"])
        ax.set_title(
            f"{row['direction']}\nV={row['arrow_value']:.3f}  d={row['delta_value']:+.3f}",
            fontsize=11,
        )
        ax.axis("off")

    ax = fig.add_subplot(grid[1, :])
    directions = [row["direction"] for row in rows]
    deltas = [row["delta_value"] for row in rows]
    colors = ["#d55e00" if direction == "down" else "#777777" for direction in directions]
    ax.axhline(0.0, color="black", linewidth=1, alpha=0.55)
    ax.bar(directions, deltas, color=colors)
    ax.set_ylabel("V(arrow direction) - V(grass)")
    ax.set_title("Value sensitivity to arrow direction at fixed position")
    ax.text(
        0.01,
        0.03,
        "arrow delta: "
        + str(tuple(arrow_delta))
        + "\nexpected hit direction: down"
        + "\nmemory: "
        + (", ".join(memory_tasks) if memory_tasks else "empty")
        + "\ninv: "
        + ", ".join(f"{k}={v}" for k, v in inventory_updates.items() if v > 0),
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
    )

    fig.suptitle("Arrow-direction dynamics counterfactual", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare all-grass baseline vs an incoming arrow aimed at the player."
    )
    parser.add_argument("--exp_name", type=str, default="ppo_achievement_memory_strong_v100_all_20m")
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--ckpt_epoch", type=int, default=3250)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument("--arrow_delta", type=str, default="0,-2")
    parser.add_argument("--arrow_direction", choices=sorted(ARROW_OBJECTS), default="down")
    parser.add_argument(
        "--direction_sweep",
        action="store_true",
        help="Score all four arrow rotations at the same arrow_delta.",
    )
    parser.add_argument(
        "--distances",
        type=str,
        default=None,
        help="Optional comma-separated sweep of cells above player, e.g. 1,2,3.",
    )
    parser.add_argument("--memory_tasks", type=str, default="")
    parser.add_argument(
        "--set_inventory",
        type=str,
        default="health=9,food=9,drink=9,energy=9,wood_pickaxe=0",
    )
    parser.add_argument("--output_dir", type=str, default="arrow_threat_counterfactual")
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
    memory_tasks = parse_csv(args.memory_tasks)
    achievement_progress = memory_vector(memory_tasks, device)
    inventory_updates = parse_inventory_assignments(args.set_inventory)

    baseline_obs, inventory = make_observation(args.eval_seed, inventory_updates)
    baseline_scores = score_observation(
        model,
        baseline_obs,
        device,
        achievement_progress=achievement_progress,
    )

    arrow_delta = parse_vec2(args.arrow_delta)
    arrow_obs, _ = make_observation(
        args.eval_seed,
        inventory_updates,
        arrow_delta=arrow_delta,
        arrow_direction=args.arrow_direction,
    )
    arrow_scores = score_observation(
        model,
        arrow_obs,
        device,
        achievement_progress=achievement_progress,
    )

    stem = f"{args.exp_name}-s{args.train_seed:02}-e{args.ckpt_epoch:04}-arrow_{args.arrow_direction}"
    figure_path = os.path.join(args.output_dir, f"{stem}.png")
    summary_path = os.path.join(args.output_dir, f"{stem}.json")
    save_figure(
        baseline_obs,
        arrow_obs,
        baseline_scores["value"],
        arrow_scores["value"],
        arrow_delta,
        args.arrow_direction,
        memory_tasks,
        inventory,
        figure_path,
    )

    summary = {
        "checkpoint": ckpt_path,
        "eval_seed": args.eval_seed,
        "arrow_delta": list(arrow_delta),
        "arrow_direction": args.arrow_direction,
        "memory_tasks": list(memory_tasks),
        "inventory": inventory,
        "baseline_value": baseline_scores["value"],
        "arrow_value": arrow_scores["value"],
        "delta_value": arrow_scores["value"] - baseline_scores["value"],
        "figure_path": figure_path,
    }

    if args.direction_sweep:
        direction_rows = []
        for direction in ("down", "left", "right", "up"):
            obs, _ = make_observation(
                args.eval_seed,
                inventory_updates,
                arrow_delta=arrow_delta,
                arrow_direction=direction,
            )
            scores = score_observation(
                model,
                obs,
                device,
                achievement_progress=achievement_progress,
            )
            direction_rows.append(
                {
                    "direction": direction,
                    "baseline_value": baseline_scores["value"],
                    "arrow_value": scores["value"],
                    "delta_value": scores["value"] - baseline_scores["value"],
                    "obs": obs,
                }
            )
        direction_stem = f"{args.exp_name}-s{args.train_seed:02}-e{args.ckpt_epoch:04}-arrow_direction_sweep"
        direction_path = os.path.join(args.output_dir, f"{direction_stem}.png")
        direction_csv_path = os.path.join(args.output_dir, f"{direction_stem}.csv")
        save_direction_sweep(
            baseline_obs,
            direction_rows,
            baseline_scores["value"],
            arrow_delta,
            memory_tasks,
            inventory,
            direction_path,
        )
        csv_rows = [{k: v for k, v in row.items() if k != "obs"} for row in direction_rows]
        with open(direction_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)
        summary["direction_sweep"] = csv_rows
        summary["direction_sweep_path"] = direction_path
        summary["direction_sweep_csv_path"] = direction_csv_path

    if args.distances:
        rows = []
        for distance in parse_distances(args.distances):
            obs, _ = make_observation(
                args.eval_seed,
                inventory_updates,
                arrow_delta=(0, -distance),
                arrow_direction=args.arrow_direction,
            )
            scores = score_observation(
                model,
                obs,
                device,
                achievement_progress=achievement_progress,
            )
            rows.append(
                {
                    "distance": distance,
                    "baseline_value": baseline_scores["value"],
                    "arrow_value": scores["value"],
                    "delta_value": scores["value"] - baseline_scores["value"],
                }
            )
        sweep_path = os.path.join(args.output_dir, f"{stem}-distance_sweep.png")
        csv_path = os.path.join(args.output_dir, f"{stem}-distance_sweep.csv")
        save_distance_sweep(rows, sweep_path)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        summary["distance_sweep"] = rows
        summary["distance_sweep_path"] = sweep_path
        summary["distance_sweep_csv_path"] = csv_path

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
