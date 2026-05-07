import argparse
import copy
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


def prepare_inventory(env, keep_vitals_full: bool = True) -> Dict[str, int]:
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
    edits_log = []
    apply_inventory_edits(env, updates, edits_log)
    return updates


def prepare_scene(env, target_delta: Tuple[int, int], target_material: str, keep_vitals_full: bool):
    clear_visible_objects(env)
    make_visible_material(env, "grass", keep_player_tile=True)
    prepare_inventory(env, keep_vitals_full=keep_vitals_full)
    env._world.daylight = 1.0
    set_target_material(env, target_delta, target_material)
    return render_env(env)


def save_figure(
    baseline_obs: np.ndarray,
    stone_obs: np.ndarray,
    baseline_value: float,
    stone_value: float,
    memory_tasks: Tuple[str, ...],
    inventory_updates: Dict[str, int],
    target_delta: Tuple[int, int],
    output_path: str,
):
    delta = stone_value - baseline_value
    shown_inventory = [
        f"{name}={inventory_updates[name]}"
        for name in ("health", "food", "drink", "energy", "wood_pickaxe")
        if name in inventory_updates and inventory_updates[name] > 0
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.2))

    panels = [
        (baseline_obs, f"Baseline: all grass\nV={baseline_value:.3f}"),
        (stone_obs, f"Counterfactual: stone in front\nV={stone_value:.3f}  d={delta:+.3f}"),
    ]
    for ax, (obs, title) in zip(axes[:2], panels):
        ax.imshow(obs)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    axes[2].bar(["all grass", "stone"], [baseline_value, stone_value], color=["#7fbf7b", "#9a9a9a"])
    axes[2].axhline(baseline_value, color="black", linewidth=1, alpha=0.45)
    axes[2].set_title("Predicted value", fontsize=11)
    axes[2].set_ylabel("V")
    axes[2].text(
        0.02,
        0.02,
        "memory: " + ", ".join(memory_tasks)
        + "\ninv: "
        + ", ".join(shown_inventory)
        + "\ntarget delta: "
        + str(tuple(target_delta)),
        transform=axes[2].transAxes,
        fontsize=9,
        va="bottom",
    )

    fig.suptitle("Achievement-memory stone counterfactual", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare an achievement-memory agent's value for all-grass vs stone-in-front scenes."
    )
    parser.add_argument("--exp_name", type=str, default="ppo_achievement_memory_strong_v100_all")
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument("--target_delta", type=str, default="0,1")
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
    parser.add_argument("--output_dir", type=str, default="achievement_memory_stone_counterfactual")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, config, ckpt_path = load_model(
        args.exp_name,
        args.timestamp,
        args.train_seed,
        args.ckpt_epoch,
        device,
    )
    target_delta = parse_vec2(args.target_delta)
    memory_tasks = tuple(task.strip() for task in args.memory_tasks.split(",") if task.strip())
    achievement_progress = memory_vector(memory_tasks, device)

    from crafter.env import Env

    env = Env(seed=args.eval_seed)
    env.reset()
    baseline_env = copy.deepcopy(env)
    stone_env = copy.deepcopy(env)

    keep_vitals_full = not args.empty_vitals
    baseline_obs = prepare_scene(baseline_env, target_delta, "grass", keep_vitals_full)
    stone_obs = prepare_scene(stone_env, target_delta, "stone", keep_vitals_full)
    inventory_updates = dict(stone_env._player.inventory)

    baseline_scores = score_observation(
        model,
        baseline_obs,
        device,
        achievement_progress=achievement_progress,
    )
    stone_scores = score_observation(
        model,
        stone_obs,
        device,
        achievement_progress=achievement_progress,
    )

    stem = f"{args.exp_name}-s{args.train_seed:02}-e{args.ckpt_epoch:03}"
    figure_path = os.path.join(args.output_dir, f"{stem}.png")
    summary_path = os.path.join(args.output_dir, f"{stem}.json")
    save_figure(
        baseline_obs,
        stone_obs,
        baseline_scores["value"],
        stone_scores["value"],
        memory_tasks,
        inventory_updates,
        target_delta,
        figure_path,
    )

    summary = {
        "checkpoint": ckpt_path,
        "eval_seed": args.eval_seed,
        "target_delta": list(target_delta),
        "memory_tasks": list(memory_tasks),
        "inventory": inventory_updates,
        "baseline_value": baseline_scores["value"],
        "stone_value": stone_scores["value"],
        "delta_value": stone_scores["value"] - baseline_scores["value"],
        "figure_path": figure_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    env.close()
    baseline_env.close()
    stone_env.close()


if __name__ == "__main__":
    main()
