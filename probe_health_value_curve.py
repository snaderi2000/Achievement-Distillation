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
    replay_to_step,
    score_observation,
)
from probe_material_value_preference import (
    clear_visible_objects,
    make_visible_material,
    set_target_texture,
)


def memory_vector(task_names: Tuple[str, ...], device: th.device) -> th.Tensor:
    unknown = sorted(set(task_names) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown achievement memory tasks: {unknown}")
    values = [1.0 if task in task_names else 0.0 for task in TASKS]
    return th.tensor([values], dtype=th.float32, device=device)


def prepare_controlled_scene(
    env,
    background_material: str,
    target_delta: Tuple[int, int] | None,
    target_texture: str | None,
    inventory_updates: Dict[str, int],
):
    clear_visible_objects(env)
    make_visible_material(env, background_material, keep_player_tile=True)
    env._world.daylight = 1.0
    edits_log = []
    apply_inventory_edits(env, inventory_updates, edits_log)
    if target_texture is not None:
        from counterfactual_env_editor import material_names

        if target_delta is None:
            raise ValueError("--target_delta is required when using --target_texture.")
        set_target_texture(env, target_delta, target_texture, set(material_names(env)))


def set_inventory_item(env, item_name: str, value: int):
    value = int(value)
    if item_name not in env._player.inventory:
        raise ValueError(f"Unknown inventory item '{item_name}'.")
    env._player.inventory[item_name] = value
    if item_name == "health":
        env._player.health = value


def score_inventory_curve(
    model,
    base_env,
    device: th.device,
    sweep_item: str,
    min_value: int,
    max_value: int,
    achievement_progress: th.Tensor | None,
    states: th.Tensor | None,
    rnn_states: th.Tensor | None,
):
    rows = []
    images = {}
    mid_value = (min_value + max_value) // 2
    example_values = {min_value, mid_value, max_value}
    for value in range(min_value, max_value + 1):
        env = copy.deepcopy(base_env)
        set_inventory_item(env, sweep_item, value)
        obs = render_env(env)
        scores = score_observation(
            model,
            obs,
            device,
            states=states,
            rnn_states=rnn_states,
            achievement_progress=achievement_progress,
        )
        row = {
            sweep_item: value,
            "value": scores["value"],
        }
        if "health_value" in scores:
            row["health_value"] = scores["health_value"]
        if "achievement_value" in scores:
            row["achievement_value"] = scores["achievement_value"]
        rows.append(row)
        if value in example_values:
            images[value] = obs
        env.close()
    return rows, images


def save_figure(rows, images, memory_tasks, inventory_updates, sweep_item: str, output_path: str):
    xs = [row[sweep_item] for row in rows]
    values = [row["value"] for row in rows]
    example_values = sorted(images)

    fig = plt.figure(figsize=(11.2, 5.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.35])
    for idx, value in enumerate(example_values):
        ax = fig.add_subplot(gs[0, idx])
        ax.imshow(images[value])
        ax.set_title(f"{sweep_item}={value}", fontsize=11)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, :])
    ax.plot(xs, values, marker="o", linewidth=2, color="#444444")
    ax.set_xticks(xs)
    ax.set_xlabel(f"rendered {sweep_item}")
    ax.set_ylabel("predicted value V(obs)")
    ax.set_title(f"Value as rendered {sweep_item} changes")
    ax.grid(True, alpha=0.25)

    memory_text = ", ".join(memory_tasks) if memory_tasks else "empty"
    inv_text = ", ".join(
        f"{name}={value}"
        for name, value in inventory_updates.items()
        if value > 0 and name != sweep_item
    )
    fig.suptitle(f"{sweep_item} counterfactual curve\nmemory: {memory_text}; inventory: {inv_text}", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_csv(path: str, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Hold a Crafter scene fixed and sweep one rendered inventory item.")
    parser.add_argument("--exp_name", type=str, default="ppo_achievement_memory_strong_v100_all")
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument(
        "--memory_tasks",
        type=str,
        default="",
        help="Comma-separated achievement bits supplied to explicit-memory models. Empty string means no achievements.",
    )
    parser.add_argument(
        "--set_inventory",
        type=str,
        default="health=9,food=9,drink=9,energy=9,wood_pickaxe=0",
        help="Comma-separated inventory values held fixed while --sweep_item is varied.",
    )
    parser.add_argument("--sweep_item", type=str, default="health")
    parser.add_argument("--min_value", type=int, default=None)
    parser.add_argument("--max_value", type=int, default=9)
    parser.add_argument("--background_material", type=str, default="grass")
    parser.add_argument("--target_texture", type=str, default=None)
    parser.add_argument("--target_delta", type=str, default="0,1")
    parser.add_argument(
        "--episode_id",
        type=int,
        default=None,
        help="Optional replay episode. If omitted, use a controlled all-background scene from env reset.",
    )
    parser.add_argument(
        "--step_id",
        type=int,
        default=None,
        help="Optional replay step. Required with --episode_id.",
    )
    parser.add_argument("--output_dir", type=str, default="health_value_curve")
    args = parser.parse_args()

    if (args.episode_id is None) != (args.step_id is None):
        raise ValueError("Use both --episode_id and --step_id, or omit both.")

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

    memory_tasks = tuple(task.strip() for task in args.memory_tasks.split(",") if task.strip())
    achievement_progress = memory_vector(memory_tasks, device)
    inventory_updates = parse_inventory_assignments(args.set_inventory)
    min_value = 1 if args.min_value is None and args.sweep_item == "health" else (0 if args.min_value is None else args.min_value)
    inventory_updates[args.sweep_item] = min_value

    states = None
    rnn_states = None
    if args.episode_id is None:
        from crafter.env import Env

        env = Env(seed=args.eval_seed)
        env.reset()
        prepare_controlled_scene(
            env,
            background_material=args.background_material,
            target_delta=parse_vec2(args.target_delta) if args.target_texture is not None else None,
            target_texture=args.target_texture,
            inventory_updates=inventory_updates,
        )
        base_env = env
    else:
        replay = replay_to_step(
            model=model,
            config=config,
            eval_seed=args.eval_seed,
            target_episode=args.episode_id,
            target_step=args.step_id,
            device=device,
        )
        base_env = replay.env
        states = replay.states
        rnn_states = replay.rnn_states
        edits_log = []
        apply_inventory_edits(base_env, inventory_updates, edits_log)

    rows, images = score_inventory_curve(
        model,
        base_env,
        device=device,
        sweep_item=args.sweep_item,
        min_value=min_value,
        max_value=args.max_value,
        achievement_progress=achievement_progress,
        states=states,
        rnn_states=rnn_states,
    )

    stem = f"{args.exp_name}-s{args.train_seed:02}-e{args.ckpt_epoch:03}-{args.sweep_item}_curve"
    figure_path = os.path.join(args.output_dir, f"{stem}.png")
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    summary_path = os.path.join(args.output_dir, f"{stem}.json")
    save_figure(rows, images, memory_tasks, inventory_updates, args.sweep_item, figure_path)
    write_csv(csv_path, rows)

    summary = {
        "checkpoint": ckpt_path,
        "eval_seed": args.eval_seed,
        "episode_id": args.episode_id,
        "step_id": args.step_id,
        "memory_tasks": list(memory_tasks),
        "sweep_item": args.sweep_item,
        "min_value": min_value,
        "max_value": args.max_value,
        "inventory": inventory_updates,
        "background_material": args.background_material,
        "target_texture": args.target_texture,
        "target_delta": None if args.target_texture is None else list(parse_vec2(args.target_delta)),
        "rows": rows,
        "figure_path": figure_path,
        "csv_path": csv_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)

    base_env.close()


if __name__ == "__main__":
    main()
