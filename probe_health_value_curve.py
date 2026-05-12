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


def parse_epochs(text: str) -> Tuple[int, ...]:
    epochs = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not epochs:
        raise ValueError("--ckpt_epochs did not contain any epochs.")
    return epochs


def parse_compare_values(text: str) -> Tuple[int, int]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if len(values) != 2:
        raise ValueError("--compare_values must contain exactly two comma-separated values, e.g. 1,9.")
    if values[0] == values[1]:
        raise ValueError("--compare_values must contain two distinct values.")
    return values


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


def render_inventory_value(base_env, sweep_item: str, value: int) -> np.ndarray:
    env = copy.deepcopy(base_env)
    set_inventory_item(env, sweep_item, value)
    obs = render_env(env)
    env.close()
    return obs


def score_compare_values_for_checkpoint(
    exp_name: str,
    timestamp: str,
    train_seed: int,
    ckpt_epoch: int,
    device: th.device,
    sweep_item: str,
    low_item_value: int,
    high_item_value: int,
    low_obs: np.ndarray,
    high_obs: np.ndarray,
    achievement_progress: th.Tensor | None,
    states: th.Tensor | None,
    rnn_states: th.Tensor | None,
):
    model, _config, ckpt_path = load_model(
        exp_name,
        timestamp,
        train_seed,
        ckpt_epoch,
        device,
    )
    low_scores = score_observation(
        model,
        low_obs,
        device,
        states=states,
        rnn_states=rnn_states,
        achievement_progress=achievement_progress,
    )
    high_scores = score_observation(
        model,
        high_obs,
        device,
        states=states,
        rnn_states=rnn_states,
        achievement_progress=achievement_progress,
    )
    row = {
        "ckpt_epoch": ckpt_epoch,
        "low_item_value": low_item_value,
        "high_item_value": high_item_value,
        "low_value": low_scores["value"],
        "high_value": high_scores["value"],
        "delta_value": high_scores["value"] - low_scores["value"],
        "checkpoint": ckpt_path,
    }
    if "health_value" in low_scores and "health_value" in high_scores:
        row["low_health_value"] = low_scores["health_value"]
        row["high_health_value"] = high_scores["health_value"]
        row["delta_health_value"] = high_scores["health_value"] - low_scores["health_value"]
    if "achievement_value" in low_scores and "achievement_value" in high_scores:
        row["low_achievement_value"] = low_scores["achievement_value"]
        row["high_achievement_value"] = high_scores["achievement_value"]
        row["delta_achievement_value"] = high_scores["achievement_value"] - low_scores["achievement_value"]
    return row


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


def save_compare_observation_figure(
    low_obs: np.ndarray,
    high_obs: np.ndarray,
    sweep_item: str,
    low_item_value: int,
    high_item_value: int,
    memory_tasks: Tuple[str, ...],
    inventory_updates: Dict[str, int],
    output_path: str,
):
    shown_inventory = [
        f"{name}={value}"
        for name, value in inventory_updates.items()
        if value > 0 and name != sweep_item
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0))
    panels = [
        (low_obs, f"{sweep_item}={low_item_value} observation"),
        (high_obs, f"{sweep_item}={high_item_value} observation"),
    ]
    for ax, (obs, title) in zip(axes[:2], panels):
        ax.imshow(obs)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    axes[2].axis("off")
    axes[2].text(
        0.0,
        0.95,
        "Fixed observations fed to every checkpoint\n\n"
        "memory:\n"
        + ("  " + "\n  ".join(memory_tasks) if memory_tasks else "  empty")
        + "\n\nfixed inventory:\n  "
        + (", ".join(shown_inventory) if shown_inventory else "empty")
        + f"\n\ncomparison:\n  {sweep_item}={low_item_value} vs {sweep_item}={high_item_value}",
        transform=axes[2].transAxes,
        fontsize=10,
        va="top",
        family="monospace",
    )

    fig.suptitle(f"Fixed {sweep_item} counterfactual observations", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_compare_sweep_figure(
    rows,
    sweep_item: str,
    low_item_value: int,
    high_item_value: int,
    output_path: str,
):
    epochs = [row["ckpt_epoch"] for row in rows]
    low_values = [row["low_value"] for row in rows]
    high_values = [row["high_value"] for row in rows]
    deltas = [row["delta_value"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.9))
    axes[0].axhline(0.0, color="black", linewidth=1, alpha=0.5)
    axes[0].plot(epochs, deltas, marker="o", color="#444444")
    axes[0].fill_between(epochs, 0.0, deltas, where=np.array(deltas) >= 0.0, color="#1b9e77", alpha=0.18)
    axes[0].fill_between(epochs, 0.0, deltas, where=np.array(deltas) < 0.0, color="#d95f02", alpha=0.18)
    axes[0].set_title(f"{sweep_item} value preference")
    axes[0].set_xlabel("checkpoint epoch")
    axes[0].set_ylabel(f"V({sweep_item}={high_item_value}) - V({sweep_item}={low_item_value})")

    axes[1].plot(epochs, low_values, marker="o", label=f"{sweep_item}={low_item_value}", color="#d95f02")
    axes[1].plot(epochs, high_values, marker="o", label=f"{sweep_item}={high_item_value}", color="#1b9e77")
    axes[1].set_title("Predicted values")
    axes[1].set_xlabel("checkpoint epoch")
    axes[1].set_ylabel("V")
    axes[1].legend(frameon=False)

    fig.suptitle(f"{sweep_item}={low_item_value} vs {sweep_item}={high_item_value} across checkpoints")
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
    parser.add_argument(
        "--ckpt_epochs",
        type=str,
        default=None,
        help="Optional comma-separated checkpoint epochs to sweep, e.g. 50,100,150.",
    )
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
    parser.add_argument(
        "--compare_values",
        type=str,
        default=None,
        help="Optional pair of item values to compare across checkpoints, e.g. 1,9.",
    )
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
    replay_ckpt_epoch = parse_epochs(args.ckpt_epochs)[0] if args.ckpt_epochs else args.ckpt_epoch
    model, config, ckpt_path = load_model(
        args.exp_name,
        args.timestamp,
        args.train_seed,
        replay_ckpt_epoch,
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

    if args.compare_values is not None:
        if args.ckpt_epochs is None:
            raise ValueError("--compare_values currently requires --ckpt_epochs.")
        low_item_value, high_item_value = parse_compare_values(args.compare_values)
        low_obs = render_inventory_value(base_env, args.sweep_item, low_item_value)
        high_obs = render_inventory_value(base_env, args.sweep_item, high_item_value)
        epochs = parse_epochs(args.ckpt_epochs)
        rows = [
            score_compare_values_for_checkpoint(
                args.exp_name,
                args.timestamp,
                args.train_seed,
                epoch,
                device,
                args.sweep_item,
                low_item_value,
                high_item_value,
                low_obs,
                high_obs,
                achievement_progress,
                states,
                rnn_states,
            )
            for epoch in epochs
        ]

        stem = (
            f"{args.exp_name}-s{args.train_seed:02}-"
            f"{args.sweep_item}_{low_item_value}_vs_{high_item_value}-sweep"
        )
        figure_path = os.path.join(args.output_dir, f"{stem}.png")
        observation_figure_path = os.path.join(args.output_dir, f"{stem}-observations.png")
        csv_path = os.path.join(args.output_dir, f"{stem}.csv")
        summary_path = os.path.join(args.output_dir, f"{stem}.json")
        save_compare_sweep_figure(
            rows,
            args.sweep_item,
            low_item_value,
            high_item_value,
            figure_path,
        )
        save_compare_observation_figure(
            low_obs,
            high_obs,
            args.sweep_item,
            low_item_value,
            high_item_value,
            memory_tasks,
            inventory_updates,
            observation_figure_path,
        )
        write_csv(csv_path, rows)

        summary = {
            "checkpoint_for_scene_setup": ckpt_path,
            "eval_seed": args.eval_seed,
            "episode_id": args.episode_id,
            "step_id": args.step_id,
            "memory_tasks": list(memory_tasks),
            "sweep_item": args.sweep_item,
            "compare_values": [low_item_value, high_item_value],
            "inventory": inventory_updates,
            "background_material": args.background_material,
            "target_texture": args.target_texture,
            "target_delta": None if args.target_texture is None else list(parse_vec2(args.target_delta)),
            "rows": rows,
            "figure_path": figure_path,
            "observation_figure_path": observation_figure_path,
            "csv_path": csv_path,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
        base_env.close()
        return

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
