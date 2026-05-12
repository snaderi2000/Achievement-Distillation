import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from achievement_distillation.constant import TASKS
from collect_value_map import load_model, observation_to_uint8_hwc, set_seed
from counterfactual_env_editor import (
    apply_material_edits,
    apply_spawn_object,
    parse_vec2,
    render_env,
    replay_to_step,
    score_observation,
)
from probe_material_value_preference import clear_visible_objects, make_visible_material


def active_memory_tasks(progress: th.Tensor) -> List[str]:
    values = progress.detach().cpu().view(-1).tolist()
    return [task for task, value in zip(TASKS, values) if value > 0.5]


def set_task(progress: th.Tensor, task: str, value: float) -> th.Tensor:
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {TASKS}")
    edited = progress.clone()
    edited[:, TASKS.index(task)] = float(value)
    return edited


def load_dataset_step(
    dataset_path: str,
    episode_id: int,
    step_id: int,
    device: th.device,
):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Rollout dataset not found: {dataset_path}")
    dataset = th.load(dataset_path, map_location="cpu")
    episode_ids = dataset["episode_ids"].cpu()
    step_ids = dataset["step_ids"].cpu()
    actions = dataset["actions"].cpu()
    progress_inputs = dataset.get("achievement_progress_inputs")

    mask = episode_ids == int(episode_id)
    episode_indices = th.nonzero(mask, as_tuple=False).view(-1)
    if episode_indices.numel() == 0:
        raise ValueError(f"Dataset has no episode_id={episode_id}.")
    episode_indices = episode_indices[th.argsort(step_ids[episode_indices])]

    target_matches = episode_indices[step_ids[episode_indices] == int(step_id)]
    if target_matches.numel() == 0:
        raise ValueError(f"Dataset has no episode={episode_id}, step={step_id}.")
    target_index = int(target_matches[0].item())

    progress = (
        progress_inputs[target_index].float().view(1, -1).to(device)
        if progress_inputs is not None
        else th.zeros(1, len(TASKS), device=device)
    )
    obs = observation_to_uint8_hwc(dataset["observations"][target_index])
    return dataset, obs, progress, target_index


def replay_dataset_actions_to_index(dataset, eval_seed: int, episode_id: int, target_index: int):
    from crafter.env import Env

    episode_ids = dataset["episode_ids"].cpu()
    step_ids = dataset["step_ids"].cpu()
    actions = dataset["actions"].cpu()
    target_step = int(step_ids[target_index].item())

    mask = episode_ids == int(episode_id)
    episode_indices = th.nonzero(mask, as_tuple=False).view(-1)
    episode_indices = episode_indices[th.argsort(step_ids[episode_indices])]

    env = Env(seed=eval_seed + int(episode_id))
    obs = env.reset()
    for idx_tensor in episode_indices:
        idx = int(idx_tensor.item())
        current_step = int(step_ids[idx].item())
        if idx == int(target_index):
            return env, obs
        obs, _, done, _info = env.step(int(actions[idx].item()))
        if done:
            raise ValueError(f"Episode ended before reaching dataset_idx={target_index}, step={target_step}.")

    env.close()
    raise ValueError(f"Could not replay dataset actions to dataset_idx={target_index}.")


def save_figure(base_obs, skeleton_obs, rows: List[Dict[str, object]], output_path: str):
    values = [float(row["value"]) for row in rows]
    labels = [str(row["condition"]).replace("_", "\n") for row in rows]

    fig = plt.figure(figsize=(13.5, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(base_obs)
    ax0.set_title("replayed step")
    ax0.axis("off")

    ax1.imshow(skeleton_obs)
    ax1.set_title("counterfactual\nskeleton in front")
    ax1.axis("off")

    bars = ax2.bar(labels, values, color=["#8da0cb", "#b3b3e6", "#fc8d62", "#66c2a5"])
    ax2.set_ylabel("V(obs, memory)")
    ax2.set_title("Skeleton value under memory edits")
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_debug_replay_figure(dataset_obs, replay_obs, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.8))
    axes[0].imshow(dataset_obs)
    axes[0].set_title("HTML/dataset obs")
    axes[0].axis("off")
    axes[1].imshow(replay_obs)
    axes[1].set_title("replayed env obs")
    axes[1].axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def insert_skeleton_pixels(obs: np.ndarray, delta: Tuple[int, int], eval_seed: int) -> np.ndarray:
    """Insert a skeleton sprite into the exact saved observation.

    This avoids simulator-state replay. It renders a clean grass scene with and without
    a skeleton, extracts only the changed sprite pixels, and pastes those pixels into
    the saved HTML/dataset observation.
    """
    from crafter.env import Env

    env = Env(seed=eval_seed)
    env.reset()
    clear_visible_objects(env)
    make_visible_material(env, "grass", keep_player_tile=True)
    clean_obs = render_env(env)
    apply_spawn_object(env, [(delta[0], delta[1], "skeleton")], [])
    skeleton_template_obs = render_env(env)

    grid = tuple(int(x) for x in env._local_view._grid)
    center_x = grid[0] // 2
    center_y = grid[1] // 2
    cell_x = center_x + int(delta[0])
    cell_y = center_y + int(delta[1])
    if not (0 <= cell_x < grid[0] and 0 <= cell_y < grid[1]):
        env.close()
        raise ValueError(f"Skeleton delta {delta} maps outside visible grid {grid}.")

    cell_w = obs.shape[1] // grid[0]
    # Crafter's 64x64 render has a 15-pixel HUD, leaving 49 world pixels = 7 rows * 7 px.
    cell_h = (obs.shape[0] - 15) // grid[1]
    x0 = cell_x * cell_w
    y0 = cell_y * cell_h

    clean_cell = clean_obs[y0 : y0 + cell_h, x0 : x0 + cell_w]
    skeleton_cell = skeleton_template_obs[y0 : y0 + cell_h, x0 : x0 + cell_w]
    sprite_mask = (skeleton_cell != clean_cell).any(axis=-1, keepdims=True)

    edited = obs.copy()
    original_cell = edited[y0 : y0 + cell_h, x0 : x0 + cell_w]
    edited[y0 : y0 + cell_h, x0 : x0 + cell_w] = np.where(sprite_mask, skeleton_cell, original_cell)
    env.close()
    return edited


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replay a memory-model state, spawn a skeleton in front, then compare value with actual memory "
            "versus memory with defeat_skeleton removed."
        )
    )
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--step_id", type=int, required=True)
    parser.add_argument("--skeleton_delta", type=str, default="0,1")
    parser.add_argument(
        "--memory_task",
        type=str,
        default="defeat_skeleton",
        help="Achievement-memory bit to compare as absent vs present.",
    )
    parser.add_argument(
        "--remove_memory_task",
        type=str,
        default=None,
        help="Deprecated alias for --memory_task.",
    )
    parser.add_argument("--target_material", type=str, default="grass")
    parser.add_argument(
        "--pixel_insert_skeleton",
        action="store_true",
        help="Insert skeleton directly into the saved observation pixels instead of editing replayed simulator state.",
    )
    parser.add_argument(
        "--rollout_dataset_path",
        type=str,
        default=None,
        help="Optional saved collect_value_map dataset. If set, replay recorded actions so the state matches the HTML.",
    )
    parser.add_argument("--output_dir", type=str, default="skeleton_memory_counterfactual")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    model, config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)
    if args.rollout_dataset_path:
        dataset, dataset_obs, actual_memory, dataset_idx = load_dataset_step(
            args.rollout_dataset_path,
            args.episode_id,
            args.step_id,
            device,
        )
        replay_env, replay_obs = replay_dataset_actions_to_index(
            dataset,
            args.eval_seed,
            args.episode_id,
            dataset_idx,
        )
        replay_states = None
        replay_rnn_states = None
        base_obs = dataset_obs.copy()
    else:
        replay = replay_to_step(
            model=model,
            config=config,
            eval_seed=args.eval_seed,
            target_episode=args.episode_id,
            target_step=args.step_id,
            device=device,
        )
        replay_env = replay.env
        replay_obs = replay.obs
        actual_memory = replay.achievement_progress
        replay_states = replay.states
        replay_rnn_states = replay.rnn_states
        dataset_idx = None
        base_obs = replay_obs.copy()

    memory_task = args.remove_memory_task or args.memory_task
    memory_without_task = set_task(actual_memory, memory_task, 0.0)
    memory_with_task = set_task(actual_memory, memory_task, 1.0)

    base_value_without_memory = score_observation(
        model,
        base_obs,
        device,
        replay_states,
        replay_rnn_states,
        memory_without_task,
    )["value"]
    base_value_with_memory = score_observation(
        model,
        base_obs,
        device,
        replay_states,
        replay_rnn_states,
        memory_with_task,
    )["value"]

    dx, dy = parse_vec2(args.skeleton_delta)
    edits_log: List[str] = []
    if args.pixel_insert_skeleton:
        skeleton_obs = insert_skeleton_pixels(base_obs, (dx, dy), args.eval_seed)
        edits_log.append(f"pixel_insert_skeleton@({dx},{dy})")
    else:
        apply_material_edits(replay_env, [(dx, dy, args.target_material)], edits_log)
        apply_spawn_object(replay_env, [(dx, dy, "skeleton")], edits_log)
        skeleton_obs = render_env(replay_env)

    skeleton_value_without_memory = score_observation(
        model,
        skeleton_obs,
        device,
        replay_states,
        replay_rnn_states,
        memory_without_task,
    )["value"]
    skeleton_value_with_memory = score_observation(
        model,
        skeleton_obs,
        device,
        replay_states,
        replay_rnn_states,
        memory_with_task,
    )["value"]

    rows = [
        {
            "condition": f"replayed_obs_{memory_task}=0",
            "value": base_value_without_memory,
            "has_skeleton_counterfactual": False,
            "memory_task": memory_task,
            "memory_task_value": 0,
        },
        {
            "condition": f"replayed_obs_{memory_task}=1",
            "value": base_value_with_memory,
            "has_skeleton_counterfactual": False,
            "memory_task": memory_task,
            "memory_task_value": 1,
        },
        {
            "condition": f"skeleton_{memory_task}=0",
            "value": skeleton_value_without_memory,
            "has_skeleton_counterfactual": True,
            "memory_task": memory_task,
            "memory_task_value": 0,
        },
        {
            "condition": f"skeleton_{memory_task}=1",
            "value": skeleton_value_with_memory,
            "has_skeleton_counterfactual": True,
            "memory_task": memory_task,
            "memory_task_value": 1,
        },
    ]

    stem = f"{args.exp_name}-s{args.train_seed:02}-ep{args.episode_id:03d}-step{args.step_id:04d}-skeleton_memory"
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    json_path = os.path.join(args.output_dir, f"{stem}.json")
    figure_path = os.path.join(args.output_dir, f"{stem}.png")
    debug_replay_path = os.path.join(args.output_dir, f"{stem}-dataset-vs-replay.png")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "value", "has_skeleton_counterfactual", "memory_task", "memory_task_value"],
        )
        writer.writeheader()
        writer.writerows(rows)
    save_figure(base_obs, skeleton_obs, rows, figure_path)
    if args.rollout_dataset_path:
        save_debug_replay_figure(base_obs, replay_obs, debug_replay_path)

    summary = {
        "checkpoint": ckpt_path,
        "eval_seed": args.eval_seed,
        "episode_id": args.episode_id,
        "step_id": args.step_id,
        "dataset_idx": dataset_idx,
        "rollout_dataset_path": args.rollout_dataset_path,
        "skeleton_delta": [dx, dy],
        "edits": edits_log,
        "actual_memory_tasks": active_memory_tasks(actual_memory),
        "memory_without_task_tasks": active_memory_tasks(memory_without_task),
        "memory_with_task_tasks": active_memory_tasks(memory_with_task),
        "memory_task": memory_task,
        "values": rows,
        "delta_skeleton_vs_replay_memory_task_0": skeleton_value_without_memory - base_value_without_memory,
        "delta_skeleton_vs_replay_memory_task_1": skeleton_value_with_memory - base_value_with_memory,
        "delta_memory_task_on_replayed_obs": base_value_with_memory - base_value_without_memory,
        "delta_memory_task_on_skeleton_obs": skeleton_value_with_memory - skeleton_value_without_memory,
        "csv_path": csv_path,
        "figure_path": figure_path,
        "debug_replay_path": debug_replay_path if args.rollout_dataset_path else None,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    replay_env.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
