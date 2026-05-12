import argparse
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import torch as th

from achievement_distillation.constant import TASKS
from collect_value_map import load_model, set_seed
from counterfactual_env_editor import (
    apply_material_edits,
    apply_spawn_object,
    parse_vec2,
    render_env,
    replay_to_step,
    score_observation,
)


def active_memory_tasks(progress: th.Tensor) -> List[str]:
    values = progress.detach().cpu().view(-1).tolist()
    return [task for task, value in zip(TASKS, values) if value > 0.5]


def remove_task(progress: th.Tensor, task: str) -> th.Tensor:
    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {TASKS}")
    edited = progress.clone()
    edited[:, TASKS.index(task)] = 0.0
    return edited


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

    bars = ax2.bar(labels, values, color=["#8da0cb", "#fc8d62", "#66c2a5"])
    ax2.set_ylabel("V(obs, memory)")
    ax2.set_title("Skeleton value under memory edits")
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom")

    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--remove_memory_task", type=str, default="defeat_skeleton")
    parser.add_argument("--target_material", type=str, default="grass")
    parser.add_argument("--output_dir", type=str, default="skeleton_memory_counterfactual")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    model, config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)
    replay = replay_to_step(
        model=model,
        config=config,
        eval_seed=args.eval_seed,
        target_episode=args.episode_id,
        target_step=args.step_id,
        device=device,
    )

    base_obs = replay.obs.copy()
    actual_memory = replay.achievement_progress
    edited_memory = remove_task(actual_memory, args.remove_memory_task)

    base_value_actual_memory = score_observation(
        model,
        base_obs,
        device,
        replay.states,
        replay.rnn_states,
        actual_memory,
    )["value"]

    dx, dy = parse_vec2(args.skeleton_delta)
    edits_log: List[str] = []
    apply_material_edits(replay.env, [(dx, dy, args.target_material)], edits_log)
    apply_spawn_object(replay.env, [(dx, dy, "skeleton")], edits_log)
    skeleton_obs = render_env(replay.env)

    skeleton_value_actual_memory = score_observation(
        model,
        skeleton_obs,
        device,
        replay.states,
        replay.rnn_states,
        actual_memory,
    )["value"]
    skeleton_value_removed_memory = score_observation(
        model,
        skeleton_obs,
        device,
        replay.states,
        replay.rnn_states,
        edited_memory,
    )["value"]

    rows = [
        {
            "condition": "replayed_obs_actual_memory",
            "value": base_value_actual_memory,
            "has_skeleton_counterfactual": False,
            "memory_task_removed": "",
        },
        {
            "condition": "skeleton_actual_memory",
            "value": skeleton_value_actual_memory,
            "has_skeleton_counterfactual": True,
            "memory_task_removed": "",
        },
        {
            "condition": f"skeleton_without_{args.remove_memory_task}_memory",
            "value": skeleton_value_removed_memory,
            "has_skeleton_counterfactual": True,
            "memory_task_removed": args.remove_memory_task,
        },
    ]

    stem = f"{args.exp_name}-s{args.train_seed:02}-ep{args.episode_id:03d}-step{args.step_id:04d}-skeleton_memory"
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    json_path = os.path.join(args.output_dir, f"{stem}.json")
    figure_path = os.path.join(args.output_dir, f"{stem}.png")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["condition", "value", "has_skeleton_counterfactual", "memory_task_removed"],
        )
        writer.writeheader()
        writer.writerows(rows)
    save_figure(base_obs, skeleton_obs, rows, figure_path)

    summary = {
        "checkpoint": ckpt_path,
        "eval_seed": args.eval_seed,
        "episode_id": args.episode_id,
        "step_id": args.step_id,
        "skeleton_delta": [dx, dy],
        "edits": edits_log,
        "actual_memory_tasks": active_memory_tasks(actual_memory),
        "edited_memory_tasks": active_memory_tasks(edited_memory),
        "removed_memory_task": args.remove_memory_task,
        "values": rows,
        "delta_skeleton_vs_replay_actual_memory": skeleton_value_actual_memory - base_value_actual_memory,
        "delta_removed_memory_on_skeleton_obs": skeleton_value_removed_memory - skeleton_value_actual_memory,
        "csv_path": csv_path,
        "figure_path": figure_path,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    replay.env.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
