import argparse
import csv
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import torch as th

from collect_value_map import load_model, set_seed
from counterfactual_env_editor import (
    apply_inventory_edits,
    replay_to_step,
    render_env,
    score_observation,
)
from achievement_distillation.constant import TASKS


def active_memory_tasks(progress: th.Tensor):
    values = progress.detach().cpu().view(-1).tolist()
    return [task for task, value in zip(TASKS, values) if value > 0.5]


def save_figure(base_obs, edited_obs, rows, memory_tasks, output_path: str):
    value_lookup: Dict[str, float] = {row["condition"]: row["value"] for row in rows}

    fig = plt.figure(figsize=(13.5, 4.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.25])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(base_obs)
    ax0.set_title("baseline\nwood_pickaxe as replayed")
    ax0.axis("off")

    ax1.imshow(edited_obs)
    ax1.set_title("counterfactual\nwood_pickaxe = 0")
    ax1.axis("off")

    labels = [
        "memory base",
        "memory no pickaxe",
        "ppo base",
        "ppo no pickaxe",
    ]
    values = [
        value_lookup["memory_base"],
        value_lookup["memory_no_pickaxe"],
        value_lookup["ppo_base"],
        value_lookup["ppo_no_pickaxe"],
    ]
    colors = ["#4c78a8", "#9ecae9", "#f58518", "#ffbf79"]
    ax2.bar(labels, values, color=colors)
    ax2.set_ylabel("V(obs)")
    ax2.set_title("value comparison")
    ax2.tick_params(axis="x", rotation=25)
    for idx, value in enumerate(values):
        ax2.text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    mem_text = ", ".join(memory_tasks) if memory_tasks else "empty"
    fig.suptitle(f"Step counterfactual: remove wood pickaxe\nmemory used for memory model: {mem_text}", y=1.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replay one explicit-memory episode step, remove wood_pickaxe from inventory, "
            "and score both observations with the memory model and a regular PPO model."
        )
    )
    parser.add_argument("--memory_exp_name", type=str, required=True)
    parser.add_argument("--memory_timestamp", type=str, default="debug")
    parser.add_argument("--memory_train_seed", type=int, required=True)
    parser.add_argument("--memory_ckpt_epoch", type=int, default=250)
    parser.add_argument("--ppo_exp_name", type=str, default="ppo")
    parser.add_argument("--ppo_timestamp", type=str, default="debug")
    parser.add_argument("--ppo_train_seed", type=int, required=True)
    parser.add_argument("--ppo_ckpt_epoch", type=int, default=250)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--step_id", type=int, required=True)
    parser.add_argument("--removed_value", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="pickaxe_memory_vs_ppo_counterfactual")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    memory_model, memory_config, memory_ckpt_path = load_model(
        args.memory_exp_name,
        args.memory_timestamp,
        args.memory_train_seed,
        args.memory_ckpt_epoch,
        device,
    )
    ppo_model, _, ppo_ckpt_path = load_model(
        args.ppo_exp_name,
        args.ppo_timestamp,
        args.ppo_train_seed,
        args.ppo_ckpt_epoch,
        device,
    )

    replay = replay_to_step(
        model=memory_model,
        config=memory_config,
        eval_seed=args.eval_seed,
        target_episode=args.episode_id,
        target_step=args.step_id,
        device=device,
    )

    base_obs = replay.obs.copy()
    achievement_progress = replay.achievement_progress
    memory_tasks = active_memory_tasks(achievement_progress)

    memory_base = score_observation(
        memory_model,
        base_obs,
        device,
        replay.states,
        replay.rnn_states,
        achievement_progress,
    )["value"]
    ppo_base = score_observation(ppo_model, base_obs, device)["value"]

    edits_log = []
    apply_inventory_edits(replay.env, {"wood_pickaxe": args.removed_value}, edits_log)
    edited_obs = render_env(replay.env)

    memory_no_pickaxe = score_observation(
        memory_model,
        edited_obs,
        device,
        replay.states,
        replay.rnn_states,
        achievement_progress,
    )["value"]
    ppo_no_pickaxe = score_observation(ppo_model, edited_obs, device)["value"]

    rows = [
        {"condition": "memory_base", "model": "memory", "wood_pickaxe_removed": False, "value": memory_base},
        {
            "condition": "memory_no_pickaxe",
            "model": "memory",
            "wood_pickaxe_removed": True,
            "value": memory_no_pickaxe,
        },
        {"condition": "ppo_base", "model": "ppo", "wood_pickaxe_removed": False, "value": ppo_base},
        {
            "condition": "ppo_no_pickaxe",
            "model": "ppo",
            "wood_pickaxe_removed": True,
            "value": ppo_no_pickaxe,
        },
    ]
    summary = {
        "memory_checkpoint": memory_ckpt_path,
        "ppo_checkpoint": ppo_ckpt_path,
        "eval_seed": args.eval_seed,
        "episode_id": args.episode_id,
        "step_id": args.step_id,
        "memory_tasks": memory_tasks,
        "edits": edits_log,
        "values": rows,
        "memory_delta_remove_pickaxe": memory_no_pickaxe - memory_base,
        "ppo_delta_remove_pickaxe": ppo_no_pickaxe - ppo_base,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"ep{args.episode_id:03d}-step{args.step_id:04d}-remove-wood-pickaxe"
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    json_path = os.path.join(args.output_dir, f"{stem}.json")
    figure_path = os.path.join(args.output_dir, f"{stem}.png")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "model", "wood_pickaxe_removed", "value"])
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w") as f:
        json.dump({**summary, "csv_path": csv_path, "figure_path": figure_path}, f, indent=2)
    save_figure(base_obs, edited_obs, rows, memory_tasks, figure_path)

    replay.env.close()
    print(json.dumps({**summary, "csv_path": csv_path, "figure_path": figure_path}, indent=2))


if __name__ == "__main__":
    main()
