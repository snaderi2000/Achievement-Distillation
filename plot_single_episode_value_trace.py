import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from achievement_distillation.constant import TASKS
from collect_value_map import load_model, set_seed
from counterfactual_env_editor import obs_to_tensor


PHASE_NAMES = ["early", "early_mid", "late_mid", "late"]


def active_memory_tasks(progress: th.Tensor) -> List[str]:
    values = progress.detach().cpu().view(-1).tolist()
    return [task for task, value in zip(TASKS, values) if value > 0.5]


def assign_phase(step_idx: int, episode_len: int) -> str:
    if episode_len <= 1:
        return "early"
    frac = step_idx / max(episode_len - 1, 1)
    if frac < 0.25:
        return "early"
    if frac < 0.50:
        return "early_mid"
    if frac < 0.75:
        return "late_mid"
    return "late"


def collect_episode_trace(model, config: Dict, device: th.device, eval_seed: int, max_steps: int):
    from crafter.env import Env

    env = Env(seed=eval_seed)
    obs = env.reset()
    hidsize = int(config.get("model_kwargs", {}).get("hidsize", getattr(model, "hidsize", 0)))
    states = th.zeros(1, hidsize, device=device) if hidsize > 0 else None
    rnn_hidsize = config.get("model_kwargs", {}).get("rnn_hidsize")
    rnn_states = th.zeros(1, int(rnn_hidsize), device=device) if rnn_hidsize is not None else None
    achievement_progress = th.zeros(1, len(TASKS), device=device)
    rows = []

    for step_id in range(max_steps):
        obs_tensor = obs_to_tensor(obs, device)
        act_kwargs = {}
        if states is not None:
            act_kwargs["states"] = states
        if rnn_states is not None:
            act_kwargs["rnn_states"] = rnn_states
        if getattr(model, "use_achievement_progress_input", False) or hasattr(model, "achievement_progress_dim"):
            act_kwargs["achievement_progress"] = achievement_progress

        with th.no_grad():
            outputs = model.act(obs_tensor, **act_kwargs)
        value = float(outputs["vpreds"].item())
        action = int(outputs["actions"].item())
        memory_before = active_memory_tasks(achievement_progress)

        next_obs, reward, done, info = env.step(action)
        achievements = info.get("achievements")
        new_achievements = []
        if achievements is not None:
            old_values = achievement_progress.detach().cpu().view(-1).numpy()
            new_values = np.array([1.0 if achievements.get(task, 0) > 0 else 0.0 for task in TASKS], dtype=np.float32)
            new_achievements = [task for task, old, new in zip(TASKS, old_values, new_values) if old < 0.5 and new > 0.5]
            achievement_progress = th.tensor([new_values], dtype=th.float32, device=device)

        rows.append(
            {
                "step": step_id,
                "value": value,
                "action": action,
                "reward": float(reward),
                "done": bool(done),
                "memory_tasks": ";".join(memory_before),
                "new_achievements": ";".join(new_achievements),
            }
        )

        if "next_states" in outputs and states is not None:
            states = outputs["next_states"]
        if "next_rnn_states" in outputs and rnn_states is not None:
            rnn_states = outputs["next_rnn_states"]

        obs = next_obs
        if done:
            break

    env.close()
    return rows


def add_phase_and_rolling_stats(rows, rolling_window: int):
    values = np.array([float(row["value"]) for row in rows], dtype=np.float64)
    episode_len = len(rows)
    for idx, row in enumerate(rows):
        row["phase"] = assign_phase(idx, episode_len)
        start = max(0, idx - rolling_window + 1)
        window = values[start : idx + 1]
        row["rolling_variance"] = float(np.var(window, ddof=0)) if len(window) else 0.0
    phase_summary = {}
    for phase in PHASE_NAMES:
        vals = np.array([float(row["value"]) for row in rows if row["phase"] == phase], dtype=np.float64)
        phase_summary[phase] = {
            "num_steps": int(len(vals)),
            "mean": float(vals.mean()) if len(vals) else None,
            "variance": float(np.var(vals, ddof=0)) if len(vals) else None,
            "min": float(vals.min()) if len(vals) else None,
            "max": float(vals.max()) if len(vals) else None,
        }
    return phase_summary


def write_csv(path: str, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_plot(path: str, rows, phase_summary):
    steps = np.array([int(row["step"]) for row in rows])
    values = np.array([float(row["value"]) for row in rows])
    rolling_var = np.array([float(row["rolling_variance"]) for row in rows])

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 7.0), sharex=True)
    axes[0].plot(steps, values, color="#4c78a8", linewidth=2)
    axes[0].scatter(
        [int(row["step"]) for row in rows if row["new_achievements"]],
        [float(row["value"]) for row in rows if row["new_achievements"]],
        color="#f58518",
        zorder=3,
        label="new achievement",
    )
    axes[0].set_ylabel("V(obs)")
    axes[0].set_title("Single episode value trace")
    axes[0].legend(frameon=False)

    phase_colors = {
        "early": "#e8f1ff",
        "early_mid": "#eef8e8",
        "late_mid": "#fff3dc",
        "late": "#fde8e8",
    }
    n = max(len(rows), 1)
    for phase_idx, phase in enumerate(PHASE_NAMES):
        start = int(round(phase_idx * n / 4))
        end = int(round((phase_idx + 1) * n / 4)) - 1
        if start <= end:
            for ax in axes:
                ax.axvspan(start, end, color=phase_colors[phase], alpha=0.5)
            axes[0].text(
                (start + end) / 2,
                axes[0].get_ylim()[1],
                f"{phase}\nvar={phase_summary[phase]['variance']:.3f}" if phase_summary[phase]["variance"] is not None else phase,
                ha="center",
                va="top",
                fontsize=9,
            )

    axes[1].plot(steps, rolling_var, color="#54a24b", linewidth=2)
    axes[1].set_ylabel("rolling value variance")
    axes[1].set_xlabel("episode step")
    axes[1].set_title("Rolling variance over time")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot value over time for one sampled Crafter episode.")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--rolling_window", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="single_episode_value_trace")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)

    rows = collect_episode_trace(model, config, device, args.eval_seed, args.max_steps)
    phase_summary = add_phase_and_rolling_stats(rows, args.rolling_window)

    stem = f"{args.exp_name}-s{args.train_seed:02}-e{args.ckpt_epoch:03}-eval{args.eval_seed}"
    csv_path = os.path.join(args.output_dir, f"{stem}-trace.csv")
    plot_path = os.path.join(args.output_dir, f"{stem}-trace.png")
    summary_path = os.path.join(args.output_dir, f"{stem}-summary.json")
    write_csv(csv_path, rows)
    save_plot(plot_path, rows, phase_summary)

    payload = {
        "checkpoint": ckpt_path,
        "eval_seed": args.eval_seed,
        "num_steps": len(rows),
        "rolling_window": args.rolling_window,
        "phase_summary": phase_summary,
        "csv_path": csv_path,
        "plot_path": plot_path,
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
