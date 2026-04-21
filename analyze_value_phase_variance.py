import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from collect_value_map import collect_value_dataset, load_model, set_seed


PHASE_NAMES = ["early", "early_mid", "late_mid", "late"]


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


def summarize_phase_variance(dataset: Dict[str, th.Tensor]) -> Dict[str, Dict]:
    values = dataset["values"].cpu().numpy()
    episode_ids = dataset["episode_ids"].cpu().numpy()
    step_ids = dataset["step_ids"].cpu().numpy()

    pooled_values = {phase: [] for phase in PHASE_NAMES}
    per_episode_values = {phase: [] for phase in PHASE_NAMES}

    for episode_id in np.unique(episode_ids):
        idx = np.where(episode_ids == episode_id)[0]
        order = np.argsort(step_ids[idx])
        ordered_idx = idx[order]
        episode_values = values[ordered_idx]
        episode_len = len(episode_values)

        grouped = defaultdict(list)
        for local_step, value in enumerate(episode_values):
            phase = assign_phase(local_step, episode_len)
            pooled_values[phase].append(float(value))
            grouped[phase].append(float(value))

        for phase in PHASE_NAMES:
            if len(grouped[phase]) >= 2:
                per_episode_values[phase].append(float(np.var(grouped[phase], ddof=0)))

    summary = {}
    for phase in PHASE_NAMES:
        phase_values = np.asarray(pooled_values[phase], dtype=np.float64)
        phase_episode_vars = np.asarray(per_episode_values[phase], dtype=np.float64)
        summary[phase] = {
            "num_states": int(len(phase_values)),
            "pooled_mean": float(phase_values.mean()) if len(phase_values) else None,
            "pooled_variance": float(np.var(phase_values, ddof=0)) if len(phase_values) else None,
            "mean_within_episode_variance": float(phase_episode_vars.mean()) if len(phase_episode_vars) else None,
            "median_within_episode_variance": float(np.median(phase_episode_vars)) if len(phase_episode_vars) else None,
            "num_episodes_with_phase_variance": int(len(phase_episode_vars)),
        }
    return summary


def plot_phase_histograms(dataset: Dict[str, th.Tensor], output_path: str):
    values = dataset["values"].cpu().numpy()
    episode_ids = dataset["episode_ids"].cpu().numpy()
    step_ids = dataset["step_ids"].cpu().numpy()

    pooled_values = {phase: [] for phase in PHASE_NAMES}
    for episode_id in np.unique(episode_ids):
        idx = np.where(episode_ids == episode_id)[0]
        order = np.argsort(step_ids[idx])
        ordered_idx = idx[order]
        episode_values = values[ordered_idx]
        episode_len = len(episode_values)
        for local_step, value in enumerate(episode_values):
            pooled_values[assign_phase(local_step, episode_len)].append(float(value))

    all_values = np.concatenate([np.asarray(pooled_values[phase], dtype=np.float64) for phase in PHASE_NAMES if pooled_values[phase]])
    bins = np.histogram_bin_edges(all_values, bins=30) if len(all_values) else 30

    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    axes = axes.flatten()
    for ax, phase in zip(axes, PHASE_NAMES):
        phase_values = np.asarray(pooled_values[phase], dtype=np.float64)
        ax.hist(phase_values, bins=bins, color="#4C78A8", alpha=0.85, edgecolor="white")
        ax.set_title(f"{phase} (n={len(phase_values)})")
        ax.set_xlabel("Predicted value")
        ax.set_ylabel("Count")
        if len(phase_values):
            ax.axvline(np.mean(phase_values), color="#F58518", linestyle="--", linewidth=1.5)
            ax.text(
                0.02,
                0.95,
                f"var={np.var(phase_values):.4f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )

    fig.suptitle("Value distributions by within-episode phase")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze value variance across early-to-late episode phases.")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, default=0)
    parser.add_argument("--ckpt_epoch", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default="value_phase_variance")
    parser.add_argument("--save_dataset_path", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.eval_seed)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, _, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)
    print(f"Loaded checkpoint: {ckpt_path}")

    dataset, _ = collect_value_dataset(
        model=model,
        device=device,
        num_episodes=args.num_episodes,
        eval_seed=args.eval_seed,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    hist_path = os.path.join(args.output_dir, f"{args.exp_name}_phase_hist.png")
    summary_path = os.path.join(args.output_dir, f"{args.exp_name}_phase_summary.json")

    summary = summarize_phase_variance(dataset)
    plot_phase_histograms(dataset, hist_path)

    payload = {
        "checkpoint": ckpt_path,
        "num_episodes": args.num_episodes,
        "eval_seed": args.eval_seed,
        "summary": summary,
        "histogram_path": hist_path,
    }
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

    if args.save_dataset_path:
        th.save(dataset, args.save_dataset_path)
        print(f"Saved dataset to {args.save_dataset_path}")

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
