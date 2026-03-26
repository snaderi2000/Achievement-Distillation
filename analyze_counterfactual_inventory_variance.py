import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch as th

from collect_value_map import load_model, set_seed
from probe_counterfactual_inventory import evaluate_value, swap_inventory_rows


def select_evenly_spaced_indices(num_points: int, target_count: int) -> np.ndarray:
    if target_count <= 0:
        raise ValueError("--num_steps must be positive.")
    if num_points == 0:
        raise ValueError("No points available to sample from.")
    if num_points <= target_count:
        return np.arange(num_points, dtype=np.int64)
    return np.unique(np.linspace(0, num_points - 1, num=target_count, dtype=np.int64))


def get_episode_indices(dataset: Dict[str, th.Tensor], episode_id: int) -> np.ndarray:
    episode_ids = dataset["episode_ids"].cpu().numpy()
    step_ids = dataset["step_ids"].cpu().numpy()
    idx = np.where(episode_ids == episode_id)[0]
    if len(idx) == 0:
        raise ValueError(f"No states found for episode {episode_id}.")
    order = np.argsort(step_ids[idx])
    return idx[order]


def observation_to_hwc(obs: th.Tensor) -> np.ndarray:
    return obs.detach().cpu().permute(1, 2, 0).numpy()


def save_selected_states_figure(
    dataset: Dict[str, th.Tensor],
    selected_indices: np.ndarray,
    output_path: str,
):
    plt = __import__("matplotlib.pyplot", fromlist=["plt"])

    values = dataset["values"].cpu().numpy()
    step_ids = dataset["step_ids"].cpu().numpy()

    n = len(selected_indices)
    cols = min(5, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for ax in axes.flat:
        ax.axis("off")

    for plot_idx, data_idx in enumerate(selected_indices):
        ax = axes.flat[plot_idx]
        ax.imshow(observation_to_hwc(dataset["observations"][data_idx]))
        ax.set_title(
            f"step={int(step_ids[data_idx])}\nvalue={float(values[data_idx]):.3f}",
            fontsize=10,
        )
        ax.axis("off")

    fig.suptitle("Evenly spaced selected states")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_heatmap(
    matrix: np.ndarray,
    step_ids: List[int],
    output_path: str,
):
    plt = __import__("matplotlib.pyplot", fromlist=["plt"])

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_title("Counterfactual value matrix")
    ax.set_xlabel("Inventory donor step")
    ax.set_ylabel("Base/world step")
    ax.set_xticks(np.arange(len(step_ids)))
    ax.set_yticks(np.arange(len(step_ids)))
    ax.set_xticklabels(step_ids, rotation=45, ha="right")
    ax.set_yticklabels(step_ids)
    fig.colorbar(im, ax=ax, label="Predicted value")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_variance_figure(
    row_vars: np.ndarray,
    col_vars: np.ndarray,
    step_ids: List[int],
    output_path: str,
):
    plt = __import__("matplotlib.pyplot", fromlist=["plt"])

    x = np.arange(len(step_ids))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.bar(x - width / 2, row_vars, width, label="Inventory variance for fixed state")
    ax.bar(x + width / 2, col_vars, width, label="State variance for fixed inventory")
    ax.set_xticks(x)
    ax.set_xticklabels(step_ids, rotation=45, ha="right")
    ax.set_xlabel("Selected step")
    ax.set_ylabel("Variance of predicted value")
    ax.set_title("Variance decomposition by selected step")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare value variability from base world state versus swapped inventory across evenly spaced episode steps."
    )
    parser.add_argument("--dataset_path", type=str, default="value_dataset.pt")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--inventory_rows", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="counterfactual_inventory_analysis")
    args = parser.parse_args()

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    th.set_num_threads(1)
    set_seed(0)

    dataset = th.load(args.dataset_path, map_location="cpu")
    episode_indices = get_episode_indices(dataset, args.episode_id)
    selected_positions = select_evenly_spaced_indices(len(episode_indices), args.num_steps)
    selected_indices = episode_indices[selected_positions]
    selected_steps = dataset["step_ids"][selected_indices].cpu().numpy().tolist()

    model, _, ckpt_path = load_model(
        args.exp_name,
        args.timestamp,
        args.train_seed,
        args.ckpt_epoch,
        device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Using device: {device}")
    print(f"Selected steps: {selected_steps}")

    n = len(selected_indices)
    value_matrix = np.zeros((n, n), dtype=np.float64)
    original_values = np.zeros(n, dtype=np.float64)

    observations = dataset["observations"]
    for row_idx, base_dataset_idx in enumerate(selected_indices):
        base_obs = observations[base_dataset_idx]
        original_values[row_idx] = evaluate_value(model, base_obs, device)
        for col_idx, donor_dataset_idx in enumerate(selected_indices):
            donor_obs = observations[donor_dataset_idx]
            hybrid_obs = swap_inventory_rows(base_obs, donor_obs, args.inventory_rows)
            value_matrix[row_idx, col_idx] = evaluate_value(model, hybrid_obs, device)

    row_vars = value_matrix.var(axis=1)
    col_vars = value_matrix.var(axis=0)

    summary = {
        "selected_steps": selected_steps,
        "original_values": original_values.tolist(),
        "mean_inventory_variance_fixed_state": float(row_vars.mean()),
        "mean_state_variance_fixed_inventory": float(col_vars.mean()),
        "row_variances": row_vars.tolist(),
        "col_variances": col_vars.tolist(),
        "inventory_rows": int(args.inventory_rows),
        "value_matrix": value_matrix.tolist(),
    }

    save_selected_states_figure(
        dataset=dataset,
        selected_indices=selected_indices,
        output_path=os.path.join(args.output_dir, "selected_states.png"),
    )
    save_heatmap(
        matrix=value_matrix,
        step_ids=selected_steps,
        output_path=os.path.join(args.output_dir, "value_matrix_heatmap.png"),
    )
    save_variance_figure(
        row_vars=row_vars,
        col_vars=col_vars,
        step_ids=selected_steps,
        output_path=os.path.join(args.output_dir, "variance_comparison.png"),
    )

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("")
    print(f"Mean inventory variance for fixed state: {row_vars.mean():.6f}")
    print(f"Mean state variance for fixed inventory: {col_vars.mean():.6f}")
    print(f"Saved outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
