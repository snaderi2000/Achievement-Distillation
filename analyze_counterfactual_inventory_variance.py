import argparse
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch as th

from collect_value_map import load_model, set_seed


def select_evenly_spaced_indices(num_points: int, target_count: int) -> np.ndarray:
    if target_count <= 0:
        raise ValueError("--num_steps must be positive.")
    if num_points == 0:
        raise ValueError("No points available to sample from.")
    if num_points <= target_count:
        return np.arange(num_points, dtype=np.int64)
    return np.unique(np.linspace(0, num_points - 1, num=target_count, dtype=np.int64))


def parse_donor_steps(text: Optional[str]) -> Optional[List[int]]:
    if text is None:
        return None
    steps = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        steps.append(int(item))
    if not steps:
        raise ValueError("Expected at least one donor step in --donor_steps.")
    return steps


def get_episode_indices(dataset: Dict[str, th.Tensor], episode_id: int) -> np.ndarray:
    episode_ids = dataset["episode_ids"].cpu().numpy()
    step_ids = dataset["step_ids"].cpu().numpy()
    idx = np.where(episode_ids == episode_id)[0]
    if len(idx) == 0:
        raise ValueError(f"No states found for episode {episode_id}.")
    order = np.argsort(step_ids[idx])
    return idx[order]


def find_dataset_index(dataset: Dict[str, th.Tensor], episode_id: int, step_id: int) -> int:
    episode_ids = dataset["episode_ids"]
    step_ids = dataset["step_ids"]
    matches = ((episode_ids == episode_id) & (step_ids == step_id)).nonzero(as_tuple=False).view(-1)
    if len(matches) == 0:
        raise ValueError(f"No state found for episode={episode_id}, step={step_id}.")
    if len(matches) > 1:
        raise ValueError(f"Multiple states found for episode={episode_id}, step={step_id}.")
    return int(matches.item())


def evaluate_value(model, observation: th.Tensor, device: th.device) -> float:
    with th.no_grad():
        outputs = model.act(observation.unsqueeze(0).to(device))
    return float(outputs["vpreds"].item())


def swap_inventory_rows(base_obs: th.Tensor, donor_obs: th.Tensor, inventory_rows: int) -> th.Tensor:
    if inventory_rows <= 0:
        raise ValueError("--inventory_rows must be positive.")
    if base_obs.shape != donor_obs.shape:
        raise ValueError(f"Observation shape mismatch: {tuple(base_obs.shape)} vs {tuple(donor_obs.shape)}")
    if inventory_rows > base_obs.shape[1]:
        raise ValueError(
            f"inventory_rows={inventory_rows} exceeds observation height {base_obs.shape[1]}"
        )

    hybrid = base_obs.clone()
    hybrid[:, -inventory_rows:, :] = donor_obs[:, -inventory_rows:, :]
    return hybrid


def swap_world_rows(base_obs: th.Tensor, donor_obs: th.Tensor, inventory_rows: int) -> th.Tensor:
    if inventory_rows <= 0:
        raise ValueError("--inventory_rows must be positive.")
    if base_obs.shape != donor_obs.shape:
        raise ValueError(f"Observation shape mismatch: {tuple(base_obs.shape)} vs {tuple(donor_obs.shape)}")
    if inventory_rows > base_obs.shape[1]:
        raise ValueError(
            f"inventory_rows={inventory_rows} exceeds observation height {base_obs.shape[1]}"
        )

    hybrid = base_obs.clone()
    hybrid[:, :-inventory_rows, :] = donor_obs[:, :-inventory_rows, :]
    return hybrid


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


def save_fixed_base_figure(
    base_obs: th.Tensor,
    donor_observations: List[th.Tensor],
    hybrid_observations: List[th.Tensor],
    donor_steps: List[int],
    base_step: int,
    base_value: float,
    donor_values: List[float],
    hybrid_values: List[float],
    output_path: str,
):
    plt = __import__("matplotlib.pyplot", fromlist=["plt"])

    n = len(donor_steps)
    fig, axes = plt.subplots(n, 3, figsize=(9.6, max(3.2 * n, 4.0)))
    axes = np.atleast_2d(axes)

    for row_idx in range(n):
        axes[row_idx, 0].imshow(observation_to_hwc(base_obs))
        axes[row_idx, 0].set_title(f"Base {base_step}\nvalue={base_value:.3f}", fontsize=10)
        axes[row_idx, 1].imshow(observation_to_hwc(donor_observations[row_idx]))
        axes[row_idx, 1].set_title(
            f"Inventory {donor_steps[row_idx]}\nvalue={donor_values[row_idx]:.3f}",
            fontsize=10,
        )
        axes[row_idx, 2].imshow(observation_to_hwc(hybrid_observations[row_idx]))
        axes[row_idx, 2].set_title(
            f"Hybrid\nvalue={hybrid_values[row_idx]:.3f}",
            fontsize=10,
        )
        for col_idx in range(3):
            axes[row_idx, col_idx].axis("off")

    fig.suptitle("Fixed-base inventory swap analysis")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_dual_fixed_base_figure(
    base_obs: th.Tensor,
    donor_observations: List[th.Tensor],
    inv_hybrid_observations: List[th.Tensor],
    world_hybrid_observations: List[th.Tensor],
    donor_steps: List[int],
    base_step: int,
    base_value: float,
    donor_values: List[float],
    inv_hybrid_values: List[float],
    world_hybrid_values: List[float],
    output_path: str,
):
    plt = __import__("matplotlib.pyplot", fromlist=["plt"])

    n = len(donor_steps)
    fig, axes = plt.subplots(n, 4, figsize=(12.8, max(3.0 * n, 4.0)))
    axes = np.atleast_2d(axes)

    for row_idx in range(n):
        panels = [
            (base_obs, f"Base {base_step}\nvalue={base_value:.3f}"),
            (donor_observations[row_idx], f"Donor {donor_steps[row_idx]}\nvalue={donor_values[row_idx]:.3f}"),
            (inv_hybrid_observations[row_idx], f"World {base_step} + inv donor\nvalue={inv_hybrid_values[row_idx]:.3f}"),
            (world_hybrid_observations[row_idx], f"Inv {base_step} + world donor\nvalue={world_hybrid_values[row_idx]:.3f}"),
        ]
        for col_idx, (obs, title) in enumerate(panels):
            axes[row_idx, col_idx].imshow(observation_to_hwc(obs))
            axes[row_idx, col_idx].set_title(title, fontsize=9)
            axes[row_idx, col_idx].axis("off")

    fig.suptitle("Fixed-step counterfactual analysis")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
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


def save_fixed_base_value_plot(
    donor_steps: List[int],
    donor_values: List[float],
    hybrid_values: List[float],
    base_value: float,
    output_path: str,
):
    plt = __import__("matplotlib.pyplot", fromlist=["plt"])

    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(donor_steps, donor_values, marker="o", label="Original donor value")
    ax.plot(donor_steps, hybrid_values, marker="o", label="Hybrid value")
    ax.axhline(base_value, color="tab:red", linestyle="--", label=f"Base value ({base_value:.3f})")
    ax.set_xlabel("Donor step")
    ax.set_ylabel("Predicted value")
    ax.set_title("Fixed-base counterfactual inventory values")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_fixed_base_dual_value_plot(
    donor_steps: List[int],
    donor_values: List[float],
    inventory_swap_values: List[float],
    world_swap_values: List[float],
    base_value: float,
    output_path: str,
):
    plt = __import__("matplotlib.pyplot", fromlist=["plt"])

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(donor_steps, donor_values, marker="o", label="Original donor value")
    ax.plot(donor_steps, inventory_swap_values, marker="o", label="Fixed world, swap inventory")
    ax.plot(donor_steps, world_swap_values, marker="o", label="Fixed inventory, swap world")
    ax.axhline(base_value, color="tab:red", linestyle="--", label=f"Base value ({base_value:.3f})")
    ax.set_xlabel("Donor step")
    ax.set_ylabel("Predicted value")
    ax.set_title("Fixed-step counterfactual values")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Counterfactual inventory analysis for one episode: either a full square matrix or a fixed-base-step sweep."
    )
    parser.add_argument("--dataset_path", type=str, default="value_dataset.pt")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--base_step", type=int, default=None)
    parser.add_argument(
        "--donor_steps",
        type=str,
        default=None,
        help="Optional comma-separated donor steps. If omitted, uses evenly spaced steps.",
    )
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
    donor_steps_arg = parse_donor_steps(args.donor_steps)

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

    observations = dataset["observations"]

    if args.base_step is not None:
        if donor_steps_arg is None:
            selected_positions = select_evenly_spaced_indices(len(episode_indices), args.num_steps)
            donor_indices = episode_indices[selected_positions]
            donor_steps = dataset["step_ids"][donor_indices].cpu().numpy().tolist()
        else:
            donor_steps = donor_steps_arg
            donor_indices = np.array(
                [find_dataset_index(dataset, args.episode_id, donor_step) for donor_step in donor_steps],
                dtype=np.int64,
            )

        base_idx = find_dataset_index(dataset, args.episode_id, args.base_step)
        base_obs = observations[base_idx]
        base_value = evaluate_value(model, base_obs, device)
        donor_values = []
        inventory_swap_values = []
        world_swap_values = []
        donor_observations = []
        inventory_swap_observations = []
        world_swap_observations = []

        print(f"Base step: {args.base_step}")
        print(f"Selected donor steps: {donor_steps}")
        print("")
        print("donor_step\t donor_value\t inv_swap\t delta_inv\t world_swap\t delta_world")
        for donor_step, donor_idx in zip(donor_steps, donor_indices.tolist()):
            donor_obs = observations[donor_idx]
            donor_value = evaluate_value(model, donor_obs, device)
            inventory_swap_obs = swap_inventory_rows(base_obs, donor_obs, args.inventory_rows)
            inventory_swap_value = evaluate_value(model, inventory_swap_obs, device)
            world_swap_obs = swap_world_rows(base_obs, donor_obs, args.inventory_rows)
            world_swap_value = evaluate_value(model, world_swap_obs, device)
            delta_inventory = inventory_swap_value - base_value
            delta_world = world_swap_value - base_value

            donor_values.append(donor_value)
            inventory_swap_values.append(inventory_swap_value)
            world_swap_values.append(world_swap_value)
            donor_observations.append(donor_obs)
            inventory_swap_observations.append(inventory_swap_obs)
            world_swap_observations.append(world_swap_obs)
            print(
                f"{donor_step:9d}\t {donor_value:11.4f}\t {inventory_swap_value:8.4f}\t "
                f"{delta_inventory:+9.4f}\t {world_swap_value:10.4f}\t {delta_world:+10.4f}"
            )

        summary = {
            "mode": "fixed_base_step",
            "base_step": int(args.base_step),
            "base_value": float(base_value),
            "donor_steps": donor_steps,
            "donor_values": donor_values,
            "inventory_swap_values": inventory_swap_values,
            "world_swap_values": world_swap_values,
            "inventory_swap_deltas": [hybrid - base_value for hybrid in inventory_swap_values],
            "world_swap_deltas": [hybrid - base_value for hybrid in world_swap_values],
            "variance_inventory_swap_values": float(np.var(inventory_swap_values)),
            "variance_world_swap_values": float(np.var(world_swap_values)),
            "variance_donor_values": float(np.var(donor_values)),
            "inventory_rows": int(args.inventory_rows),
        }

        save_dual_fixed_base_figure(
            base_obs=base_obs,
            donor_observations=donor_observations,
            inv_hybrid_observations=inventory_swap_observations,
            world_hybrid_observations=world_swap_observations,
            donor_steps=donor_steps,
            base_step=args.base_step,
            base_value=base_value,
            donor_values=donor_values,
            inv_hybrid_values=inventory_swap_values,
            world_hybrid_values=world_swap_values,
            output_path=os.path.join(args.output_dir, "fixed_base_swaps.png"),
        )
        save_fixed_base_dual_value_plot(
            donor_steps=donor_steps,
            donor_values=donor_values,
            inventory_swap_values=inventory_swap_values,
            world_swap_values=world_swap_values,
            base_value=base_value,
            output_path=os.path.join(args.output_dir, "fixed_base_values.png"),
        )
        with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("")
        print(f"Variance of donor values: {np.var(donor_values):.6f}")
        print(f"Variance of fixed-world inventory swaps: {np.var(inventory_swap_values):.6f}")
        print(f"Variance of fixed-inventory world swaps: {np.var(world_swap_values):.6f}")
        print(f"Saved outputs to {args.output_dir}")
        return

    selected_positions = select_evenly_spaced_indices(len(episode_indices), args.num_steps)
    selected_indices = episode_indices[selected_positions]
    selected_steps = dataset["step_ids"][selected_indices].cpu().numpy().tolist()
    print(f"Selected steps: {selected_steps}")

    n = len(selected_indices)
    value_matrix = np.zeros((n, n), dtype=np.float64)
    original_values = np.zeros(n, dtype=np.float64)

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
        "mode": "square_matrix",
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
