import argparse
import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from achievement_distillation.constant import TASKS
from collect_value_map import load_model, set_seed
from counterfactual_env_editor import (
    apply_inventory_edits,
    material_names,
    parse_inventory_assignments,
    render_env,
    score_observation,
    visible_world_cells,
)
from probe_material_value_preference import clear_visible_objects, make_visible_material, set_target_texture


def parse_csv(text: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def memory_vector(task_names: Tuple[str, ...], device: th.device) -> th.Tensor:
    unknown = sorted(set(task_names) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown achievement memory tasks: {unknown}")
    values = [1.0 if task in task_names else 0.0 for task in TASKS]
    return th.tensor([values], dtype=th.float32, device=device)


def prepare_base_env(eval_seed: int, background_material: str, inventory_updates: Dict[str, int]):
    from crafter.env import Env

    env = Env(seed=eval_seed)
    env.reset()
    clear_visible_objects(env)
    make_visible_material(env, background_material, keep_player_tile=True)
    apply_inventory_edits(env, inventory_updates, [])
    env._world.daylight = 1.0
    return env


def visible_deltas(env) -> List[Tuple[int, int, int, int]]:
    local_grid = np.asarray(env._local_view._grid, dtype=np.int64)
    center = local_grid // 2
    rows = []
    for ix in range(int(local_grid[0])):
        for iy in range(int(local_grid[1])):
            dx = int(ix - center[0])
            dy = int(iy - center[1])
            if dx == 0 and dy == 0:
                continue
            rows.append((ix, iy, dx, dy))
    return rows


def write_csv(path: str, rows: Sequence[Dict[str, object]]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_heatmap(path: str, rows: Sequence[Dict[str, object]], grid_shape: Tuple[int, int], texture: str, baseline_value: float):
    values = np.full((grid_shape[1], grid_shape[0]), np.nan, dtype=np.float32)
    deltas = np.full((grid_shape[1], grid_shape[0]), np.nan, dtype=np.float32)
    for row in rows:
        ix = int(row["cell_x"])
        iy = int(row["cell_y"])
        values[iy, ix] = float(row["value"])
        deltas[iy, ix] = float(row["delta_from_background"])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    im0 = axes[0].imshow(values, cmap="viridis")
    axes[0].set_title(f"V({texture} at cell)")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    vmax = np.nanmax(np.abs(deltas))
    vmax = max(float(vmax), 1e-6)
    im1 = axes[1].imshow(deltas, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[1].set_title(f"V({texture}) - V(all grass)\nbase={baseline_value:.3f}")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("visible x")
        ax.set_ylabel("visible y")
        ax.set_xticks(range(grid_shape[0]))
        ax.set_yticks(range(grid_shape[1]))
        ax.grid(color="white", linewidth=0.6, alpha=0.7)
        for row in rows:
            ix = int(row["cell_x"])
            iy = int(row["cell_y"])
            ax.text(ix, iy, f"{float(row['delta_from_background']):+.2f}", ha="center", va="center", fontsize=7)

    fig.suptitle(f"Spatial value map for one {texture} on all-grass background")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_montage(path: str, images: Sequence[Tuple[str, float, np.ndarray]], cols: int = 9):
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.0 * cols, 2.25 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (label, value, obs) in zip(axes, images):
        ax.imshow(obs)
        ax.set_title(f"{label}\nV={value:.2f}", fontsize=8)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Move one material/object around the visible grid and map how value changes with position."
    )
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument("--texture", type=str, default="tree")
    parser.add_argument("--background_material", type=str, default="grass")
    parser.add_argument("--memory_tasks", type=str, default="")
    parser.add_argument("--set_inventory", type=str, default="health=9,food=9,drink=9,energy=9,wood_pickaxe=0")
    parser.add_argument("--output_dir", type=str, default="material_position_value_map")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, _config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)

    inventory_updates = parse_inventory_assignments(args.set_inventory)
    memory_tasks = parse_csv(args.memory_tasks)
    achievement_progress = memory_vector(memory_tasks, device)

    base_env = prepare_base_env(args.eval_seed, args.background_material, inventory_updates)
    valid_materials = set(material_names(base_env))
    if args.texture not in valid_materials:
        raise ValueError(f"This script currently expects a material texture. Unknown material: {args.texture}")
    if args.background_material not in valid_materials:
        raise ValueError(f"Unknown background material: {args.background_material}")

    base_obs = render_env(base_env)
    baseline_value = score_observation(model, base_obs, device, achievement_progress=achievement_progress)["value"]
    grid_shape = tuple(int(x) for x in np.asarray(base_env._local_view._grid, dtype=np.int64))

    rows: List[Dict[str, object]] = []
    images: List[Tuple[str, float, np.ndarray]] = [("all grass", baseline_value, base_obs)]
    for cell_x, cell_y, dx, dy in visible_deltas(base_env):
        env = prepare_base_env(args.eval_seed, args.background_material, inventory_updates)
        set_target_texture(env, (dx, dy), args.texture, valid_materials)
        obs = render_env(env)
        value = score_observation(model, obs, device, achievement_progress=achievement_progress)["value"]
        rows.append(
            {
                "cell_x": cell_x,
                "cell_y": cell_y,
                "dx": dx,
                "dy": dy,
                "texture": args.texture,
                "value": value,
                "background_value": baseline_value,
                "delta_from_background": value - baseline_value,
            }
        )
        images.append((f"({dx},{dy})", value, obs))
        env.close()

    stem = f"{args.exp_name}-s{args.train_seed:02}-{args.texture}-position"
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    heatmap_path = os.path.join(args.output_dir, f"{stem}-heatmap.png")
    montage_path = os.path.join(args.output_dir, f"{stem}-observations.png")
    summary_path = os.path.join(args.output_dir, f"{stem}.json")

    write_csv(csv_path, rows)
    save_heatmap(heatmap_path, rows, grid_shape, args.texture, baseline_value)
    save_montage(montage_path, images)

    summary = {
        "checkpoint": ckpt_path,
        "eval_seed": args.eval_seed,
        "texture": args.texture,
        "background_material": args.background_material,
        "grid_shape": list(grid_shape),
        "memory_tasks": list(memory_tasks),
        "inventory": inventory_updates,
        "background_value": baseline_value,
        "csv_path": csv_path,
        "heatmap_path": heatmap_path,
        "montage_path": montage_path,
        "num_positions": len(rows),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    base_env.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
