import argparse
import copy
import csv
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from achievement_distillation.constant import TASKS
from collect_value_map import load_model, set_seed
from counterfactual_env_editor import (
    apply_inventory_edits,
    material_names,
    obs_to_tensor,
    render_env,
    score_observation,
    valid_spawn_objects,
    visible_world_cells,
)
from probe_material_value_preference import parse_texture_list, set_target_texture


def parse_csv(text: str) -> Tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def memory_vector(task_names: Tuple[str, ...], device: th.device) -> th.Tensor:
    unknown = sorted(set(task_names) - set(TASKS))
    if unknown:
        raise ValueError(f"Unknown achievement memory tasks: {unknown}")
    values = [1.0 if task in task_names else 0.0 for task in TASKS]
    return th.tensor([values], dtype=th.float32, device=device)


def choose_visible_delta(env, rng: np.random.Generator) -> Tuple[int, int]:
    candidates = []
    for cell_index, _world_pos, _material, obj in visible_world_cells(env):
        if obj is env._player:
            continue
        local_grid = np.asarray(env._local_view._grid, dtype=np.int64)
        center = local_grid // 2
        dx = int(cell_index[0] - center[0])
        dy = int(cell_index[1] - center[1])
        if dx == 0 and dy == 0:
            continue
        candidates.append((dx, dy))
    if not candidates:
        raise ValueError("No visible non-player cells available for texture insertion.")
    return candidates[int(rng.integers(0, len(candidates)))]


def world_cell_bounds(obs: np.ndarray, env, dx: int, dy: int):
    grid = tuple(int(x) for x in np.asarray(env._local_view._grid, dtype=np.int64))
    center = (grid[0] // 2, grid[1] // 2)
    cell_x = center[0] + int(dx)
    cell_y = center[1] + int(dy)
    if not (0 <= cell_x < grid[0] and 0 <= cell_y < grid[1]):
        raise ValueError(f"Delta ({dx},{dy}) maps outside visible grid {grid}.")
    cell_w = obs.shape[1] // grid[0]
    cell_h = (obs.shape[0] - 15) // grid[1]
    return cell_x * cell_w, cell_y * cell_h, cell_w, cell_h


def synthetic_texture_patch(name: str, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    palettes = {
        "novel_purple": np.array([145, 55, 210], dtype=np.int16),
        "novel_red": np.array([210, 35, 45], dtype=np.int16),
        "novel_cyan": np.array([40, 195, 220], dtype=np.int16),
        "novel_yellow": np.array([225, 210, 45], dtype=np.int16),
    }
    if name not in palettes:
        raise ValueError(f"Unknown synthetic texture '{name}'.")
    base = palettes[name]
    yy, xx = np.mgrid[:h, :w]
    checker = ((xx // 2 + yy // 2) % 2)[:, :, None]
    noise = rng.integers(-18, 19, size=(h, w, 3), dtype=np.int16)
    patch = base + noise + checker * 24
    border = np.zeros((h, w, 1), dtype=bool)
    border[0, :, 0] = True
    border[-1, :, 0] = True
    border[:, 0, 0] = True
    border[:, -1, 0] = True
    patch = np.where(border, np.maximum(patch - 55, 0), patch)
    return np.clip(patch, 0, 255).astype(np.uint8)


def insert_synthetic_texture(obs: np.ndarray, env, dx: int, dy: int, texture: str, rng: np.random.Generator) -> np.ndarray:
    x0, y0, cell_w, cell_h = world_cell_bounds(obs, env, dx, dy)
    edited = obs.copy()
    edited[y0 : y0 + cell_h, x0 : x0 + cell_w] = synthetic_texture_patch(texture, cell_h, cell_w, rng)
    return edited


def active_memory_tasks(progress: th.Tensor) -> List[str]:
    values = progress.detach().cpu().view(-1).tolist()
    return [task for task, value in zip(TASKS, values) if value > 0.5]


def sample_rollout_states(
    model,
    config: Dict,
    eval_seed: int,
    device: th.device,
    num_states: int,
    max_steps: int,
    rng: np.random.Generator,
):
    from crafter.env import Env

    env = Env(seed=eval_seed)
    obs = env.reset()
    hidsize = int(config.get("model_kwargs", {}).get("hidsize", getattr(model, "hidsize", 0)))
    states = th.zeros(1, hidsize, device=device) if hidsize > 0 else None
    rnn_hidsize = config.get("model_kwargs", {}).get("rnn_hidsize")
    rnn_states = th.zeros(1, int(rnn_hidsize), device=device) if rnn_hidsize is not None else None
    achievement_progress = th.zeros(1, len(TASKS), device=device)

    sample_steps = sorted(rng.choice(np.arange(max_steps), size=min(num_states, max_steps), replace=False).tolist())
    sample_step_set = set(int(step) for step in sample_steps)
    samples = []

    for step_id in range(max_steps):
        if step_id in sample_step_set:
            samples.append(
                {
                    "step_id": step_id,
                    "env": copy.deepcopy(env),
                    "obs": obs.copy(),
                    "states": None if states is None else states.clone(),
                    "rnn_states": None if rnn_states is None else rnn_states.clone(),
                    "achievement_progress": achievement_progress.clone(),
                }
            )
            if len(samples) >= num_states:
                break

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
            action = int(outputs["actions"].item())
            if "next_states" in outputs:
                states = outputs["next_states"]
            if "next_rnn_states" in outputs:
                rnn_states = outputs["next_rnn_states"]

        obs, _reward, done, info = env.step(action)
        achievements = info.get("achievements")
        if achievements is not None:
            achievement_progress = th.tensor(
                [[1.0 if achievements.get(task, 0) > 0 else 0.0 for task in TASKS]],
                dtype=th.float32,
                device=device,
            )
        if done:
            obs = env.reset()
            if states is not None:
                states.zero_()
            if rnn_states is not None:
                rnn_states.zero_()
            achievement_progress.zero_()

    env.close()
    return samples, sample_steps


def write_csv(path: str, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_summary_plot(path: str, rows, textures):
    means = []
    stds = []
    for texture in textures:
        vals = [float(row["delta_value"]) for row in rows if row["texture"] == texture]
        means.append(float(np.mean(vals)) if vals else 0.0)
        stds.append(float(np.std(vals)) if vals else 0.0)
    colors = ["#d95f02" if value > 0 else "#1b9e77" for value in means]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(textures)), 4.2))
    ax.bar(textures, means, yerr=stds, color=colors, alpha=0.85)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("mean delta V after insertion")
    ax.set_title("Value response to inserting one texture into random rollout states")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_montage(path: str, examples):
    cols = 4
    rows = int(np.ceil(len(examples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.4 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (title, obs) in zip(axes, examples):
        ax.imshow(obs)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Insert candidate textures into random rollout states and measure whether value rises or falls."
    )
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument("--num_states", type=int, default=5)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument(
        "--textures",
        type=str,
        default="novel_purple,novel_red,stone,tree,water,sand,coal,iron,cow,skeleton,zombie,plant",
        help=(
            "Comma-separated existing textures or synthetic novel textures. "
            "Synthetic options: novel_purple, novel_red, novel_cyan, novel_yellow."
        ),
    )
    parser.add_argument("--memory_tasks", type=str, default=None, help="Override explicit memory bits; default uses rollout memory.")
    parser.add_argument("--set_inventory", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="novel_texture_value_response")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.eval_seed)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

    model, config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)
    samples, sample_steps = sample_rollout_states(
        model,
        config,
        args.eval_seed,
        device,
        args.num_states,
        args.max_steps,
        rng,
    )
    if not samples:
        raise ValueError("No rollout states were sampled.")

    valid_materials = set(material_names(samples[0]["env"]))
    textures = parse_texture_list(args.textures, sorted(valid_materials))
    synthetic_textures = {"novel_purple", "novel_red", "novel_cyan", "novel_yellow"}
    valid_textures = valid_materials | set(valid_spawn_objects()) | synthetic_textures
    unknown = sorted(set(textures) - valid_textures)
    if unknown:
        raise ValueError(f"Unknown textures {unknown}. Valid materials: {sorted(valid_materials)}; objects: {valid_spawn_objects()}")

    inventory_updates = {}
    if args.set_inventory:
        from counterfactual_env_editor import parse_inventory_assignments

        inventory_updates = parse_inventory_assignments(args.set_inventory)
    memory_override = None
    if args.memory_tasks is not None:
        memory_override = memory_vector(parse_csv(args.memory_tasks), device)

    rows = []
    examples = []
    for state_idx, sample in enumerate(samples):
        base_env = sample["env"]
        if inventory_updates:
            apply_inventory_edits(base_env, inventory_updates, [])
        base_obs = render_env(base_env)
        achievement_progress = memory_override if memory_override is not None else sample["achievement_progress"]
        base_value = score_observation(
            model,
            base_obs,
            device,
            sample["states"],
            sample["rnn_states"],
            achievement_progress,
        )["value"]
        dx, dy = choose_visible_delta(base_env, rng)
        state_examples = [(f"state {state_idx} base\nV={base_value:.2f}", base_obs)]
        for texture in textures:
            env_copy = copy.deepcopy(base_env)
            if texture in synthetic_textures:
                texture_kind = "synthetic_pixel"
                edited_obs = insert_synthetic_texture(base_obs, env_copy, dx, dy, texture, rng)
            else:
                texture_kind = set_target_texture(env_copy, (dx, dy), texture, valid_materials)
                edited_obs = render_env(env_copy)
            edited_value = score_observation(
                model,
                edited_obs,
                device,
                sample["states"],
                sample["rnn_states"],
                achievement_progress,
            )["value"]
            rows.append(
                {
                    "state_idx": state_idx,
                    "step_id": sample["step_id"],
                    "texture": texture,
                    "texture_kind": texture_kind,
                    "dx": dx,
                    "dy": dy,
                    "base_value": base_value,
                    "edited_value": edited_value,
                    "delta_value": edited_value - base_value,
                    "memory_tasks": ";".join(active_memory_tasks(achievement_progress)),
                }
            )
            state_examples.append((f"s{state_idx} {texture}@({dx},{dy})\nd={edited_value - base_value:+.2f}", edited_obs))
            env_copy.close()
        base_env.close()
        examples.extend(state_examples)

    csv_path = os.path.join(args.output_dir, "novel_texture_value_rows.csv")
    plot_path = os.path.join(args.output_dir, "novel_texture_value_summary.png")
    montage_path = os.path.join(args.output_dir, "novel_texture_examples.png")
    summary_path = os.path.join(args.output_dir, "summary.json")
    write_csv(csv_path, rows)
    save_summary_plot(plot_path, rows, textures)
    save_montage(montage_path, examples)

    texture_summary = {}
    for texture in textures:
        vals = [float(row["delta_value"]) for row in rows if row["texture"] == texture]
        texture_summary[texture] = {
            "mean_delta": float(np.mean(vals)),
            "std_delta": float(np.std(vals)),
            "fraction_positive": float(np.mean(np.array(vals) > 0.0)),
            "fraction_negative": float(np.mean(np.array(vals) < 0.0)),
        }

    summary = {
        "checkpoint": ckpt_path,
        "eval_seed": args.eval_seed,
        "sample_steps": sample_steps,
        "num_states": len(samples),
        "textures": textures,
        "memory_override": args.memory_tasks,
        "inventory_updates": inventory_updates,
        "texture_summary": texture_summary,
        "csv_path": csv_path,
        "plot_path": plot_path,
        "montage_path": montage_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
