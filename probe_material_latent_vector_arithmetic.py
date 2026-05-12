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
    obs_to_tensor,
    parse_inventory_assignments,
    parse_vec2,
    render_env,
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


def make_obs(
    eval_seed: int,
    texture: str | None,
    target_delta: Tuple[int, int],
    background_material: str,
    inventory_updates: Dict[str, int],
) -> np.ndarray:
    from crafter.env import Env

    env = Env(seed=eval_seed)
    env.reset()
    valid_materials = set(material_names(env))
    clear_visible_objects(env)
    make_visible_material(env, background_material, keep_player_tile=True)
    apply_inventory_edits(env, inventory_updates, [])
    env._world.daylight = 1.0
    if texture is not None:
        set_target_texture(env, target_delta, texture, valid_materials)
    obs = render_env(env)
    env.close()
    return obs


def encode(model, obs: np.ndarray, device: th.device) -> th.Tensor:
    with th.no_grad():
        return model.encode(obs_to_tensor(obs, device))


def value_from_image_latent(
    model,
    image_latent: th.Tensor,
    achievement_progress: th.Tensor | None,
) -> float:
    with th.no_grad():
        if hasattr(model, "progress_encoder"):
            if achievement_progress is None:
                raise ValueError("achievement_progress is required for this explicit-memory model.")
            memory_latent = model.progress_encoder(achievement_progress.float())
            shared = th.cat([image_latent, memory_latent], dim=-1)
            vf_features = model.vf_backbone(shared)
            vpred = model.vf_head(vf_features)
            return float(model.vf_head.denormalize(vpred).item())

        model_latents = image_latent
        if getattr(model, "use_achievement_progress_input", False):
            if achievement_progress is None:
                raise ValueError("achievement_progress is required when use_achievement_progress_input=True.")
            model_latents = th.cat([image_latent, achievement_progress.float()], dim=-1)
        vf_latents = model.vf_tower(model_latents) if getattr(model, "vf_tower", None) is not None else model_latents
        vpred = model.vf_head(vf_latents)
        return float(model.vf_head.denormalize(vpred).item())


def mean_direction(
    model,
    device: th.device,
    seeds: Sequence[int],
    texture: str,
    target_delta: Tuple[int, int],
    background_material: str,
    inventory_updates: Dict[str, int],
) -> th.Tensor:
    diffs: List[th.Tensor] = []
    for seed in seeds:
        base_obs = make_obs(seed, None, target_delta, background_material, inventory_updates)
        target_obs = make_obs(seed, texture, target_delta, background_material, inventory_updates)
        base_latent = encode(model, base_obs, device)
        target_latent = encode(model, target_obs, device)
        diffs.append(target_latent - base_latent)
    return th.stack(diffs, dim=0).mean(dim=0)


def write_csv(path: str, rows: Sequence[Dict[str, object]]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path: str, rows: Sequence[Dict[str, object]], textures: Sequence[str], alphas: Sequence[float]):
    fig, axes = plt.subplots(1, len(textures), figsize=(5.0 * len(textures), 4.0), sharey=True)
    axes = np.asarray(axes).reshape(-1)
    for ax, texture in zip(axes, textures):
        texture_rows = [row for row in rows if row["patch_direction"] == texture]
        mean_by_alpha = []
        for alpha in alphas:
            vals = [float(row["patched_value"]) for row in texture_rows if float(row["alpha"]) == float(alpha)]
            mean_by_alpha.append(float(np.mean(vals)))
        actual_vals = [float(row[f"actual_{texture}_value"]) for row in texture_rows if float(row["alpha"]) == 1.0]
        base_vals = [float(row["base_value"]) for row in texture_rows if float(row["alpha"]) == 1.0]
        ax.plot(alphas, mean_by_alpha, marker="o", label=f"base + alpha*d_{texture}")
        ax.axhline(float(np.mean(actual_vals)), color="#333333", linestyle="--", label=f"actual {texture}")
        ax.axhline(float(np.mean(base_vals)), color="#999999", linestyle=":", label="base")
        ax.set_title(f"{texture} direction")
        ax.set_xlabel("alpha")
        ax.set_ylabel("V")
        ax.legend(frameon=False)
    fig.suptitle("Latent vector arithmetic: material-at-position directions")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether material-at-position directions in image latents causally move value "
            "toward the actual rendered material counterfactual."
        )
    )
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--direction_seed_start", type=int, default=1000)
    parser.add_argument("--num_direction_seeds", type=int, default=20)
    parser.add_argument("--test_seed_start", type=int, default=2000)
    parser.add_argument("--num_test_seeds", type=int, default=10)
    parser.add_argument("--textures", type=str, default="tree,stone")
    parser.add_argument("--target_delta", type=str, default="0,-2")
    parser.add_argument("--background_material", type=str, default="grass")
    parser.add_argument("--alphas", type=str, default="0,0.5,1,1.5,2")
    parser.add_argument("--memory_tasks", type=str, default="")
    parser.add_argument(
        "--set_inventory",
        type=str,
        default="health=9,food=9,drink=9,energy=9,wood_pickaxe=0",
    )
    parser.add_argument("--output_dir", type=str, default="material_latent_vector_arithmetic")
    args = parser.parse_args()

    set_seed(args.direction_seed_start)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, _config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)

    textures = parse_csv(args.textures)
    target_delta = parse_vec2(args.target_delta)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    inventory_updates = parse_inventory_assignments(args.set_inventory)
    memory_tasks = parse_csv(args.memory_tasks)
    achievement_progress = memory_vector(memory_tasks, device)

    direction_seeds = list(range(args.direction_seed_start, args.direction_seed_start + args.num_direction_seeds))
    test_seeds = list(range(args.test_seed_start, args.test_seed_start + args.num_test_seeds))

    directions = {
        texture: mean_direction(
            model,
            device,
            direction_seeds,
            texture,
            target_delta,
            args.background_material,
            inventory_updates,
        )
        for texture in textures
    }

    rows: List[Dict[str, object]] = []
    for seed in test_seeds:
        base_obs = make_obs(seed, None, target_delta, args.background_material, inventory_updates)
        base_latent = encode(model, base_obs, device)
        base_value = value_from_image_latent(model, base_latent, achievement_progress)
        actual_values = {}
        for texture in textures:
            actual_obs = make_obs(seed, texture, target_delta, args.background_material, inventory_updates)
            actual_latent = encode(model, actual_obs, device)
            actual_values[texture] = value_from_image_latent(model, actual_latent, achievement_progress)
        for patch_texture, direction in directions.items():
            for alpha in alphas:
                patched_latent = base_latent + alpha * direction
                patched_value = value_from_image_latent(model, patched_latent, achievement_progress)
                row = {
                    "seed": seed,
                    "patch_direction": patch_texture,
                    "alpha": alpha,
                    "base_value": base_value,
                    "patched_value": patched_value,
                    "target_delta": list(target_delta),
                    "background_material": args.background_material,
                }
                for texture, value in actual_values.items():
                    row[f"actual_{texture}_value"] = value
                    row[f"patched_minus_actual_{texture}"] = patched_value - value
                rows.append(row)

    csv_path = os.path.join(args.output_dir, "latent_vector_arithmetic_rows.csv")
    figure_path = os.path.join(args.output_dir, "latent_vector_arithmetic.png")
    summary_path = os.path.join(args.output_dir, "summary.json")
    write_csv(csv_path, rows)
    save_figure(figure_path, rows, textures, alphas)

    summary = {
        "checkpoint": ckpt_path,
        "textures": list(textures),
        "target_delta": list(target_delta),
        "background_material": args.background_material,
        "direction_seeds": direction_seeds,
        "test_seeds": test_seeds,
        "memory_tasks": list(memory_tasks),
        "inventory": inventory_updates,
        "alphas": alphas,
        "csv_path": csv_path,
        "figure_path": figure_path,
    }
    for texture in textures:
        alpha_one = [
            row
            for row in rows
            if row["patch_direction"] == texture and abs(float(row["alpha"]) - 1.0) < 1e-8
        ]
        summary[f"{texture}_alpha1_mean_patched_value"] = float(
            np.mean([float(row["patched_value"]) for row in alpha_one])
        )
        summary[f"{texture}_mean_actual_value"] = float(
            np.mean([float(row[f"actual_{texture}_value"]) for row in alpha_one])
        )
        summary[f"{texture}_mean_base_value"] = float(np.mean([float(row["base_value"]) for row in alpha_one]))
        summary[f"{texture}_alpha1_mean_abs_error_to_actual"] = float(
            np.mean([abs(float(row["patched_value"]) - float(row[f"actual_{texture}_value"])) for row in alpha_one])
        )

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
