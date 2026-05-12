import argparse
import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn.functional as F

from achievement_distillation.constant import TASKS
from collect_value_map import load_model, set_seed
from counterfactual_env_editor import apply_inventory_edits, material_names, obs_to_tensor, parse_inventory_assignments, parse_vec2, render_env
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
    texture: str,
    delta: Tuple[int, int],
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
    set_target_texture(env, delta, texture, valid_materials)
    obs = render_env(env)
    env.close()
    return obs


def encode(model, obs: np.ndarray, device: th.device) -> th.Tensor:
    with th.no_grad():
        return model.encode(obs_to_tensor(obs, device))


def value_from_image_latent(model, image_latent: th.Tensor, achievement_progress: th.Tensor | None) -> float:
    with th.no_grad():
        if hasattr(model, "progress_encoder"):
            if achievement_progress is None:
                raise ValueError("achievement_progress is required for explicit-memory models.")
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


def tensor_distance(a: th.Tensor, b: th.Tensor) -> Dict[str, float]:
    diff = a - b
    return {
        "latent_l2": float(th.linalg.vector_norm(diff).item()),
        "latent_cosine": float(F.cosine_similarity(a, b, dim=-1).item()),
    }


def write_csv(path: str, rows: Sequence[Dict[str, object]]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path: str, rows: Sequence[Dict[str, object]]):
    labels = []
    value_errors = []
    l2s = []
    for direction in sorted({str(row["analogy"]) for row in rows}):
        subset = [row for row in rows if row["analogy"] == direction]
        labels.append(direction.replace("_", "\n"))
        value_errors.append(float(np.mean([abs(float(row["patched_value"]) - float(row["actual_value"])) for row in subset])))
        l2s.append(float(np.mean([float(row["latent_l2"]) for row in subset])))

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    axes[0].bar(labels, value_errors, color="#4c78a8")
    axes[0].set_title("Value match error")
    axes[0].set_ylabel("mean |V(patched) - V(actual)|")
    axes[1].bar(labels, l2s, color="#f58518")
    axes[1].set_title("Latent match error")
    axes[1].set_ylabel("mean ||z_patched - z_actual||")
    fig.suptitle("Material/location latent analogy")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_analogy(
    model,
    device: th.device,
    seed: int,
    source_texture: str,
    target_texture: str,
    ref_delta: Tuple[int, int],
    query_delta: Tuple[int, int],
    background_material: str,
    inventory_updates: Dict[str, int],
    achievement_progress: th.Tensor,
) -> Dict[str, object]:
    source_ref = encode(model, make_obs(seed, source_texture, ref_delta, background_material, inventory_updates), device)
    source_query = encode(model, make_obs(seed, source_texture, query_delta, background_material, inventory_updates), device)
    target_ref = encode(model, make_obs(seed, target_texture, ref_delta, background_material, inventory_updates), device)
    target_query = encode(model, make_obs(seed, target_texture, query_delta, background_material, inventory_updates), device)

    patched = source_query - source_ref + target_ref
    patched_value = value_from_image_latent(model, patched, achievement_progress)
    actual_value = value_from_image_latent(model, target_query, achievement_progress)
    source_query_value = value_from_image_latent(model, source_query, achievement_progress)
    target_ref_value = value_from_image_latent(model, target_ref, achievement_progress)

    distances = tensor_distance(patched, target_query)
    return {
        "seed": seed,
        "analogy": f"{source_texture}_query_minus_{source_texture}_ref_plus_{target_texture}_ref_to_{target_texture}_query",
        "source_texture": source_texture,
        "target_texture": target_texture,
        "ref_delta": list(ref_delta),
        "query_delta": list(query_delta),
        "patched_value": patched_value,
        "actual_value": actual_value,
        "value_error": patched_value - actual_value,
        "abs_value_error": abs(patched_value - actual_value),
        "source_query_value": source_query_value,
        "target_ref_value": target_ref_value,
        **distances,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test latent analogies like z(stone_above) - z(stone_front) + z(tree_front) ~= z(tree_above)."
    )
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--eval_seed_start", type=int, default=2000)
    parser.add_argument("--num_eval_seeds", type=int, default=20)
    parser.add_argument("--source_texture", type=str, default="stone")
    parser.add_argument("--target_texture", type=str, default="tree")
    parser.add_argument("--ref_delta", type=str, default="0,1")
    parser.add_argument("--query_delta", type=str, default="0,-2")
    parser.add_argument("--background_material", type=str, default="grass")
    parser.add_argument("--memory_tasks", type=str, default="")
    parser.add_argument("--set_inventory", type=str, default="health=9,food=9,drink=9,energy=9,wood_pickaxe=0")
    parser.add_argument("--output_dir", type=str, default="material_latent_analogy")
    args = parser.parse_args()

    set_seed(args.eval_seed_start)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, _config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)

    ref_delta = parse_vec2(args.ref_delta)
    query_delta = parse_vec2(args.query_delta)
    inventory_updates = parse_inventory_assignments(args.set_inventory)
    memory_tasks = parse_csv(args.memory_tasks)
    achievement_progress = memory_vector(memory_tasks, device)
    seeds = list(range(args.eval_seed_start, args.eval_seed_start + args.num_eval_seeds))

    rows: List[Dict[str, object]] = []
    for seed in seeds:
        rows.append(
            run_analogy(
                model,
                device,
                seed,
                args.source_texture,
                args.target_texture,
                ref_delta,
                query_delta,
                args.background_material,
                inventory_updates,
                achievement_progress,
            )
        )
        rows.append(
            run_analogy(
                model,
                device,
                seed,
                args.target_texture,
                args.source_texture,
                ref_delta,
                query_delta,
                args.background_material,
                inventory_updates,
                achievement_progress,
            )
        )

    csv_path = os.path.join(args.output_dir, "latent_analogy_rows.csv")
    figure_path = os.path.join(args.output_dir, "latent_analogy.png")
    summary_path = os.path.join(args.output_dir, "summary.json")
    write_csv(csv_path, rows)
    save_figure(figure_path, rows)

    summary = {
        "checkpoint": ckpt_path,
        "source_texture": args.source_texture,
        "target_texture": args.target_texture,
        "ref_delta": list(ref_delta),
        "query_delta": list(query_delta),
        "background_material": args.background_material,
        "eval_seeds": seeds,
        "memory_tasks": list(memory_tasks),
        "inventory": inventory_updates,
        "csv_path": csv_path,
        "figure_path": figure_path,
    }
    for analogy in sorted({str(row["analogy"]) for row in rows}):
        subset = [row for row in rows if row["analogy"] == analogy]
        summary[f"{analogy}_mean_abs_value_error"] = float(np.mean([float(row["abs_value_error"]) for row in subset]))
        summary[f"{analogy}_mean_latent_l2"] = float(np.mean([float(row["latent_l2"]) for row in subset]))
        summary[f"{analogy}_mean_latent_cosine"] = float(np.mean([float(row["latent_cosine"]) for row in subset]))
        summary[f"{analogy}_mean_patched_value"] = float(np.mean([float(row["patched_value"]) for row in subset]))
        summary[f"{analogy}_mean_actual_value"] = float(np.mean([float(row["actual_value"]) for row in subset]))

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
