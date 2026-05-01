import argparse
import copy
import csv
import json
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from collect_value_map import load_model, set_seed
from counterfactual_env_editor import (
    apply_inventory_edits,
    apply_spawn_object,
    material_names,
    parse_inventory_assignments,
    render_env,
    replay_to_step,
    score_observation,
    valid_spawn_objects,
    visible_world_cells,
    world_pos_from_delta,
)


DEFAULT_MATERIALS = [
    "water",
    "sand",
    "grass",
    "tree",
    "path",
    "stone",
    "coal",
    "iron",
    "diamond",
    "lava",
]


def default_textures() -> List[str]:
    return DEFAULT_MATERIALS + valid_spawn_objects()


def parse_vec2(text: str) -> Tuple[int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate '{text}'. Expected dx,dy.")
    return int(parts[0]), int(parts[1])


def clear_visible_objects(env):
    removed = 0
    for _, _, _, obj in visible_world_cells(env):
        if obj is None or obj is env._player:
            continue
        env._world.remove(obj)
        removed += 1
    return removed


def make_visible_material(env, material: str, keep_player_tile: bool = True) -> int:
    changed = 0
    for _, world_pos, _, obj in visible_world_cells(env):
        if keep_player_tile and obj is env._player:
            continue
        env._world[tuple(world_pos)] = material
        changed += 1
    return changed


def set_target_material(env, target_delta: Tuple[int, int], material: str):
    pos = tuple(world_pos_from_delta(env, target_delta[0], target_delta[1]))
    _, obj = env._world[pos]
    if obj is not None and obj is not env._player:
        env._world.remove(obj)
    if obj is env._player:
        raise ValueError("Target delta points at the player tile; use a nearby tile such as 0,1.")
    env._world[pos] = material


def set_target_texture(env, target_delta: Tuple[int, int], texture: str, valid_materials: set[str]):
    pos = tuple(world_pos_from_delta(env, target_delta[0], target_delta[1]))
    _, obj = env._world[pos]
    if obj is env._player:
        raise ValueError("Target delta points at the player tile; use a nearby tile such as 0,1.")
    if obj is not None:
        env._world.remove(obj)
    env._world[pos] = "grass"
    if texture in valid_materials:
        env._world[pos] = texture
        return "material"
    edits_log: List[str] = []
    apply_spawn_object(env, [(target_delta[0], target_delta[1], texture)], edits_log)
    return "object"


def parse_texture_list(text: str, valid_materials: Sequence[str]) -> List[str]:
    if text == "all":
        return list(valid_materials)
    if text == "all_textures":
        textures = list(valid_materials)
        for name in valid_spawn_objects():
            if name not in textures:
                textures.append(name)
        return textures
    return [part.strip() for part in text.split(",") if part.strip()]


def score_textures(
    model,
    replay,
    device: th.device,
    textures: Sequence[str],
    target_delta: Tuple[int, int],
    background_material: str,
    clear_objects: bool,
    inventory_updates: Dict[str, int],
    noise_texture: str | None,
    noise_delta: Tuple[int, int],
):
    prepared_env = copy.deepcopy(replay.env)
    valid_materials = set(material_names(prepared_env))
    edits_log: List[str] = []
    apply_inventory_edits(prepared_env, inventory_updates, edits_log)
    removed_objects = clear_visible_objects(prepared_env) if clear_objects else 0
    changed_tiles = make_visible_material(prepared_env, background_material)
    if noise_texture:
        set_target_texture(prepared_env, noise_delta, noise_texture, valid_materials)
    base_obs = render_env(prepared_env)
    base_scores = score_observation(model, base_obs, device, replay.states, replay.rnn_states)

    rows: List[Dict[str, object]] = []
    images = []
    for texture in textures:
        env_copy = copy.deepcopy(prepared_env)
        texture_kind = set_target_texture(env_copy, target_delta, texture, valid_materials)
        obs = render_env(env_copy)
        scores = score_observation(model, obs, device, replay.states, replay.rnn_states)
        row = {
            "texture": texture,
            "texture_kind": texture_kind,
            "value": scores["value"],
            "delta_from_background": scores["value"] - base_scores["value"],
            "background_value": base_scores["value"],
            "target_delta": list(target_delta),
            "background_material": background_material,
            "inventory_edits": ";".join(edits_log),
            "noise_texture": noise_texture,
            "noise_delta": None if noise_texture is None else list(noise_delta),
            "visible_objects_removed": removed_objects,
            "visible_tiles_set_to_background": changed_tiles,
        }
        if "health_value" in scores:
            row["health_value"] = scores["health_value"]
        if "achievement_value" in scores:
            row["achievement_value"] = scores["achievement_value"]
        rows.append(row)
        images.append((texture, scores["value"], row["delta_from_background"], obs))
        env_copy.close()

    prepared_env.close()
    rows.sort(key=lambda row: float(row["value"]), reverse=True)
    return rows, images, base_scores, base_obs


def write_csv(path: str, rows: Sequence[Dict[str, object]]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_montage(path: str, images, base_obs, base_value: float, cols: int = 4):
    panels = [("all_grass", base_value, 0.0, base_obs)] + list(images)
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.0 * cols, 4.4 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, (name, value, delta, obs) in zip(axes, panels):
        ax.imshow(obs)
        ax.set_title(f"{name}\nV={value:.3f}  d={delta:+.3f}", fontsize=11)
        ax.axis("off")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Probe value preference for one texture tile near the Crafter player.")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--step_id", type=int, default=0)
    parser.add_argument("--target_delta", type=str, default="0,1", help="Relative dx,dy tile to edit. Default is below.")
    parser.add_argument("--background_material", type=str, default="grass")
    parser.add_argument("--set_inventory", type=str, default=None, help="Comma-separated item=value assignments.")
    parser.add_argument(
        "--materials",
        type=str,
        default="all_textures",
        help=(
            "Comma-separated material/object texture names to test. "
            "Use 'all' for materials only, or 'all_textures' for materials plus spawnable object sprites."
        ),
    )
    parser.add_argument(
        "--noise_texture",
        type=str,
        default=None,
        help="Optional extra texture placed at --noise_delta before sweeping target textures.",
    )
    parser.add_argument("--noise_delta", type=str, default="1,1", help="Relative dx,dy for optional noise texture.")
    parser.add_argument("--keep_objects", action="store_true", help="Do not clear visible non-player objects.")
    parser.add_argument("--output_dir", type=str, default="material_value_probe")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)
    print(f"Loaded checkpoint: {ckpt_path}", flush=True)

    replay = replay_to_step(
        model=model,
        config=config,
        eval_seed=args.eval_seed,
        target_episode=args.episode_id,
        target_step=args.step_id,
        device=device,
    )
    valid_materials = material_names(replay.env)
    target_delta = parse_vec2(args.target_delta)
    noise_delta = parse_vec2(args.noise_delta)
    inventory_updates = parse_inventory_assignments(args.set_inventory)
    if args.background_material not in valid_materials:
        raise ValueError(f"Unknown background material '{args.background_material}'. Valid: {valid_materials}")
    textures = parse_texture_list(args.materials, valid_materials)
    valid_textures = set(valid_materials) | set(valid_spawn_objects())
    unknown = sorted(set(textures) - valid_textures)
    if args.noise_texture is not None and args.noise_texture not in valid_textures:
        unknown.append(args.noise_texture)
    if unknown:
        raise ValueError(
            f"Unknown textures {sorted(set(unknown))}. "
            f"Valid materials: {valid_materials}. Valid objects: {valid_spawn_objects()}"
        )

    rows, images, base_scores, base_obs = score_textures(
        model=model,
        replay=replay,
        device=device,
        textures=textures,
        target_delta=target_delta,
        background_material=args.background_material,
        clear_objects=not args.keep_objects,
        inventory_updates=inventory_updates,
        noise_texture=args.noise_texture,
        noise_delta=noise_delta,
    )

    csv_path = os.path.join(args.output_dir, "texture_values.csv")
    summary_path = os.path.join(args.output_dir, "summary.json")
    montage_path = os.path.join(args.output_dir, "texture_values.png")
    write_csv(csv_path, rows)
    save_montage(montage_path, images, base_obs, base_scores["value"])
    summary = {
        "checkpoint": ckpt_path,
        "eval_seed": args.eval_seed,
        "episode_id": args.episode_id,
        "step_id": args.step_id,
        "target_delta": list(target_delta),
        "background_material": args.background_material,
        "inventory_updates": inventory_updates,
        "noise_texture": args.noise_texture,
        "noise_delta": None if args.noise_texture is None else list(noise_delta),
        "background_value": base_scores["value"],
        "ranked_textures": rows,
        "csv_path": csv_path,
        "montage_path": montage_path,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    replay.env.close()
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
