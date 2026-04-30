import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from collect_value_map import load_model, observation_to_uint8_hwc, set_seed


@dataclass
class ReplayState:
    env: object
    obs: np.ndarray
    states: th.Tensor
    rnn_states: Optional[th.Tensor]
    step_id: int
    episode_id: int


def parse_inventory_assignments(text: Optional[str]) -> Dict[str, int]:
    if not text:
        return {}
    assignments: Dict[str, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Invalid inventory assignment '{item}'. Expected item=value.")
        name, value = item.split("=", 1)
        assignments[name.strip()] = int(value.strip())
    return assignments


def parse_vec2(text: str) -> Tuple[int, int]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Invalid coordinate '{text}'. Expected x,y.")
    return int(parts[0]), int(parts[1])


def parse_material_edit(text: str) -> Tuple[int, int, str]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid material edit '{text}'. Expected dx,dy,material.")
    return int(parts[0]), int(parts[1]), parts[2]


def parse_object_edit(text: str) -> Tuple[int, int, str]:
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Invalid object edit '{text}'. Expected dx,dy,object.")
    return int(parts[0]), int(parts[1]), parts[2]


def obs_to_tensor(obs: np.ndarray, device: th.device) -> th.Tensor:
    obs = th.from_numpy(np.transpose(obs, (2, 0, 1))).unsqueeze(0).to(device)
    return obs.float() / 255.0


def render_env(env) -> np.ndarray:
    return env.render()


def rotate_world_observation(obs: np.ndarray, degrees: int, inventory_rows: int) -> np.ndarray:
    if degrees % 360 == 0:
        return obs.copy()
    if degrees % 360 != 180:
        raise ValueError(
            "--rotate_world_degrees currently supports 180 only. "
            "The Crafter world panel is rectangular once the HUD rows are excluded, so 90/270 need a crop/pad policy."
        )
    if inventory_rows <= 0 or inventory_rows >= obs.shape[0]:
        raise ValueError(f"Invalid inventory_rows={inventory_rows} for observation height {obs.shape[0]}.")
    rotated = obs.copy()
    rotated[:-inventory_rows] = np.rot90(obs[:-inventory_rows], k=2)
    return rotated


def get_hidsize(config: Dict) -> int:
    return int(config.get("model_kwargs", {}).get("hidsize", 512))


def score_observation(
    model,
    obs: np.ndarray,
    device: th.device,
    states: Optional[th.Tensor] = None,
    rnn_states: Optional[th.Tensor] = None,
) -> Dict[str, float]:
    obs_tensor = obs_to_tensor(obs, device)
    kwargs = {}
    if states is not None:
        kwargs["states"] = states
    if rnn_states is not None:
        kwargs["rnn_states"] = rnn_states
    with th.no_grad():
        outputs = model.act(obs_tensor, **kwargs)
    result = {
        "value": float(outputs["vpreds"].item()),
    }
    if "health_vpreds" in outputs:
        result["health_value"] = float(outputs["health_vpreds"].item())
    if "achievement_vpreds" in outputs:
        result["achievement_value"] = float(outputs["achievement_vpreds"].item())
    return result


def rnn_state_norm(rnn_states: Optional[th.Tensor]) -> Optional[float]:
    if rnn_states is None:
        return None
    return float(th.linalg.vector_norm(rnn_states).item())


def replay_to_step(
    model,
    config: Dict,
    eval_seed: int,
    target_episode: int,
    target_step: int,
    device: th.device,
) -> ReplayState:
    from crafter.env import Env

    env = Env(seed=eval_seed)
    hidsize = get_hidsize(config)
    rnn_hidsize = config.get("model_kwargs", {}).get("rnn_hidsize")

    for episode_idx in range(target_episode + 1):
        obs = env.reset()
        states = th.zeros(1, hidsize, device=device)
        rnn_states = None
        if rnn_hidsize is not None:
            rnn_states = th.zeros(1, int(rnn_hidsize), device=device)
        step_idx = 0

        while True:
            if episode_idx == target_episode and step_idx == target_step:
                return ReplayState(
                    env=env,
                    obs=obs,
                    states=states.clone(),
                    rnn_states=None if rnn_states is None else rnn_states.clone(),
                    step_id=step_idx,
                    episode_id=episode_idx,
                )

            obs_tensor = obs_to_tensor(obs, device)
            act_kwargs = {"states": states}
            if rnn_states is not None:
                act_kwargs["rnn_states"] = rnn_states
            with th.no_grad():
                outputs = model.act(obs_tensor, **act_kwargs)
                action = int(outputs["actions"].item())
                if "next_states" in outputs:
                    states = outputs["next_states"]
                if "next_rnn_states" in outputs:
                    rnn_states = outputs["next_rnn_states"]

            obs, _, done, _ = env.step(action)
            step_idx += 1
            if done:
                break

    env.close()
    raise ValueError(f"Could not reach episode={target_episode}, step={target_step}.")


def world_pos_from_delta(env, dx: int, dy: int) -> np.ndarray:
    player_pos = np.array(env._player.pos, dtype=np.int64)
    pos = player_pos + np.array([dx, dy], dtype=np.int64)
    world_area = np.array(env._world.area, dtype=np.int64)
    if not (0 <= pos[0] < world_area[0] and 0 <= pos[1] < world_area[1]):
        raise ValueError(f"Target world position {tuple(pos)} is out of bounds for area {tuple(world_area)}.")
    return pos


def material_names(env) -> List[str]:
    mat_ids = getattr(env._world, "_mat_ids", {})
    return sorted(name for name in mat_ids.keys() if isinstance(name, str))


def visible_world_positions(env) -> List[Tuple[Tuple[int, int], Tuple[int, int], str]]:
    local_grid = np.asarray(env._local_view._grid, dtype=np.int64)
    offset = local_grid // 2
    center = np.asarray(env._player.pos, dtype=np.int64)
    world_area = np.array(env._world.area, dtype=np.int64)

    positions = []
    for dx in range(-int(offset[0]), int(offset[0]) + 1):
        for dy in range(-int(offset[1]), int(offset[1]) + 1):
            pos = center + np.array([dx, dy], dtype=np.int64)
            if 0 <= pos[0] < world_area[0] and 0 <= pos[1] < world_area[1]:
                material, _ = env._world[tuple(pos)]
                positions.append(((dx, dy), (int(pos[0]), int(pos[1])), material))
    return positions


def apply_inventory_edits(env, inventory_updates: Dict[str, int], edits_log: List[str]):
    if not inventory_updates:
        return
    items = getattr(env._player, "inventory", None)
    if items is None:
        raise RuntimeError("Crafter player inventory is unavailable.")
    for name, value in inventory_updates.items():
        if name not in items:
            raise ValueError(f"Unknown inventory item '{name}'. Known items include: {sorted(items.keys())}")
        items[name] = value
        if name == "health":
            env._player.health = value
        edits_log.append(f"inventory[{name}]={value}")


def apply_health_edit(env, value: Optional[int], edits_log: List[str]):
    if value is None:
        return
    env._player.health = value
    env._player.inventory["health"] = value
    edits_log.append(f"health={value}")


def apply_daylight_edit(env, value: Optional[float], edits_log: List[str]):
    if value is None:
        return
    env._world.daylight = float(value)
    edits_log.append(f"daylight={float(value):.3f}")


def apply_move_player(env, move_delta: Optional[Tuple[int, int]], absolute_pos: Optional[Tuple[int, int]], edits_log: List[str]):
    if move_delta is not None and absolute_pos is not None:
        raise ValueError("Use only one of --move_player or --set_player_pos.")
    if absolute_pos is not None:
        new_pos = np.array(absolute_pos, dtype=np.int64)
    elif move_delta is not None:
        new_pos = world_pos_from_delta(env, move_delta[0], move_delta[1])
    else:
        return

    material, obj = env._world[tuple(new_pos)]
    if obj is not None and obj is not env._player:
        raise ValueError(f"Cannot move player onto occupied tile {tuple(new_pos)} containing {type(obj).__name__}.")
    if material not in env._player.walkable:
        raise ValueError(f"Cannot move player onto non-walkable material '{material}' at {tuple(new_pos)}.")
    env._world.move(env._player, new_pos)
    edits_log.append(f"player_pos={tuple(int(x) for x in new_pos)}")


def apply_clear_object(env, clear_specs: Sequence[Tuple[int, int]], edits_log: List[str]):
    for dx, dy in clear_specs:
        pos = world_pos_from_delta(env, dx, dy)
        _, obj = env._world[tuple(pos)]
        if obj is None or obj is env._player:
            edits_log.append(f"clear_object@({dx},{dy})=none")
            continue
        env._world.remove(obj)
        edits_log.append(f"clear_object@({dx},{dy})={type(obj).__name__}")


def apply_material_edits(env, material_specs: Sequence[Tuple[int, int, str]], edits_log: List[str]):
    valid_materials = set(material_names(env))
    for dx, dy, material in material_specs:
        if material not in valid_materials:
            raise ValueError(f"Unknown material '{material}'. Valid materials: {sorted(valid_materials)}")
        pos = world_pos_from_delta(env, dx, dy)
        _, obj = env._world[tuple(pos)]
        if obj is env._player:
            raise ValueError(f"Cannot overwrite the player's current tile with material '{material}'.")
        env._world[tuple(pos)] = material
        edits_log.append(f"material@({dx},{dy})={material}")


def apply_replace_visible_material(env, source_material: Optional[str], target_material: Optional[str], edits_log: List[str]):
    if not source_material:
        return
    if not target_material:
        raise ValueError("--replace_visible_material_with is required when using --replace_visible_material.")

    valid_materials = set(material_names(env))
    if source_material not in valid_materials:
        raise ValueError(f"Unknown source material '{source_material}'. Valid materials: {sorted(valid_materials)}")
    if target_material not in valid_materials:
        raise ValueError(f"Unknown target material '{target_material}'. Valid materials: {sorted(valid_materials)}")

    replaced = 0
    for (dx, dy), world_pos, material in visible_world_positions(env):
        if material != source_material:
            continue
        _, obj = env._world[world_pos]
        if obj is env._player:
            continue
        env._world[world_pos] = target_material
        replaced += 1
        edits_log.append(f"material@({dx},{dy})={target_material}")

    edits_log.append(f"replace_visible_material {source_material}->{target_material} count={replaced}")


def valid_spawn_objects() -> List[str]:
    return ["cow", "zombie", "skeleton", "plant", "fence", "arrow_left", "arrow_right", "arrow_up", "arrow_down"]


def apply_spawn_object(env, object_specs: Sequence[Tuple[int, int, str]], edits_log: List[str]):
    if not object_specs:
        return

    from crafter.objects import Arrow, Cow, Fence, Plant, Skeleton, Zombie

    constructors = {
        "cow": lambda world, pos: Cow(world, pos),
        "zombie": lambda world, pos: Zombie(world, pos, env._player),
        "skeleton": lambda world, pos: Skeleton(world, pos, env._player),
        "plant": lambda world, pos: Plant(world, pos),
        "fence": lambda world, pos: Fence(world, pos),
        "arrow_left": lambda world, pos: Arrow(world, pos, np.array([-1, 0], dtype=np.int64)),
        "arrow_right": lambda world, pos: Arrow(world, pos, np.array([1, 0], dtype=np.int64)),
        "arrow_up": lambda world, pos: Arrow(world, pos, np.array([0, -1], dtype=np.int64)),
        "arrow_down": lambda world, pos: Arrow(world, pos, np.array([0, 1], dtype=np.int64)),
    }

    valid_materials = set(material_names(env))
    for dx, dy, object_name in object_specs:
        if object_name in valid_materials:
            pos = world_pos_from_delta(env, dx, dy)
            _, obj = env._world[tuple(pos)]
            if obj is env._player:
                raise ValueError(f"Cannot set the player's current tile to material '{object_name}'.")
            if obj is not None:
                env._world.remove(obj)
            env._world[tuple(pos)] = object_name
            edits_log.append(f"material@({dx},{dy})={object_name}")
            continue
        if object_name not in constructors:
            raise ValueError(
                f"Unknown spawn object or material '{object_name}'. "
                f"Valid objects: {valid_spawn_objects()}. Valid materials: {sorted(valid_materials)}"
            )
        pos = world_pos_from_delta(env, dx, dy)
        material, obj = env._world[tuple(pos)]
        if obj is env._player:
            raise ValueError(f"Cannot spawn '{object_name}' on the player's current tile.")
        if obj is not None:
            env._world.remove(obj)
        spawned = constructors[object_name](env._world, pos)
        if hasattr(spawned, "is_free") and not isinstance(spawned, Arrow):
            # For static creatures/objects, respect walkability on the underlying material.
            if material not in spawned.walkable:
                raise ValueError(
                    f"Cannot spawn '{object_name}' on non-walkable material '{material}' at {tuple(pos)}."
                )
        env._world.add(spawned)
        edits_log.append(f"spawn@({dx},{dy})={object_name}")


def save_side_by_side(
    base_obs: np.ndarray,
    edited_obs: np.ndarray,
    base_scores: Dict[str, float],
    edited_scores: Dict[str, float],
    output_path: str,
    title_suffix: str,
):
    def panel_title(prefix: str, scores: Dict[str, float]) -> str:
        lines = [f"{prefix}", f"value={scores['value']:.3f}"]
        if "health_value" in scores:
            lines.append(f"health={scores['health_value']:.3f}")
        if "achievement_value" in scores:
            lines.append(f"ach={scores['achievement_value']:.3f}")
        return "\n".join(lines)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.2))
    axes[0].imshow(base_obs)
    axes[0].set_title(panel_title("Base", base_scores))
    axes[0].axis("off")

    axes[1].imshow(edited_obs)
    axes[1].set_title(panel_title("Edited", edited_scores))
    axes[1].axis("off")

    fig.suptitle(title_suffix)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Edit live Crafter env state at a chosen replay step and re-score the value.")
    parser.add_argument("--exp_name", type=str, required=True, help="Checkpoint used for scoring unless replay_* overrides are provided.")
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, required=True)
    parser.add_argument("--replay_exp_name", type=str, default=None, help="Optional checkpoint used only to replay and recover the base scene.")
    parser.add_argument("--replay_timestamp", type=str, default=None)
    parser.add_argument("--replay_train_seed", type=int, default=None)
    parser.add_argument("--replay_ckpt_epoch", type=int, default=None)
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--episode_id", type=int, required=True)
    parser.add_argument("--step_id", type=int, required=True)
    parser.add_argument(
        "--score_rnn_state_episode_id",
        type=int,
        default=None,
        help="Optional donor episode whose recurrent state is used only for scoring.",
    )
    parser.add_argument(
        "--score_rnn_state_step_id",
        type=int,
        default=None,
        help="Optional donor step whose recurrent state is used only for scoring.",
    )

    parser.add_argument("--set_health", type=int, default=None)
    parser.add_argument("--set_inventory", type=str, default=None, help="Comma-separated item=value assignments.")
    parser.add_argument("--move_player", type=str, default=None, help="Relative move dx,dy from current player position.")
    parser.add_argument("--set_player_pos", type=str, default=None, help="Absolute world position x,y.")
    parser.add_argument("--set_daylight", type=float, default=None)
    parser.add_argument(
        "--rotate_world_degrees",
        type=int,
        default=0,
        help="Rotate only the visible world pixels in observation space. Inventory/HUD rows are left unchanged.",
    )
    parser.add_argument(
        "--inventory_rows",
        type=int,
        default=16,
        help="Number of bottom observation rows treated as inventory/HUD for observation-space edits.",
    )
    parser.add_argument(
        "--clear_object",
        action="append",
        default=[],
        help="Remove an object at relative visible coordinate dx,dy. Can be repeated.",
    )
    parser.add_argument(
        "--set_material",
        action="append",
        default=[],
        help="Set material at relative visible coordinate dx,dy,material. Can be repeated.",
    )
    parser.add_argument(
        "--replace_visible_material",
        type=str,
        default=None,
        help="Replace every visible tile of this material with --replace_visible_material_with.",
    )
    parser.add_argument(
        "--replace_visible_material_with",
        type=str,
        default=None,
        help="Target material used by --replace_visible_material.",
    )
    parser.add_argument(
        "--spawn_object",
        action="append",
        default=[],
        help=(
            "Spawn a real Crafter object at relative visible coordinate dx,dy,object. "
            f"Valid objects: {', '.join(valid_spawn_objects())}. Can be repeated."
        ),
    )
    parser.add_argument("--output_dir", type=str, default="counterfactual_env_editor")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    score_model, score_config, score_ckpt_path = load_model(
        args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device
    )
    print(f"Loaded scoring checkpoint: {score_ckpt_path}")

    replay_exp_name = args.replay_exp_name or args.exp_name
    replay_timestamp = args.replay_timestamp or args.timestamp
    replay_train_seed = args.replay_train_seed if args.replay_train_seed is not None else args.train_seed
    replay_ckpt_epoch = args.replay_ckpt_epoch if args.replay_ckpt_epoch is not None else args.ckpt_epoch

    if (
        replay_exp_name == args.exp_name
        and replay_timestamp == args.timestamp
        and replay_train_seed == args.train_seed
        and replay_ckpt_epoch == args.ckpt_epoch
    ):
        replay_model = score_model
        replay_config = score_config
        replay_ckpt_path = score_ckpt_path
    else:
        replay_model, replay_config, replay_ckpt_path = load_model(
            replay_exp_name,
            replay_timestamp,
            replay_train_seed,
            replay_ckpt_epoch,
            device,
        )
    print(f"Loaded replay checkpoint: {replay_ckpt_path}")

    replay = replay_to_step(
        model=replay_model,
        config=replay_config,
        eval_seed=args.eval_seed,
        target_episode=args.episode_id,
        target_step=args.step_id,
        device=device,
    )
    base_score_states = replay.states
    base_score_rnn_states = replay.rnn_states
    edited_score_states = replay.states
    edited_score_rnn_states = replay.rnn_states
    donor_state = None
    if args.score_rnn_state_episode_id is not None or args.score_rnn_state_step_id is not None:
        if args.score_rnn_state_episode_id is None or args.score_rnn_state_step_id is None:
            raise ValueError(
                "Use both --score_rnn_state_episode_id and --score_rnn_state_step_id when swapping recurrent state."
            )
        donor_state = replay_to_step(
            model=score_model,
            config=score_config,
            eval_seed=args.eval_seed,
            target_episode=args.score_rnn_state_episode_id,
            target_step=args.score_rnn_state_step_id,
            device=device,
        )
        if donor_state.rnn_states is None:
            raise ValueError("Scoring model does not expose recurrent state; hidden-state swapping is unavailable.")
        edited_score_states = donor_state.states
        edited_score_rnn_states = donor_state.rnn_states

    base_obs = replay.obs.copy()
    base_scores = score_observation(score_model, base_obs, device, base_score_states, base_score_rnn_states)

    edits_log: List[str] = []
    if donor_state is not None:
        edits_log.append(
            f"score_rnn_state=episode{args.score_rnn_state_episode_id}:step{args.score_rnn_state_step_id}"
        )
    inventory_updates = parse_inventory_assignments(args.set_inventory)
    move_player = parse_vec2(args.move_player) if args.move_player else None
    set_player_pos = parse_vec2(args.set_player_pos) if args.set_player_pos else None
    clear_specs = [parse_vec2(text) for text in args.clear_object]
    material_specs = [parse_material_edit(text) for text in args.set_material]
    object_specs = [parse_object_edit(text) for text in args.spawn_object]

    apply_health_edit(replay.env, args.set_health, edits_log)
    apply_inventory_edits(replay.env, inventory_updates, edits_log)
    apply_daylight_edit(replay.env, args.set_daylight, edits_log)
    apply_move_player(replay.env, move_player, set_player_pos, edits_log)
    apply_clear_object(replay.env, clear_specs, edits_log)
    apply_material_edits(replay.env, material_specs, edits_log)
    apply_replace_visible_material(
        replay.env,
        args.replace_visible_material,
        args.replace_visible_material_with,
        edits_log,
    )
    apply_spawn_object(replay.env, object_specs, edits_log)

    edited_obs = render_env(replay.env)
    if args.rotate_world_degrees % 360 != 0:
        edited_obs = rotate_world_observation(edited_obs, args.rotate_world_degrees, args.inventory_rows)
        edits_log.append(f"rotate_world_degrees={args.rotate_world_degrees}")
    edited_scores = score_observation(score_model, edited_obs, device, edited_score_states, edited_score_rnn_states)

    os.makedirs(args.output_dir, exist_ok=True)
    stem = f"{args.exp_name}-e{args.episode_id:03d}-s{args.step_id:04d}"
    figure_path = os.path.join(args.output_dir, f"{stem}.png")
    summary_path = os.path.join(args.output_dir, f"{stem}.json")

    title_suffix = ", ".join(edits_log) if edits_log else "No edits"
    save_side_by_side(base_obs, edited_obs, base_scores, edited_scores, figure_path, title_suffix)

    summary = {
        "scoring_checkpoint": score_ckpt_path,
        "replay_checkpoint": replay_ckpt_path,
        "eval_seed": args.eval_seed,
        "episode_id": args.episode_id,
        "step_id": args.step_id,
        "base_value": base_scores["value"],
        "edited_value": edited_scores["value"],
        "delta_value": edited_scores["value"] - base_scores["value"],
        "edits": edits_log,
        "score_state_source": (
            {
                "episode_id": args.score_rnn_state_episode_id,
                "step_id": args.score_rnn_state_step_id,
                "rnn_state_norm": rnn_state_norm(edited_score_rnn_states),
            }
            if donor_state is not None
            else {
                "episode_id": args.episode_id,
                "step_id": args.step_id,
                "rnn_state_norm": rnn_state_norm(edited_score_rnn_states),
            }
        ),
        "base_state_source": {
            "episode_id": args.episode_id,
            "step_id": args.step_id,
            "rnn_state_norm": rnn_state_norm(base_score_rnn_states),
        },
        "player_pos_after": tuple(int(x) for x in replay.env._player.pos),
        "inventory_after": dict(replay.env._player.inventory),
        "daylight_after": float(replay.env._world.daylight),
        "inventory_rows_for_obs_edits": int(args.inventory_rows),
        "figure_path": figure_path,
    }
    if "health_value" in base_scores and "health_value" in edited_scores:
        summary["base_health_value"] = base_scores["health_value"]
        summary["edited_health_value"] = edited_scores["health_value"]
        summary["delta_health_value"] = edited_scores["health_value"] - base_scores["health_value"]
    if "achievement_value" in base_scores and "achievement_value" in edited_scores:
        summary["base_achievement_value"] = base_scores["achievement_value"]
        summary["edited_achievement_value"] = edited_scores["achievement_value"]
        summary["delta_achievement_value"] = edited_scores["achievement_value"] - base_scores["achievement_value"]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    replay.env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
