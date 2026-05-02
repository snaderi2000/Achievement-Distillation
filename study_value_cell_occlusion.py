import argparse
import copy
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from collect_value_map import load_model, set_seed
from counterfactual_env_editor import (
    crafter_inventory_rows,
    get_hidsize,
    obs_to_tensor,
    replay_to_step,
    score_observation,
)


@dataclass
class CandidateState:
    episode_id: int
    step_id: int
    obs: np.ndarray
    env: object
    states: th.Tensor
    rnn_states: Optional[th.Tensor]
    base_value: float


def visible_grid_shape(env) -> Tuple[int, int]:
    grid = np.asarray(env._local_view._grid, dtype=np.int64)
    return int(grid[0]), int(grid[1])


def cell_pixel_bounds(obs: np.ndarray, grid_shape: Tuple[int, int], ix: int, iy: int) -> Tuple[int, int, int, int]:
    inventory_rows = crafter_inventory_rows(size=obs.shape[:2])
    cell_w = obs.shape[1] // grid_shape[0]
    cell_h = (obs.shape[0] - inventory_rows) // grid_shape[1]
    x0 = ix * cell_w
    x1 = min((ix + 1) * cell_w, cell_w * grid_shape[0])
    y0 = iy * cell_h
    y1 = min((iy + 1) * cell_h, cell_h * grid_shape[1])
    return x0, x1, y0, y1


def inventory_items() -> List[str]:
    try:
        from crafter import constants as crafter_constants

        return list(crafter_constants.items)
    except Exception:
        return [
            "health",
            "food",
            "drink",
            "energy",
            "wood",
            "stone",
            "coal",
            "iron",
            "diamond",
            "sapling",
            "wood_pickaxe",
            "stone_pickaxe",
            "iron_pickaxe",
            "wood_sword",
            "stone_sword",
            "iron_sword",
        ]


def inventory_slot_grid(obs: np.ndarray) -> Tuple[int, int, int, int]:
    items = inventory_items()
    view_cols = 9
    rows = int(np.ceil(len(items) / view_cols))
    inventory_rows = crafter_inventory_rows(size=obs.shape[:2])
    slot_w = obs.shape[1] // view_cols
    slot_h = max(inventory_rows // rows, 1)
    y_start = obs.shape[0] - inventory_rows
    return view_cols, rows, slot_w, slot_h, y_start


def inventory_slot_bounds(obs: np.ndarray, slot_idx: int) -> Tuple[int, int, int, int]:
    cols, rows, slot_w, slot_h, y_start = inventory_slot_grid(obs)
    row = slot_idx // cols
    col = slot_idx % cols
    x0 = col * slot_w
    x1 = min((col + 1) * slot_w, obs.shape[1])
    y0 = y_start + row * slot_h
    y1 = min(y_start + (row + 1) * slot_h, obs.shape[0])
    return x0, x1, y0, y1


def occlusion_fill(obs: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    inventory_rows = crafter_inventory_rows(size=obs.shape[:2])
    world = obs[:-inventory_rows]
    if mode == "black":
        return np.zeros(3, dtype=np.uint8)
    if mode == "gray":
        return np.full(3, 127, dtype=np.uint8)
    if mode == "world_mean":
        return np.clip(np.rint(world.reshape(-1, 3).mean(axis=0)), 0, 255).astype(np.uint8)
    if mode == "noise":
        return rng.integers(0, 256, size=3, dtype=np.uint8)
    raise ValueError("--occlusion_mode must be one of: black, gray, world_mean, noise.")


def occlude_cell(
    obs: np.ndarray,
    grid_shape: Tuple[int, int],
    ix: int,
    iy: int,
    mode: str,
    rng: np.random.Generator,
) -> np.ndarray:
    occluded = obs.copy()
    x0, x1, y0, y1 = cell_pixel_bounds(obs, grid_shape, ix, iy)
    fill = occlusion_fill(obs, mode, rng)
    if mode == "noise":
        occluded[y0:y1, x0:x1] = rng.integers(0, 256, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)
    else:
        occluded[y0:y1, x0:x1] = fill
    return occluded


def occlude_inventory_slot(obs: np.ndarray, slot_idx: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    occluded = obs.copy()
    x0, x1, y0, y1 = inventory_slot_bounds(obs, slot_idx)
    fill = occlusion_fill(obs, mode, rng)
    if mode == "noise":
        occluded[y0:y1, x0:x1] = rng.integers(0, 256, size=(y1 - y0, x1 - x0, 3), dtype=np.uint8)
    else:
        occluded[y0:y1, x0:x1] = fill
    return occluded


def world_cell_metadata(env, ix: int, iy: int) -> Dict[str, object]:
    grid = np.asarray(env._local_view._grid, dtype=np.int64)
    center_index = grid // 2
    center_world = np.asarray(env._player.pos, dtype=np.int64)
    delta = np.array([ix, iy], dtype=np.int64) - center_index
    pos = center_world + delta
    material, obj = env._world[tuple(pos)]
    return {
        "region": "world",
        "cell_ix": ix,
        "cell_iy": iy,
        "inventory_slot": None,
        "inventory_item": None,
        "delta_x": int(delta[0]),
        "delta_y": int(delta[1]),
        "world_x": int(pos[0]),
        "world_y": int(pos[1]),
        "material": material,
        "object": None if obj is None else type(obj).__name__,
        "is_player_cell": bool(obj is env._player),
    }


def inventory_slot_metadata(slot_idx: int, item_name: str) -> Dict[str, object]:
    cols = 9
    return {
        "region": "inventory",
        "cell_ix": None,
        "cell_iy": None,
        "inventory_slot": slot_idx,
        "inventory_item": item_name,
        "delta_x": None,
        "delta_y": None,
        "world_x": None,
        "world_y": None,
        "material": None,
        "object": None,
        "is_player_cell": False,
        "inventory_col": slot_idx % cols,
        "inventory_row": slot_idx // cols,
    }


def collect_candidate_states(args, model, config, device) -> List[CandidateState]:
    from crafter.env import Env

    if args.episode_id is not None or args.step_id is not None:
        if args.episode_id is None or args.step_id is None:
            raise ValueError("--episode_id and --step_id must be provided together.")
        replay = replay_to_step(
            model,
            config,
            args.eval_seed,
            args.episode_id,
            args.step_id,
            device,
        )
        scores = score_observation(model, replay.obs, device, replay.states, replay.rnn_states)
        return [
            CandidateState(
                episode_id=args.episode_id,
                step_id=args.step_id,
                obs=replay.obs.copy(),
                env=replay.env,
                states=replay.states.clone(),
                rnn_states=None if replay.rnn_states is None else replay.rnn_states.clone(),
                base_value=scores["value"],
            )
        ]

    env = Env(seed=args.eval_seed)
    hidsize = get_hidsize(config)
    rnn_hidsize = config.get("model_kwargs", {}).get("rnn_hidsize")
    candidates: List[CandidateState] = []

    for episode_idx in range(args.num_episodes):
        obs = env.reset()
        states = th.zeros(1, hidsize, device=device)
        rnn_states = None
        if rnn_hidsize is not None:
            rnn_states = th.zeros(1, int(rnn_hidsize), device=device)

        step_idx = 0
        while True:
            should_sample = (
                step_idx >= args.start_step
                and (step_idx - args.start_step) % args.step_stride == 0
                and (args.max_step is None or step_idx <= args.max_step)
            )
            if should_sample:
                scores = score_observation(model, obs, device, states.clone(), None if rnn_states is None else rnn_states.clone())
                candidates.append(
                    CandidateState(
                        episode_id=episode_idx,
                        step_id=step_idx,
                        obs=obs.copy(),
                        env=copy.deepcopy(env),
                        states=states.clone(),
                        rnn_states=None if rnn_states is None else rnn_states.clone(),
                        base_value=scores["value"],
                    )
                )
                if len(candidates) % args.print_every == 0:
                    print(
                        f"sampled {len(candidates)} states "
                        f"(episode={episode_idx}, step={step_idx}, value={scores['value']:.3f})",
                        flush=True,
                    )
                if args.max_states is not None and len(candidates) >= args.max_states:
                    break

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
            if done or (args.max_step is not None and step_idx > args.max_step):
                break
            if args.max_states is not None and len(candidates) >= args.max_states:
                break

        if args.max_states is not None and len(candidates) >= args.max_states:
            break

    env.close()
    candidates.sort(key=lambda state: state.base_value, reverse=True)
    return candidates[: args.top_k]


def score_cell_occlusions(
    candidate: CandidateState,
    model,
    device,
    occlusion_mode: str,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
    grid_shape = visible_grid_shape(candidate.env)
    deltas = np.zeros((grid_shape[1], grid_shape[0]), dtype=np.float32)
    items = inventory_items()
    inventory_cols, inventory_rows, _, _, _ = inventory_slot_grid(candidate.obs)
    inventory_deltas = np.full((inventory_rows, inventory_cols), np.nan, dtype=np.float32)
    rows: List[Dict[str, object]] = []
    for iy in range(grid_shape[1]):
        for ix in range(grid_shape[0]):
            occluded = occlude_cell(candidate.obs, grid_shape, ix, iy, occlusion_mode, rng)
            value = score_observation(
                model,
                occluded,
                device,
                candidate.states,
                candidate.rnn_states,
            )["value"]
            delta = value - candidate.base_value
            deltas[iy, ix] = delta
            row = {
                "episode_id": candidate.episode_id,
                "step_id": candidate.step_id,
                "base_value": candidate.base_value,
                "occluded_value": value,
                "delta": delta,
                "abs_delta": abs(delta),
            }
            row.update(world_cell_metadata(candidate.env, ix, iy))
            rows.append(row)
    for slot_idx, item_name in enumerate(items):
        occluded = occlude_inventory_slot(candidate.obs, slot_idx, occlusion_mode, rng)
        value = score_observation(
            model,
            occluded,
            device,
            candidate.states,
            candidate.rnn_states,
        )["value"]
        delta = value - candidate.base_value
        slot_row = slot_idx // inventory_cols
        slot_col = slot_idx % inventory_cols
        inventory_deltas[slot_row, slot_col] = delta
        row = {
            "episode_id": candidate.episode_id,
            "step_id": candidate.step_id,
            "base_value": candidate.base_value,
            "occluded_value": value,
            "delta": delta,
            "abs_delta": abs(delta),
        }
        row.update(inventory_slot_metadata(slot_idx, item_name))
        rows.append(row)
    return deltas, inventory_deltas, rows


def draw_grid(ax, obs: np.ndarray, grid_shape: Tuple[int, int]):
    inventory_rows = crafter_inventory_rows(size=obs.shape[:2])
    cell_w = obs.shape[1] // grid_shape[0]
    cell_h = (obs.shape[0] - inventory_rows) // grid_shape[1]
    world_w = cell_w * grid_shape[0]
    world_h = cell_h * grid_shape[1]
    for ix in range(grid_shape[0] + 1):
        x = ix * cell_w
        ax.plot([x, x], [0, world_h], color="white", linewidth=0.45, alpha=0.7)
    for iy in range(grid_shape[1] + 1):
        y = iy * cell_h
        ax.plot([0, grid_shape[0] * cell_w], [y, y], color="white", linewidth=0.45, alpha=0.7)


def draw_inventory_grid(ax, obs: np.ndarray):
    cols, rows, slot_w, slot_h, y_start = inventory_slot_grid(obs)
    for col in range(cols + 1):
        x = col * slot_w
        ax.plot([x, x], [y_start, min(y_start + rows * slot_h, obs.shape[0])], color="white", linewidth=0.45, alpha=0.7)
    for row in range(rows + 1):
        y = y_start + row * slot_h
        ax.plot([0, cols * slot_w], [y, y], color="white", linewidth=0.45, alpha=0.7)


def save_heatmap(candidate: CandidateState, deltas: np.ndarray, inventory_deltas: np.ndarray, output_path: str, title: str):
    grid_shape = visible_grid_shape(candidate.env)
    inventory_rows = crafter_inventory_rows(size=candidate.obs.shape[:2])
    cell_w = candidate.obs.shape[1] // grid_shape[0]
    cell_h = (candidate.obs.shape[0] - inventory_rows) // grid_shape[1]
    world_w = cell_w * grid_shape[0]
    world_h = cell_h * grid_shape[1]
    full_heatmap = np.full(candidate.obs.shape[:2], np.nan, dtype=np.float32)
    world_heatmap = np.repeat(np.repeat(deltas, cell_h, axis=0), cell_w, axis=1)
    full_heatmap[:world_h, :world_w] = world_heatmap[:world_h, :world_w]

    items = inventory_items()
    for slot_idx, _ in enumerate(items):
        row = slot_idx // inventory_deltas.shape[1]
        col = slot_idx % inventory_deltas.shape[1]
        if row >= inventory_deltas.shape[0] or np.isnan(inventory_deltas[row, col]):
            continue
        x0, x1, y0, y1 = inventory_slot_bounds(candidate.obs, slot_idx)
        full_heatmap[y0:y1, x0:x1] = inventory_deltas[row, col]

    vmax = max(float(np.nanmax(np.abs(full_heatmap))), 1e-6)
    masked_heatmap = np.ma.masked_invalid(full_heatmap)

    fig, ax = plt.subplots(1, 1, figsize=(6.4, 6.4), constrained_layout=True)
    ax.imshow(candidate.obs)
    im = ax.imshow(masked_heatmap, cmap="coolwarm", alpha=0.62, vmin=-vmax, vmax=vmax)
    ax.set_title(f"{title}\nV={candidate.base_value:.3f}")
    ax.axis("off")
    draw_grid(ax, candidate.obs, grid_shape)
    draw_inventory_grid(ax, candidate.obs)
    fig.colorbar(im, ax=ax, shrink=0.78, label="V(occluded) - V(base)")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(path: str, rows: List[Dict[str, object]]):
    if not rows:
        return
    fieldnames = sorted({field for row in rows for field in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Cell-level occlusion heatmaps for Crafter value functions.")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument("--episode_id", type=int, default=None, help="Analyze one exact replayed episode. Use with --step_id.")
    parser.add_argument("--step_id", type=int, default=None, help="Analyze one exact replayed step. Use with --episode_id.")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--step_stride", type=int, default=20)
    parser.add_argument("--max_states", type=int, default=100)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument(
        "--occlusion_mode",
        choices=["black", "gray", "world_mean", "noise"],
        default="world_mean",
        help="How to replace the removed visible map cell.",
    )
    parser.add_argument("--output_dir", type=str, default="value_cell_occlusion")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)
    print(f"Loaded checkpoint: {ckpt_path}", flush=True)

    rng = np.random.default_rng(args.eval_seed)
    candidates = collect_candidate_states(args, model, config, device)
    if not candidates:
        raise ValueError("No candidate states sampled.")

    all_rows: List[Dict[str, object]] = []
    summary_rows = []
    for rank, candidate in enumerate(candidates, start=1):
        stem = f"rank{rank:03d}-ep{candidate.episode_id:03d}-step{candidate.step_id:04d}"
        deltas, inventory_deltas, rows = score_cell_occlusions(candidate, model, device, args.occlusion_mode, rng)
        all_rows.extend(rows)
        heatmap_path = os.path.join(args.output_dir, f"{stem}-cell-occlusion.png")
        save_heatmap(
            candidate,
            deltas,
            inventory_deltas,
            heatmap_path,
            title=f"{args.occlusion_mode} occlusion\nrank {rank}, ep {candidate.episode_id}, step {candidate.step_id}",
        )
        np.save(os.path.join(args.output_dir, f"{stem}-deltas.npy"), deltas)
        np.save(os.path.join(args.output_dir, f"{stem}-inventory-deltas.npy"), inventory_deltas)
        summary_rows.append(
            {
                "rank": rank,
                "episode_id": candidate.episode_id,
                "step_id": candidate.step_id,
                "base_value": candidate.base_value,
                "mean_abs_delta": float(np.mean(np.abs(deltas))),
                "max_abs_delta": float(np.max(np.abs(deltas))),
                "inventory_mean_abs_delta": float(np.nanmean(np.abs(inventory_deltas))),
                "inventory_max_abs_delta": float(np.nanmax(np.abs(inventory_deltas))),
                "min_delta": float(np.min(deltas)),
                "max_delta": float(np.max(deltas)),
                "heatmap_path": heatmap_path,
            }
        )
        candidate.env.close()
        print(
            f"saved {heatmap_path} "
            f"(base_value={candidate.base_value:.3f}, max_abs_delta={summary_rows[-1]['max_abs_delta']:.3f})",
            flush=True,
        )

    write_csv(os.path.join(args.output_dir, "cell_occlusion_rows.csv"), all_rows)
    write_csv(os.path.join(args.output_dir, "state_summary.csv"), summary_rows)
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(
            {
                "checkpoint": ckpt_path,
                "eval_seed": args.eval_seed,
                "occlusion_mode": args.occlusion_mode,
                "num_states": len(candidates),
                "states": summary_rows,
            },
            f,
            indent=2,
        )
    print(f"Wrote occlusion outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
