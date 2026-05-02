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
    world_h = obs.shape[0] - inventory_rows
    world_w = obs.shape[1]
    cell_w = world_w // grid_shape[0]
    cell_h = world_h // grid_shape[1]
    x0 = ix * cell_w
    x1 = min((ix + 1) * cell_w, world_w)
    y0 = iy * cell_h
    y1 = min((iy + 1) * cell_h, world_h)
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


def world_cell_metadata(env, ix: int, iy: int) -> Dict[str, object]:
    grid = np.asarray(env._local_view._grid, dtype=np.int64)
    center_index = grid // 2
    center_world = np.asarray(env._player.pos, dtype=np.int64)
    delta = np.array([ix, iy], dtype=np.int64) - center_index
    pos = center_world + delta
    material, obj = env._world[tuple(pos)]
    return {
        "cell_ix": ix,
        "cell_iy": iy,
        "delta_x": int(delta[0]),
        "delta_y": int(delta[1]),
        "world_x": int(pos[0]),
        "world_y": int(pos[1]),
        "material": material,
        "object": None if obj is None else type(obj).__name__,
        "is_player_cell": bool(obj is env._player),
    }


def collect_candidate_states(args, model, config, device) -> List[CandidateState]:
    from crafter.env import Env

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
) -> Tuple[np.ndarray, List[Dict[str, object]]]:
    grid_shape = visible_grid_shape(candidate.env)
    deltas = np.zeros((grid_shape[1], grid_shape[0]), dtype=np.float32)
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
    return deltas, rows


def draw_grid(ax, obs: np.ndarray, grid_shape: Tuple[int, int]):
    inventory_rows = crafter_inventory_rows(size=obs.shape[:2])
    world_h = obs.shape[0] - inventory_rows
    world_w = obs.shape[1]
    cell_w = world_w // grid_shape[0]
    cell_h = world_h // grid_shape[1]
    for ix in range(grid_shape[0] + 1):
        x = ix * cell_w
        ax.plot([x, x], [0, world_h], color="white", linewidth=0.45, alpha=0.7)
    for iy in range(grid_shape[1] + 1):
        y = iy * cell_h
        ax.plot([0, grid_shape[0] * cell_w], [y, y], color="white", linewidth=0.45, alpha=0.7)


def save_heatmap(candidate: CandidateState, deltas: np.ndarray, output_path: str, title: str):
    grid_shape = visible_grid_shape(candidate.env)
    inventory_rows = crafter_inventory_rows(size=candidate.obs.shape[:2])
    world_h = candidate.obs.shape[0] - inventory_rows
    world_w = candidate.obs.shape[1]
    cell_w = world_w // grid_shape[0]
    cell_h = world_h // grid_shape[1]
    heatmap = np.repeat(np.repeat(deltas, cell_h, axis=0), cell_w, axis=1)
    heatmap = np.pad(
        heatmap,
        ((0, max(world_h - heatmap.shape[0], 0)), (0, max(world_w - heatmap.shape[1], 0))),
        mode="edge",
    )[:world_h, :world_w]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    axes[0].imshow(candidate.obs)
    axes[0].set_title(f"Original\nV={candidate.base_value:.3f}")
    axes[0].axis("off")
    draw_grid(axes[0], candidate.obs, grid_shape)

    vmax = max(float(np.max(np.abs(deltas))), 1e-6)
    im = axes[1].imshow(candidate.obs)
    axes[1].imshow(heatmap, cmap="coolwarm", alpha=0.62, vmin=-vmax, vmax=vmax)
    axes[1].set_title(title)
    axes[1].axis("off")
    draw_grid(axes[1], candidate.obs, grid_shape)

    cell_im = axes[2].imshow(deltas, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[2].set_title("Cell delta\nV(occluded) - V(base)")
    axes[2].set_xticks(range(grid_shape[0]))
    axes[2].set_yticks(range(grid_shape[1]))
    axes[2].set_xlabel("visible x")
    axes[2].set_ylabel("visible y")
    for iy in range(grid_shape[1]):
        for ix in range(grid_shape[0]):
            axes[2].text(ix, iy, f"{deltas[iy, ix]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(cell_im, ax=axes[2], shrink=0.8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_csv(path: str, rows: List[Dict[str, object]]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Cell-level occlusion heatmaps for Crafter value functions.")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, default=67)
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
        deltas, rows = score_cell_occlusions(candidate, model, device, args.occlusion_mode, rng)
        all_rows.extend(rows)
        heatmap_path = os.path.join(args.output_dir, f"{stem}-cell-occlusion.png")
        save_heatmap(
            candidate,
            deltas,
            heatmap_path,
            title=f"{args.occlusion_mode} occlusion\nrank {rank}, ep {candidate.episode_id}, step {candidate.step_id}",
        )
        np.save(os.path.join(args.output_dir, f"{stem}-deltas.npy"), deltas)
        summary_rows.append(
            {
                "rank": rank,
                "episode_id": candidate.episode_id,
                "step_id": candidate.step_id,
                "base_value": candidate.base_value,
                "mean_abs_delta": float(np.mean(np.abs(deltas))),
                "max_abs_delta": float(np.max(np.abs(deltas))),
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
