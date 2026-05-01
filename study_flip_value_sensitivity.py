import argparse
import copy
import csv
import json
import os
from typing import Dict, List, Optional

import numpy as np
import torch as th

from collect_value_map import load_model, set_seed
from counterfactual_env_editor import (
    apply_flip_visible_world_state,
    get_hidsize,
    render_env,
    save_side_by_side,
    score_observation,
    visible_object_orientation_report,
)


def obs_to_tensor(obs: np.ndarray, device: th.device) -> th.Tensor:
    obs = th.from_numpy(np.transpose(obs, (2, 0, 1))).unsqueeze(0).to(device)
    return obs.float() / 255.0


def scalar_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "mean_abs": None,
            "median_abs": None,
            "max_abs": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "mean_abs": float(np.abs(arr).mean()),
        "median_abs": float(np.median(np.abs(arr))),
        "max_abs": float(np.abs(arr).max()),
    }


def state_object_counts(report: List[Dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in report:
        name = str(row.get("object", "unknown"))
        counts[name] = counts.get(name, 0) + 1
    return counts


def score_flip_modes(model, base_env, base_obs, device, states, rnn_states, modes, save_dir=None, stem=None):
    base_scores = score_observation(model, base_obs, device, states, rnn_states)
    row = {
        "base_value": base_scores["value"],
        "visible_objects": json.dumps(state_object_counts(visible_object_orientation_report(base_env)), sort_keys=True),
    }
    for mode in modes:
        env_copy = copy.deepcopy(base_env)
        edits_log: List[str] = []
        apply_flip_visible_world_state(env_copy, mode, edits_log)
        flipped_obs = render_env(env_copy)
        flipped_scores = score_observation(model, flipped_obs, device, states, rnn_states)
        value = flipped_scores["value"]
        row[f"{mode}_value"] = value
        row[f"{mode}_delta"] = value - base_scores["value"]
        row[f"{mode}_abs_delta"] = abs(value - base_scores["value"])
        row[f"{mode}_edits"] = ";".join(edits_log)
        if save_dir is not None and stem is not None:
            save_side_by_side(
                base_obs,
                flipped_obs,
                base_scores,
                flipped_scores,
                os.path.join(save_dir, f"{stem}-{mode}.png"),
                ", ".join(edits_log),
            )
        env_copy.close()
    return row


def collect_study_rows(args, model, config, device):
    from crafter.env import Env

    modes = ["horizontal", "vertical", "both"]
    env = Env(seed=args.eval_seed)
    hidsize = get_hidsize(config)
    rnn_hidsize = config.get("model_kwargs", {}).get("rnn_hidsize")
    rows = []

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
                base_env = copy.deepcopy(env)
                stem = f"ep{episode_idx:03d}-step{step_idx:04d}" if len(rows) < args.save_top_k_candidates else None
                figure_dir = os.path.join(args.output_dir, "candidate_figures") if stem is not None else None
                if figure_dir is not None:
                    os.makedirs(figure_dir, exist_ok=True)
                row = {
                    "episode_id": episode_idx,
                    "step_id": step_idx,
                    "player_pos": json.dumps([int(x) for x in env._player.pos]),
                    "player_facing": json.dumps(list(getattr(env._player, "facing", []))),
                    "inventory": json.dumps(dict(env._player.inventory), sort_keys=True),
                }
                row.update(
                    score_flip_modes(
                        model,
                        base_env,
                        obs.copy(),
                        device,
                        states.clone(),
                        None if rnn_states is None else rnn_states.clone(),
                        modes,
                        save_dir=figure_dir,
                        stem=stem,
                    )
                )
                row["max_abs_delta"] = max(row["horizontal_abs_delta"], row["vertical_abs_delta"], row["both_abs_delta"])
                rows.append(row)
                base_env.close()
                if len(rows) % args.print_every == 0:
                    print(
                        f"scored {len(rows)} states "
                        f"(episode={episode_idx}, step={step_idx}, max_abs_delta={row['max_abs_delta']:.3f})",
                        flush=True,
                    )
                if args.max_states is not None and len(rows) >= args.max_states:
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
            if args.max_states is not None and len(rows) >= args.max_states:
                break

        print(f"finished episode {episode_idx} with {len(rows)} sampled states so far", flush=True)
        if args.max_states is not None and len(rows) >= args.max_states:
            break

    env.close()
    return rows


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Measure PPO value sensitivity to Crafter horizontal/vertical flips.")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=None)
    parser.add_argument("--step_stride", type=int, default=20)
    parser.add_argument("--max_states", type=int, default=None)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--save_top_k", type=int, default=20)
    parser.add_argument(
        "--save_top_k_candidates",
        type=int,
        default=0,
        help="Debug option: save figures for the first N sampled states during rollout.",
    )
    parser.add_argument("--output_dir", type=str, default="flip_value_sensitivity")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)
    print(f"Loaded checkpoint: {ckpt_path}", flush=True)

    rows = collect_study_rows(args, model, config, device)
    if not rows:
        raise ValueError("No states were sampled. Try lowering --start_step or --step_stride.")

    by_horizontal = sorted(rows, key=lambda row: row["horizontal_abs_delta"], reverse=True)
    by_vertical = sorted(rows, key=lambda row: row["vertical_abs_delta"], reverse=True)
    by_both = sorted(rows, key=lambda row: row["both_abs_delta"], reverse=True)
    by_max = sorted(rows, key=lambda row: row["max_abs_delta"], reverse=True)

    write_csv(os.path.join(args.output_dir, "all_states.csv"), rows)
    write_csv(os.path.join(args.output_dir, "sorted_by_horizontal.csv"), by_horizontal)
    write_csv(os.path.join(args.output_dir, "sorted_by_vertical.csv"), by_vertical)
    write_csv(os.path.join(args.output_dir, "sorted_by_both.csv"), by_both)
    write_csv(os.path.join(args.output_dir, "sorted_by_max_abs_delta.csv"), by_max)

    top_dir = os.path.join(args.output_dir, "top_figures")
    os.makedirs(top_dir, exist_ok=True)
    for rank, row in enumerate(by_max[: args.save_top_k], start=1):
        # Re-score the exact state to save clean figures for the largest shifts.
        from counterfactual_env_editor import replay_to_step

        replay = replay_to_step(
            model=model,
            config=config,
            eval_seed=args.eval_seed,
            target_episode=int(row["episode_id"]),
            target_step=int(row["step_id"]),
            device=device,
        )
        stem = f"rank{rank:03d}-ep{int(row['episode_id']):03d}-step{int(row['step_id']):04d}"
        score_flip_modes(
            model,
            replay.env,
            replay.obs.copy(),
            device,
            replay.states,
            replay.rnn_states,
            ["horizontal", "vertical", "both"],
            save_dir=top_dir,
            stem=stem,
        )
        replay.env.close()

    summary = {
        "checkpoint": ckpt_path,
        "eval_seed": args.eval_seed,
        "num_states": len(rows),
        "horizontal_delta": scalar_stats([row["horizontal_delta"] for row in rows]),
        "vertical_delta": scalar_stats([row["vertical_delta"] for row in rows]),
        "both_delta": scalar_stats([row["both_delta"] for row in rows]),
        "top_by_horizontal": by_horizontal[: args.save_top_k],
        "top_by_vertical": by_vertical[: args.save_top_k],
        "top_by_max_abs_delta": by_max[: args.save_top_k],
    }
    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("top_")}, indent=2), flush=True)
    print(f"Wrote study outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
