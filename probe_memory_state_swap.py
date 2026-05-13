import argparse
import base64
import csv
import json
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from achievement_distillation.constant import TASKS
from collect_value_map import load_model, observation_to_uint8_hwc, set_seed


def active_memory_tasks(progress: th.Tensor) -> List[str]:
    values = progress.detach().cpu().view(-1).tolist()
    return [task for task, value in zip(TASKS, values) if value > 0.5]


def obs_batch_from_uint8(obs_hwc, device: th.device) -> th.Tensor:
    obs = th.from_numpy(obs_hwc).permute(2, 0, 1).unsqueeze(0).to(device)
    return obs.float() / 255.0


def load_step(dataset: Dict[str, th.Tensor], episode_id: int, step_id: int, device: th.device):
    episode_ids = dataset["episode_ids"].cpu()
    step_ids = dataset["step_ids"].cpu()
    matches = th.nonzero((episode_ids == episode_id) & (step_ids == step_id), as_tuple=False).view(-1)
    if matches.numel() == 0:
        raise ValueError(f"Dataset has no episode={episode_id}, step={step_id}.")
    idx = int(matches[0].item())
    obs_hwc = observation_to_uint8_hwc(dataset["observations"][idx])
    progress = dataset.get("achievement_progress_inputs")
    if progress is not None:
        memory = progress[idx].float().view(1, -1).to(device)
    else:
        memory = th.zeros(1, len(TASKS), device=device)
    return {
        "idx": idx,
        "source_value": None,
        "episode_id": int(episode_ids[idx].item()),
        "step_id": int(step_ids[idx].item()),
        "obs_hwc": obs_hwc,
        "memory": memory,
    }


def load_html_payload(html_path: str) -> Dict:
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    match = re.search(r"const DATA = (\{.*?\});\n\n\s+const canvas", html, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find embedded DATA payload in {html_path}.")
    return json.loads(match.group(1))


def html_node_to_state(node: Dict, device: th.device):
    raw = base64.b64decode(node["image_bytes"])
    height = int(node["image_height"])
    width = int(node["image_width"])
    rgba = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 4)
    obs_hwc = np.ascontiguousarray(rgba[..., :3])
    progress = node.get("achievement_progress_input")
    if progress is not None:
        memory = th.tensor(progress, dtype=th.float32, device=device).view(1, -1)
    else:
        memory = th.zeros(1, len(TASKS), device=device)
    return {
        "idx": int(node["dataset_index"]),
        "source_value": None if node.get("value") is None else float(node["value"]),
        "episode_id": int(node["episode_id"]),
        "step_id": int(node["step_id"]),
        "obs_hwc": obs_hwc,
        "memory": memory,
    }


def load_step_from_html(html_path: str, episode_id: int, step_id: int, device: th.device):
    payload = load_html_payload(html_path)
    nodes = payload.get("nodes", [])
    matches = [
        node
        for node in nodes
        if int(node.get("episode_id", -1)) == episode_id and int(node.get("step_id", -1)) == step_id
    ]
    if not matches:
        available = sorted(
            (int(node.get("episode_id", -1)), int(node.get("step_id", -1)))
            for node in nodes
        )
        preview = ", ".join(f"ep{ep}:step{step}" for ep, step in available[:12])
        raise ValueError(
            f"HTML has no episode={episode_id}, step={step_id}. "
            f"First available nodes: {preview}"
        )
    return html_node_to_state(matches[0], device)


def score(model, obs_hwc, memory: th.Tensor, device: th.device) -> float:
    kwargs = {}
    if getattr(model, "use_achievement_progress_input", False) or hasattr(model, "achievement_progress_dim"):
        kwargs["achievement_progress"] = memory
    with th.no_grad():
        outputs = model.act(obs_batch_from_uint8(obs_hwc, device), **kwargs)
    return float(outputs["vpreds"].item())


def swap_inventory_pixels(world_obs_hwc: np.ndarray, inventory_obs_hwc: np.ndarray, inventory_y: int) -> np.ndarray:
    swapped = np.array(world_obs_hwc, copy=True)
    swapped[inventory_y:, :, :] = inventory_obs_hwc[inventory_y:, :, :]
    return swapped


def write_csv(path: str, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(path: str, state_a, state_b, state_a_inv_b, state_b_inv_a, rows):
    labels = [row["condition"].replace("_", "\n") for row in rows]
    values = [float(row["value"]) for row in rows]

    fig = plt.figure(figsize=(15.5, 7.0))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.0, 1.0, 1.0, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    ax3 = fig.add_subplot(gs[0, 3])
    ax4 = fig.add_subplot(gs[1, :])

    ax0.imshow(state_a["obs_hwc"])
    ax0.set_title(f"episode {state_a['episode_id']}, step {state_a['step_id']}\nmem bits={len(active_memory_tasks(state_a['memory']))}")
    ax0.axis("off")

    ax1.imshow(state_b["obs_hwc"])
    ax1.set_title(f"episode {state_b['episode_id']}, step {state_b['step_id']}\nmem bits={len(active_memory_tasks(state_b['memory']))}")
    ax1.axis("off")

    ax2.imshow(state_a_inv_b["obs_hwc"])
    ax2.set_title(f"step {state_a['step_id']} world\nstep {state_b['step_id']} inventory")
    ax2.axis("off")

    ax3.imshow(state_b_inv_a["obs_hwc"])
    ax3.set_title(f"step {state_b['step_id']} world\nstep {state_a['step_id']} inventory")
    ax3.axis("off")

    colors = ["#4c78a8", "#9ecae9", "#72b7b2", "#b6d6d3", "#f58518", "#ffbf79", "#54a24b", "#a7d99b"]
    bars = ax4.bar(labels, values, color=colors[: len(values)])
    ax4.set_title("Value under world / inventory / memory swaps")
    ax4.set_ylabel("V(obs, memory)")
    ax4.tick_params(axis="x", rotation=24)
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Load exact HTML/dataset states and swap explicit-memory vectors for value scoring."
    )
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, required=True)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument(
        "--html_path",
        type=str,
        default=None,
        help="Optional value-graph HTML to read exact displayed observations and memory from.",
    )
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--step_a", type=int, required=True)
    parser.add_argument("--step_b", type=int, required=True)
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument(
        "--inventory_y",
        type=int,
        default=49,
        help="Pixel row where the inventory/HUD starts. Crafter 64x64 observations use 49.",
    )
    parser.add_argument("--output_dir", type=str, default="memory_state_swap")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, _config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)

    if args.html_path is not None:
        if not os.path.exists(args.html_path):
            raise FileNotFoundError(f"HTML not found: {args.html_path}")
        state_a = load_step_from_html(args.html_path, args.episode_id, args.step_a, device)
        state_b = load_step_from_html(args.html_path, args.episode_id, args.step_b, device)
        data_source = args.html_path
    else:
        if args.dataset_path is None:
            raise ValueError("Pass either --html_path or --dataset_path.")
        if not os.path.exists(args.dataset_path):
            raise FileNotFoundError(f"Dataset not found: {args.dataset_path}")
        dataset = th.load(args.dataset_path, map_location="cpu")
        state_a = load_step(dataset, args.episode_id, args.step_a, device)
        state_b = load_step(dataset, args.episode_id, args.step_b, device)
        data_source = args.dataset_path

    uses_memory = getattr(model, "use_achievement_progress_input", False) or hasattr(model, "achievement_progress_dim")
    state_a_inv_b = {
        **state_a,
        "obs_hwc": swap_inventory_pixels(state_a["obs_hwc"], state_b["obs_hwc"], args.inventory_y),
    }
    state_b_inv_a = {
        **state_b,
        "obs_hwc": swap_inventory_pixels(state_b["obs_hwc"], state_a["obs_hwc"], args.inventory_y),
    }
    rows = [
        {
            "condition": f"step_{args.step_a}_own_inv_own_mem",
            "world_step": args.step_a,
            "inventory_step": args.step_a,
            "memory_step": args.step_a if uses_memory else None,
            "html_or_dataset_value": state_a["source_value"],
            "value": score(model, state_a["obs_hwc"], state_a["memory"], device),
        },
        {
            "condition": f"step_{args.step_a}_own_inv_mem_from_{args.step_b}",
            "world_step": args.step_a,
            "inventory_step": args.step_a,
            "memory_step": args.step_b if uses_memory else None,
            "html_or_dataset_value": None,
            "value": score(model, state_a["obs_hwc"], state_b["memory"], device),
        },
        {
            "condition": f"step_{args.step_a}_inv_from_{args.step_b}_own_mem",
            "world_step": args.step_a,
            "inventory_step": args.step_b,
            "memory_step": args.step_a if uses_memory else None,
            "html_or_dataset_value": None,
            "value": score(model, state_a_inv_b["obs_hwc"], state_a["memory"], device),
        },
        {
            "condition": f"step_{args.step_a}_inv_and_mem_from_{args.step_b}",
            "world_step": args.step_a,
            "inventory_step": args.step_b,
            "memory_step": args.step_b if uses_memory else None,
            "html_or_dataset_value": None,
            "value": score(model, state_a_inv_b["obs_hwc"], state_b["memory"], device),
        },
        {
            "condition": f"step_{args.step_b}_own_inv_own_mem",
            "world_step": args.step_b,
            "inventory_step": args.step_b,
            "memory_step": args.step_b if uses_memory else None,
            "html_or_dataset_value": state_b["source_value"],
            "value": score(model, state_b["obs_hwc"], state_b["memory"], device),
        },
        {
            "condition": f"step_{args.step_b}_own_inv_mem_from_{args.step_a}",
            "world_step": args.step_b,
            "inventory_step": args.step_b,
            "memory_step": args.step_a if uses_memory else None,
            "html_or_dataset_value": None,
            "value": score(model, state_b["obs_hwc"], state_a["memory"], device),
        },
        {
            "condition": f"step_{args.step_b}_inv_from_{args.step_a}_own_mem",
            "world_step": args.step_b,
            "inventory_step": args.step_a,
            "memory_step": args.step_b if uses_memory else None,
            "html_or_dataset_value": None,
            "value": score(model, state_b_inv_a["obs_hwc"], state_b["memory"], device),
        },
        {
            "condition": f"step_{args.step_b}_inv_and_mem_from_{args.step_a}",
            "world_step": args.step_b,
            "inventory_step": args.step_a,
            "memory_step": args.step_a if uses_memory else None,
            "html_or_dataset_value": None,
            "value": score(model, state_b_inv_a["obs_hwc"], state_a["memory"], device),
        },
    ]

    stem = f"{args.exp_name}-s{args.train_seed:02}-ep{args.episode_id:03d}-swap{args.step_a}-{args.step_b}"
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    fig_path = os.path.join(args.output_dir, f"{stem}.png")
    json_path = os.path.join(args.output_dir, f"{stem}.json")
    write_csv(csv_path, rows)
    save_figure(fig_path, state_a, state_b, state_a_inv_b, state_b_inv_a, rows)

    summary = {
        "checkpoint": ckpt_path,
        "dataset_path": args.dataset_path,
        "html_path": args.html_path,
        "data_source": data_source,
        "episode_id": args.episode_id,
        "step_a": args.step_a,
        "step_b": args.step_b,
        "uses_memory": uses_memory,
        "step_a_dataset_idx": state_a["idx"],
        "step_b_dataset_idx": state_b["idx"],
        "inventory_y": args.inventory_y,
        "step_a_memory_tasks": active_memory_tasks(state_a["memory"]),
        "step_b_memory_tasks": active_memory_tasks(state_b["memory"]),
        "rows": rows,
        "csv_path": csv_path,
        "figure_path": fig_path,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
