import argparse
import base64
import csv
import json
import os
import re
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from achievement_distillation.constant import TASKS
from collect_value_map import load_model, set_seed
from counterfactual_env_editor import (
    apply_inventory_edits,
    crafter_inventory_rows,
    parse_inventory_assignments,
    replay_to_step,
)
from collect_value_map import observation_to_uint8_hwc


def normalize_path(path: str) -> str:
    if path.startswith("file://"):
        return path[len("file://") :]
    return path


def load_html_payload(html_path: str) -> Dict:
    html_path = normalize_path(html_path)
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()
    match = re.search(r"const DATA = (\{.*?\});\n\n\s+const canvas", html, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find embedded DATA payload in {html_path}.")
    return json.loads(match.group(1))


def load_html_step(html_path: str, episode_id: int, step_id: int, device: th.device) -> Dict:
    payload = load_html_payload(html_path)
    matches = [
        node
        for node in payload.get("nodes", [])
        if int(node.get("episode_id", -1)) == episode_id and int(node.get("step_id", -1)) == step_id
    ]
    if not matches:
        raise ValueError(f"HTML has no node for episode={episode_id}, step={step_id}.")
    node = matches[0]
    raw = base64.b64decode(node["image_bytes"])
    h = int(node["image_height"])
    w = int(node["image_width"])
    rgba = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 4)
    progress = node.get("achievement_progress_input")
    if progress is None:
        memory = th.zeros(1, len(TASKS), device=device)
    else:
        memory = th.tensor(progress, dtype=th.float32, device=device).view(1, -1)
    return {
        "dataset_index": int(node["dataset_index"]),
        "html_value": None if node.get("value") is None else float(node["value"]),
        "episode_id": int(node["episode_id"]),
        "step_id": int(node["step_id"]),
        "obs_hwc": np.ascontiguousarray(rgba[..., :3]),
        "memory": memory,
        "memory_tasks": node.get("memory_tasks"),
    }


def load_replay_step(model, config: Dict, eval_seed: int, episode_id: int, step_id: int, device: th.device) -> Dict:
    replay = replay_to_step(model, config, eval_seed, episode_id, step_id, device)
    return {
        "dataset_index": None,
        "html_value": None,
        "episode_id": int(replay.episode_id),
        "step_id": int(replay.step_id),
        "obs_hwc": replay.obs.copy(),
        "memory": replay.achievement_progress.clone(),
        "memory_tasks": active_memory_tasks(replay.achievement_progress),
        "env": replay.env,
    }


def load_dataset_step(dataset_path: str, episode_id: int, step_id: int, device: th.device) -> Dict:
    dataset = th.load(normalize_path(dataset_path), map_location="cpu")
    episode_ids = dataset["episode_ids"].cpu()
    step_ids = dataset["step_ids"].cpu()
    matches = th.nonzero((episode_ids == episode_id) & (step_ids == step_id), as_tuple=False).view(-1)
    if matches.numel() == 0:
        raise ValueError(f"Dataset has no episode={episode_id}, step={step_id}.")
    idx = int(matches[0].item())
    progress = dataset.get("achievement_progress_inputs")
    if progress is None:
        memory = th.zeros(1, len(TASKS), device=device)
    else:
        memory = progress[idx].float().view(1, -1).to(device)
    values = dataset.get("values")
    source_value = None
    if values is not None:
        source_value = float(values[idx].item())
    return {
        "dataset_index": idx,
        "html_value": source_value,
        "episode_id": int(episode_ids[idx].item()),
        "step_id": int(step_ids[idx].item()),
        "obs_hwc": observation_to_uint8_hwc(dataset["observations"][idx]),
        "memory": memory,
        "memory_tasks": active_memory_tasks(memory),
    }


def load_dataset_replay_step(dataset_path: str, eval_seed: int, episode_id: int, step_id: int, device: th.device) -> Dict:
    from crafter.env import Env

    dataset = th.load(normalize_path(dataset_path), map_location="cpu")
    episode_ids = dataset["episode_ids"].cpu()
    step_ids = dataset["step_ids"].cpu()
    matches = th.nonzero((episode_ids == episode_id) & (step_ids == step_id), as_tuple=False).view(-1)
    if matches.numel() == 0:
        raise ValueError(f"Dataset has no episode={episode_id}, step={step_id}.")
    idx = int(matches[0].item())

    episode_matches = th.nonzero(episode_ids == episode_id, as_tuple=False).view(-1)
    ordered = episode_matches[th.argsort(step_ids[episode_matches])]
    action_by_step = {
        int(step_ids[data_idx].item()): int(dataset["actions"][data_idx].item())
        for data_idx in ordered
    }

    env = Env(seed=eval_seed + episode_id)
    obs = env.reset()
    for replay_step in range(step_id):
        if replay_step not in action_by_step:
            env.close()
            raise ValueError(f"Dataset is missing action for episode={episode_id}, step={replay_step}.")
        obs, _reward, done, _info = env.step(action_by_step[replay_step])
        if done:
            env.close()
            raise ValueError(f"Episode ended before requested step {step_id}; ended after step {replay_step}.")

    progress = dataset.get("achievement_progress_inputs")
    if progress is None:
        memory = th.zeros(1, len(TASKS), device=device)
    else:
        memory = progress[idx].float().view(1, -1).to(device)
    values = dataset.get("values")
    source_value = None if values is None else float(values[idx].item())
    saved_obs = observation_to_uint8_hwc(dataset["observations"][idx])
    obs_l1 = float(np.mean(np.abs(saved_obs.astype(np.float32) - obs.astype(np.float32))))
    return {
        "dataset_index": idx,
        "html_value": source_value,
        "episode_id": int(episode_id),
        "step_id": int(step_id),
        "obs_hwc": obs.copy(),
        "memory": memory,
        "memory_tasks": active_memory_tasks(memory),
        "env": env,
        "dataset_replay_obs_l1": obs_l1,
    }


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


def inventory_slot_bounds(obs: np.ndarray, item_name: str):
    items = inventory_items()
    if item_name not in items:
        raise ValueError(f"Unknown inventory item '{item_name}'. Known items: {items}")
    slot_idx = items.index(item_name)
    cols = 9
    inv_rows = int(np.ceil(len(items) / cols))
    hud_rows = crafter_inventory_rows(size=obs.shape[:2])
    slot_w = obs.shape[1] // cols
    slot_h = max(hud_rows // inv_rows, 1)
    y_start = obs.shape[0] - hud_rows
    row = slot_idx // cols
    col = slot_idx % cols
    x0 = col * slot_w
    x1 = min((col + 1) * slot_w, obs.shape[1])
    y0 = y_start + row * slot_h
    y1 = min(y_start + (row + 1) * slot_h, obs.shape[0])
    return x0, x1, y0, y1


def parse_csv_names(text: str) -> List[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def remove_inventory_item_pixels(obs: np.ndarray, item_names: List[str]) -> np.ndarray:
    edited = np.array(obs, copy=True)
    for item_name in item_names:
        x0, x1, y0, y1 = inventory_slot_bounds(obs, item_name)
        edited[y0:y1, x0:x1] = 0
    return edited


def render_replay_inventory_edit(state: Dict, inventory_items_to_remove: List[str], set_inventory: str) -> np.ndarray:
    updates = {item_name: 0 for item_name in inventory_items_to_remove}
    updates.update(parse_inventory_assignments(set_inventory))
    edits_log: List[str] = []
    apply_inventory_edits(state["env"], updates, edits_log)
    return state["env"].render()


def remove_memory_tasks(memory: th.Tensor, task_names: List[str]) -> th.Tensor:
    edited = memory.clone()
    for task_name in task_names:
        if task_name not in TASKS:
            raise ValueError(f"Unknown task '{task_name}'.")
        edited[:, TASKS.index(task_name)] = 0.0
    return edited


def active_memory_tasks(memory: th.Tensor) -> List[str]:
    values = memory.detach().cpu().view(-1).tolist()
    return [task for task, value in zip(TASKS, values) if value > 0.5]


def obs_batch_from_uint8(obs_hwc: np.ndarray, device: th.device) -> th.Tensor:
    obs = th.from_numpy(np.ascontiguousarray(obs_hwc)).permute(2, 0, 1).unsqueeze(0).to(device)
    return obs.float() / 255.0


def score(model, obs_hwc: np.ndarray, memory: th.Tensor, device: th.device) -> float:
    kwargs = {}
    if getattr(model, "use_achievement_progress_input", False) or hasattr(model, "achievement_progress_dim"):
        kwargs["achievement_progress"] = memory
    with th.no_grad():
        outputs = model.act(obs_batch_from_uint8(obs_hwc, device), **kwargs)
    return float(outputs["vpreds"].item())


def write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_figure(
    path: str,
    state: Dict,
    obs_no_inventory: np.ndarray,
    rows: List[Dict],
    item_names: List[str],
    task_names: List[str],
    inventory_edit_title: str,
):
    labels = [row["condition"].replace("_", "\n") for row in rows]
    values = [float(row["value"]) for row in rows]

    fig = plt.figure(figsize=(13.5, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.45])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    ax0.imshow(state["obs_hwc"])
    ax0.set_title(
        f"HTML step {state['step_id']}\noriginal HUD, mem bits={len(active_memory_tasks(state['memory']))}"
    )
    ax0.axis("off")

    ax1.imshow(obs_no_inventory)
    ax1.set_title(f"inventory edited\n{inventory_edit_title}")
    ax1.axis("off")

    bars = ax2.bar(labels, values, color=["#4c78a8", "#9ecae9", "#f58518", "#ffbf79"])
    ax2.set_title(f"Remove inventory / memory\n{', '.join(item_names)} / {', '.join(task_names)}")
    ax2.set_ylabel("V(obs, memory)")
    ax2.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Inventory/memory counterfactual from HTML pixels or replayed env render.")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, default="debug")
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, required=True)
    parser.add_argument("--source", choices=["html", "dataset", "dataset_replay", "replay"], default="html")
    parser.add_argument("--html_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--step_id", type=int, required=True)
    parser.add_argument("--inventory_item", type=str, default="stone", help="Comma-separated inventory item(s).")
    parser.add_argument(
        "--set_inventory",
        type=str,
        default="",
        help="Replay mode only. Comma-separated item=value assignments applied after removed items are set to 0.",
    )
    parser.add_argument("--memory_task", type=str, default="collect_stone", help="Comma-separated memory task(s).")
    parser.add_argument("--eval_seed", type=int, default=67)
    parser.add_argument("--output_dir", type=str, default="expirements/stone_memory_inventory_counterfactual")
    args = parser.parse_args()

    set_seed(args.eval_seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model, config, ckpt_path = load_model(args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device)

    if args.source == "html":
        if args.html_path is None:
            raise ValueError("--html_path is required when --source html.")
        state = load_html_step(args.html_path, args.episode_id, args.step_id, device)
    elif args.source == "dataset":
        if args.dataset_path is None:
            raise ValueError("--dataset_path is required when --source dataset.")
        state = load_dataset_step(args.dataset_path, args.episode_id, args.step_id, device)
    elif args.source == "dataset_replay":
        if args.dataset_path is None:
            raise ValueError("--dataset_path is required when --source dataset_replay.")
        state = load_dataset_replay_step(args.dataset_path, args.eval_seed, args.episode_id, args.step_id, device)
    else:
        state = load_replay_step(model, config, args.eval_seed, args.episode_id, args.step_id, device)
    inventory_items_to_remove = parse_csv_names(args.inventory_item)
    memory_tasks_to_remove = parse_csv_names(args.memory_task)
    if args.source in {"html", "dataset"}:
        obs_no_inventory = remove_inventory_item_pixels(state["obs_hwc"], inventory_items_to_remove)
        inventory_edit_title = f"blanked: {', '.join(inventory_items_to_remove)}"
    else:
        obs_no_inventory = render_replay_inventory_edit(state, inventory_items_to_remove, args.set_inventory)
        inventory_assignments = parse_inventory_assignments(args.set_inventory)
        assignment_text = ", ".join(
            [f"{item}=0" for item in inventory_items_to_remove]
            + [f"{key}={value}" for key, value in inventory_assignments.items()]
        )
        inventory_edit_title = assignment_text if assignment_text else "none"
    memory_no_task = remove_memory_tasks(state["memory"], memory_tasks_to_remove)

    rows = [
        {
            "condition": "base",
            "inventory_removed": False,
            "memory_removed": False,
            "html_value": state["html_value"],
            "value": score(model, state["obs_hwc"], state["memory"], device),
        },
        {
            "condition": "memory_removed",
            "inventory_removed": False,
            "memory_removed": True,
            "html_value": None,
            "value": score(model, state["obs_hwc"], memory_no_task, device),
        },
        {
            "condition": "inventory_removed",
            "inventory_removed": True,
            "memory_removed": False,
            "html_value": None,
            "value": score(model, obs_no_inventory, state["memory"], device),
        },
        {
            "condition": "inventory_and_memory_removed",
            "inventory_removed": True,
            "memory_removed": True,
            "html_value": None,
            "value": score(model, obs_no_inventory, memory_no_task, device),
        },
    ]

    item_stem = "-".join(inventory_items_to_remove)
    stem = f"{args.exp_name}-s{args.train_seed:02d}-ep{args.episode_id:03d}-step{args.step_id:04d}-{item_stem}-inventory-memory"
    csv_path = os.path.join(args.output_dir, f"{stem}.csv")
    fig_path = os.path.join(args.output_dir, f"{stem}.png")
    json_path = os.path.join(args.output_dir, f"{stem}.json")
    write_csv(csv_path, rows)
    save_figure(
        fig_path,
        state,
        obs_no_inventory,
        rows,
        inventory_items_to_remove,
        memory_tasks_to_remove,
        inventory_edit_title,
    )

    summary = {
        "checkpoint": ckpt_path,
        "source": args.source,
        "html_path": None if args.html_path is None else normalize_path(args.html_path),
        "dataset_path": None if args.dataset_path is None else normalize_path(args.dataset_path),
        "episode_id": args.episode_id,
        "step_id": args.step_id,
        "dataset_index": state["dataset_index"],
        "dataset_replay_obs_l1": state.get("dataset_replay_obs_l1"),
        "inventory_items": inventory_items_to_remove,
        "set_inventory": parse_inventory_assignments(args.set_inventory),
        "inventory_edit_title": inventory_edit_title,
        "memory_tasks_removed": memory_tasks_to_remove,
        "base_memory_tasks": active_memory_tasks(state["memory"]),
        "memory_tasks_after_removal": active_memory_tasks(memory_no_task),
        "rows": rows,
        "csv_path": csv_path,
        "figure_path": fig_path,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
