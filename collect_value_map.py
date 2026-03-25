import argparse
import base64
import importlib
import json
import os
import random
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th
import yaml

ACTION_NAMES = [
    "noop",
    "left",
    "right",
    "up",
    "down",
    "grab_or_attack",
    "sleep",
    "place_table",
    "place_stone",
    "place_furnace",
    "place_plant",
    "make_wood_pickaxe",
    "make_stone_pickaxe",
    "make_iron_pickaxe",
    "make_wood_sword",
    "make_stone_sword",
    "make_iron_sword",
]


def load_config(exp_name: str) -> Dict:
    config_path = f"configs/{exp_name}.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)


def load_model(exp_name: str, timestamp: str, train_seed: int, ckpt_epoch: int, device: th.device):
    from crafter.env import Env
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

    from achievement_distillation.model import BaseModel
    import achievement_distillation.model as model_module
    from achievement_distillation.wrapper import VecPyTorch

    config = load_config(exp_name)

    temp_venv = VecPyTorch(DummyVecEnv([lambda: Env(seed=train_seed)]), device=device)
    try:
        model_cls = getattr(model_module, config["model_cls"])
        model: BaseModel = model_cls(
            observation_space=temp_venv.observation_space,
            action_space=temp_venv.action_space,
            **config["model_kwargs"],
        )
        model.to(device)
    finally:
        temp_venv.close()

    run_name = f"{exp_name}-{timestamp}-s{train_seed:02}"
    ckpt_path = os.path.join("models", run_name, f"agent-e{ckpt_epoch:03}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    state_dict = th.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config, ckpt_path


def pca_2d(latents: np.ndarray) -> np.ndarray:
    latents = latents.astype(np.float64, copy=False)
    mean = latents.mean(axis=0, keepdims=True)
    centered = latents - mean
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    components = vh[:2].T
    return centered @ components

def observation_to_uint8_hwc(observation: th.Tensor) -> np.ndarray:
    obs = observation.detach().cpu().numpy()
    if obs.ndim != 3:
        raise ValueError(f"Expected observation with 3 dims, got shape {obs.shape}")
    obs = np.transpose(obs, (1, 2, 0))
    obs = np.clip(np.rint(obs * 255.0), 0, 255).astype(np.uint8)
    return obs


def encode_observation_bytes(observation: th.Tensor) -> Tuple[str, int, int]:
    obs = observation_to_uint8_hwc(observation)
    alpha = np.full((*obs.shape[:2], 1), 255, dtype=np.uint8)
    obs = np.concatenate([obs, alpha], axis=2)
    height, width, _ = obs.shape
    encoded = base64.b64encode(obs.tobytes()).decode("ascii")
    return encoded, width, height


def select_dataset_indices(num_states: int, max_states: Optional[int]) -> np.ndarray:
    if max_states is None or max_states <= 0 or num_states <= max_states:
        return np.arange(num_states, dtype=np.int64)
    return np.linspace(0, num_states - 1, num=max_states, dtype=np.int64)


def compute_value_ring_layout(values: np.ndarray, step_ids: np.ndarray, episode_ids: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    value_min = float(values.min())
    value_max = float(values.max())
    denom = max(value_max - value_min, 1e-8)
    value_norm = (values - value_min) / denom

    coords = np.zeros((len(values), 2), dtype=np.float32)
    for episode_id in np.unique(episode_ids):
        mask = episode_ids == episode_id
        episode_idx = np.where(mask)[0]
        if len(episode_idx) == 0:
            continue

        local_steps = step_ids[episode_idx].astype(np.float64)
        if len(episode_idx) == 1:
            angle = np.array([0.0], dtype=np.float64)
        else:
            step_min = float(local_steps.min())
            step_span = max(float(local_steps.max()) - step_min, 1.0)
            angle = 2.0 * np.pi * ((local_steps - step_min) / step_span)

        radius = 0.2 + 0.85 * (1.0 - value_norm[episode_idx])
        coords[episode_idx, 0] = radius * np.cos(angle)
        coords[episode_idx, 1] = radius * np.sin(angle)

    return coords


def build_value_edges(values: np.ndarray, num_neighbors: int, value_threshold: Optional[float]) -> List[Tuple[int, int]]:
    num_states = len(values)
    if num_states <= 1 or num_neighbors <= 0:
        return []

    sorted_idx = np.argsort(values)
    edges = set()
    for rank, node_idx in enumerate(sorted_idx):
        max_offset = min(num_neighbors, num_states - 1)
        for offset in range(1, max_offset + 1):
            for neighbor_rank in (rank - offset, rank + offset):
                if neighbor_rank < 0 or neighbor_rank >= num_states:
                    continue
                neighbor_idx = int(sorted_idx[neighbor_rank])
                diff = abs(float(values[node_idx]) - float(values[neighbor_idx]))
                if value_threshold is not None and diff > value_threshold:
                    continue
                edge = (int(min(node_idx, neighbor_idx)), int(max(node_idx, neighbor_idx)))
                edges.add(edge)

    return sorted(edges)


def build_temporal_edges(episode_ids: np.ndarray, step_ids: np.ndarray) -> List[Tuple[int, int]]:
    edges = []
    for episode_id in np.unique(episode_ids):
        mask = episode_ids == episode_id
        episode_idx = np.where(mask)[0]
        if len(episode_idx) <= 1:
            continue
        order = np.argsort(step_ids[episode_idx])
        ordered_idx = episode_idx[order]
        for src, dst in zip(ordered_idx[:-1], ordered_idx[1:]):
            edges.append((int(src), int(dst)))
    return edges


def find_latest_mp4(directory: str) -> Optional[str]:
    if not os.path.isdir(directory):
        return None
    candidates = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".mp4")
    ]
    if not candidates:
        return None
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def compute_first_unlocks(
    successes: np.ndarray,
    episode_ids: np.ndarray,
    task_names: Sequence[str],
) -> List[List[str]]:
    first_unlocks: List[List[str]] = []
    prev_success = None
    prev_episode = None
    for idx in range(len(successes)):
        current_success = successes[idx].astype(bool)
        if prev_success is None or prev_episode != int(episode_ids[idx]):
            previous = np.zeros_like(current_success, dtype=bool)
        else:
            previous = prev_success
        new_unlock_mask = np.logical_and(current_success, np.logical_not(previous))
        first_unlocks.append(
            [task_names[task_idx] for task_idx in np.where(new_unlock_mask)[0].tolist()]
        )
        prev_success = current_success
        prev_episode = int(episode_ids[idx])
    return first_unlocks


def save_value_graph_viewer(
    dataset: Dict[str, th.Tensor],
    output_path: str,
    max_states: Optional[int],
    num_neighbors: int,
    value_threshold: Optional[float],
):
    selected_idx = select_dataset_indices(len(dataset["values"]), max_states)
    observations = dataset["observations"][selected_idx]
    values = dataset["values"][selected_idx].cpu().numpy()
    actions = dataset["actions"][selected_idx].cpu().numpy()
    rewards = dataset["rewards"][selected_idx].cpu().numpy()
    dones = dataset["dones"][selected_idx].cpu().numpy()
    episode_ids = dataset["episode_ids"][selected_idx].cpu().numpy()
    step_ids = dataset["step_ids"][selected_idx].cpu().numpy()

    coords = compute_value_ring_layout(values, step_ids, episode_ids)
    temporal_edges = build_temporal_edges(episode_ids, step_ids)
    value_edges = build_value_edges(values, num_neighbors=num_neighbors, value_threshold=value_threshold)

    task_names: Sequence[str] = dataset["task_names"]
    achievements = dataset["achievements"][selected_idx].cpu().numpy()
    successes = dataset["successes"][selected_idx].cpu().numpy()
    full_episode_ids = dataset["episode_ids"].cpu().numpy()
    full_successes = dataset["successes"].cpu().numpy()
    full_first_unlocks = compute_first_unlocks(full_successes, full_episode_ids, task_names)

    value_min = float(values.min()) if len(values) else 0.0
    value_max = float(values.max()) if len(values) else 1.0

    nodes = []
    for idx, obs in enumerate(observations):
        image_bytes, width, height = encode_observation_bytes(obs)
        achieved = [
            task_names[task_idx]
            for task_idx, flag in enumerate(successes[idx].tolist())
            if int(flag) > 0
        ]
        nodes.append(
            {
                "id": int(idx),
                "x": float(coords[idx, 0]),
                "y": float(coords[idx, 1]),
                "value": float(values[idx]),
                "action": int(actions[idx]),
                "reward": float(rewards[idx]),
                "done": bool(dones[idx]),
                "episode_id": int(episode_ids[idx]),
                "step_id": int(step_ids[idx]),
                "action_name": ACTION_NAMES[int(actions[idx])] if 0 <= int(actions[idx]) < len(ACTION_NAMES) else f"action_{int(actions[idx])}",
                "image_bytes": image_bytes,
                "image_width": int(width),
                "image_height": int(height),
                "achieved_tasks": achieved,
                "new_achievements": full_first_unlocks[int(selected_idx[idx])],
                "achievement_counts": achievements[idx].tolist(),
            }
        )

    payload = {
        "meta": {
            "num_nodes": len(nodes),
            "num_temporal_edges": len(temporal_edges),
            "num_value_edges": len(value_edges),
            "value_min": value_min,
            "value_max": value_max,
            "value_threshold": None if value_threshold is None else float(value_threshold),
            "num_neighbors": int(num_neighbors),
            "max_states": None if max_states is None else int(max_states),
            "reward_nodes": int(np.count_nonzero(np.abs(rewards) > 1e-8)),
            "first_unlock_nodes": int(sum(1 for unlocks in full_first_unlocks if unlocks)),
            "action_names": ACTION_NAMES,
        },
        "nodes": nodes,
        "edges": {
            "temporal": temporal_edges,
            "value": value_edges,
        },
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Crafter Value Graph</title>
  <style>
    :root {{
      --paper: #f5f1e8;
      --ink: #223127;
      --moss: #66806a;
      --sage: #d8dfcf;
      --accent: #ca7f45;
      --panel: rgba(255, 251, 242, 0.9);
      --line: rgba(55, 74, 59, 0.16);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #f9f4ea 0%, #efe7d7 45%, #e5dcc8 100%);
      height: 100vh;
      overflow: hidden;
    }}
    .app {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      height: 100vh;
    }}
    .stage {{
      position: relative;
      overflow: hidden;
      border-right: 1px solid rgba(34, 49, 39, 0.1);
    }}
    canvas {{
      width: 100%;
      height: 100%;
      display: block;
      cursor: grab;
    }}
    canvas.dragging {{
      cursor: grabbing;
    }}
    .overlay {{
      position: absolute;
      top: 18px;
      left: 18px;
      max-width: 380px;
      padding: 14px 16px;
      background: var(--panel);
      backdrop-filter: blur(8px);
      border: 1px solid rgba(34, 49, 39, 0.08);
      border-radius: 16px;
      box-shadow: 0 16px 48px rgba(74, 67, 48, 0.12);
      line-height: 1.45;
    }}
    .overlay h1 {{
      margin: 0 0 6px;
      font-size: 20px;
      font-weight: 600;
    }}
    .overlay p {{
      margin: 0;
      font-size: 14px;
    }}
    .sidebar {{
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
      background: linear-gradient(180deg, rgba(255,255,255,0.42), rgba(248,242,231,0.88));
      height: 100vh;
      overflow-y: auto;
      overflow-x: hidden;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(34, 49, 39, 0.08);
      border-radius: 18px;
      padding: 16px;
      box-shadow: 0 14px 40px rgba(66, 59, 44, 0.08);
    }}
    .meta {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      font-size: 13px;
    }}
    .meta strong {{
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--moss);
      margin-bottom: 4px;
    }}
    .preview-wrap {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .preview-canvas {{
      width: 100%;
      aspect-ratio: 1 / 1;
      border-radius: 16px;
      background:
        linear-gradient(135deg, rgba(120, 144, 113, 0.2), rgba(202, 127, 69, 0.18));
      border: 1px solid rgba(34, 49, 39, 0.08);
    }}
    .hint {{
      font-size: 13px;
      color: #4a5d4d;
      line-height: 1.5;
    }}
    .controls {{
      display: flex;
      flex-direction: column;
      gap: 10px;
      font-size: 13px;
      color: #3e4e40;
    }}
    .controls label {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .controls input[type="text"] {{
      width: 100%;
      padding: 9px 10px;
      border-radius: 10px;
      border: 1px solid rgba(34, 49, 39, 0.14);
      background: rgba(255, 252, 246, 0.95);
      color: var(--ink);
      font: inherit;
    }}
    .button-row {{
      display: flex;
      gap: 8px;
    }}
    .button-row button {{
      border: 0;
      border-radius: 999px;
      padding: 8px 12px;
      font: inherit;
      background: #e2d4bb;
      color: #36483a;
      cursor: pointer;
    }}
    .button-row button:hover {{
      background: #d6c4a6;
    }}
    .status {{
      min-height: 18px;
      color: #6c533f;
    }}
    .task-list {{
      font-size: 13px;
      color: #3e4e40;
      min-height: 20px;
    }}
    .legend {{
      display: flex;
      flex-direction: column;
      gap: 8px;
      font-size: 13px;
    }}
    .legend-row {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .swatch {{
      width: 16px;
      height: 4px;
      border-radius: 999px;
      flex: 0 0 auto;
    }}
    @media (max-width: 960px) {{
      .app {{
        grid-template-columns: 1fr;
        grid-template-rows: minmax(0, 1fr) auto;
      }}
      .sidebar {{
        max-height: 42vh;
        overflow: auto;
      }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <div class="stage">
      <canvas id="graph"></canvas>
      <div class="overlay">
        <h1>Value Graph Explorer</h1>
        <p>Drag to pan, scroll to zoom, hover to preview a state, and click a node to pin it. The ring layout is value-driven: higher-value states sit nearer the center, while edges show rollout order and value-neighborhood links.</p>
      </div>
    </div>
    <aside class="sidebar">
      <section class="card">
        <div class="meta">
          <div><strong>Nodes</strong><span id="meta-nodes"></span></div>
          <div><strong>Temporal Edges</strong><span id="meta-temporal"></span></div>
          <div><strong>Value Edges</strong><span id="meta-value"></span></div>
          <div><strong>Value Range</strong><span id="meta-range"></span></div>
          <div><strong>Value Threshold</strong><span id="meta-threshold"></span></div>
          <div><strong>Neighbors Per Node</strong><span id="meta-neighbors"></span></div>
          <div><strong>Reward States</strong><span id="meta-rewards"></span></div>
          <div><strong>First Unlocks</strong><span id="meta-first-unlocks"></span></div>
          <div><strong>Action Matches</strong><span id="meta-action-matches">none</span></div>
        </div>
      </section>
      <section class="card controls">
        <label><input id="reward-ring-toggle" type="checkbox" checked> Show reward-state rings</label>
        <div>
          <strong style="display:block;font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:#66806a;margin-bottom:6px;">Action Highlight</strong>
          <select id="action-filter-select" style="width:100%;padding:9px 10px;border-radius:10px;border:1px solid rgba(34, 49, 39, 0.14);background:rgba(255, 252, 246, 0.95);color:#223127;font:inherit;">
            <option value="">No action filter</option>
          </select>
        </div>
        <div>
          <strong style="display:block;font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:#66806a;margin-bottom:6px;">Step Highlight</strong>
          <input id="step-range-input" type="text" placeholder="e.g. 181-187 or 42">
        </div>
        <div class="button-row">
          <button id="apply-step-range" type="button">Highlight</button>
          <button id="clear-step-range" type="button">Clear</button>
        </div>
        <div id="step-range-status" class="status">No step range highlighted.</div>
      </section>
      <section class="card preview-wrap">
        <canvas id="preview" class="preview-canvas" width="320" height="320"></canvas>
        <div class="meta">
          <div><strong>Episode</strong><span id="node-episode">-</span></div>
          <div><strong>Step</strong><span id="node-step">-</span></div>
          <div><strong>Value</strong><span id="node-value">-</span></div>
          <div><strong>Reward</strong><span id="node-reward">-</span></div>
          <div><strong>Action</strong><span id="node-action">-</span></div>
          <div><strong>Done</strong><span id="node-done">-</span></div>
        </div>
        <div>
          <strong style="display:block;font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:#66806a;margin-bottom:6px;">New Achievements At Step</strong>
          <div id="node-new-tasks" class="task-list">No new achievements at this step.</div>
        </div>
        <div>
          <strong style="display:block;font-size:11px;text-transform:uppercase;letter-spacing:0.08em;color:#66806a;margin-bottom:6px;">Achievements Unlocked</strong>
          <div id="node-tasks" class="task-list">Hover a node to inspect its state.</div>
        </div>
      </section>
      <section class="card legend">
        <div class="legend-row"><span class="swatch" style="background:#768b78;"></span><span>Temporal rollout edges</span></div>
        <div class="legend-row"><span class="swatch" style="background:#ca7f45;"></span><span>Value-neighbor edges</span></div>
        <div class="legend-row"><span class="swatch" style="background:linear-gradient(90deg,#2c7c7a,#e8c15a,#c95c34);height:10px;"></span><span>Node color tracks predicted value</span></div>
        <div class="legend-row"><span class="swatch" style="background:#caa24a;height:10px;"></span><span>Reward-state ring</span></div>
        <div class="legend-row"><span class="swatch" style="background:#2d3441;height:10px;"></span><span>Step-range highlight</span></div>
        <div class="legend-row"><span class="swatch" style="background:#b5524f;height:10px;"></span><span>First-time achievement unlock</span></div>
        <div class="legend-row"><span class="swatch" style="background:#5b4fa8;height:10px;"></span><span>Selected action highlight</span></div>
      </section>
      <section class="card hint">
        Graph settings: states are arranged by predicted value on concentric rings, temporal edges connect adjacent rollout states, and value-neighbor edges connect each node to its nearest states in value space subject to the threshold shown above.
      </section>
      <section class="card hint">
        Click a node to keep it selected while you move around the graph. Clicking the background clears the selection. Use <code>+</code> and <code>-</code> to zoom, <code>0</code> to refit, and <code>Esc</code> to clear the pinned node.
      </section>
    </aside>
  </div>
  <script>
    const DATA = {json.dumps(payload)};

    const canvas = document.getElementById("graph");
    const ctx = canvas.getContext("2d");
    const previewCanvas = document.getElementById("preview");
    const previewCtx = previewCanvas.getContext("2d");

    const nodeEpisode = document.getElementById("node-episode");
    const nodeStep = document.getElementById("node-step");
    const nodeValue = document.getElementById("node-value");
    const nodeReward = document.getElementById("node-reward");
    const nodeAction = document.getElementById("node-action");
    const nodeDone = document.getElementById("node-done");
    const nodeNewTasks = document.getElementById("node-new-tasks");
    const nodeTasks = document.getElementById("node-tasks");
    const rewardRingToggle = document.getElementById("reward-ring-toggle");
    const actionFilterSelect = document.getElementById("action-filter-select");
    const stepRangeInput = document.getElementById("step-range-input");
    const stepRangeStatus = document.getElementById("step-range-status");
    const applyStepRangeButton = document.getElementById("apply-step-range");
    const clearStepRangeButton = document.getElementById("clear-step-range");
    const actionMatchMeta = document.getElementById("meta-action-matches");

    document.getElementById("meta-nodes").textContent = DATA.meta.num_nodes;
    document.getElementById("meta-temporal").textContent = DATA.meta.num_temporal_edges;
    document.getElementById("meta-value").textContent = DATA.meta.num_value_edges;
    document.getElementById("meta-range").textContent = `${{DATA.meta.value_min.toFixed(2)}} to ${{DATA.meta.value_max.toFixed(2)}}`;
    document.getElementById("meta-threshold").textContent = DATA.meta.value_threshold === null ? "none" : DATA.meta.value_threshold.toFixed(3);
    document.getElementById("meta-neighbors").textContent = String(DATA.meta.num_neighbors);
    document.getElementById("meta-rewards").textContent = String(DATA.meta.reward_nodes);
    document.getElementById("meta-first-unlocks").textContent = String(DATA.meta.first_unlock_nodes);
    for (const actionName of DATA.meta.action_names) {{
      const option = document.createElement("option");
      option.value = actionName;
      option.textContent = actionName;
      actionFilterSelect.appendChild(option);
    }}

    let dpr = window.devicePixelRatio || 1;
    let transform = {{
      scale: 220,
      offsetX: 0,
      offsetY: 0,
    }};
    const MIN_SCALE = 20;
    const MAX_SCALE = 20000;
    let dragging = false;
    let lastMouse = null;
    let hoveredNode = null;
    let pinnedNode = null;
    let activeStepRange = null;
    let activeActionFilter = "";

    const adjacency = new Map();
    function registerEdge(a, b, type) {{
      if (!adjacency.has(a)) adjacency.set(a, []);
      if (!adjacency.has(b)) adjacency.set(b, []);
      adjacency.get(a).push({{ id: b, type }});
      adjacency.get(b).push({{ id: a, type }});
    }}

    for (const [a, b] of DATA.edges.temporal) registerEdge(a, b, "temporal");
    for (const [a, b] of DATA.edges.value) registerEdge(a, b, "value");

    function resizeCanvas() {{
      const rect = canvas.getBoundingClientRect();
      dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(rect.width * dpr);
      canvas.height = Math.floor(rect.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (transform.offsetX === 0 && transform.offsetY === 0) {{
        let minX = Infinity;
        let maxX = -Infinity;
        let minY = Infinity;
        let maxY = -Infinity;
        for (const node of DATA.nodes) {{
          minX = Math.min(minX, node.x);
          maxX = Math.max(maxX, node.x);
          minY = Math.min(minY, node.y);
          maxY = Math.max(maxY, node.y);
        }}
        const spanX = Math.max(maxX - minX, 1e-6);
        const spanY = Math.max(maxY - minY, 1e-6);
        const fitScale = Math.min(rect.width / (spanX * 1.15), rect.height / (spanY * 1.15));
        transform.scale = Math.min(1200, Math.max(120, fitScale * 1.35));
        transform.offsetX = rect.width / 2 - ((minX + maxX) / 2) * transform.scale;
        transform.offsetY = rect.height / 2 - ((minY + maxY) / 2) * transform.scale;
      }}
      draw();
    }}

    function getValueNorm(value) {{
      const span = Math.max(DATA.meta.value_max - DATA.meta.value_min, 1e-8);
      return (value - DATA.meta.value_min) / span;
    }}

    function valueColor(value) {{
      const t = Math.max(0, Math.min(1, getValueNorm(value)));
      const hue = 188 - 166 * t;
      const sat = 54 + 24 * t;
      const light = 38 + 24 * t;
      return `hsl(${{hue}}, ${{sat}}%, ${{light}}%)`;
    }}

    function hasReward(node) {{
      return Math.abs(node.reward) > 1e-8;
    }}

    function hasNewAchievement(node) {{
      return node.new_achievements && node.new_achievements.length > 0;
    }}

    function isInActiveStepRange(node) {{
      if (!activeStepRange) {{
        return false;
      }}
      return node.step_id >= activeStepRange.start && node.step_id <= activeStepRange.end;
    }}

    function updateStepRangeStatus(message) {{
      stepRangeStatus.textContent = message;
    }}

    function updateActionMatchStatus() {{
      if (!activeActionFilter) {{
        actionMatchMeta.textContent = "none";
        return;
      }}
      const matchCount = DATA.nodes.filter((node) => node.action_name === activeActionFilter).length;
      actionMatchMeta.textContent = `${{matchCount}} (${{activeActionFilter}})`;
    }}

    function parseStepRange(text) {{
      const cleaned = text.trim();
      if (!cleaned) {{
        return null;
      }}
      const match = cleaned.match(/^(\\d+)(?:\\s*-\\s*(\\d+))?$/);
      if (!match) {{
        return "invalid";
      }}
      const start = Number(match[1]);
      const end = match[2] ? Number(match[2]) : start;
      return {{
        start: Math.min(start, end),
        end: Math.max(start, end),
      }};
    }}

    function worldToScreen(x, y) {{
      return {{
        x: x * transform.scale + transform.offsetX,
        y: y * transform.scale + transform.offsetY,
      }};
    }}

    function screenToWorld(x, y) {{
      return {{
        x: (x - transform.offsetX) / transform.scale,
        y: (y - transform.offsetY) / transform.scale,
      }};
    }}

    function zoomAtPoint(mouseX, mouseY, zoomFactor) {{
      const before = screenToWorld(mouseX, mouseY);
      transform.scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, transform.scale * zoomFactor));
      transform.offsetX = mouseX - before.x * transform.scale;
      transform.offsetY = mouseY - before.y * transform.scale;
      draw();
    }}

    function drawEdge(edge, strokeStyle, alpha, lineWidth) {{
      const [a, b] = edge;
      const src = DATA.nodes[a];
      const dst = DATA.nodes[b];
      const p1 = worldToScreen(src.x, src.y);
      const p2 = worldToScreen(dst.x, dst.y);
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.globalAlpha = alpha;
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = lineWidth;
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    }}

    function drawNode(node, isHovered, isPinned, isNeighbor, hasRewardRing, inStepRange, isFirstUnlock, actionMatched) {{
      const p = worldToScreen(node.x, node.y);
      const radius = isPinned ? 8.5 : isHovered ? 7.5 : isNeighbor ? 6.8 : 5.8;
      if (inStepRange) {{
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius + 6.5, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(45, 52, 65, 0.92)";
        ctx.lineWidth = 3.2;
        ctx.stroke();
      }}
      if (hasRewardRing) {{
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius + (inStepRange ? 11 : 6.5), 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(202, 162, 74, 0.95)";
        ctx.lineWidth = 2.4;
        ctx.stroke();
      }}
      if (isFirstUnlock) {{
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius + (hasRewardRing ? 15 : 10), 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(181, 82, 79, 0.96)";
        ctx.lineWidth = 3.2;
        ctx.stroke();
      }}
      if (actionMatched) {{
        ctx.beginPath();
        ctx.arc(p.x, p.y, radius + 18.5, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(91, 79, 168, 0.92)";
        ctx.lineWidth = 2.8;
        ctx.stroke();
      }}
      ctx.beginPath();
      ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
      ctx.fillStyle = isFirstUnlock ? "#b5524f" : valueColor(node.value);
      ctx.fill();
      if (isHovered || isPinned) {{
        ctx.lineWidth = 2.2;
        ctx.strokeStyle = "#223127";
        ctx.stroke();
      }}
    }}

    function draw() {{
      const rect = canvas.getBoundingClientRect();
      ctx.clearRect(0, 0, rect.width, rect.height);

      const focusNode = pinnedNode ?? hoveredNode;
      const highlighted = new Set();
      if (focusNode) {{
        highlighted.add(focusNode.id);
        for (const neighbor of adjacency.get(focusNode.id) || []) {{
          highlighted.add(neighbor.id);
        }}
      }}

      for (const edge of DATA.edges.temporal) {{
        const strong = focusNode && (edge[0] === focusNode.id || edge[1] === focusNode.id);
        drawEdge(edge, "#768b78", strong ? 0.85 : 0.18, strong ? 1.8 : 1.0);
      }}
      for (const edge of DATA.edges.value) {{
        const strong = focusNode && (edge[0] === focusNode.id || edge[1] === focusNode.id);
        drawEdge(edge, "#ca7f45", strong ? 0.78 : 0.12, strong ? 1.7 : 0.9);
      }}

      for (const node of DATA.nodes) {{
        drawNode(
          node,
          hoveredNode && node.id === hoveredNode.id,
          pinnedNode && node.id === pinnedNode.id,
          focusNode ? highlighted.has(node.id) && node.id !== focusNode.id : false,
          rewardRingToggle.checked && hasReward(node),
          isInActiveStepRange(node),
          hasNewAchievement(node),
          activeActionFilter && node.action_name === activeActionFilter,
        );
      }}
    }}

    function decodeImageBytes(base64String) {{
      const binary = atob(base64String);
      const bytes = new Uint8ClampedArray(binary.length);
      for (let i = 0; i < binary.length; i += 1) {{
        bytes[i] = binary.charCodeAt(i);
      }}
      return bytes;
    }}

    function renderPreview(node) {{
      previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
      previewCtx.fillStyle = "#efe6d5";
      previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);

      if (!node) {{
        previewCtx.fillStyle = "#536454";
        previewCtx.font = "18px Georgia";
        previewCtx.textAlign = "center";
        previewCtx.fillText("Hover a node", previewCanvas.width / 2, previewCanvas.height / 2);
        nodeEpisode.textContent = "-";
        nodeStep.textContent = "-";
        nodeValue.textContent = "-";
        nodeReward.textContent = "-";
        nodeAction.textContent = "-";
        nodeDone.textContent = "-";
        nodeNewTasks.textContent = "No new achievements at this step.";
        nodeTasks.textContent = "Hover a node to inspect its state.";
        return;
      }}

      const bytes = decodeImageBytes(node.image_bytes);
      const imageData = new ImageData(bytes, node.image_width, node.image_height);
      const tempCanvas = document.createElement("canvas");
      tempCanvas.width = node.image_width;
      tempCanvas.height = node.image_height;
      tempCanvas.getContext("2d").putImageData(imageData, 0, 0);

      const scale = Math.min(previewCanvas.width / node.image_width, previewCanvas.height / node.image_height);
      const drawW = node.image_width * scale;
      const drawH = node.image_height * scale;
      const drawX = (previewCanvas.width - drawW) / 2;
      const drawY = (previewCanvas.height - drawH) / 2;
      previewCtx.imageSmoothingEnabled = false;
      previewCtx.drawImage(tempCanvas, drawX, drawY, drawW, drawH);

      nodeEpisode.textContent = String(node.episode_id);
      nodeStep.textContent = String(node.step_id);
      nodeValue.textContent = node.value.toFixed(3);
      nodeReward.textContent = node.reward.toFixed(3);
      nodeAction.textContent = node.action_name;
      nodeDone.textContent = node.done ? "yes" : "no";
      nodeNewTasks.textContent = node.new_achievements.length ? node.new_achievements.join(", ") : "No new achievements at this step.";
      nodeTasks.textContent = node.achieved_tasks.length ? node.achieved_tasks.join(", ") : "No achievements unlocked yet.";
    }}

    function findNodeAtScreenPoint(x, y) {{
      let bestNode = null;
      let bestDist = Infinity;
      for (const node of DATA.nodes) {{
        const p = worldToScreen(node.x, node.y);
        const dx = p.x - x;
        const dy = p.y - y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 10 && dist < bestDist) {{
          bestDist = dist;
          bestNode = node;
        }}
      }}
      return bestNode;
    }}

    canvas.addEventListener("mousedown", (event) => {{
      dragging = true;
      lastMouse = {{ x: event.clientX, y: event.clientY }};
      canvas.classList.add("dragging");
    }});

    window.addEventListener("mouseup", () => {{
      dragging = false;
      lastMouse = null;
      canvas.classList.remove("dragging");
    }});

    window.addEventListener("mousemove", (event) => {{
      const rect = canvas.getBoundingClientRect();
      if (dragging && lastMouse) {{
        transform.offsetX += event.clientX - lastMouse.x;
        transform.offsetY += event.clientY - lastMouse.y;
        lastMouse = {{ x: event.clientX, y: event.clientY }};
        draw();
        return;
      }}

      if (
        event.clientX < rect.left || event.clientX > rect.right ||
        event.clientY < rect.top || event.clientY > rect.bottom
      ) {{
        return;
      }}

      hoveredNode = findNodeAtScreenPoint(event.clientX - rect.left, event.clientY - rect.top);
      renderPreview(pinnedNode ?? hoveredNode);
      draw();
    }});

    canvas.addEventListener("mouseleave", () => {{
      hoveredNode = null;
      renderPreview(pinnedNode);
      draw();
    }});

    canvas.addEventListener("click", (event) => {{
      const rect = canvas.getBoundingClientRect();
      const node = findNodeAtScreenPoint(event.clientX - rect.left, event.clientY - rect.top);
      pinnedNode = node;
      renderPreview(pinnedNode ?? hoveredNode);
      draw();
    }});

    canvas.addEventListener("wheel", (event) => {{
      event.preventDefault();
      const rect = canvas.getBoundingClientRect();
      const mouseX = event.clientX - rect.left;
      const mouseY = event.clientY - rect.top;
      const intensity = event.ctrlKey ? 0.0035 : 0.0018;
      const zoomFactor = Math.exp(-event.deltaY * intensity);
      zoomAtPoint(mouseX, mouseY, zoomFactor);
    }}, {{ passive: false }});

    window.addEventListener("keydown", (event) => {{
      if (event.key === "Escape") {{
        pinnedNode = null;
        renderPreview(hoveredNode);
        draw();
        return;
      }}

      const rect = canvas.getBoundingClientRect();
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      if (event.key === "+" || event.key === "=") {{
        zoomAtPoint(centerX, centerY, 1.25);
        return;
      }}
      if (event.key === "-" || event.key === "_") {{
        zoomAtPoint(centerX, centerY, 0.8);
        return;
      }}
      if (event.key === "0") {{
        transform.offsetX = 0;
        transform.offsetY = 0;
        resizeCanvas();
      }}
    }});

    rewardRingToggle.addEventListener("change", () => {{
      draw();
    }});

    actionFilterSelect.addEventListener("change", () => {{
      activeActionFilter = actionFilterSelect.value;
      updateActionMatchStatus();
      draw();
    }});

    applyStepRangeButton.addEventListener("click", () => {{
      const parsed = parseStepRange(stepRangeInput.value);
      if (parsed === "invalid") {{
        activeStepRange = null;
        updateStepRangeStatus("Use a single step like 42 or a range like 181-187.");
        draw();
        return;
      }}
      if (parsed === null) {{
        activeStepRange = null;
        updateStepRangeStatus("No step range highlighted.");
        draw();
        return;
      }}
      activeStepRange = parsed;
      updateStepRangeStatus(`Highlighting steps ${{parsed.start}}-${{parsed.end}}.`);
      draw();
    }});

    clearStepRangeButton.addEventListener("click", () => {{
      activeStepRange = null;
      stepRangeInput.value = "";
      updateStepRangeStatus("No step range highlighted.");
      draw();
    }});

    stepRangeInput.addEventListener("keydown", (event) => {{
      if (event.key === "Enter") {{
        applyStepRangeButton.click();
      }}
    }});

    renderPreview(null);
    updateActionMatchStatus();
    resizeCanvas();
    window.addEventListener("resize", resizeCanvas);
  </script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def collect_value_dataset(
    model,
    device: th.device,
    num_episodes: int,
    eval_seed: int,
    video_directory: Optional[str] = None,
) -> Tuple[Dict[str, th.Tensor], Optional[str]]:
    from crafter.env import Env
    from crafter.recorder import VideoRecorder
    from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

    from achievement_distillation.constant import TASKS
    from achievement_distillation.wrapper import VecPyTorch

    latest_video_path = None

    def make_env():
        env = Env(seed=eval_seed)
        if video_directory:
            env = VideoRecorder(env, directory=video_directory)
        return env

    venv = DummyVecEnv([make_env])
    venv = VecPyTorch(venv, device=device)

    observations: List[th.Tensor] = []
    latents: List[th.Tensor] = []
    values: List[th.Tensor] = []
    actions: List[th.Tensor] = []
    rewards: List[float] = []
    dones: List[bool] = []
    achievements: List[th.Tensor] = []
    success_flags: List[th.Tensor] = []
    episode_ids: List[int] = []
    step_ids: List[int] = []

    for episode_idx in range(num_episodes):
        try:
            venv.env_method("seed", eval_seed + episode_idx)
        except Exception:
            pass

        obs = venv.reset()
        done = False
        step_idx = 0
        while not done:
            with th.no_grad():
                outputs = model.act(obs)
                action = outputs["actions"]
                value = outputs["vpreds"]
                latent = outputs["latents"]

            next_obs, reward, done_tensor, infos = venv.step(action)

            observations.append(obs.squeeze(0).detach().cpu())
            latents.append(latent.squeeze(0).detach().cpu())
            values.append(value.squeeze(0).detach().cpu())
            actions.append(action.squeeze(0).detach().cpu())
            rewards.append(float(reward.item()))
            dones.append(bool(done_tensor.item()))
            achievements.append(infos["achievements"].squeeze(0).detach().cpu())
            success_flags.append(infos["successes"].squeeze(0).detach().cpu())
            episode_ids.append(episode_idx)
            step_ids.append(step_idx)

            obs = next_obs
            done = bool(done_tensor.item())
            step_idx += 1

        if video_directory:
            latest_video_path = find_latest_mp4(video_directory)
    venv.close()

    dataset = {
        "observations": th.stack(observations),
        "latents": th.stack(latents),
        "values": th.stack(values).view(-1),
        "actions": th.stack(actions).view(-1),
        "rewards": th.tensor(rewards, dtype=th.float32),
        "dones": th.tensor(dones, dtype=th.bool),
        "achievements": th.stack(achievements),
        "successes": th.stack(success_flags),
        "episode_ids": th.tensor(episode_ids, dtype=th.long),
        "step_ids": th.tensor(step_ids, dtype=th.long),
        "task_names": TASKS,
    }
    return dataset, latest_video_path


def save_value_map(dataset: Dict[str, th.Tensor], output_path: str, max_points: int = 5000):
    plt = importlib.import_module("matplotlib.pyplot")

    latents = dataset["latents"].cpu().numpy()
    values = dataset["values"].cpu().numpy()
    episode_ids = dataset["episode_ids"].cpu().numpy()

    if len(latents) > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(latents), size=max_points, replace=False)
        latents = latents[idx]
        values = values[idx]
        episode_ids = episode_ids[idx]

    coords = pca_2d(latents)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        coords[:, 0],
        coords[:, 1],
        c=values,
        cmap="viridis",
        s=12,
        alpha=0.8,
        linewidths=0,
    )
    plt.colorbar(scatter, label="Predicted value")
    plt.xlabel("Latent PC 1")
    plt.ylabel("Latent PC 2")
    plt.title("State-value map from rollout states")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()

    base, ext = os.path.splitext(output_path)
    if ext.lower() not in {".png", ".jpg", ".jpeg", ".pdf"}:
        base = output_path
    np.savez_compressed(
        f"{base}_embedding.npz",
        coords=coords,
        values=values,
        episode_ids=episode_ids,
    )


def main():
    parser = argparse.ArgumentParser(description="Collect rollout states and export predicted values from a trained Crafter agent.")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True)
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--output_dataset_path", type=str, default="value_dataset.pt")
    parser.add_argument("--value_map_path", type=str, default=None)
    parser.add_argument("--value_map_max_points", type=int, default=5000)
    parser.add_argument("--value_graph_html_path", type=str, default=None)
    parser.add_argument("--value_graph_max_states", type=int, default=400)
    parser.add_argument("--value_graph_num_neighbors", type=int, default=4)
    parser.add_argument("--value_graph_value_threshold", type=float, default=None)
    parser.add_argument("--episode_video_dir", type=str, default=None)
    args = parser.parse_args()

    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    th.set_num_threads(1)
    set_seed(args.eval_seed)

    model, _, ckpt_path = load_model(
        args.exp_name,
        args.timestamp,
        args.train_seed,
        args.ckpt_epoch,
        device,
    )
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Using device: {device}")

    dataset, video_path = collect_value_dataset(
        model=model,
        device=device,
        num_episodes=args.num_episodes,
        eval_seed=args.eval_seed,
        video_directory=args.episode_video_dir,
    )

    th.save(dataset, args.output_dataset_path)
    print(f"Saved dataset with {len(dataset['values'])} states to {args.output_dataset_path}")
    print(f"Latents shape: {tuple(dataset['latents'].shape)}")
    print(
        f"Value stats: mean={dataset['values'].mean().item():.4f}, "
        f"std={dataset['values'].std().item():.4f}, "
        f"min={dataset['values'].min().item():.4f}, "
        f"max={dataset['values'].max().item():.4f}"
    )

    if args.value_map_path:
        save_value_map(dataset, args.value_map_path, max_points=args.value_map_max_points)
        print(f"Saved value-map figure to {args.value_map_path}")
        print(f"Saved embedding data to {os.path.splitext(args.value_map_path)[0]}_embedding.npz")

    if args.value_graph_html_path:
        save_value_graph_viewer(
            dataset,
            output_path=args.value_graph_html_path,
            max_states=args.value_graph_max_states,
            num_neighbors=args.value_graph_num_neighbors,
            value_threshold=args.value_graph_value_threshold,
        )
        print(f"Saved interactive value-graph viewer to {args.value_graph_html_path}")

    if video_path:
        print(f"Saved rollout video to {video_path}")

if __name__ == "__main__":
    main()
