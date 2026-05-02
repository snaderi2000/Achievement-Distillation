import copy
from typing import Dict, List, Sequence, Tuple

import numpy as np


def mirrored_cell_index(ix: int, iy: int, grid_shape: Sequence[int], mode: str) -> Tuple[int, int]:
    if mode == "horizontal":
        return int(grid_shape[0] - 1 - ix), int(iy)
    if mode == "vertical":
        return int(ix), int(grid_shape[1] - 1 - iy)
    if mode == "both":
        return int(grid_shape[0] - 1 - ix), int(grid_shape[1] - 1 - iy)
    raise ValueError("Semantic flip mode must be one of: none, horizontal, vertical, both.")


def flip_vector(vec, mode: str):
    arr = np.asarray(vec)
    if arr.shape == () or arr.size < 2 or not np.issubdtype(arr.dtype, np.number):
        return vec
    flipped = arr.copy()
    if mode in ("horizontal", "both"):
        flipped[0] *= -1
    if mode in ("vertical", "both"):
        flipped[1] *= -1
    if isinstance(vec, np.ndarray):
        return flipped.astype(vec.dtype, copy=False)
    if isinstance(vec, tuple):
        return tuple(flipped.tolist())
    if isinstance(vec, list):
        return flipped.tolist()
    return flipped


def flip_object_direction_attrs(obj, mode: str):
    for attr in ("direction", "_direction", "facing", "_facing", "dir", "_dir"):
        if not hasattr(obj, attr):
            continue
        value = getattr(obj, attr)
        try:
            setattr(obj, attr, flip_vector(value, mode))
        except Exception:
            pass


def visible_world_cells(env) -> List[Tuple[Tuple[int, int], Tuple[int, int], str, object]]:
    local_grid = np.asarray(env._local_view._grid, dtype=np.int64)
    center_index = local_grid // 2
    center_world = np.asarray(env._player.pos, dtype=np.int64)
    world_area = np.asarray(env._world.area, dtype=np.int64)

    cells = []
    for ix in range(int(local_grid[0])):
        for iy in range(int(local_grid[1])):
            delta = np.array([ix, iy], dtype=np.int64) - center_index
            pos = center_world + delta
            if 0 <= pos[0] < world_area[0] and 0 <= pos[1] < world_area[1]:
                material, obj = env._world[tuple(pos)]
                cells.append(((ix, iy), (int(pos[0]), int(pos[1])), material, obj))
    return cells


def apply_flip_visible_world_state(env, mode: str, edits_log: List[str] | None = None):
    if mode == "none":
        return
    local_grid = np.asarray(env._local_view._grid, dtype=np.int64)
    cells = visible_world_cells(env)
    materials: Dict[Tuple[int, int], str] = {}
    objects = {}
    cell_to_world = {}
    for cell_index, world_pos, material, obj in cells:
        materials[cell_index] = material
        cell_to_world[cell_index] = tuple(world_pos)
        if obj is not None and obj is not env._player:
            objects[cell_index] = obj

    for obj in objects.values():
        env._world.remove(obj)
        # Crafter marks removed objects as inert; reset before re-inserting.
        if hasattr(obj, "removed"):
            obj.removed = False

    for cell_index, material in materials.items():
        target_cell = mirrored_cell_index(cell_index[0], cell_index[1], local_grid, mode)
        target_pos = cell_to_world.get(target_cell)
        if target_pos is not None:
            env._world[target_pos] = material

    moved = 0
    for cell_index, obj in objects.items():
        target_cell = mirrored_cell_index(cell_index[0], cell_index[1], local_grid, mode)
        target_pos = cell_to_world.get(target_cell)
        if target_pos is None:
            continue
        _, target_obj = env._world[target_pos]
        if target_obj is not None:
            raise ValueError(
                f"Cannot flip {type(obj).__name__} from visible cell {cell_index} to occupied cell "
                f"{target_cell} at {target_pos} containing {type(target_obj).__name__}."
            )
        flip_object_direction_attrs(obj, mode)
        obj.pos = np.array(target_pos, dtype=np.int64)
        env._world.add(obj)
        moved += 1

    flip_object_direction_attrs(env._player, mode)
    if edits_log is not None:
        edits_log.append(f"flip_visible_world_state={mode}, objects_moved={moved}")


class SemanticFlipEnv:
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, "metadata", {})
        self.reward_range = getattr(env, "reward_range", None)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def step(self, action):
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def close(self):
        return self.env.close()

    def seed(self, seed=None):
        if hasattr(self.env, "seed"):
            return self.env.seed(seed)
        return None

    def render_semantic_flip(self, mode: str):
        env_copy = copy.deepcopy(self.env)
        apply_flip_visible_world_state(env_copy, mode)
        obs = env_copy.render()
        env_copy.close()
        return obs
