from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch as th


@dataclass
class HealthTemplates:
    templates: th.Tensor
    mask: th.Tensor


_HEALTH_TEMPLATES_CACHE: dict[tuple[int, int], HealthTemplates] = {}


def _render_health_templates(size_hw: tuple[int, int]) -> HealthTemplates:
    from crafter.env import Env

    env = Env(seed=0)
    try:
        env.reset()
        # Fix other vitals so only health changes across renders.
        env._player.inventory["food"] = 9
        env._player.inventory["drink"] = 9
        env._player.inventory["energy"] = 9

        templates = []
        for health in range(10):
            env._player.health = health
            env._player.inventory["health"] = health
            obs = env.render()
            if obs.shape[:2] != size_hw:
                raise ValueError(
                    f"Crafter render size {obs.shape[:2]} does not match expected {size_hw}."
                )
            obs_chw = np.transpose(obs, (2, 0, 1)).astype(np.float32) / 255.0
            templates.append(th.from_numpy(obs_chw))

        templates_t = th.stack(templates, dim=0)  # [10, C, H, W]
        ref = templates_t[0]
        diffs = (templates_t - ref.unsqueeze(0)).abs()
        mask = diffs.max(dim=0).values > 1e-6
        return HealthTemplates(templates=templates_t, mask=mask)
    finally:
        env.close()


def get_health_templates(size_hw: tuple[int, int], device: Optional[th.device] = None) -> HealthTemplates:
    cache_key = tuple(size_hw)
    if cache_key not in _HEALTH_TEMPLATES_CACHE:
        _HEALTH_TEMPLATES_CACHE[cache_key] = _render_health_templates(cache_key)

    cached = _HEALTH_TEMPLATES_CACHE[cache_key]
    if device is None:
        return cached
    return HealthTemplates(
        templates=cached.templates.to(device),
        mask=cached.mask.to(device),
    )


def apply_health_counterfactual(
    obs: th.Tensor,
    target_health: th.Tensor,
    templates: HealthTemplates,
) -> th.Tensor:
    if obs.dim() != 4:
        raise ValueError(f"Expected obs shape [B,C,H,W], got {tuple(obs.shape)}")

    target_health = target_health.long().clamp(min=0, max=9)
    edited = obs.clone()
    template_obs = templates.templates[target_health]  # [B,C,H,W]
    mask = templates.mask.unsqueeze(0).expand_as(edited)
    edited[mask] = template_obs[mask]
    return edited
