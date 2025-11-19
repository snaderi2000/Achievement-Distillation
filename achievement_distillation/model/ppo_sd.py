from __future__ import annotations

from achievement_distillation.model.ppo_ad import PPOADModel


class PPOSDModel(PPOADModel):
    """
    Survival-distillation variant of the PPO model.

    The architecture and auxiliary heads mirror the achievement-distillation
    model. The main difference now lives in how the algorithm constructs goals
    (via vitals) rather than within the network itself, so we simply inherit the
    full PPOADModel implementation.
    """

    pass
