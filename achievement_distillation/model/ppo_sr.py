from __future__ import annotations

from typing import Dict

import torch as th
import torch.nn.functional as F

from gym import spaces

from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.mlp import MLP
from achievement_distillation.model.ppo import PPOModel
from achievement_distillation.mse_head import ScaledMSEHead


class PPOSRModel(PPOModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        impala_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        action_head_kwargs: Dict = {},
        mse_head_kwargs: Dict = {}
    ):
        super().__init__(
            observation_space,
            action_space,
            hidsize,
            impala_kwargs=impala_kwargs,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
        )

        # Heads
        num_actions = getattr(self.action_space, "n")
        pi_latent_size = hidsize
        vf_latent_size = hidsize
        self.pi_head = CategoricalActionHead(
            insize=pi_latent_size,
            num_actions=num_actions,
            **action_head_kwargs,
        )
        self.vf_head = ScaledMSEHead(
            insize=vf_latent_size,
            outsize=1,
            **mse_head_kwargs,
        )
        self.survival_risk_head = nn.Linear(hidsize, 3)

       

    def forward(self, obs: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        # Pass through encoder
        latents = self.encode(obs)

        # Pass through heads
        pi_latents = vf_latents = latents
        pi_logits = self.pi_head(pi_latents)
        vpreds = self.vf_head(vf_latents)

        vrisk_preds = self.survival_risk_head(latents)


        # Define outputs
        outputs = {
            "latents": latents,
            "pi_latents": pi_latents,
            "vf_latents": vf_latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
            "vrisk_preds": vrisk_preds,
        }

        return outputs

    def compute_survival_loss(
        self,
        obs: th.Tensor,
        risk_targets: th.Tensor,
        old_model: PPOSRModel,
    ) -> Dict[str, th.Tensor]:
        # 1. Forward pass with current model
        # We need gradients here to update the encoder
        outputs = self.forward(obs)
        risk_logits = outputs["vrisk_preds"]
        pi_logits = outputs["pi_logits"]
        vpreds = outputs["vpreds"]

        # 2. Forward pass with old model
        # We do NOT want gradients here; this is our frozen reference point
        with th.no_grad():
            old_outputs = old_model.forward(obs)
            old_pi_logits = old_outputs["pi_logits"]
            old_vpreds = old_outputs["vpreds"]

        # 3. Survival Risk Loss (Cross Entropy)
        # Compares predicted risk class logits vs ground truth class (0, 1, 2)
        loss_survival = F.cross_entropy(risk_logits, risk_targets)

        # 4. Policy Regularizer (KL Divergence)
        # Ensures the new encoder features don't break the existing policy behavior
        pi_dist = self.pi_head.kl_divergence(pi_logits, old_pi_logits).mean()

        # 5. Value Regularizer (MSE)
        # Ensures the value function estimates remain stable
        vf_dist = F.mse_loss(vpreds, old_vpreds)

        return {
            "loss_survival": loss_survival,
            "pi_dist": pi_dist,
            "vf_dist": vf_dist,
        }
    
