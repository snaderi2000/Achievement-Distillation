from __future__ import annotations

from typing import Dict

import torch as th
import torch.nn.functional as F

from gym import spaces

from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.mlp import MLP
from achievement_distillation.model.ppo import PPOModel
from achievement_distillation.mse_head import ScaledMSEHead


class PPOSDModel(PPOModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        impala_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        action_head_kwargs: Dict = {},
        mse_head_kwargs: Dict = {},
        nhidlayer: int = 1,
        temperature: float = 0.1,
        use_memory: bool = True,
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
        pi_latent_size = 2 * hidsize if use_memory else hidsize
        vf_latent_size = 2 * hidsize if use_memory else hidsize
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

        # SD params
        self.use_memory = use_memory
        self.temperature = temperature

        # SD layers
        self.next_goal_pred_mlp = MLP(
            insize=2 * hidsize,
            nhidlayer=nhidlayer,
            outsize=hidsize,
            hidsize=hidsize,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
        )

    def forward(self, obs: th.Tensor, states: th.Tensor) -> Dict[str, th.Tensor]:
        # Pass through encoder
        latents = self.encode(obs)

        # Concatenate latents and states
        if self.use_memory:
            # Assuming states are passed and handled appropriately upstream
            pi_latents = vf_latents = th.cat([latents, states], dim=-1)
        else:
            pi_latents = vf_latents = latents

        # Pass through heads
        pi_logits = self.pi_head(pi_latents)
        vpreds = self.vf_head(vf_latents)

        # Define outputs
        outputs = {
            "latents": latents,
            "pi_latents": pi_latents,
            "vf_latents": vf_latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
        }

        return outputs

    def compute_pred_losses(
        self,
        obs: th.Tensor,
        next_obs: th.Tensor,
        old_model: PPOADModel,
    ) -> Dict[str, th.Tensor]:
        # Encode observations
        latents = self.encode(obs)
        
        with th.no_grad():
            next_latents = old_model.encode(next_obs)
            states = F.normalize(next_latents - latents, dim=-1)

        # Make predictions
        # For memory=True, we should ideally pass a state. Using zeros for now.
        dummy_states = th.zeros_like(latents)
        preds_input = th.cat([latents, dummy_states], dim=-1)
        preds = self.next_goal_pred_mlp(preds_input)
        preds = F.normalize(preds, dim=-1)

        # Create negative samples by shifting the states tensor
        neg_states = th.roll(states, shifts=1, dims=0)

        # Compute prediction loss (contrastive)
        pos_logits = th.einsum("bk,bk->b", states, preds)
        neg_logits = th.einsum("bk,bk->b", neg_states, preds)
        logits = th.stack([pos_logits, neg_logits], dim=-1)
        logits = logits / self.temperature
        targets = th.zeros(len(logits), device=logits.device).long()
        pred_loss = F.cross_entropy(logits, targets)

        # Policy and value distillation are not directly applicable
        # with this simplified data loader. Returning zero losses.
        pi_dist = th.tensor(0.0, device=obs.device)
        vf_dist = th.tensor(0.0, device=obs.device)

        return {
            "pred_loss": pred_loss,
            "pi_dist": pi_dist,
            "vf_dist": vf_dist,
        }
