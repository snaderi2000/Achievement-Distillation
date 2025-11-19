from __future__ import annotations

from typing import Dict

import torch as th
import torch.nn.functional as F

from achievement_distillation.model.ppo_ad import PPOADModel


class PPOSDModel(PPOADModel):
    """
    Survival-distillation variant of the PPO model.
    """

    def compute_pred_losses(
        self,
        anc_goal_obs: th.Tensor,
        anc_goal_next_obs: th.Tensor,
        pos_obs: th.Tensor,
        pos_actions: th.Tensor,
        pos_old_states: th.Tensor,
        pos_old_vtargs: th.Tensor,
        neg_obs: th.Tensor,
        neg_actions: th.Tensor,
        neg_old_states: th.Tensor,
        neg_old_vtargs: th.Tensor,
        old_model: PPOADModel,
    ) -> Dict[str, th.Tensor]:
        # Anchor goal representations
        anc_states = self.get_states(anc_goal_obs, anc_goal_next_obs)

        # Positive predictions from needy states
        pos_latents = self.encode(pos_obs)
        pos_preds = self.get_next_goal_preds(pos_latents, pos_actions, pos_old_states)

        # Negative predictions
        neg_latents = self.encode(neg_obs)
        neg_preds = self.get_next_goal_preds(neg_latents, neg_actions, neg_old_states)

        # Contrastive loss
        pos_logits = th.einsum("bk,bk->b", anc_states, pos_preds)
        neg_logits = th.einsum("bk,bk->b", anc_states, neg_preds)
        logits = th.stack([pos_logits, neg_logits], dim=-1)
        logits = logits / self.temperature
        targets = th.zeros(len(logits), device=logits.device).long()
        pred_loss = F.cross_entropy(logits, targets)

        # Policy distillation regularizer
        outputs = self.forward(pos_obs, states=pos_old_states)
        old_outputs = old_model.act(pos_obs, states=pos_old_states)
        pi_logits = outputs["pi_logits"]
        old_pi_logits = old_outputs["pi_logits"]
        pi_dist = self.pi_head.kl_divergence(pi_logits, old_pi_logits).mean()

        # Value distillation regularizer
        vpreds = outputs["vpreds"]
        vf_dist = self.vf_head.mse_loss(vpreds, pos_old_vtargs).mean()

        return {
            "pred_loss": pred_loss,
            "pi_dist": pi_dist,
            "vf_dist": vf_dist,
        }
