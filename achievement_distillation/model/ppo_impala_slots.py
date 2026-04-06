from typing import Dict

import torch as th
import torch.nn as nn

from gym import spaces

from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.impala_cnn import ImpalaCNN
from achievement_distillation.model.base import BaseModel
from achievement_distillation.mse_head import ScaledMSEHead
from achievement_distillation.slot_attention import SlotAttention
from achievement_distillation.torch_util import FanInInitReLULayer


class PPOImpalaSlotsModel(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        impala_kwargs: Dict = {},
        slot_dim: int = 256,
        num_slots: int = 8,
        slot_iterations: int = 3,
        dense_init_norm_kwargs: Dict = {},
        action_head_kwargs: Dict = {},
        mse_head_kwargs: Dict = {},
    ):
        super().__init__(observation_space, action_space)

        obs_shape = getattr(self.observation_space, "shape")
        self.enc = ImpalaCNN(
            obs_shape,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **impala_kwargs,
        )
        c, h, w = self.enc.final_spatial_shape
        self.feature_channels = c
        self.feature_hw = (h, w)
        self.num_tokens = h * w

        self.token_proj = nn.Linear(c, slot_dim)
        self.pos_embedding = nn.Parameter(th.zeros(1, self.num_tokens, slot_dim))
        self.slot_attention = SlotAttention(
            input_dim=slot_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
            num_iterations=slot_iterations,
        )
        self.linear = FanInInitReLULayer(
            slot_dim,
            hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )
        self.hidsize = hidsize

        num_actions = getattr(self.action_space, "n")
        self.pi_head = CategoricalActionHead(
            insize=hidsize,
            num_actions=num_actions,
            **action_head_kwargs,
        )
        self.vf_head = ScaledMSEHead(
            insize=hidsize,
            outsize=1,
            **mse_head_kwargs,
        )

    @th.no_grad()
    def act(self, obs: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        assert not self.training
        outputs = self.forward(obs, **kwargs)
        pi_logits = outputs["pi_logits"]
        actions = self.pi_head.sample(pi_logits)
        log_probs = self.pi_head.log_prob(pi_logits, actions)
        vpreds = self.vf_head.denormalize(outputs["vpreds"])
        outputs.update({"actions": actions, "log_probs": log_probs, "vpreds": vpreds})
        return outputs

    def forward(self, obs: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        latents, slots, slot_attn = self.encode(obs, return_slots=True)
        pi_latents = vf_latents = latents
        pi_logits = self.pi_head(pi_latents)
        vpreds = self.vf_head(vf_latents)
        return {
            "latents": latents,
            "slots": slots,
            "slot_attn": slot_attn,
            "pi_latents": pi_latents,
            "vf_latents": vf_latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
        }

    def encode(self, obs: th.Tensor, return_slots: bool = False):
        x = self.enc.forward_features(obs)
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), self.num_tokens, self.feature_channels)
        x = self.token_proj(x) + self.pos_embedding
        slots, slot_attn = self.slot_attention(x)
        pooled_slots = slots.mean(dim=1)
        latents = self.linear(pooled_slots)
        if return_slots:
            return latents, slots, slot_attn
        return latents

    def compute_losses(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        log_probs: th.Tensor,
        vtargs: th.Tensor,
        advs: th.Tensor,
        clip_param: float = 0.2,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        outputs = self.forward(obs, **kwargs)
        pi_logits = outputs["pi_logits"]
        new_log_probs = self.pi_head.log_prob(pi_logits, actions)
        ratio = th.exp(new_log_probs - log_probs)
        ratio_clipped = th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pi_loss = -th.min(advs * ratio, advs * ratio_clipped).mean()
        entropy = self.pi_head.entropy(pi_logits).mean()
        vpreds = outputs["vpreds"]
        vf_loss = self.vf_head.mse_loss(vpreds, vtargs).mean()
        return {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}
