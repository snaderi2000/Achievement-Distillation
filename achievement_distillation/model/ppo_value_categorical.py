from typing import Dict

import torch as th

from gym import spaces

from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.categorical_value_head import CategoricalValueHead
from achievement_distillation.impala_cnn import ImpalaCNN
from achievement_distillation.model.base import BaseModel
from achievement_distillation.torch_util import FanInInitReLULayer


class PPOValueCategoricalModel(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        impala_kwargs: Dict = {},
        value_hidsize: int = 0,
        dense_init_norm_kwargs: Dict = {},
        action_head_kwargs: Dict = {},
        value_head_kwargs: Dict = {},
    ):
        super().__init__(observation_space, action_space)

        obs_shape = getattr(self.observation_space, "shape")
        self.enc = ImpalaCNN(
            obs_shape,
            dense_init_norm_kwargs=dense_init_norm_kwargs,
            **impala_kwargs,
        )
        outsize = impala_kwargs["outsize"]
        self.linear = FanInInitReLULayer(
            outsize,
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

        self.vf_tower = None
        vf_insize = hidsize
        if value_hidsize > 0:
            self.vf_tower = FanInInitReLULayer(
                hidsize,
                value_hidsize,
                layer_type="linear",
                **dense_init_norm_kwargs,
            )
            vf_insize = value_hidsize
        self.vf_head = CategoricalValueHead(
            insize=vf_insize,
            **value_head_kwargs,
        )

    @th.no_grad()
    def act(self, obs: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        assert not self.training
        outputs = self.forward(obs, **kwargs)
        pi_logits = outputs["pi_logits"]
        actions = self.pi_head.sample(pi_logits)
        log_probs = self.pi_head.log_prob(pi_logits, actions)
        outputs.update({"actions": actions, "log_probs": log_probs})
        return outputs

    def forward(self, obs: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        latents = self.encode(obs)
        pi_latents = latents
        vf_latents = self.vf_tower(latents) if self.vf_tower is not None else latents
        pi_logits = self.pi_head(pi_latents)
        value_logits = self.vf_head(vf_latents)
        vpreds = self.vf_head.expected_value(value_logits)
        return {
            "latents": latents,
            "pi_latents": pi_latents,
            "vf_latents": vf_latents,
            "pi_logits": pi_logits,
            "value_logits": value_logits,
            "vpreds": vpreds,
        }

    def encode(self, obs: th.Tensor) -> th.Tensor:
        x = self.enc(obs)
        x = self.linear(x)
        return x

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
        vf_loss = self.vf_head.cross_entropy_loss(outputs["value_logits"], vtargs).mean()

        return {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}
