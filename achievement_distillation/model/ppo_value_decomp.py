from typing import Dict

import torch as th
import torch.nn.functional as F

from gym import spaces

from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.impala_cnn import ImpalaCNN
from achievement_distillation.model.base import BaseModel
from achievement_distillation.mse_head import ScaledMSEHead
from achievement_distillation.torch_util import FanInInitReLULayer


class PPOValueDecompModel(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        value_hidsize: int = 1024,
        use_survival_targets: bool = False,
        impala_kwargs: Dict = {},
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
        outsize = impala_kwargs["outsize"]
        self.linear = FanInInitReLULayer(
            outsize,
            hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )
        self.hidsize = hidsize
        self.use_survival_targets = use_survival_targets

        num_actions = getattr(self.action_space, "n")
        self.pi_head = CategoricalActionHead(
            insize=hidsize,
            num_actions=num_actions,
            **action_head_kwargs,
        )

        self.health_vf_tower = FanInInitReLULayer(
            hidsize,
            value_hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )
        self.achievement_vf_tower = FanInInitReLULayer(
            hidsize,
            value_hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )
        self.health_vf_head = ScaledMSEHead(
            insize=value_hidsize,
            outsize=1,
            **mse_head_kwargs,
        )
        self.achievement_vf_head = ScaledMSEHead(
            insize=value_hidsize,
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
        outputs.update({"actions": actions, "log_probs": log_probs})
        return outputs

    def forward(self, obs: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        latents = self.encode(obs)
        pi_latents = latents
        health_vf_latents = self.health_vf_tower(latents)
        achievement_vf_latents = self.achievement_vf_tower(latents)
        pi_logits = self.pi_head(pi_latents)
        health_vpreds_raw = self.health_vf_head(health_vf_latents)
        achievement_vpreds_raw = self.achievement_vf_head(achievement_vf_latents)
        health_vpreds = self.health_vf_head.denormalize(health_vpreds_raw)
        achievement_vpreds = self.achievement_vf_head.denormalize(achievement_vpreds_raw)
        vpreds = health_vpreds + achievement_vpreds
        return {
            "latents": latents,
            "pi_latents": pi_latents,
            "health_vf_latents": health_vf_latents,
            "achievement_vf_latents": achievement_vf_latents,
            "pi_logits": pi_logits,
            "health_vpreds_raw": health_vpreds_raw,
            "achievement_vpreds_raw": achievement_vpreds_raw,
            "health_vpreds": health_vpreds,
            "achievement_vpreds": achievement_vpreds,
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
        achievement_vtargs: th.Tensor,
        health_vtargs: th.Tensor,
        survival_vtargs: th.Tensor | None = None,
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

        total_vpreds = outputs["vpreds"]
        vf_loss = F.mse_loss(total_vpreds, vtargs)
        health_targets = survival_vtargs if (self.use_survival_targets and survival_vtargs is not None) else health_vtargs
        health_vf_loss = self.health_vf_head.mse_loss(outputs["health_vpreds_raw"], health_targets).mean()
        achievement_vf_loss = self.achievement_vf_head.mse_loss(
            outputs["achievement_vpreds_raw"],
            achievement_vtargs,
        ).mean()

        return {
            "pi_loss": pi_loss,
            "vf_loss": vf_loss,
            "entropy": entropy,
            "health_vf_loss": health_vf_loss,
            "achievement_vf_loss": achievement_vf_loss,
        }
