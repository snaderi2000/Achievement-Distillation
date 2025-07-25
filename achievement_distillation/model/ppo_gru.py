from typing import Dict

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from gym import spaces

from achievement_distillation.model.base import BaseModel
from achievement_distillation.impala_cnn import ImpalaCNN
from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.mse_head import ScaledMSEHead
from achievement_distillation.torch_util import FanInInitReLULayer


class PPOGRUModel(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        rnn_hidsize: int,
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
        self.layer_norm = nn.LayerNorm(outsize)
        self.rnn_linear = nn.Linear(outsize, rnn_hidsize)
        self.gru = nn.GRUCell(rnn_hidsize, rnn_hidsize)
        self.rnn_hidsize = rnn_hidsize

        concat_size = outsize + rnn_hidsize
        self.linear = FanInInitReLULayer(
            concat_size,
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
    def act(self, obs: th.Tensor, rnn_states: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        assert not self.training
        outputs = self.forward(obs, rnn_states=rnn_states, **kwargs)
        pi_logits = outputs["pi_logits"]
        actions = self.pi_head.sample(pi_logits)
        log_probs = self.pi_head.log_prob(pi_logits, actions)
        vpreds = self.vf_head.denormalize(outputs["vpreds"])
        outputs.update({
            "actions": actions,
            "log_probs": log_probs,
            "vpreds": vpreds,
            "rnn_states": outputs["rnn_states"],
        })
        return outputs

    def forward(self, obs: th.Tensor, rnn_states: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        cnn_latents = self.encode(obs)
        rnn_input = F.relu(self.rnn_linear(cnn_latents))
        next_rnn_states = self.gru(rnn_input, rnn_states)
        rnn_output = F.relu(next_rnn_states)
        latents_in = th.cat([cnn_latents, rnn_output], dim=-1)
        latents = self.linear(latents_in)
        pi_logits = self.pi_head(latents)
        vpreds = self.vf_head(latents)
        outputs = {
            "latents": latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
            "rnn_states": next_rnn_states,
        }
        return outputs

    def encode(self, obs: th.Tensor) -> th.Tensor:
        x = self.enc(obs)
        x = self.layer_norm(x)
        return x

    def compute_losses(
        self,
        obs: th.Tensor,
        actions: th.Tensor,
        log_probs: th.Tensor,
        vtargs: th.Tensor,
        advs: th.Tensor,
        rnn_states: th.Tensor,
        clip_param: float = 0.2,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        outputs = self.forward(obs, rnn_states=rnn_states, **kwargs)
        pi_logits = outputs["pi_logits"]
        new_log_probs = self.pi_head.log_prob(pi_logits, actions)
        ratio = th.exp(new_log_probs - log_probs)
        ratio_clipped = th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pi_loss = -th.min(advs * ratio, advs * ratio_clipped).mean()

        entropy = self.pi_head.entropy(pi_logits).mean()

        vpreds = outputs["vpreds"]
        vf_loss = self.vf_head.mse_loss(vpreds, vtargs).mean()

        losses = {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}
        return losses
