from typing import Dict

import torch as th
import torch.nn as nn

from gym import spaces

from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.impala_cnn import ImpalaCNN
from achievement_distillation.model.base import BaseModel
from achievement_distillation.mse_head import ScaledMSEHead
from achievement_distillation.torch_util import FanInInitReLULayer


class PPOGRUModel(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        rnn_hidsize: int = 256,
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
        self.rnn_hidsize = rnn_hidsize
        self.rnn_input = FanInInitReLULayer(
            hidsize,
            rnn_hidsize,
            layer_type="linear",
            **dense_init_norm_kwargs,
        )
        self.gru = nn.GRUCell(rnn_hidsize, rnn_hidsize)

        head_insize = hidsize + rnn_hidsize
        num_actions = getattr(self.action_space, "n")
        self.pi_head = CategoricalActionHead(
            insize=head_insize,
            num_actions=num_actions,
            **action_head_kwargs,
        )
        self.vf_head = ScaledMSEHead(
            insize=head_insize,
            outsize=1,
            **mse_head_kwargs,
        )

    @th.no_grad()
    def act(self, obs: th.Tensor, rnn_states: th.Tensor | None = None, **kwargs) -> Dict[str, th.Tensor]:
        assert not self.training
        outputs = self.forward(obs, rnn_states=rnn_states, **kwargs)
        pi_logits = outputs["pi_logits"]
        actions = self.pi_head.sample(pi_logits)
        log_probs = self.pi_head.log_prob(pi_logits, actions)
        vpreds = self.vf_head.denormalize(outputs["vpreds"])
        outputs.update(
            {
                "actions": actions,
                "log_probs": log_probs,
                "vpreds": vpreds,
            }
        )
        return outputs

    def forward(self, obs: th.Tensor, rnn_states: th.Tensor | None = None, **kwargs) -> Dict[str, th.Tensor]:
        latents = self.encode(obs)
        if rnn_states is None:
            rnn_states = th.zeros(latents.shape[0], self.rnn_hidsize, device=latents.device)

        rnn_inputs = self.rnn_input(latents)
        next_rnn_states = self.gru(rnn_inputs, rnn_states)
        next_rnn_states = th.relu(next_rnn_states)

        head_latents = th.cat([latents, next_rnn_states], dim=-1)
        pi_logits = self.pi_head(head_latents)
        vpreds = self.vf_head(head_latents)
        return {
            "latents": latents,
            "pi_latents": head_latents,
            "vf_latents": head_latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
            "next_rnn_states": next_rnn_states,
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
        rnn_states: th.Tensor | None = None,
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
        vf_loss = self.vf_head.mse_loss(outputs["vpreds"], vtargs).mean()
        return {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}
