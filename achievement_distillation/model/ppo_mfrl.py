from typing import Dict

import torch as th
import torch.nn as nn

from gym import spaces

from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.impala_cnn import ImpalaCNN
from achievement_distillation.model.base import BaseModel
from achievement_distillation.mse_head import ScaledMSEHead
from achievement_distillation.torch_util import FanInInitReLULayer


class DenseResidualBlock(nn.Module):
    def __init__(self, hidsize: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidsize)
        self.fc1 = nn.Linear(hidsize, hidsize)
        self.norm2 = nn.LayerNorm(hidsize)
        self.fc2 = nn.Linear(hidsize, hidsize)

    def forward(self, x: th.Tensor) -> th.Tensor:
        residual = x
        x = self.norm1(x)
        x = th.relu(self.fc1(x))
        x = self.norm2(x)
        x = th.relu(self.fc2(x))
        return residual + x


class ResidualMLPHead(nn.Module):
    def __init__(self, insize: int, hidsize: int, nres_blocks: int, outsize: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(insize)
        self.fc = nn.Linear(insize, hidsize)
        self.resblocks = nn.ModuleList([DenseResidualBlock(hidsize) for _ in range(nres_blocks)])
        self.output_norm = nn.LayerNorm(hidsize)
        self.outsize = outsize

    def forward_features(self, x: th.Tensor) -> th.Tensor:
        x = self.input_norm(x)
        x = th.relu(self.fc(x))
        for block in self.resblocks:
            x = block(x)
            x = th.relu(x)
        x = self.output_norm(x)
        return x


class PPOMFRLModel(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        rnn_hidsize: int = 256,
        head_hidsize: int = 2048,
        nres_blocks: int = 2,
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
        self.rnn_input_norm = nn.LayerNorm(hidsize)
        self.rnn_input_proj = nn.Linear(hidsize, rnn_hidsize)
        self.gru = nn.GRUCell(rnn_hidsize, rnn_hidsize)

        head_insize = hidsize + rnn_hidsize
        self.pi_backbone = ResidualMLPHead(
            insize=head_insize,
            hidsize=head_hidsize,
            nres_blocks=nres_blocks,
            outsize=getattr(self.action_space, "n"),
        )
        self.vf_backbone = ResidualMLPHead(
            insize=head_insize,
            hidsize=head_hidsize,
            nres_blocks=nres_blocks,
            outsize=1,
        )

        num_actions = getattr(self.action_space, "n")
        self.pi_head = CategoricalActionHead(
            insize=head_hidsize,
            num_actions=num_actions,
            **action_head_kwargs,
        )
        self.vf_head = ScaledMSEHead(
            insize=head_hidsize,
            outsize=1,
            **mse_head_kwargs,
        )

    @th.no_grad()
    def act(self, obs: th.Tensor, rnn_states: th.Tensor | None = None, **kwargs) -> Dict[str, th.Tensor]:
        assert not self.training
        outputs = self.forward(obs, rnn_states=rnn_states, **kwargs)
        actions = self.pi_head.sample(outputs["pi_logits"])
        log_probs = self.pi_head.log_prob(outputs["pi_logits"], actions)
        vpreds = self.vf_head.denormalize(outputs["vpreds"])
        outputs.update({"actions": actions, "log_probs": log_probs, "vpreds": vpreds})
        return outputs

    def forward(self, obs: th.Tensor, rnn_states: th.Tensor | None = None, **kwargs) -> Dict[str, th.Tensor]:
        z = self.encode(obs)
        if rnn_states is None:
            rnn_states = th.zeros(z.shape[0], self.rnn_hidsize, device=z.device)

        rnn_input = self.rnn_input_norm(z)
        rnn_input = th.relu(self.rnn_input_proj(rnn_input))
        y = th.relu(self.gru(rnn_input, rnn_states))

        shared = th.cat([z, y], dim=-1)
        pi_features = self.pi_backbone.forward_features(shared)
        vf_features = self.vf_backbone.forward_features(shared)
        pi_logits = self.pi_head(pi_features)
        vpreds = self.vf_head(vf_features)

        return {
            "latents": z,
            "memory_latents": y,
            "pi_latents": pi_features,
            "vf_latents": vf_features,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
            "next_rnn_states": y,
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
        new_log_probs = self.pi_head.log_prob(outputs["pi_logits"], actions)
        ratio = th.exp(new_log_probs - log_probs)
        ratio_clipped = th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pi_loss = -th.min(advs * ratio, advs * ratio_clipped).mean()
        entropy = self.pi_head.entropy(outputs["pi_logits"]).mean()
        vf_loss = self.vf_head.mse_loss(outputs["vpreds"], vtargs).mean()
        return {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}
