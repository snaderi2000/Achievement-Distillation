from typing import Dict

import torch as th
import torch.nn as nn

from gym import spaces

from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.impala_cnn import ImpalaCNN
from achievement_distillation.model.base import BaseModel
from achievement_distillation.mse_head import ScaledMSEHead
from achievement_distillation.torch_util import FanInInitReLULayer


class PPOAchievementMemoryStrongModel(BaseModel):
    use_recurrent_loader = True

    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        rnn_hidsize: int = 256,
        achievement_progress_dim: int = 22,
        head_hidsize: int = 1024,
        vf_head_hidsize: int | None = None,
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
        self.achievement_progress_dim = int(achievement_progress_dim)
        self.progress_encoder = nn.Sequential(
            nn.LayerNorm(self.achievement_progress_dim),
            nn.Linear(self.achievement_progress_dim, rnn_hidsize),
            nn.ReLU(),
            nn.Linear(rnn_hidsize, rnn_hidsize),
            nn.ReLU(),
        )

        head_insize = hidsize + rnn_hidsize
        vf_head_hidsize = head_hidsize if vf_head_hidsize is None else vf_head_hidsize
        self.pi_backbone = nn.Sequential(
            nn.LayerNorm(head_insize),
            nn.Linear(head_insize, head_hidsize),
            nn.ReLU(),
        )
        self.vf_backbone = nn.Sequential(
            nn.LayerNorm(head_insize),
            nn.Linear(head_insize, vf_head_hidsize),
            nn.ReLU(),
        )

        num_actions = getattr(self.action_space, "n")
        self.pi_head = CategoricalActionHead(
            insize=head_hidsize,
            num_actions=num_actions,
            **action_head_kwargs,
        )
        self.vf_head = ScaledMSEHead(
            insize=vf_head_hidsize,
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
        achievement_progress = kwargs.get("achievement_progress")
        if achievement_progress is None:
            raise ValueError("achievement_progress input is required for PPOAchievementMemoryStrongModel")

        latents = self.encode(obs)
        memory_latents = self.progress_encoder(achievement_progress.float())

        shared = th.cat([latents, memory_latents], dim=-1)
        pi_features = self.pi_backbone(shared)
        vf_features = self.vf_backbone(shared)
        pi_logits = self.pi_head(pi_features)
        vpreds = self.vf_head(vf_features)
        return {
            "latents": latents,
            "memory_latents": memory_latents,
            "pi_latents": pi_features,
            "vf_latents": vf_features,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
            "next_rnn_states": memory_latents,
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
        achievement_progress: th.Tensor,
        masks: th.Tensor | None = None,
        init_rnn_states: th.Tensor | None = None,
        rnn_states: th.Tensor | None = None,
        clip_param: float = 0.2,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        if obs.dim() == 5:
            seq_len, batch_size = obs.shape[:2]
            obs = obs.reshape(seq_len * batch_size, *obs.shape[2:])
            achievement_progress = achievement_progress.reshape(
                seq_len * batch_size,
                *achievement_progress.shape[2:],
            )
            actions = actions.reshape(seq_len * batch_size, *actions.shape[2:])
            log_probs = log_probs.reshape(seq_len * batch_size, *log_probs.shape[2:])
            vtargs = vtargs.reshape(seq_len * batch_size, *vtargs.shape[2:])
            advs = advs.reshape(seq_len * batch_size, *advs.shape[2:])

        outputs = self.forward(
            obs,
            rnn_states=rnn_states,
            achievement_progress=achievement_progress,
            **kwargs,
        )
        pi_logits = outputs["pi_logits"]
        vpreds = outputs["vpreds"]

        new_log_probs = self.pi_head.log_prob(pi_logits, actions)
        ratio = th.exp(new_log_probs - log_probs)
        ratio_clipped = th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pi_loss = -th.min(advs * ratio, advs * ratio_clipped).mean()
        entropy = self.pi_head.entropy(pi_logits).mean()
        vf_loss = self.vf_head.mse_loss(vpreds, vtargs).mean()
        return {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}
