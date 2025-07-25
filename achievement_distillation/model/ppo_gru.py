from typing import Dict, Optional, Tuple

import torch as th
import torch.nn as nn

from gym import spaces

from achievement_distillation.model.ppo import PPOModel


class PPOGRUModel(PPOModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        gru_layers: int = 1,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, hidsize, **kwargs)

        # Recurrent layer operating on latent features
        self.gru_layers = gru_layers
        self.gru = nn.GRU(hidsize, hidsize, num_layers=gru_layers)

    def get_init_rnn_states(self, batch_size: int) -> th.Tensor:
        return th.zeros(self.gru_layers, batch_size, self.hidsize)

    @th.no_grad()
    def act(
        self,
        obs: th.Tensor,
        rnn_states: Optional[th.Tensor] = None,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        assert not self.training
        outputs = self.forward(obs, rnn_states=rnn_states, **kwargs)
        pi_logits = outputs["pi_logits"]
        actions = self.pi_head.sample(pi_logits)
        log_probs = self.pi_head.log_prob(pi_logits, actions)
        vpreds = outputs["vpreds"]
        vpreds = self.vf_head.denormalize(vpreds)
        outputs.update(
            {
                "actions": actions,
                "log_probs": log_probs,
                "vpreds": vpreds,
                "rnn_states": outputs["rnn_states"],
            }
        )
        return outputs

    def forward(
        self, obs: th.Tensor, rnn_states: Optional[th.Tensor] = None, **kwargs
    ) -> Dict[str, th.Tensor]:
        latents, next_states = self.encode(obs, rnn_states=rnn_states)
        pi_logits = self.pi_head(latents)
        vpreds = self.vf_head(latents)
        outputs = {
            "latents": latents,
            "pi_latents": latents,
            "vf_latents": latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
            "rnn_states": next_states,
        }
        return outputs

    def encode(
        self, obs: th.Tensor, rnn_states: Optional[th.Tensor] = None
    ) -> Tuple[th.Tensor, th.Tensor]:
        x = self.enc(obs)
        x = self.linear(x)
        x = x.unsqueeze(0)
        if rnn_states is None:
            rnn_states = th.zeros(
                self.gru_layers, x.shape[1], self.hidsize, device=x.device
            )
        x, next_states = self.gru(x, rnn_states)
        x = x.squeeze(0)
        return x, next_states
