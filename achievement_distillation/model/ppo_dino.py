from typing import Dict

import torch as th

from gym import spaces

from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.dino_encoder import DinoV3Encoder
from achievement_distillation.model.base import BaseModel
from achievement_distillation.mse_head import ScaledMSEHead
from achievement_distillation.torch_util import FanInInitReLULayer


class PPODinoModel(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        dino_model_name: str,
        dino_image_size: int = 224,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        dino_readout_type: str = "cls",
        dino_attention_hidden_size: int | None = None,
        dense_init_norm_kwargs: Dict = {},
        action_head_kwargs: Dict = {},
        mse_head_kwargs: Dict = {},
    ):
        super().__init__(observation_space, action_space)

        self.enc = DinoV3Encoder(
            model_name=dino_model_name,
            image_size=dino_image_size,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            readout_type=dino_readout_type,
            attention_hidden_size=dino_attention_hidden_size,
        )
        self.linear = FanInInitReLULayer(
            self.enc.output_dim,
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
        latents = self.encode(obs)
        pi_latents = vf_latents = latents
        pi_logits = self.pi_head(pi_latents)
        vpreds = self.vf_head(vf_latents)
        return {
            "latents": latents,
            "pi_latents": pi_latents,
            "vf_latents": vf_latents,
            "pi_logits": pi_logits,
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
        vpreds = outputs["vpreds"]
        vf_loss = self.vf_head.mse_loss(vpreds, vtargs).mean()
        return {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}

    def get_param_groups(self):
        backbone_params = list(self.enc.trainable_backbone_parameters())
        backbone_ids = {id(param) for param in backbone_params}
        head_params = [param for param in self.parameters() if param.requires_grad and id(param) not in backbone_ids]

        groups = []
        if backbone_params:
            groups.append({"name": "backbone", "params": backbone_params})
        if head_params:
            groups.append({"name": "heads", "params": head_params})
        return groups
