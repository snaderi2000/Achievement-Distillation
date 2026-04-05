from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel


class DinoV3Encoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        image_size: int = 224,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.image_size = image_size
        self.hidden_size = int(self.model.config.hidden_size)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False

        mean = th.tensor([0.485, 0.456, 0.406], dtype=th.float32).view(1, 3, 1, 1)
        std = th.tensor([0.229, 0.224, 0.225], dtype=th.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

    def preprocess(self, obs: th.Tensor) -> th.Tensor:
        x = F.interpolate(
            obs,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )
        x = (x - self.pixel_mean) / self.pixel_std
        return x

    def forward(self, obs: th.Tensor) -> th.Tensor:
        pixel_values = self.preprocess(obs)
        if self.freeze_backbone:
            with th.no_grad():
                outputs = self.model(pixel_values=pixel_values)
        else:
            outputs = self.model(pixel_values=pixel_values)
        cls_token = outputs.last_hidden_state[:, 0]
        return cls_token

    def output_shape(self) -> Tuple[int]:
        return (self.hidden_size,)
