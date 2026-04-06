from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from achievement_distillation.lora import apply_lora_to_model, lora_parameters
from transformers import AutoModel


class DinoV3Encoder(nn.Module):
    def __init__(
        self,
        model_name: str,
        image_size: int = 224,
        freeze_backbone: bool = True,
        unfreeze_last_n_blocks: int = 0,
        readout_type: str = "cls",
        attention_hidden_size: int | None = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"),
    ):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.image_size = image_size
        self.hidden_size = int(self.model.config.hidden_size)
        self.freeze_backbone = freeze_backbone
        self.unfreeze_last_n_blocks = unfreeze_last_n_blocks
        self.readout_type = readout_type
        self.attention_hidden_size = attention_hidden_size or self.hidden_size
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = tuple(lora_target_modules)
        self.lora_module_paths: list[str] = []

        if self.readout_type not in {"cls", "patch_mean", "cls_mean", "cls_attn"}:
            raise ValueError(f"Unsupported readout_type: {self.readout_type}")

        if self.use_lora:
            for param in self.model.parameters():
                param.requires_grad = False
            self.lora_module_paths = apply_lora_to_model(
                self.model,
                target_module_names=self.lora_target_modules,
                rank=self.lora_rank,
                alpha=self.lora_alpha,
                dropout=self.lora_dropout,
            )
        elif freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        elif unfreeze_last_n_blocks > 0:
            self._freeze_all_then_unfreeze_top_blocks(unfreeze_last_n_blocks)

        if self.readout_type == "cls_attn":
            self.patch_attention = nn.Sequential(
                nn.Linear(self.hidden_size, self.attention_hidden_size),
                nn.GELU(),
                nn.Linear(self.attention_hidden_size, 1),
            )
            self.output_dim = 2 * self.hidden_size
        else:
            self.patch_attention = None
            self.output_dim = self.hidden_size

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
        tokens = outputs.last_hidden_state
        cls_token = tokens[:, 0]
        patch_tokens = tokens[:, 1:]
        if self.readout_type == "cls":
            return cls_token
        patch_mean = patch_tokens.mean(dim=1)
        if self.readout_type == "patch_mean":
            return patch_mean
        if self.readout_type == "cls_mean":
            return 0.5 * (cls_token + patch_mean)
        attention_scores = self.patch_attention(patch_tokens)
        attention_weights = th.softmax(attention_scores, dim=1)
        attention_pool = (attention_weights * patch_tokens).sum(dim=1)
        return th.cat([cls_token, attention_pool], dim=-1)

    def output_shape(self) -> Tuple[int]:
        return (self.output_dim,)

    def backbone_parameters(self):
        return self.model.parameters()

    def trainable_backbone_parameters(self):
        if self.use_lora:
            return list(lora_parameters(self.model))
        return [param for param in self.model.parameters() if param.requires_grad]

    def _get_backbone_blocks(self):
        candidates = [
            self.model,
            getattr(self.model, "encoder", None),
            getattr(self.model, "vision_model", None),
            getattr(getattr(self.model, "vision_model", None), "encoder", None),
        ]

        for candidate in candidates:
            if candidate is None:
                continue
            for attr_name in ["layer", "layers", "block", "blocks"]:
                if hasattr(candidate, attr_name):
                    blocks = getattr(candidate, attr_name)
                    if isinstance(blocks, (nn.ModuleList, list, tuple)) and len(blocks) > 0:
                        return blocks

        for _, module in self.model.named_modules():
            for attr_name in ["layer", "layers", "block", "blocks"]:
                if hasattr(module, attr_name):
                    blocks = getattr(module, attr_name)
                    if isinstance(blocks, (nn.ModuleList, list, tuple)) and len(blocks) > 0:
                        return blocks

        raise AttributeError("Could not find transformer blocks on this DINO model.")

    def _freeze_all_then_unfreeze_top_blocks(self, unfreeze_last_n_blocks: int):
        for param in self.model.parameters():
            param.requires_grad = False

        blocks = self._get_backbone_blocks()
        for block in blocks[-unfreeze_last_n_blocks:]:
            for param in block.parameters():
                param.requires_grad = True

        for norm_attr in ["layernorm", "post_layernorm", "norm"]:
            if hasattr(self.model, norm_attr) and isinstance(getattr(self.model, norm_attr), nn.Module):
                for param in getattr(self.model, norm_attr).parameters():
                    param.requires_grad = True
