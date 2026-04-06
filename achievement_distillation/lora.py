from __future__ import annotations

import math
from typing import Iterable, Sequence

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive.")
        self.base = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_a = nn.Parameter(th.zeros(rank, in_features))
        self.lora_b = nn.Parameter(th.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: th.Tensor) -> th.Tensor:
        base_out = self.base(x)
        lora_out = F.linear(self.dropout(x), self.lora_a)
        lora_out = F.linear(lora_out, self.lora_b)
        return base_out + self.scaling * lora_out


def _get_module_by_path(root: nn.Module, path: str):
    module = root
    if not path:
        return None
    for part in path.split("."):
        if not hasattr(module, part):
            return None
        module = getattr(module, part)
    return module


def _set_module_by_path(root: nn.Module, path: str, new_module: nn.Module):
    parent_path, _, leaf_name = path.rpartition(".")
    parent = _get_module_by_path(root, parent_path) if parent_path else root
    if parent is None:
        raise AttributeError(f"Could not locate parent module for path '{path}'")
    setattr(parent, leaf_name, new_module)


def apply_lora_to_model(
    model: nn.Module,
    target_module_names: Sequence[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    module_name_filter: str | None = None,
) -> list[str]:
    replaced_paths: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        leaf_name = name.rsplit(".", 1)[-1]
        if leaf_name not in target_module_names:
            continue
        if module_name_filter is not None and module_name_filter not in name:
            continue
        wrapped = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        _set_module_by_path(model, name, wrapped)
        replaced_paths.append(name)
    return replaced_paths


def lora_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_a
            yield module.lora_b
