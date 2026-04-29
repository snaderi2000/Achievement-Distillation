from typing import Dict

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CategoricalValueHead(nn.Module):
    def __init__(
        self,
        insize: int,
        num_bins: int = 101,
        vmin: float = -5.0,
        vmax: float = 25.0,
        init_scale: float = 0.1,
    ):
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if vmax <= vmin:
            raise ValueError("vmax must be greater than vmin")

        self.num_bins = int(num_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.linear = nn.Linear(insize, self.num_bins)
        init.orthogonal_(self.linear.weight, gain=init_scale)
        init.constant_(self.linear.bias, val=0.0)

        support = th.linspace(self.vmin, self.vmax, steps=self.num_bins)
        self.register_buffer("support", support)
        self.bin_width = (self.vmax - self.vmin) / (self.num_bins - 1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(x)

    def expected_value(self, logits: th.Tensor) -> th.Tensor:
        probs = F.softmax(logits, dim=-1)
        values = probs @ self.support
        return values.unsqueeze(-1)

    def _target_distribution(self, targ: th.Tensor) -> th.Tensor:
        targ = targ.squeeze(-1).clamp(self.vmin, self.vmax)
        scaled = (targ - self.vmin) / self.bin_width
        lower = th.floor(scaled).long().clamp(0, self.num_bins - 1)
        upper = th.ceil(scaled).long().clamp(0, self.num_bins - 1)

        upper_weight = (scaled - lower.float()).clamp(0.0, 1.0)
        lower_weight = 1.0 - upper_weight

        target = th.zeros(targ.shape[0], self.num_bins, device=targ.device, dtype=targ.dtype)
        target.scatter_add_(1, lower.unsqueeze(-1), lower_weight.unsqueeze(-1))
        target.scatter_add_(1, upper.unsqueeze(-1), upper_weight.unsqueeze(-1))
        return target

    def cross_entropy_loss(self, logits: th.Tensor, targ: th.Tensor) -> th.Tensor:
        target_dist = self._target_distribution(targ)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_dist * log_probs).sum(dim=-1, keepdim=True)
