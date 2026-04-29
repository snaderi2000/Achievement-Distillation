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
        target_mode: str = "two_hot",
        sigma_ratio: float = 0.75,
        init_scale: float = 0.1,
        init_value: float = 0.0,
    ):
        super().__init__()
        if num_bins < 2:
            raise ValueError("num_bins must be at least 2")
        if vmax <= vmin:
            raise ValueError("vmax must be greater than vmin")
        if target_mode not in ("two_hot", "hl_gauss"):
            raise ValueError("target_mode must be 'two_hot' or 'hl_gauss'")
        if sigma_ratio <= 0:
            raise ValueError("sigma_ratio must be positive")

        self.num_bins = int(num_bins)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.target_mode = target_mode
        self.sigma_ratio = float(sigma_ratio)
        self.init_value = float(init_value)
        self.linear = nn.Linear(insize, self.num_bins)
        init.orthogonal_(self.linear.weight, gain=init_scale)

        if self.target_mode == "hl_gauss":
            edges = th.linspace(self.vmin, self.vmax, steps=self.num_bins + 1)
            support = 0.5 * (edges[:-1] + edges[1:])
            self.bin_width = (self.vmax - self.vmin) / self.num_bins
            self.register_buffer("edges", edges)
        else:
            support = th.linspace(self.vmin, self.vmax, steps=self.num_bins)
            self.bin_width = (self.vmax - self.vmin) / (self.num_bins - 1)
            self.register_buffer("edges", th.empty(0))
        self.register_buffer("support", support)
        self._init_bias_to_value(self.init_value)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(x)

    def expected_value(self, logits: th.Tensor) -> th.Tensor:
        probs = F.softmax(logits, dim=-1)
        values = probs @ self.support
        return values.unsqueeze(-1)

    def _target_distribution(self, targ: th.Tensor) -> th.Tensor:
        if self.target_mode == "hl_gauss":
            return self._hl_gauss_target_distribution(targ)
        return self._two_hot_target_distribution(targ)

    def _two_hot_target_distribution(self, targ: th.Tensor) -> th.Tensor:
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

    def _hl_gauss_target_distribution(self, targ: th.Tensor) -> th.Tensor:
        targ = targ.squeeze(-1).clamp(self.vmin, self.vmax)
        sigma = self.sigma_ratio * self.bin_width
        z = (self.edges.unsqueeze(0) - targ.unsqueeze(-1)) / (sigma * (2.0 ** 0.5))
        cdf = th.erf(z)
        probs = cdf[:, 1:] - cdf[:, :-1]
        probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return probs

    def cross_entropy_loss(self, logits: th.Tensor, targ: th.Tensor) -> th.Tensor:
        target_dist = self._target_distribution(targ)
        log_probs = F.log_softmax(logits, dim=-1)
        return -(target_dist * log_probs).sum(dim=-1, keepdim=True)

    def _init_bias_to_value(self, value: float):
        with th.no_grad():
            target = self._target_distribution(th.tensor([[value]], dtype=self.support.dtype))
            self.linear.bias.copy_(target.squeeze(0).clamp(min=1e-8).log())

    def target_stats(self, targ: th.Tensor) -> dict[str, th.Tensor]:
        targ = targ.detach()
        return {
            "vf_target_min": targ.min(),
            "vf_target_max": targ.max(),
            "vf_target_mean": targ.mean(),
            "vf_target_clamp_low": (targ < self.vmin).float().mean(),
            "vf_target_clamp_high": (targ > self.vmax).float().mean(),
        }
