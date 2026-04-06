from typing import Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SlotAttention(nn.Module):
    def __init__(
        self,
        input_dim: int,
        slot_dim: int,
        num_slots: int = 8,
        num_iterations: int = 3,
        mlp_hidden_dim: int | None = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.slot_dim = slot_dim
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.eps = eps
        mlp_hidden_dim = mlp_hidden_dim or 2 * slot_dim

        self.norm_inputs = nn.LayerNorm(input_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

        self.slots_mu = nn.Parameter(th.zeros(1, 1, slot_dim))
        self.slots_log_sigma = nn.Parameter(th.zeros(1, 1, slot_dim))

        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(input_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(input_dim, slot_dim, bias=False)

        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, slot_dim),
        )

        self.scale = slot_dim ** -0.5

    def forward(self, inputs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        batch_size = inputs.shape[0]
        inputs = self.norm_inputs(inputs)
        k = self.project_k(inputs)
        v = self.project_v(inputs)

        mu = self.slots_mu.expand(batch_size, self.num_slots, -1)
        sigma = th.exp(self.slots_log_sigma).expand(batch_size, self.num_slots, -1)
        slots = mu + sigma * th.randn_like(mu)

        attn = None
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)

            attn_logits = th.einsum("bsd,bnd->bsn", q, k) * self.scale
            attn = F.softmax(attn_logits, dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = th.einsum("bsn,bnd->bsd", attn, v)
            slots = self.gru(
                updates.reshape(-1, self.slot_dim),
                slots_prev.reshape(-1, self.slot_dim),
            )
            slots = slots.reshape(batch_size, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn
