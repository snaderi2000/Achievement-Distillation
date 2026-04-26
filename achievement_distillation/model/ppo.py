from typing import Dict

import torch as th
import torch.nn.functional as F

from gym import spaces

from achievement_distillation.model.base import BaseModel
from achievement_distillation.impala_cnn import ImpalaCNN
from achievement_distillation.action_head import CategoricalActionHead
from achievement_distillation.mse_head import PlainMSEHead, ScaledMSEHead
from achievement_distillation.torch_util import FanInInitReLULayer


class PPOModel(BaseModel):
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        impala_kwargs: Dict = {},
        dense_init_norm_kwargs: Dict = {},
        action_head_kwargs: Dict = {},
        mse_head_kwargs: Dict = {},
        use_phase_vf_head: bool = False,
        use_short_reward_head: bool = False,
        use_health_event_heads: bool = False,
        use_death_event_head: bool = False,
        aux_head_hidsize: int = 0,
        value_hidsize: int = 0,
        aux_on_value_features: bool = False,
    ):
        super().__init__(observation_space, action_space)

        # Encoder
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

        # Heads
        num_actions = getattr(self.action_space, "n")
        self.pi_head = CategoricalActionHead(
            insize=hidsize,
            num_actions=num_actions,
            **action_head_kwargs,
        )
        self.vf_tower = None
        vf_insize = hidsize
        if value_hidsize > 0:
            self.vf_tower = FanInInitReLULayer(
                hidsize,
                value_hidsize,
                layer_type="linear",
                **dense_init_norm_kwargs,
            )
            vf_insize = value_hidsize
        self.vf_head = ScaledMSEHead(
            insize=vf_insize,
            outsize=1,
            **mse_head_kwargs,
        )
        self.use_phase_vf_head = use_phase_vf_head
        if use_phase_vf_head:
            self.phase_vf_head = PlainMSEHead(
                insize=hidsize,
                outsize=1,
            )
        self.use_short_reward_head = use_short_reward_head
        if use_short_reward_head:
            self.short_reward_mlp = FanInInitReLULayer(
                hidsize,
                hidsize // 2,
                layer_type="linear",
                **dense_init_norm_kwargs,
            )
            self.short_reward_head = PlainMSEHead(
                insize=hidsize // 2,
                outsize=1,
            )
        self.use_health_event_heads = use_health_event_heads
        self.use_death_event_head = use_death_event_head
        self.aux_head_hidsize = aux_head_hidsize
        self.aux_on_value_features = aux_on_value_features
        aux_source_insize = vf_insize if aux_on_value_features else hidsize
        if use_health_event_heads:
            if aux_head_hidsize > 0:
                self.health_decrease_mlp = FanInInitReLULayer(
                    aux_source_insize,
                    aux_head_hidsize,
                    layer_type="linear",
                    **dense_init_norm_kwargs,
                )
                self.health_increase_mlp = FanInInitReLULayer(
                    aux_source_insize,
                    aux_head_hidsize,
                    layer_type="linear",
                    **dense_init_norm_kwargs,
                )
                aux_insize = aux_head_hidsize
            else:
                self.health_decrease_mlp = None
                self.health_increase_mlp = None
                aux_insize = aux_source_insize
            self.health_decrease_head = PlainMSEHead(insize=aux_insize, outsize=1)
            self.health_increase_head = PlainMSEHead(insize=aux_insize, outsize=1)
        if use_death_event_head:
            if aux_head_hidsize > 0:
                self.death_event_mlp = FanInInitReLULayer(
                    aux_source_insize,
                    aux_head_hidsize,
                    layer_type="linear",
                    **dense_init_norm_kwargs,
                )
                death_insize = aux_head_hidsize
            else:
                self.death_event_mlp = None
                death_insize = aux_source_insize
            self.death_event_head = PlainMSEHead(insize=death_insize, outsize=1)

    @th.no_grad()
    def act(self, obs: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        # Check training mode
        assert not self.training

        # Pass through model
        outputs = self.forward(obs, **kwargs)

        # Sample actions
        pi_logits = outputs["pi_logits"]
        actions = self.pi_head.sample(pi_logits)

        # Compute log probs
        log_probs = self.pi_head.log_prob(pi_logits, actions)

        # Denormalize vpreds
        vpreds = outputs["vpreds"]
        vpreds = self.vf_head.denormalize(vpreds)

        # Update outputs
        outputs.update({"actions": actions, "log_probs": log_probs, "vpreds": vpreds})

        return outputs

    def forward(self, obs: th.Tensor, **kwargs) -> Dict[str, th.Tensor]:
        # Pass through encoder
        latents = self.encode(obs)

        # Pass through heads
        pi_latents = latents
        vf_latents = self.vf_tower(latents) if self.vf_tower is not None else latents
        pi_logits = self.pi_head(pi_latents)
        vpreds = self.vf_head(vf_latents)
        phase_vpreds = self.phase_vf_head(vf_latents) if self.use_phase_vf_head else None
        short_reward_preds = (
            self.short_reward_head(self.short_reward_mlp(latents))
            if self.use_short_reward_head
            else None
        )
        aux_source = vf_latents if self.aux_on_value_features else latents
        if self.use_health_event_heads:
            health_decrease_feats = (
                self.health_decrease_mlp(aux_source) if self.health_decrease_mlp is not None else aux_source
            )
            health_increase_feats = (
                self.health_increase_mlp(aux_source) if self.health_increase_mlp is not None else aux_source
            )
            health_decrease_logits = self.health_decrease_head(health_decrease_feats)
            health_increase_logits = self.health_increase_head(health_increase_feats)
        else:
            health_decrease_logits = None
            health_increase_logits = None
        death_event_logits = None
        if self.use_death_event_head:
            death_feats = self.death_event_mlp(aux_source) if self.death_event_mlp is not None else aux_source
            death_event_logits = self.death_event_head(death_feats)

        # Define outputs
        outputs = {
            "latents": latents,
            "pi_latents": pi_latents,
            "vf_latents": vf_latents,
            "pi_logits": pi_logits,
            "vpreds": vpreds,
        }
        if phase_vpreds is not None:
            outputs["phase_vpreds"] = phase_vpreds
        if short_reward_preds is not None:
            outputs["short_reward_preds"] = short_reward_preds
        if health_decrease_logits is not None:
            outputs["health_decrease_logits"] = health_decrease_logits
        if health_increase_logits is not None:
            outputs["health_increase_logits"] = health_increase_logits
        if death_event_logits is not None:
            outputs["death_event_logits"] = death_event_logits

        return outputs

    def encode(self, obs: th.Tensor) -> th.Tensor:
        # Pass through encoder
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
        phase_ids: th.Tensor | None = None,
        phase_mask: th.Tensor | None = None,
        progress_bins: th.Tensor | None = None,
        short_reward_targets: th.Tensor | None = None,
        short_reward_mask: th.Tensor | None = None,
        health_decrease_targets: th.Tensor | None = None,
        health_increase_targets: th.Tensor | None = None,
        health_event_mask: th.Tensor | None = None,
        death_targets: th.Tensor | None = None,
        death_event_mask: th.Tensor | None = None,
        rank_margin: float = 0.05,
        rank_delta: float = 0.1,
        rank_max_pairs_per_group: int = 8,
        rank_num_progress_bins: int = 4,
        **kwargs,
    ) -> Dict[str, th.Tensor]:
        # Pass through model
        outputs = self.forward(obs, **kwargs)

        # Compute policy loss
        pi_logits = outputs["pi_logits"]
        new_log_probs = self.pi_head.log_prob(pi_logits, actions)
        ratio = th.exp(new_log_probs - log_probs)
        ratio_clipped = th.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pi_loss = -th.min(advs * ratio, advs * ratio_clipped).mean()

        # Compute entropy
        entropy = self.pi_head.entropy(pi_logits).mean()

        # Compute value loss
        vpreds = outputs["vpreds"]
        vf_loss = self.vf_head.mse_loss(vpreds, vtargs).mean()

        # Define losses
        losses = {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}
        rank_loss = self.compute_rank_loss(
            vpreds=vpreds,
            vtargs=vtargs,
            phase_ids=phase_ids,
            progress_bins=progress_bins,
            rank_margin=rank_margin,
            rank_delta=rank_delta,
            rank_max_pairs_per_group=rank_max_pairs_per_group,
            rank_num_progress_bins=rank_num_progress_bins,
        )
        losses["rank_loss"] = rank_loss
        if self.use_phase_vf_head:
            phase_vf_loss = th.zeros((), device=vtargs.device)
            if phase_ids is not None and phase_mask is not None:
                phase_targets, valid_mask = self.compute_phase_normalized_targets(
                    vtargs=vtargs,
                    phase_ids=phase_ids,
                    phase_mask=phase_mask,
                )
                if valid_mask.any():
                    phase_vpreds = outputs["phase_vpreds"]
                    phase_vf_loss = self.phase_vf_head.mse_loss(
                        phase_vpreds[valid_mask],
                        phase_targets[valid_mask],
                    ).mean()
            losses["phase_vf_loss"] = phase_vf_loss
        if self.use_short_reward_head:
            short_reward_loss = th.zeros((), device=vtargs.device)
            if short_reward_targets is not None and short_reward_mask is not None:
                valid_mask = short_reward_mask.bool()
                if valid_mask.any():
                    short_reward_preds = outputs["short_reward_preds"]
                    short_reward_loss = self.short_reward_head.mse_loss(
                        short_reward_preds[valid_mask],
                        short_reward_targets[valid_mask],
                    ).mean()
            losses["short_reward_loss"] = short_reward_loss
        if self.use_health_event_heads:
            health_decrease_loss = th.zeros((), device=vtargs.device)
            health_increase_loss = th.zeros((), device=vtargs.device)
            if (
                health_decrease_targets is not None
                and health_increase_targets is not None
                and health_event_mask is not None
            ):
                valid_mask = health_event_mask.bool()
                if valid_mask.any():
                    health_decrease_logits = outputs["health_decrease_logits"][valid_mask]
                    health_increase_logits = outputs["health_increase_logits"][valid_mask]
                    health_decrease_loss = F.binary_cross_entropy_with_logits(
                        health_decrease_logits,
                        health_decrease_targets[valid_mask],
                    )
                    health_increase_loss = F.binary_cross_entropy_with_logits(
                        health_increase_logits,
                        health_increase_targets[valid_mask],
                    )
            losses["health_decrease_loss"] = health_decrease_loss
            losses["health_increase_loss"] = health_increase_loss
        if self.use_death_event_head:
            death_event_loss = th.zeros((), device=vtargs.device)
            if death_targets is not None and death_event_mask is not None:
                valid_mask = death_event_mask.bool()
                if valid_mask.any():
                    death_event_logits = outputs["death_event_logits"][valid_mask]
                    death_event_loss = F.binary_cross_entropy_with_logits(
                        death_event_logits,
                        death_targets[valid_mask],
                    )
            losses["death_event_loss"] = death_event_loss

        return losses

    def compute_rank_loss(
        self,
        vpreds: th.Tensor,
        vtargs: th.Tensor,
        phase_ids: th.Tensor | None,
        progress_bins: th.Tensor | None,
        rank_margin: float,
        rank_delta: float,
        rank_max_pairs_per_group: int,
        rank_num_progress_bins: int,
    ) -> th.Tensor:
        if phase_ids is None or progress_bins is None:
            return vpreds.new_zeros(())

        pred_values = self.vf_head.denormalize(vpreds).squeeze(-1)
        target_values = vtargs.squeeze(-1)
        phase_values = phase_ids.squeeze(-1)
        progress_values = progress_bins.squeeze(-1)

        valid = phase_values >= 0
        if not valid.any():
            return vpreds.new_zeros(())

        device = vpreds.device
        group_ids = phase_values * rank_num_progress_bins + progress_values
        unique_groups = th.unique(group_ids[valid])

        pair_losses = []
        for group_id in unique_groups.tolist():
            group_mask = valid & (group_ids == group_id)
            group_idx = group_mask.nonzero(as_tuple=False).squeeze(-1)
            if group_idx.numel() < 2:
                continue

            perm = group_idx[th.randperm(group_idx.numel(), device=device)]
            npairs = min(perm.numel() // 2, rank_max_pairs_per_group)
            if npairs == 0:
                continue

            left_idx = perm[: 2 * npairs : 2]
            right_idx = perm[1 : 2 * npairs : 2]
            left_targ = target_values[left_idx]
            right_targ = target_values[right_idx]
            gap = left_targ - right_targ
            keep = gap.abs() > rank_delta
            if not keep.any():
                continue

            left_idx = left_idx[keep]
            right_idx = right_idx[keep]
            gap = gap[keep]

            winners = th.where(gap > 0, left_idx, right_idx)
            losers = th.where(gap > 0, right_idx, left_idx)
            margin_gap = pred_values[winners] - pred_values[losers]
            pair_losses.append(th.relu(rank_margin - margin_gap))

        if not pair_losses:
            return vpreds.new_zeros(())
        return th.cat(pair_losses).mean()

    def compute_phase_normalized_targets(
        self,
        vtargs: th.Tensor,
        phase_ids: th.Tensor,
        phase_mask: th.Tensor,
        num_phases: int = 4,
        eps: float = 1e-6,
    ) -> tuple[th.Tensor, th.Tensor]:
        phase_targets = th.zeros_like(vtargs)
        valid_mask = phase_mask.bool()
        for phase in range(num_phases):
            phase_sel = valid_mask & (phase_ids == phase)
            if not phase_sel.any():
                continue
            phase_values = vtargs[phase_sel]
            mean = phase_values.mean()
            std = phase_values.std(unbiased=False)
            phase_targets[phase_sel] = (phase_values - mean) / (std + eps)
        return phase_targets, valid_mask
