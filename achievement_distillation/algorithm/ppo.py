from contextlib import nullcontext

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from achievement_distillation.model.ppo import PPOModel
from achievement_distillation.algorithm.base import BaseAlgorithm
from achievement_distillation.storage import RolloutStorage


class PPOAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        model: PPOModel,
        ppo_nepoch: int,
        ppo_nbatch: int,
        clip_param: float,
        vf_loss_coef: float,
        ent_coef: float,
        lr: float,
        max_grad_norm: float,
        backbone_lr: float | None = None,
        phase_vf_loss_coef: float = 0.0,
        short_reward_loss_coef: float = 0.0,
        short_reward_horizon: int = 0,
        health_event_horizon: int = 0,
        health_decrease_loss_coef: float = 0.0,
        health_increase_loss_coef: float = 0.0,
        death_event_horizon: int = 0,
        death_event_loss_coef: float = 0.0,
        health_vf_loss_coef: float = 0.0,
        achievement_vf_loss_coef: float = 0.0,
        health_cf_loss_coef: float = 0.0,
        health_cf_margin: float = 0.1,
        rank_loss_coef: float = 0.0,
        rank_margin: float = 0.05,
        rank_delta: float = 0.1,
        rank_num_phase_bins: int = 4,
        rank_num_progress_bins: int = 4,
        rank_max_pairs_per_group: int = 8,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
        value_aug_coef: float = 0.0,
        value_aug_start_step: int = 0,
        value_aug_modes: list[str] | tuple[str, ...] = ("horizontal", "vertical", "both"),
        value_aug_modes_per_batch: int = 1,
        value_aug_inventory_rows: int = 15,
        value_aug_detach_target: bool = True,
        value_aug_source: str = "pixel",
    ):
        super().__init__(model)
        self.model: PPOModel

        # PPO params
        self.clip_param = clip_param
        self.ppo_nepoch = ppo_nepoch
        self.ppo_nbatch = ppo_nbatch
        self.vf_loss_coef = vf_loss_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.phase_vf_loss_coef = phase_vf_loss_coef
        self.short_reward_loss_coef = short_reward_loss_coef
        self.short_reward_horizon = short_reward_horizon
        self.health_event_horizon = health_event_horizon
        self.health_decrease_loss_coef = health_decrease_loss_coef
        self.health_increase_loss_coef = health_increase_loss_coef
        self.death_event_horizon = death_event_horizon
        self.death_event_loss_coef = death_event_loss_coef
        self.health_vf_loss_coef = health_vf_loss_coef
        self.achievement_vf_loss_coef = achievement_vf_loss_coef
        self.health_cf_loss_coef = health_cf_loss_coef
        self.health_cf_margin = health_cf_margin
        self.rank_loss_coef = rank_loss_coef
        self.rank_margin = rank_margin
        self.rank_delta = rank_delta
        self.rank_num_phase_bins = rank_num_phase_bins
        self.rank_num_progress_bins = rank_num_progress_bins
        self.rank_max_pairs_per_group = rank_max_pairs_per_group
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and th.cuda.is_available()
        self.scaler = th.amp.GradScaler("cuda", enabled=self.use_amp)
        self.value_aug_coef = float(value_aug_coef)
        self.value_aug_start_step = int(value_aug_start_step)
        self.value_aug_modes = tuple(value_aug_modes)
        self.value_aug_modes_per_batch = int(value_aug_modes_per_batch)
        self.value_aug_inventory_rows = int(value_aug_inventory_rows)
        self.value_aug_detach_target = bool(value_aug_detach_target)
        self.value_aug_source = value_aug_source
        self.num_env_steps = 0

        # Optimizer
        if hasattr(model, "get_param_groups"):
            param_groups = model.get_param_groups()
            if param_groups:
                optimizer_groups = []
                for group in param_groups:
                    group_lr = lr
                    if group.get("name") == "backbone" and backbone_lr is not None:
                        group_lr = backbone_lr
                    optimizer_groups.append({"params": group["params"], "lr": group_lr})
                self.optimizer = optim.Adam(optimizer_groups, lr=lr)
            else:
                self.optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def update(self, storage: RolloutStorage):
        # Set model to training mode
        self.model.train()

        if getattr(self.model, "use_recurrent_loader", False):
            return self._update_recurrent(storage)

        # Run PPO
        pi_loss_epoch = 0
        vf_loss_epoch = 0
        entropy_epoch = 0
        phase_vf_loss_epoch = 0
        short_reward_loss_epoch = 0
        health_decrease_loss_epoch = 0
        health_increase_loss_epoch = 0
        death_event_loss_epoch = 0
        health_vf_loss_epoch = 0
        achievement_vf_loss_epoch = 0
        health_cf_loss_epoch = 0
        rank_loss_epoch = 0
        value_aug_loss_epoch = 0
        value_aug_loss_keys = ("semantic",) if self.value_aug_source == "semantic" else self.value_aug_modes
        value_aug_mode_loss_epoch = {mode: 0.0 for mode in value_aug_loss_keys}
        extra_stat_sums = {}
        nupdate = 0
        value_aug_active = self._value_aug_active(storage)

        for _ in range(self.ppo_nepoch):
            # Get data loader
            data_loader = storage.get_data_loader(
                self.ppo_nbatch,
                short_reward_horizon=self.short_reward_horizon,
                health_event_horizon=self.health_event_horizon,
                death_event_horizon=self.death_event_horizon,
                num_phase_bins=self.rank_num_phase_bins,
                num_progress_bins=self.rank_num_progress_bins,
            )

            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(data_loader):
                # Compute loss
                with self._autocast_context():
                    losses = self.model.compute_losses(
                        **batch,
                        clip_param=self.clip_param,
                        rank_margin=self.rank_margin,
                        rank_delta=self.rank_delta,
                        rank_max_pairs_per_group=self.rank_max_pairs_per_group,
                        rank_num_progress_bins=self.rank_num_progress_bins,
                        health_cf_margin=self.health_cf_margin,
                    )
                    pi_loss = losses["pi_loss"]
                    vf_loss = losses["vf_loss"]
                    entropy = losses["entropy"]
                    phase_vf_loss = losses.get("phase_vf_loss")
                    if phase_vf_loss is None:
                        phase_vf_loss = pi_loss.new_zeros(())
                    short_reward_loss = losses.get("short_reward_loss")
                    if short_reward_loss is None:
                        short_reward_loss = pi_loss.new_zeros(())
                    health_decrease_loss = losses.get("health_decrease_loss")
                    if health_decrease_loss is None:
                        health_decrease_loss = pi_loss.new_zeros(())
                    health_increase_loss = losses.get("health_increase_loss")
                    if health_increase_loss is None:
                        health_increase_loss = pi_loss.new_zeros(())
                    death_event_loss = losses.get("death_event_loss")
                    if death_event_loss is None:
                        death_event_loss = pi_loss.new_zeros(())
                    health_vf_loss = losses.get("health_vf_loss")
                    if health_vf_loss is None:
                        health_vf_loss = pi_loss.new_zeros(())
                    achievement_vf_loss = losses.get("achievement_vf_loss")
                    if achievement_vf_loss is None:
                        achievement_vf_loss = pi_loss.new_zeros(())
                    health_cf_loss = losses.get("health_cf_loss")
                    if health_cf_loss is None:
                        health_cf_loss = pi_loss.new_zeros(())
                    rank_loss = losses.get("rank_loss")
                    if rank_loss is None:
                        rank_loss = pi_loss.new_zeros(())
                    value_aug_loss = pi_loss.new_zeros(())
                    value_aug_mode_losses = {}
                    if value_aug_active:
                        value_aug_loss, value_aug_mode_losses = self._compute_value_aug_loss(batch)
                    loss = (
                        pi_loss
                        + self.vf_loss_coef * vf_loss
                        + self.phase_vf_loss_coef * phase_vf_loss
                        + self.short_reward_loss_coef * short_reward_loss
                        + self.health_decrease_loss_coef * health_decrease_loss
                        + self.health_increase_loss_coef * health_increase_loss
                        + self.death_event_loss_coef * death_event_loss
                        + self.health_vf_loss_coef * health_vf_loss
                        + self.achievement_vf_loss_coef * achievement_vf_loss
                        + self.health_cf_loss_coef * health_cf_loss
                        + self.rank_loss_coef * rank_loss
                        + self.value_aug_coef * value_aug_loss
                        - self.ent_coef * entropy
                    )

                loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()

                should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or (batch_idx + 1 == self.ppo_nbatch)
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                # Update stats
                pi_loss_epoch += pi_loss.item()
                vf_loss_epoch += vf_loss.item()
                entropy_epoch += entropy.item()
                phase_vf_loss_epoch += phase_vf_loss.item()
                short_reward_loss_epoch += short_reward_loss.item()
                health_decrease_loss_epoch += health_decrease_loss.item()
                health_increase_loss_epoch += health_increase_loss.item()
                death_event_loss_epoch += death_event_loss.item()
                health_vf_loss_epoch += health_vf_loss.item()
                achievement_vf_loss_epoch += achievement_vf_loss.item()
                health_cf_loss_epoch += health_cf_loss.item()
                rank_loss_epoch += rank_loss.item()
                value_aug_loss_epoch += value_aug_loss.item()
                for mode in value_aug_loss_keys:
                    mode_loss = value_aug_mode_losses.get(mode)
                    if mode_loss is not None:
                        value_aug_mode_loss_epoch[mode] += mode_loss.detach().item()
                for key, value in losses.items():
                    if (
                        key.startswith("vf_target_")
                        or key.startswith("vf_pred_")
                        or key.startswith("vf_error_")
                        or key == "vf_explained_var"
                    ):
                        extra_stat_sums[key] = extra_stat_sums.get(key, 0.0) + value.detach().item()
                nupdate += 1

        # Compute average stats
        pi_loss_epoch /= nupdate
        vf_loss_epoch /= nupdate
        entropy_epoch /= nupdate
        phase_vf_loss_epoch /= nupdate
        short_reward_loss_epoch /= nupdate
        health_decrease_loss_epoch /= nupdate
        health_increase_loss_epoch /= nupdate
        death_event_loss_epoch /= nupdate
        health_vf_loss_epoch /= nupdate
        achievement_vf_loss_epoch /= nupdate
        health_cf_loss_epoch /= nupdate
        rank_loss_epoch /= nupdate
        value_aug_loss_epoch /= nupdate
        for mode in value_aug_mode_loss_epoch:
            value_aug_mode_loss_epoch[mode] /= nupdate

        # Define train stats
        train_stats = {
            "pi_loss": pi_loss_epoch,
            "vf_loss": vf_loss_epoch,
            "entropy": entropy_epoch,
        }
        if self.phase_vf_loss_coef > 0:
            train_stats["phase_vf_loss"] = phase_vf_loss_epoch
        if self.short_reward_loss_coef > 0:
            train_stats["short_reward_loss"] = short_reward_loss_epoch
        if self.health_decrease_loss_coef > 0:
            train_stats["health_decrease_loss"] = health_decrease_loss_epoch
        if self.health_increase_loss_coef > 0:
            train_stats["health_increase_loss"] = health_increase_loss_epoch
        if self.death_event_loss_coef > 0:
            train_stats["death_event_loss"] = death_event_loss_epoch
        if self.health_vf_loss_coef > 0:
            train_stats["health_vf_loss"] = health_vf_loss_epoch
        if self.achievement_vf_loss_coef > 0:
            train_stats["achievement_vf_loss"] = achievement_vf_loss_epoch
        if self.health_cf_loss_coef > 0:
            train_stats["health_cf_loss"] = health_cf_loss_epoch
        if self.rank_loss_coef > 0:
            train_stats["rank_loss"] = rank_loss_epoch
        if self.value_aug_coef > 0:
            train_stats["value_aug_loss"] = value_aug_loss_epoch
            train_stats["value_aug_active"] = float(value_aug_active)
            train_stats["value_aug_env_steps"] = float(self.num_env_steps)
            for mode, value in value_aug_mode_loss_epoch.items():
                train_stats[f"value_aug_loss_{mode}"] = value
        for key, value in extra_stat_sums.items():
            train_stats[key] = value / nupdate

        self.num_env_steps += storage.nstep * storage.nproc
        return train_stats

    def _update_recurrent(self, storage: RolloutStorage):
        pi_loss_epoch = 0
        vf_loss_epoch = 0
        entropy_epoch = 0
        nupdate = 0

        for _ in range(self.ppo_nepoch):
            data_loader = storage.get_recurrent_data_loader(self.ppo_nbatch)

            self.optimizer.zero_grad()
            for batch_idx, batch in enumerate(data_loader):
                with self._autocast_context():
                    losses = self.model.compute_losses(
                        **batch,
                        clip_param=self.clip_param,
                    )
                    pi_loss = losses["pi_loss"]
                    vf_loss = losses["vf_loss"]
                    entropy = losses["entropy"]
                    loss = pi_loss + self.vf_loss_coef * vf_loss - self.ent_coef * entropy

                loss = loss / self.grad_accum_steps
                self.scaler.scale(loss).backward()

                should_step = ((batch_idx + 1) % self.grad_accum_steps == 0) or (batch_idx + 1 == self.ppo_nbatch)
                if should_step:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()

                pi_loss_epoch += pi_loss.item()
                vf_loss_epoch += vf_loss.item()
                entropy_epoch += entropy.item()
                nupdate += 1

        pi_loss_epoch /= nupdate
        vf_loss_epoch /= nupdate
        entropy_epoch /= nupdate
        return {
            "pi_loss": pi_loss_epoch,
            "vf_loss": vf_loss_epoch,
            "entropy": entropy_epoch,
        }

    def _autocast_context(self):
        if self.use_amp:
            return th.amp.autocast("cuda")
        return nullcontext()

    def _value_aug_active(self, storage: RolloutStorage) -> bool:
        if self.value_aug_coef <= 0:
            return False
        next_env_steps = self.num_env_steps + storage.nstep * storage.nproc
        return next_env_steps >= self.value_aug_start_step

    def _flip_obs_world(self, obs: th.Tensor, mode: str) -> th.Tensor:
        if mode not in ("horizontal", "vertical", "both"):
            raise ValueError(f"Unknown value augmentation flip mode: {mode}")
        if obs.ndim != 4:
            raise ValueError(f"Expected obs shape [B,C,H,W] for value augmentation, got {tuple(obs.shape)}")
        inventory_rows = self.value_aug_inventory_rows
        if inventory_rows <= 0:
            world = obs
            hud = None
        elif inventory_rows >= obs.shape[-2]:
            raise ValueError(
                f"value_aug_inventory_rows={inventory_rows} is invalid for observation height {obs.shape[-2]}"
            )
        else:
            world = obs[:, :, :-inventory_rows, :]
            hud = obs[:, :, -inventory_rows:, :]

        flipped_world = world
        if mode in ("horizontal", "both"):
            flipped_world = th.flip(flipped_world, dims=(-1,))
        if mode in ("vertical", "both"):
            flipped_world = th.flip(flipped_world, dims=(-2,))
        if hud is None:
            return flipped_world
        return th.cat([flipped_world, hud], dim=-2)

    def _compute_value_aug_loss(self, batch: dict[str, th.Tensor]) -> tuple[th.Tensor, dict[str, th.Tensor]]:
        forward_kwargs = {}
        if "achievement_progress" in batch:
            forward_kwargs["achievement_progress"] = batch["achievement_progress"]

        if self.value_aug_detach_target:
            with th.no_grad():
                target_vpreds = self.model.forward(batch["obs"], **forward_kwargs)["vpreds"]
        else:
            target_vpreds = self.model.forward(batch["obs"], **forward_kwargs)["vpreds"]

        mode_losses = {}
        selected_modes = self._select_value_aug_modes()
        aug_loss = target_vpreds.new_zeros(())
        if self.value_aug_source == "semantic":
            if "value_aug_obs" not in batch:
                raise RuntimeError("value_aug_source='semantic' requires RolloutStorage value_aug_obs.")
            flipped_outputs = self.model.forward(batch["value_aug_obs"], **forward_kwargs)
            aug_loss = F.mse_loss(flipped_outputs["vpreds"], target_vpreds)
            mode_losses["semantic"] = aug_loss
        elif self.value_aug_source == "pixel":
            for mode in selected_modes:
                flipped_obs = self._flip_obs_world(batch["obs"], mode)
                flipped_outputs = self.model.forward(flipped_obs, **forward_kwargs)
                mode_loss = F.mse_loss(flipped_outputs["vpreds"], target_vpreds)
                mode_losses[mode] = mode_loss
                aug_loss = aug_loss + mode_loss
            if selected_modes:
                aug_loss = aug_loss / len(selected_modes)
        else:
            raise ValueError("value_aug_source must be 'pixel' or 'semantic'.")
        return aug_loss, mode_losses

    def _select_value_aug_modes(self) -> tuple[str, ...]:
        if self.value_aug_modes_per_batch <= 0 or self.value_aug_modes_per_batch >= len(self.value_aug_modes):
            return self.value_aug_modes
        perm = th.randperm(len(self.value_aug_modes))[: self.value_aug_modes_per_batch]
        return tuple(self.value_aug_modes[int(idx)] for idx in perm)
