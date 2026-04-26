from contextlib import nullcontext

import torch as th
import torch.nn as nn
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
        rank_loss_coef: float = 0.0,
        rank_margin: float = 0.05,
        rank_delta: float = 0.1,
        rank_num_phase_bins: int = 4,
        rank_num_progress_bins: int = 4,
        rank_max_pairs_per_group: int = 8,
        grad_accum_steps: int = 1,
        use_amp: bool = False,
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
        self.rank_loss_coef = rank_loss_coef
        self.rank_margin = rank_margin
        self.rank_delta = rank_delta
        self.rank_num_phase_bins = rank_num_phase_bins
        self.rank_num_progress_bins = rank_num_progress_bins
        self.rank_max_pairs_per_group = rank_max_pairs_per_group
        self.grad_accum_steps = grad_accum_steps
        self.use_amp = use_amp and th.cuda.is_available()
        self.scaler = th.amp.GradScaler("cuda", enabled=self.use_amp)

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
        rank_loss_epoch = 0
        nupdate = 0

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
                    rank_loss = losses.get("rank_loss")
                    if rank_loss is None:
                        rank_loss = pi_loss.new_zeros(())
                    loss = (
                        pi_loss
                        + self.vf_loss_coef * vf_loss
                        + self.phase_vf_loss_coef * phase_vf_loss
                        + self.short_reward_loss_coef * short_reward_loss
                        + self.health_decrease_loss_coef * health_decrease_loss
                        + self.health_increase_loss_coef * health_increase_loss
                        + self.death_event_loss_coef * death_event_loss
                        + self.rank_loss_coef * rank_loss
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
                rank_loss_epoch += rank_loss.item()
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
        rank_loss_epoch /= nupdate

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
        if self.rank_loss_coef > 0:
            train_stats["rank_loss"] = rank_loss_epoch

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
