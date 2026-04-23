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

        # Run PPO
        pi_loss_epoch = 0
        vf_loss_epoch = 0
        entropy_epoch = 0
        phase_vf_loss_epoch = 0
        short_reward_loss_epoch = 0
        nupdate = 0

        for _ in range(self.ppo_nepoch):
            # Get data loader
            data_loader = storage.get_data_loader(
                self.ppo_nbatch,
                short_reward_horizon=self.short_reward_horizon,
            )

            for batch in data_loader:
                # Compute loss
                losses = self.model.compute_losses(**batch, clip_param=self.clip_param)
                pi_loss = losses["pi_loss"]
                vf_loss = losses["vf_loss"]
                entropy = losses["entropy"]
                phase_vf_loss = losses.get("phase_vf_loss")
                if phase_vf_loss is None:
                    phase_vf_loss = pi_loss.new_zeros(())
                short_reward_loss = losses.get("short_reward_loss")
                if short_reward_loss is None:
                    short_reward_loss = pi_loss.new_zeros(())
                loss = (
                    pi_loss
                    + self.vf_loss_coef * vf_loss
                    + self.phase_vf_loss_coef * phase_vf_loss
                    + self.short_reward_loss_coef * short_reward_loss
                    - self.ent_coef * entropy
                )

                # Update parameter
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update stats
                pi_loss_epoch += pi_loss.item()
                vf_loss_epoch += vf_loss.item()
                entropy_epoch += entropy.item()
                phase_vf_loss_epoch += phase_vf_loss.item()
                short_reward_loss_epoch += short_reward_loss.item()
                nupdate += 1

        # Compute average stats
        pi_loss_epoch /= nupdate
        vf_loss_epoch /= nupdate
        entropy_epoch /= nupdate
        phase_vf_loss_epoch /= nupdate
        short_reward_loss_epoch /= nupdate

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

        return train_stats
