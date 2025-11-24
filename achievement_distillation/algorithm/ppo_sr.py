import copy
from typing import Dict

import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from achievement_distillation.algorithm.base import BaseAlgorithm
from achievement_distillation.model.ppo_sr import PPOSRModel
from achievement_distillation.storage import RolloutStorage


class PPOSRAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        model: PPOSRModel,
        ppo_nepoch: int,
        ppo_nbatch: int,
        clip_param: float,
        vf_loss_coef: float,
        ent_coef: float,
        lr: float,
        max_grad_norm: float,
        # Aux params
        aux_freq: int,
        aux_nepoch: int,
        aux_batch_size: int, # New param for aux batch size
        pi_dist_coef: float,
        vf_dist_coef: float,
    ):
        super().__init__(model)
        self.model: PPOSRModel = model

        # PPO params
        self.ppo_nepoch = ppo_nepoch
        self.ppo_nbatch = ppo_nbatch
        self.clip_param = clip_param
        self.vf_loss_coef = vf_loss_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_count = 0

        # Aux params
        self.aux_freq = aux_freq
        self.aux_nepoch = aux_nepoch
        self.aux_batch_size = aux_batch_size
        self.pi_dist_coef = pi_dist_coef
        self.vf_dist_coef = vf_dist_coef

        # Optimizers
        # Main optimizer for Policy/Value
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        
        # Aux optimizer for the Survival Head + Encoder
        # We use a separate optimizer to avoid momentum interference with PPO
        self.aux_optimizer = optim.Adam(model.parameters(), lr=lr)

    def update(self, storage: RolloutStorage):
        # Set model to training mode
        self.model.train()

        # -----------------------------------------------------
        # Phase 1: Standard PPO (Maximize Reward)
        # -----------------------------------------------------
        pi_loss_epoch = 0
        vf_loss_epoch = 0
        entropy_epoch = 0
        nupdate = 0

        for _ in range(self.ppo_nepoch):
            data_loader = storage.get_data_loader(self.ppo_nbatch)

            for batch in data_loader:
                losses = self.model.compute_losses(**batch, clip_param=self.clip_param)
                pi_loss = losses["pi_loss"]
                vf_loss = losses["vf_loss"]
                entropy = losses["entropy"]
                
                loss = pi_loss + self.vf_loss_coef * vf_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                pi_loss_epoch += pi_loss.item()
                vf_loss_epoch += vf_loss.item()
                entropy_epoch += entropy.item()
                nupdate += 1

        # Compute average PPO stats
        train_stats = {
            "pi_loss": pi_loss_epoch / nupdate,
            "vf_loss": vf_loss_epoch / nupdate,
            "entropy": entropy_epoch / nupdate,
        }

        self.ppo_count += 1

        # -----------------------------------------------------
        # Phase 2: Survival Risk Distillation (Auxiliary)
        # -----------------------------------------------------
        if self.ppo_count % self.aux_freq == 0:
            
            # 1. Snapshot the current model (Frozen Target)
            old_model = copy.deepcopy(self.model)
            old_model.eval()

            loss_survival_epoch = 0
            pi_dist_epoch = 0
            vf_dist_epoch = 0
            aux_nupdate = 0

            for _ in range(self.aux_nepoch):
                # Use the dedicated survival loader we added to Storage
                # We calculate nbatch based on desired batch size
                nbatch = (storage.nstep * storage.nproc) // self.aux_batch_size
                survival_loader = storage.get_survival_loader(nbatch)

                for batch in survival_loader:
                    obs = batch["obs"]
                    vitals = batch["vitals"] # [Batch, 4] (Food, Drink, Energy, Health)

                    # --- A. Calculate Ground Truth Targets ---
                    # Find the weakest vital for each agent
                    min_vital, _ = th.min(vitals, dim=-1) # [Batch]

                    # 0: Critical (< 3)
                    # 1: Caution (3-4)
                    # 2: Safe (>= 5)
                    risk_targets = th.zeros_like(min_vital).long()
                    risk_targets[min_vital >= 3] = 1
                    risk_targets[min_vital >= 5] = 2

                    # --- B. Compute Loss ---
                    # Note: We pass old_model for regularization
                    aux_losses = self.model.compute_survival_loss(
                        obs=obs,
                        risk_targets=risk_targets,
                        old_model=old_model
                    )

                    loss_survival = aux_losses["loss_survival"]
                    pi_dist = aux_losses["pi_dist"]
                    vf_dist = aux_losses["vf_dist"]

                    # Total Aux Loss
                    loss = (
                        loss_survival
                        + self.pi_dist_coef * pi_dist
                        + self.vf_dist_coef * vf_dist
                    )

                    # --- C. Optimize ---
                    self.aux_optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.aux_optimizer.step()

                    # Stats
                    loss_survival_epoch += loss_survival.item()
                    pi_dist_epoch += pi_dist.item()
                    vf_dist_epoch += vf_dist.item()
                    aux_nupdate += 1

            # Compute average Aux stats
            aux_stats = {
                "loss_survival": loss_survival_epoch / aux_nupdate,
                "pi_dist": pi_dist_epoch / aux_nupdate,
                "vf_dist": vf_dist_epoch / aux_nupdate,
            }
            train_stats.update(aux_stats)

        return train_stats