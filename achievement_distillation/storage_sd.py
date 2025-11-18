from typing import Dict, Iterator

import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from gym import spaces

from achievement_distillation.model.base import BaseModel


class RolloutStorage:
    def __init__(
        self,
        nstep: int,
        nproc: int,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        device: th.device,
    ):
        # Params
        self.nstep = nstep
        self.nproc = nproc
        self.device = device

        # Get obs shape and action dim
        assert isinstance(observation_space, spaces.Box)
        assert isinstance(action_space, spaces.Discrete)
        obs_shape = getattr(observation_space, "shape")
        action_shape = (1,)

        # Tensors
        self.obs = th.zeros(nstep + 1, nproc, *obs_shape, device=device)
        self.actions = th.zeros(nstep, nproc, *action_shape, device=device).long()
        self.rewards = th.zeros(nstep, nproc, 1, device=device)
        self.masks = th.ones(nstep + 1, nproc, 1, device=device)
        self.vpreds = th.zeros(nstep + 1, nproc, 1, device=device)
        self.log_probs = th.zeros(nstep, nproc, 1, device=device)
        self.returns = th.zeros(nstep, nproc, 1, device=device)
        self.advs = th.zeros(nstep, nproc, 1, device=device)
        self.vitals = th.zeros(nstep + 1, nproc, 3, device=device).long()
        self.timesteps = th.zeros(nstep + 1, nproc, 1, device=device).long()
        self.states = th.zeros(nstep + 1, nproc, hidsize, device=device)

        # Step
        self.step = 0

    def __getitem__(self, key: str) -> th.Tensor:
        return getattr(self, key)

    def get_inputs(self, step: int):
        inputs = {"obs": self.obs[step], "states": self.states[step]}
        return inputs

    def insert(
        self,
        obs: th.Tensor,
        latents: th.Tensor,
        actions: th.Tensor,
        rewards: th.Tensor,
        masks: th.Tensor,
        vpreds: th.Tensor,
        log_probs: th.Tensor,
        vitals: th.Tensor,
        model: BaseModel,
        **kwargs,
    ):
        # Get prev successes, timesteps, and states
        prev_vitals = self.vitals[self.step]
        prev_states = self.states[self.step]
        prev_timesteps = self.timesteps[self.step]

        # Update timesteps
        timesteps = prev_timesteps + 1

        # Update states if new achievment is unlocked
        # For survival distillation, we don't update states based on achievements
        states = prev_states

        # Update vitals, timesteps, and states if done
        done_conds = masks == 0
        vitals = th.where(done_conds, 0, vitals)
        timesteps = th.where(done_conds, 0, timesteps)
        states = th.where(done_conds, 0, states)

        # Update tensors
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.vpreds[self.step].copy_(vpreds)
        self.log_probs[self.step].copy_(log_probs)
        self.vitals[self.step + 1].copy_(vitals)
        self.timesteps[self.step + 1].copy_(timesteps)
        self.states[self.step + 1].copy_(states)

        # Update step
        self.step = (self.step + 1) % self.nstep

    def reset(self):
        # Reset tensors
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.vitals[0].copy_(self.vitals[-1])
        self.timesteps[0].copy_(self.timesteps[-1])
        self.states[0].copy_(self.states[-1])

        # Reset step
        self.step = 0

    def compute_returns(self, gamma: float, gae_lambda: float):
        # Compute returns
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = (
                self.rewards[step]
                + gamma * self.vpreds[step + 1] * self.masks[step + 1]
                - self.vpreds[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.vpreds[step]
            self.advs[step] = gae

        # Compute advantages
        self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-8)

    def get_goals(self):
        # Calculate difference in vitals
        vitals_diff = self.vitals[1:] - self.vitals[:-1]

        # Identify restoration events
        restoration_events = (vitals_diff > 0) & (self.vitals[:-1] < 4)

        # Get goal steps and corresponding observations
        goal_steps = restoration_events.any(dim=-1).nonzero(as_tuple=False)[:, 0]
        goal_obs = self.obs[:-1][goal_steps]
        goal_next_obs = self.obs[1:][goal_steps]

        return goal_obs, goal_next_obs, goal_steps

    def get_pred_data_loader(self, nbatch: int) -> Iterator[Dict[str, th.Tensor]]:
        goal_obs, goal_next_obs, goal_steps = self.get_goals()

        # Get restoration indices for food and water
        food_restored_indices = (self.vitals[1:, :, 0] > self.vitals[:-1, :, 0]).nonzero(as_tuple=False)
        water_restored_indices = (self.vitals[1:, :, 1] > self.vitals[:-1, :, 1]).nonzero(as_tuple=False)

        training_pairs = []
        for t in range(self.nstep):
            for proc in range(self.nproc):
                current_vitals = self.vitals[t, proc]
                is_hungry = current_vitals[0] < 4
                is_thirsty = current_vitals[1] < 4

                if is_hungry or is_thirsty:
                    future_food_restorations = food_restored_indices[
                        (food_restored_indices[:, 0] > t) & (food_restored_indices[:, 1] == proc)
                    ]
                    future_water_restorations = water_restored_indices[
                        (water_restored_indices[:, 0] > t) & (water_restored_indices[:, 1] == proc)
                    ]

                    nearest_restoration_step = float('inf')
                    
                    if is_hungry and len(future_food_restorations) > 0:
                        nearest_restoration_step = min(nearest_restoration_step, future_food_restorations[0, 0])
                    
                    if is_thirsty and len(future_water_restorations) > 0:
                        nearest_restoration_step = min(nearest_restoration_step, future_water_restorations[0, 0])

                    if nearest_restoration_step != float('inf'):
                        current_state_idx = t * self.nproc + proc
                        target_state_obs = self.obs[nearest_restoration_step + 1, proc]
                        training_pairs.append((self.obs[t, proc], target_state_obs))

        if not training_pairs:
            # Create a dummy loader if no pairs are found, to avoid errors
            return iter([])

        # Create dataset and dataloader
        obs_pairs, next_obs_pairs = zip(*training_pairs)
        obs_tensor = th.stack(obs_pairs)
        next_obs_tensor = th.stack(next_obs_pairs)

        dataset = th.utils.data.TensorDataset(obs_tensor, next_obs_tensor)
        sampler = BatchSampler(SubsetRandomSampler(range(len(dataset))), batch_size=nbatch, drop_last=True)
        
        def collate_fn(batch):
            obs, next_obs = zip(*batch)
            return {"obs": th.stack(obs), "next_obs": th.stack(next_obs)}

        return th.utils.data.DataLoader(dataset, sampler=sampler, collate_fn=collate_fn)

    def get_data_loader(self, nbatch: int) -> Iterator[Dict[str, th.Tensor]]:
        # Get sampler
        ndata = self.nstep * self.nproc
        assert ndata >= nbatch
        batch_size = ndata // nbatch
        sampler = SubsetRandomSampler(range(ndata))
        sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

        # Sample batch
        obs = self.obs[:-1].view(-1, *self.obs.shape[2:])
        states = self.states[:-1].view(-1, *self.states.shape[2:])
        actions = self.actions.view(-1, *self.actions.shape[2:])
        vtargs = self.returns.view(-1, *self.returns.shape[2:])
        log_probs = self.log_probs.view(-1, *self.log_probs.shape[2:])
        advs = self.advs.view(-1, *self.advs.shape[2:])

        for indices in sampler:
            batch = {
                "obs": obs[indices],
                "states": states[indices],
                "actions": actions[indices],
                "vtargs": vtargs[indices],
                "log_probs": log_probs[indices],
                "advs": advs[indices],
            }
            yield batch
