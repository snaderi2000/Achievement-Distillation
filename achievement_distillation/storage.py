from typing import Dict, Iterator

import torch as th
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from gym import spaces

from achievement_distillation.model.base import BaseModel

from typing import Optional



class RolloutStorage:
    def __init__(
        self,
        nstep: int,
        nproc: int,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        hidsize: int,
        rnn_hidsize: int | None,
        device: th.device,
    ):
        # Params
        self.nstep = nstep
        self.nproc = nproc
        self.device = device
        self.rnn_hidsize = hidsize if rnn_hidsize is None else rnn_hidsize

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
        self.successes = th.zeros(nstep + 1, nproc, 22, device=device).long()
        self.timesteps = th.zeros(nstep + 1, nproc, 1, device=device).long()
        self.states = th.zeros(nstep + 1, nproc, hidsize, device=device)
        self.rnn_states = th.zeros(nstep + 1, nproc, self.rnn_hidsize, device=device)
        self.vitals = th.zeros(nstep + 1, nproc, 4, device=device)
        self.done_episode_lengths = th.zeros(nstep, nproc, 1, device=device).long()
        self.progress_bonus_counts = th.zeros(23, device=device).float()

        # Step
        self.step = 0

    def __getitem__(self, key: str) -> th.Tensor:
        return getattr(self, key)

    def get_inputs(self, step: int):
        inputs = {
            "obs": self.obs[step],
            "states": self.states[step],
            "rnn_states": self.rnn_states[step],
        }
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
        successes: th.Tensor,
        model: BaseModel,
        vitals: Optional[th.Tensor] = None,
        episode_lengths: Optional[th.Tensor] = None,
        next_rnn_states: Optional[th.Tensor] = None,
        **kwargs,
    ):
        # Get prev successes, timesteps, and states
        prev_successes = self.successes[self.step]
        prev_states = self.states[self.step]
        prev_timesteps = self.timesteps[self.step]
        # Update timesteps
        timesteps = prev_timesteps + 1

        # Update states if new achievment is unlocked
        success_conds = successes != prev_successes
        success_conds = success_conds.any(dim=-1, keepdim=True)
        if success_conds.any():
            with th.no_grad():
                next_latents = model.encode(obs)
            states = next_latents - latents
            states = F.normalize(states, dim=-1)
            states = th.where(success_conds, states, prev_states)
        else:
            states = prev_states

        # Update successes, timesteps, and states if done
        done_conds = masks == 0
        successes = th.where(done_conds, 0, successes)
        timesteps = th.where(done_conds, 0, timesteps)
        states = th.where(done_conds, 0, states)
    
        # Update tensors
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.vpreds[self.step].copy_(vpreds)
        self.log_probs[self.step].copy_(log_probs)
        self.successes[self.step + 1].copy_(successes)
        self.timesteps[self.step + 1].copy_(timesteps)
        self.states[self.step + 1].copy_(states)
        if vitals is not None:
            self.vitals[self.step + 1].copy_(vitals)
        if next_rnn_states is not None:
            next_rnn_states = th.where(done_conds, 0, next_rnn_states)
            self.rnn_states[self.step + 1].copy_(next_rnn_states)
        else:
            self.rnn_states[self.step + 1].zero_()
        if episode_lengths is not None:
            self.done_episode_lengths[self.step].copy_(episode_lengths.view(-1, 1).long())
        else:
            self.done_episode_lengths[self.step].zero_()
        # Update step
        self.step = (self.step + 1) % self.nstep

    def reset(self):
        # Reset tensors
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.successes[0].copy_(self.successes[-1])
        self.timesteps[0].copy_(self.timesteps[-1])
        self.states[0].copy_(self.states[-1])
        self.rnn_states[0].copy_(self.rnn_states[-1])
        self.vitals[0].copy_(self.vitals[-1])
        self.done_episode_lengths.zero_()
        # Reset step
        self.step = 0

    def get_phase_labels(self, num_phases: int = 4):
        phase_ids = th.full((self.nstep, self.nproc, 1), -1, device=self.device).long()
        phase_mask = th.zeros((self.nstep, self.nproc, 1), device=self.device).bool()

        for env_idx in range(self.nproc):
            segment_start = 0
            while segment_start < self.nstep:
                done_indices = (self.done_episode_lengths[segment_start:, env_idx, 0] > 0).nonzero(as_tuple=False)
                if len(done_indices) == 0:
                    break
                segment_end = segment_start + int(done_indices[0].item())
                episode_length = int(self.done_episode_lengths[segment_end, env_idx, 0].item())
                if episode_length <= 0:
                    segment_start = segment_end + 1
                    continue

                episode_steps = self.timesteps[segment_start:segment_end + 1, env_idx, 0]
                denom = max(episode_length, 1)
                phase_values = th.clamp((episode_steps * num_phases) // denom, max=num_phases - 1)
                phase_ids[segment_start:segment_end + 1, env_idx, 0] = phase_values
                phase_mask[segment_start:segment_end + 1, env_idx, 0] = True
                segment_start = segment_end + 1

        return phase_ids, phase_mask

    def get_progress_bins(self, num_bins: int = 4):
        success_counts = self.successes[:-1].sum(dim=-1, keepdim=True).long()
        if num_bins <= 1:
            return th.zeros_like(success_counts)
        max_successes = self.successes.shape[-1]
        progress_bins = (success_counts * num_bins) // max_successes
        progress_bins = th.clamp(progress_bins, max=num_bins - 1)
        return progress_bins

    def get_short_reward_targets(self, horizon: int):
        short_targets = th.zeros((self.nstep, self.nproc, 1), device=self.device)
        short_mask = th.zeros((self.nstep, self.nproc, 1), device=self.device).bool()

        if horizon <= 0:
            return short_targets, short_mask

        for env_idx in range(self.nproc):
            rewards = self.rewards[:, env_idx, 0]
            next_masks = self.masks[1:, env_idx, 0]
            for step in range(self.nstep):
                target = rewards.new_zeros(())
                terminated = False
                for offset in range(horizon):
                    future_step = step + offset
                    if future_step >= self.nstep:
                        break
                    target = target + rewards[future_step]
                    if next_masks[future_step] == 0:
                        terminated = True
                        break

                # Exact target if we saw the full horizon in-buffer, or the episode ended
                # before the horizon elapsed.
                if step + horizon <= self.nstep or terminated:
                    short_targets[step, env_idx, 0] = target
                    short_mask[step, env_idx, 0] = True

        return short_targets, short_mask

    def get_health_event_targets(self, horizon: int, reward_mag: float = 0.1, tol: float = 0.05):
        decrease_targets = th.zeros((self.nstep, self.nproc, 1), device=self.device)
        increase_targets = th.zeros((self.nstep, self.nproc, 1), device=self.device)
        event_mask = th.zeros((self.nstep, self.nproc, 1), device=self.device).bool()

        if horizon <= 0:
            return decrease_targets, increase_targets, event_mask

        lower_pos, upper_pos = reward_mag - tol, reward_mag + tol
        lower_neg, upper_neg = -reward_mag - tol, -reward_mag + tol

        for env_idx in range(self.nproc):
            rewards = self.rewards[:, env_idx, 0]
            next_masks = self.masks[1:, env_idx, 0]
            for step in range(self.nstep):
                terminated = False
                window = []
                for offset in range(horizon):
                    future_step = step + offset
                    if future_step >= self.nstep:
                        break
                    reward = rewards[future_step]
                    window.append(reward)
                    if next_masks[future_step] == 0:
                        terminated = True
                        break

                if step + horizon <= self.nstep or terminated:
                    if window:
                        window = th.stack(window)
                        increase_targets[step, env_idx, 0] = ((window >= lower_pos) & (window <= upper_pos)).any().float()
                        decrease_targets[step, env_idx, 0] = ((window >= lower_neg) & (window <= upper_neg)).any().float()
                    event_mask[step, env_idx, 0] = True

        return decrease_targets, increase_targets, event_mask

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

    def get_data_loader(
        self,
        nbatch: int,
        short_reward_horizon: int = 0,
        health_event_horizon: int = 0,
        num_phase_bins: int = 4,
        num_progress_bins: int = 4,
    ) -> Iterator[Dict[str, th.Tensor]]:
        # Get sampler
        ndata = self.nstep * self.nproc
        assert ndata >= nbatch
        batch_size = ndata // nbatch
        sampler = SubsetRandomSampler(range(ndata))
        sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

        # Sample batch
        obs = self.obs[:-1].view(-1, *self.obs.shape[2:])
        states = self.states[:-1].view(-1, *self.states.shape[2:])
        rnn_states = self.rnn_states[:-1].view(-1, *self.rnn_states.shape[2:])
        actions = self.actions.view(-1, *self.actions.shape[2:])
        vtargs = self.returns.view(-1, *self.returns.shape[2:])
        log_probs = self.log_probs.view(-1, *self.log_probs.shape[2:])
        advs = self.advs.view(-1, *self.advs.shape[2:])
        phase_ids, phase_mask = self.get_phase_labels(num_phase_bins)
        phase_ids = phase_ids.view(-1, 1)
        phase_mask = phase_mask.view(-1, 1)
        progress_bins = self.get_progress_bins(num_progress_bins).view(-1, 1)
        short_reward_targets, short_reward_mask = self.get_short_reward_targets(short_reward_horizon)
        short_reward_targets = short_reward_targets.view(-1, 1)
        short_reward_mask = short_reward_mask.view(-1, 1)
        health_decrease_targets, health_increase_targets, health_event_mask = self.get_health_event_targets(health_event_horizon)
        health_decrease_targets = health_decrease_targets.view(-1, 1)
        health_increase_targets = health_increase_targets.view(-1, 1)
        health_event_mask = health_event_mask.view(-1, 1)

        for indices in sampler:
            batch = {
                "obs": obs[indices],
                "states": states[indices],
                "rnn_states": rnn_states[indices],
                "actions": actions[indices],
                "vtargs": vtargs[indices],
                "log_probs": log_probs[indices],
                "advs": advs[indices],
                "phase_ids": phase_ids[indices],
                "phase_mask": phase_mask[indices],
                "progress_bins": progress_bins[indices],
                "short_reward_targets": short_reward_targets[indices],
                "short_reward_mask": short_reward_mask[indices],
                "health_decrease_targets": health_decrease_targets[indices],
                "health_increase_targets": health_increase_targets[indices],
                "health_event_mask": health_event_mask[indices],
            }
            yield batch

    def get_recurrent_data_loader(self, nbatch: int) -> Iterator[Dict[str, th.Tensor]]:
        assert self.nproc >= nbatch
        batch_envs = self.nproc // nbatch
        sampler = SubsetRandomSampler(range(self.nproc))
        sampler = BatchSampler(sampler, batch_size=batch_envs, drop_last=True)

        obs = self.obs[:-1]
        actions = self.actions
        vtargs = self.returns
        log_probs = self.log_probs
        advs = self.advs
        masks = self.masks[1:]
        init_rnn_states = self.rnn_states[0]

        for env_indices in sampler:
            env_indices = th.tensor(env_indices, device=self.device, dtype=th.long)
            batch = {
                "obs": obs[:, env_indices],
                "actions": actions[:, env_indices],
                "vtargs": vtargs[:, env_indices],
                "log_probs": log_probs[:, env_indices],
                "advs": advs[:, env_indices],
                "masks": masks[:, env_indices],
                "init_rnn_states": init_rnn_states[env_indices],
            }
            yield batch

    # Load vital states

    def get_survival_loader(self, nbatch: int) -> Iterator[Dict[str, th.Tensor]]:
        # 1. Flatten dimensions (merge nstep and nproc)
        # We assume self.vitals is shape (nstep+1, nproc, 4)
        # We slice [:-1] to ignore the final observation (which has no action/reward following it)
        obs = self.obs[:-1].view(-1, *self.obs.shape[2:])
        vitals = self.vitals[:-1].view(-1, *self.vitals.shape[2:])

        # 2. Create Sampler
        ndata = obs.shape[0]
        batch_size = ndata // nbatch
        sampler = SubsetRandomSampler(range(ndata))
        sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=True)

        # 3. Yield Batches
        for indices in sampler:
            batch = {
                "obs": obs[indices],
                "vitals": vitals[indices],
            }
            yield batch
