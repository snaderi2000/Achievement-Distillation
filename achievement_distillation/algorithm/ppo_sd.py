from collections import deque
import copy
from typing import Dict, Iterator, List, Tuple

import numpy as np
import torch as th
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from ot.partial import entropic_partial_wasserstein

from achievement_distillation.algorithm.base import BaseAlgorithm
from achievement_distillation.model.ppo_sd import PPOSDModel
from achievement_distillation.storage_sd import RolloutStorage


class Buffer:
    def __init__(self, maxlen: int, vital_threshold: int):
        self.segs: List[Dict[str, th.Tensor]] = deque(maxlen=maxlen)
        self.trajs: List[Dict[str, th.Tensor]] = []
        self.vital_threshold = vital_threshold
        self.cross_traj_goals: Dict[int, List[Dict[str, th.Tensor]]] = {
            0: [],
            1: [],
            2: [],
        }
        self.reset_counters()

    def reset_counters(self):
        self.needy_count = 0
        self.restoration_count = 0
        self.pair_count = 0

    def __len__(self):
        return len(self.segs)

    def insert(self, seg: Dict[str, th.Tensor]):
        self.segs.append(seg)

    def parse_segs(self):
        # Clear trajectories
        self.trajs.clear()
        self.cross_traj_goals = {0: [], 1: [], 2: []}

        if len(self.segs) == 0:
            return

        # Concatenate segments
        obs = th.cat([seg["obs"][:-1] for seg in self.segs], dim=0)
        actions = th.cat([seg["actions"] for seg in self.segs], dim=0)
        states = th.cat([seg["states"][:-1] for seg in self.segs], dim=0)
        returns = th.cat([seg["returns"] for seg in self.segs], dim=0)
        masks = th.cat([seg["masks"][:-1] for seg in self.segs], dim=0)
        rewards = th.cat([seg["rewards"] for seg in self.segs], dim=0)
        vitals = th.cat([seg["vitals"][:-1] for seg in self.segs], dim=0)
        next_vitals = th.cat([seg["vitals"][1:] for seg in self.segs], dim=0)

        # Sanity check
        assert (
            len(obs)
            == len(actions)
            == len(states)
            == len(returns)
            == len(masks)
            == len(rewards)
            == len(vitals)
            == len(next_vitals)
        )

        # Split into trajectories
        nproc = obs.shape[1]

        for p in range(nproc):
            # Get segment per process
            obs_p = obs[:, p]
            actions_p = actions[:, p]
            states_p = states[:, p]
            returns_p = returns[:, p]
            masks_p = masks[:, p]
            rewards_p = rewards[:, p]
            vitals_p = vitals[:, p]
            next_vitals_p = next_vitals[:, p]

            # Get done steps
            done_conds_p = (masks_p == 0).squeeze(dim=-1)
            done_steps_p = done_conds_p.nonzero(as_tuple=False).flatten()
            done_steps_p = done_steps_p.tolist()
            done_steps_p = sorted(done_steps_p)

            for start, end in zip(done_steps_p[:-1], done_steps_p[1:]):
                # Get trajectory
                obs_t = obs_p[start:end]
                actions_t = actions_p[start:end]
                states_t = states_p[start:end]
                returns_t = returns_p[start:end]
                rewards_t = rewards_p[start:end]
                vitals_t = vitals_p[start:end]
                next_vitals_t = next_vitals_p[start:end]

                # Store trajectory
                traj = {
                    "obs": obs_t,
                    "actions": actions_t,
                    "old_states": states_t,
                    "old_vtargs": returns_t,
                    "rewards": rewards_t,
                    "vitals": vitals_t,
                    "next_vitals": next_vitals_t,
                }
                self.trajs.append(traj)
                self.register_cross_traj_goals(traj)

    def register_cross_traj_goals(self, traj: Dict[str, th.Tensor]):
        goal_obs = traj["goal_obs"]
        goal_next_obs = traj["goal_next_obs"]
        goal_vital_idx = traj["goal_vital_idx"]

        if len(goal_obs) == 0:
            return

        for idx in range(len(goal_obs)):
            vital_idx = int(goal_vital_idx[idx].item())
            self.cross_traj_goals[vital_idx].append(
                {
                    "goal_obs": goal_obs[idx].cpu(),
                    "goal_next_obs": goal_next_obs[idx].cpu(),
                }
            )

    def preprocess_trajs(self):
        # Loop over trajectories
        for traj in self.trajs:
            obs = traj["obs"]
            vitals = traj["vitals"]
            next_vitals = traj["next_vitals"]

            goals = self.get_goals(obs, vitals, next_vitals)
            traj.update(goals)
            traj["pred_pairs"] = self.get_pred_pairs(traj)

    def get_goals(
        self,
        obs: th.Tensor,
        vitals: th.Tensor,
        next_vitals: th.Tensor,
    ) -> Dict[str, th.Tensor]:
        device = obs.device
        dtype = obs.dtype

        if len(obs) <= 1:
            return {
                "goal_steps": th.zeros(0, dtype=th.long, device=device),
                "goal_obs": th.zeros(0, *obs.shape[1:], dtype=dtype, device=device),
                "goal_next_obs": th.zeros(
                    0, *obs.shape[1:], dtype=dtype, device=device
                ),
            }

        vitals_t = vitals[:-1]
        next_vitals_t = next_vitals[:-1]

        deficit = vitals_t < self.vital_threshold
        restoration = (next_vitals_t > vitals_t) & deficit
        goal_conds = restoration.any(dim=-1)
        goal_steps = goal_conds.nonzero(as_tuple=False).flatten()
        goal_steps = goal_steps + 1

        if len(goal_steps) == 0:
            goal_obs = th.zeros(0, *obs.shape[1:], dtype=dtype, device=device)
            goal_next_obs = th.zeros(0, *obs.shape[1:], dtype=dtype, device=device)
            goal_vital_idx = th.zeros(0, dtype=th.long, device=device)
        else:
            goal_obs = obs[goal_steps - 1]
            goal_next_obs = obs[goal_steps]
            restoration_events = restoration[goal_steps - 1]
            goal_vital_idx = restoration_events.float().argmax(dim=-1)

        goals = {
            "goal_steps": goal_steps,
            "goal_obs": goal_obs,
            "goal_next_obs": goal_next_obs,
            "goal_vital_idx": goal_vital_idx,
        }
        return goals

    def get_pred_pairs(self, traj: Dict[str, th.Tensor]) -> List[Dict[str, int]]:
        obs = traj["obs"]
        vitals = traj["vitals"]
        next_vitals = traj["next_vitals"]
        goal_steps = traj["goal_steps"]
        goal_vital_idx = traj["goal_vital_idx"]

        if len(obs) <= 1 or len(goal_steps) == 0:
            return []

        goal_step_to_idx = {int(step.item()): idx for idx, step in enumerate(goal_steps)}

        deficit_mask = vitals < self.vital_threshold
        restoration_mask = (next_vitals > vitals) & deficit_mask
        self.restoration_count += restoration_mask.any(dim=-1).float().sum().item()
        pairs: List[Dict[str, int]] = []

        for t in range(len(obs) - 1):
            needy_dims = deficit_mask[t].nonzero(as_tuple=False).flatten()
            if len(needy_dims) > 0:
                self.needy_count += 1
            if len(needy_dims) == 0:
                continue

            best_step = None
            best_dim = None

            for dim in needy_dims.tolist():
                future_events = restoration_mask[t:, dim]
                future_steps = future_events.nonzero(as_tuple=False).flatten()
                if len(future_steps) == 0:
                    continue
                candidate_step = t + future_steps[0].item()
                if best_step is None or candidate_step < best_step:
                    best_step = candidate_step
                    best_dim = dim

            if best_step is None:
                alt_pair = self.get_cross_traj_pair(needy_dims.tolist(), need_step=t)
                if alt_pair is None:
                    continue
                pairs.append(alt_pair)
                self.pair_count += 1
                continue

            goal_step_value = best_step + 1
            goal_idx = goal_step_to_idx.get(goal_step_value)
            if goal_idx is None:
                alt_pair = self.get_cross_traj_pair(
                    [best_dim] if best_dim is not None else needy_dims.tolist(),
                    need_step=t,
                )
                if alt_pair is None:
                    continue
                pairs.append(alt_pair)
                self.pair_count += 1
                continue

            pairs.append(
                {
                    "need_step": t,
                    "goal_idx": goal_idx,
                    "cross": False,
                }
            )
            self.pair_count += 1

        return pairs
    def get_cross_traj_pair(
        self, needy_dims: List[int], need_step: int
    ) -> Dict[str, th.Tensor] | None:
        for dim in needy_dims:
            goals = self.cross_traj_goals.get(dim, [])
            if len(goals) == 0:
                continue
            goal = goals.pop()
            return {
                "need_step": need_step,
                "cross": True,
                "cross_goal_obs": goal["goal_obs"],
                "cross_goal_next_obs": goal["goal_next_obs"],
            }

        return None

    def get_next_goals(
        self,
        goal_steps: th.Tensor,
        goal_obs: th.Tensor,
        goal_next_obs: th.Tensor,
        obs: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        next_goal_obs = []
        next_goal_next_obs = []
        goal_steps = goal_steps.tolist()
        goal_steps = sorted(set([0] + goal_steps + [len(obs)]))

        for i, (start, end) in enumerate(zip(goal_steps[:-1], goal_steps[1:])):
            if i == len(goal_steps) - 2:
                next_goal_ob = obs[-1].unsqueeze(dim=0)
                next_goal_next_ob = th.zeros_like(obs[-1]).unsqueeze(dim=0)
            else:
                next_goal_ob = goal_obs[i].unsqueeze(dim=0)
                next_goal_next_ob = goal_next_obs[i].unsqueeze(dim=0)

            next_goal_ob = next_goal_ob.repeat_interleave(end - start, dim=0)
            next_goal_obs.append(next_goal_ob)

            next_goal_next_ob = next_goal_next_ob.repeat_interleave(
                end - start, dim=0
            )
            next_goal_next_obs.append(next_goal_next_ob)

        next_goal_obs = th.cat(next_goal_obs, dim=0)
        next_goal_next_obs = th.cat(next_goal_next_obs, dim=0)

        return next_goal_obs, next_goal_next_obs

    def get_pred_data_loader(
        self,
        max_batch_size: int = 512,
    ) -> Iterator[Dict[str, th.Tensor]]:
        trajs = [traj for traj in self.trajs if len(traj["pred_pairs"]) > 0]
        ntraj = len(trajs)

        if ntraj == 0:
            return iter([])

        for i in th.randperm(ntraj):
            traj = trajs[i]
            pairs = traj["pred_pairs"]
            obs = traj["obs"]
            actions = traj["actions"]
            old_states = traj["old_states"]
            old_vtargs = traj["old_vtargs"]
            goal_obs = traj["goal_obs"]
            goal_next_obs = traj["goal_next_obs"]

            anc_goal_obs_list = []
            anc_goal_next_obs_list = []
            pos_obs_list = []
            pos_actions_list = []
            pos_states_list = []
            pos_vtargs_list = []

            for pair in pairs:
                need_step = pair["need_step"]
                pos_obs_list.append(obs[need_step])
                pos_actions_list.append(actions[need_step])
                pos_states_list.append(old_states[need_step])
                pos_vtargs_list.append(old_vtargs[need_step])

                if pair.get("cross", False):
                    anc_goal_obs_list.append(pair["cross_goal_obs"].to(obs.device))
                    anc_goal_next_obs_list.append(pair["cross_goal_next_obs"].to(obs.device))
                else:
                    goal_idx = pair["goal_idx"]
                    anc_goal_obs_list.append(goal_obs[goal_idx])
                    anc_goal_next_obs_list.append(goal_next_obs[goal_idx])

            anc_goal_obs = th.stack(anc_goal_obs_list)
            anc_goal_next_obs = th.stack(anc_goal_next_obs_list)
            pos_obs = th.stack(pos_obs_list)
            pos_actions = th.stack(pos_actions_list)
            pos_old_states = th.stack(pos_states_list)
            pos_old_vtargs = th.stack(pos_vtargs_list)

            ndata = len(pairs)
            rand_inds = th.randint(len(obs), (ndata,))
            neg_obs = obs[rand_inds]
            neg_actions = actions[rand_inds]
            neg_old_states = old_states[rand_inds]
            neg_old_vtargs = old_vtargs[rand_inds]

            sampler = SubsetRandomSampler(range(ndata))
            sampler = BatchSampler(sampler, batch_size=max_batch_size, drop_last=False)

            for inds in sampler:
                batch = {
                    "anc_goal_obs": anc_goal_obs[inds].cuda(),
                    "anc_goal_next_obs": anc_goal_next_obs[inds].cuda(),
                    "pos_obs": pos_obs[inds].cuda(),
                    "pos_actions": pos_actions[inds].cuda(),
                    "pos_old_states": pos_old_states[inds].cuda(),
                    "pos_old_vtargs": pos_old_vtargs[inds].cuda(),
                    "neg_obs": neg_obs[inds].cuda(),
                    "neg_actions": neg_actions[inds].cuda(),
                    "neg_old_states": neg_old_states[inds].cuda(),
                    "neg_old_vtargs": neg_old_vtargs[inds].cuda(),
                }
                yield batch

    def get_match_data_loader(
        self,
        model: PPOSDModel,
        max_batch_size: int = 512,
    ) -> Iterator[Dict[str, th.Tensor]]:
        trajs = [traj for traj in self.trajs if len(traj["goal_steps"]) > 0]
        ntraj = len(trajs)

        if ntraj <= 1:
            return iter([])

        for i in th.randperm(ntraj):
            traj_s = trajs[i]
            obs_s = traj_s["obs"]
            old_states_s = traj_s["old_states"]
            old_vtargs_s = traj_s["old_vtargs"]
            goal_obs_s = traj_s["goal_obs"]
            goal_next_obs_s = traj_s["goal_next_obs"]

            with th.no_grad():
                goal_obs_s = goal_obs_s.cuda()
                goal_next_obs_s = goal_next_obs_s.cuda()
                states_s = model.get_states(goal_obs_s, goal_next_obs_s)

            anc_goal_obs = []
            anc_goal_next_obs = []
            pos_goal_obs = []
            pos_goal_next_obs = []
            neg_goal_obs = []
            neg_goal_next_obs = []

            inds = th.randperm(ntraj - 1)[:16]

            for j in inds:
                if ntraj <= 1:
                    break
                if j >= i:
                    j += 1
                    if j >= ntraj:
                        j = j % ntraj
                        if j == i:
                            continue

                traj_t = trajs[j]
                goal_obs_t = traj_t["goal_obs"]
                goal_next_obs_t = traj_t["goal_next_obs"]

                with th.no_grad():
                    goal_obs_t = goal_obs_t.cuda()
                    goal_next_obs_t = goal_next_obs_t.cuda()
                    states_t = model.get_states(goal_obs_t, goal_next_obs_t)

                a = np.ones(len(states_s))
                b = np.ones(len(states_t))
                M = 1 - th.einsum("ik,jk->ij", states_s, states_t).cpu().numpy()
                T = entropic_partial_wasserstein(a, b, M, reg=0.05, numItermax=100)
                T = th.from_numpy(T).float()
                row_inds, col_inds = th.where(T > 0.5)

                if len(row_inds) == 0:
                    continue

                anc_goal_obs.append(goal_obs_s[row_inds])
                anc_goal_next_obs.append(goal_next_obs_s[row_inds])

                pos_goal_obs.append(goal_obs_t[col_inds])
                pos_goal_next_obs.append(goal_next_obs_t[col_inds])

                rand_inds = th.randint(len(goal_obs_t), (len(col_inds),))
                neg_goal_obs.append(goal_obs_t[rand_inds])
                neg_goal_next_obs.append(goal_next_obs_t[rand_inds])

            if len(anc_goal_obs) == 0:
                continue

            anc_goal_obs = th.cat(anc_goal_obs, dim=0)
            anc_goal_next_obs = th.cat(anc_goal_next_obs, dim=0)
            pos_goal_obs = th.cat(pos_goal_obs, dim=0)
            pos_goal_next_obs = th.cat(pos_goal_next_obs, dim=0)
            neg_goal_obs = th.cat(neg_goal_obs, dim=0)
            neg_goal_next_obs = th.cat(neg_goal_next_obs, dim=0)

            ndata = len(anc_goal_obs)
            sampler = SubsetRandomSampler(range(ndata))
            sampler = BatchSampler(sampler, batch_size=max_batch_size, drop_last=False)

            rand_inds = th.randint(len(obs_s), (ndata,))
            obs = obs_s[rand_inds]
            old_states = old_states_s[rand_inds]
            old_vtargs = old_vtargs_s[rand_inds]

            for inds in sampler:
                batch = {
                    "anc_goal_obs": anc_goal_obs[inds].cuda(),
                    "anc_goal_next_obs": anc_goal_next_obs[inds].cuda(),
                    "pos_goal_obs": pos_goal_obs[inds].cuda(),
                    "pos_goal_next_obs": pos_goal_next_obs[inds].cuda(),
                    "neg_goal_obs": neg_goal_obs[inds].cuda(),
                    "neg_goal_next_obs": neg_goal_next_obs[inds].cuda(),
                    "obs": obs[inds].cuda(),
                    "old_states": old_states[inds].cuda(),
                    "old_vtargs": old_vtargs[inds].cuda(),
                }
                yield batch


class PPOSDAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        model: PPOSDModel,
        ppo_nepoch: int,
        ppo_nbatch: int,
        clip_param: float,
        vf_loss_coef: float,
        ent_coef: float,
        lr: float,
        max_grad_norm: float,
        aux_freq: int,
        aux_nepoch: int,
        pi_dist_coef: int,
        vf_dist_coef: int,
        vital_threshold: int = 4,
    ):
        super().__init__(model)
        self.model: PPOSDModel

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
        self.pi_dist_coef = pi_dist_coef
        self.vf_dist_coef = vf_dist_coef

        # Buffer
        self.buffer = Buffer(maxlen=aux_freq, vital_threshold=vital_threshold)

        # Optimizers
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.match_optimizer = optim.Adam(model.parameters(), lr=lr)
        self.pred_optimizer = optim.Adam(model.parameters(), lr=lr)

    def update(self, storage: RolloutStorage):
        # Set model to training mode
        self.model.train()

        # Insert data to buffer
        keys = ["obs", "actions", "states", "returns", "masks", "rewards", "vitals"]
        seg = {key: storage[key].cpu() for key in keys}
        self.buffer.insert(seg)

        # Run PPO
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

        pi_loss_epoch /= nupdate
        vf_loss_epoch /= nupdate
        entropy_epoch /= nupdate

        train_stats = {
            "pi_loss": pi_loss_epoch,
            "vf_loss": vf_loss_epoch,
            "entropy": entropy_epoch,
        }

        self.ppo_count += 1

        if self.ppo_count % self.aux_freq == 0 and len(self.buffer) > 0:
            self.buffer.parse_segs()
            self.buffer.preprocess_trajs()
            print(
                "[SD Buffer] needy_count="
                f"{int(self.buffer.needy_count)} "
                f"restoration_count={int(self.buffer.restoration_count)} "
                f"pair_count={int(self.buffer.pair_count)} "
                f"cross_pool_sizes={[len(v) for v in self.buffer.cross_traj_goals.values()]}"
            )
            self.buffer.reset_counters()

            old_model = copy.deepcopy(self.model)
            old_model.eval()

            match_loss_epoch = 0
            pred_loss_epoch = 0
            pi_dist_epoch = 0
            vf_dist_epoch = 0
            match_nupdate = 0
            pred_nupdate = 0

            for _ in range(self.aux_nepoch):
                match_data_loader = self.buffer.get_match_data_loader(self.model)

                for batch in match_data_loader:
                    match_losses = self.model.compute_match_losses(
                        **batch,
                        old_model=old_model,
                    )
                    match_loss = match_losses["match_loss"]
                    pi_dist = match_losses["pi_dist"]
                    vf_dist = match_losses["vf_dist"]
                    loss = (
                        match_loss
                        + self.pi_dist_coef * pi_dist
                        + self.vf_dist_coef * vf_dist
                    )

                    self.match_optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.match_optimizer.step()

                    match_loss_epoch += match_loss.item()
                    pi_dist_epoch += pi_dist.item()
                    vf_dist_epoch += vf_dist.item()
                    match_nupdate += 1

                pred_data_loader = self.buffer.get_pred_data_loader()

                for batch in pred_data_loader:
                    pred_losses = self.model.compute_pred_losses(
                        **batch,
                        old_model=old_model,
                    )
                    pred_loss = pred_losses["pred_loss"]
                    pi_dist = pred_losses["pi_dist"]
                    vf_dist = pred_losses["vf_dist"]
                    loss = (
                        pred_loss
                        + self.pi_dist_coef * pi_dist
                        + self.vf_dist_coef * vf_dist
                    )

                    self.pred_optimizer.zero_grad()
                    loss.backward()
                    clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.pred_optimizer.step()

                    pred_loss_epoch += pred_loss.item()
                    pi_dist_epoch += pi_dist.item()
                    vf_dist_epoch += vf_dist.item()
                    pred_nupdate += 1

            if match_nupdate > 0:
                match_loss_epoch /= match_nupdate
            if pred_nupdate > 0:
                pred_loss_epoch /= pred_nupdate
            total_updates = max(match_nupdate + pred_nupdate, 1)
            pi_dist_epoch /= total_updates
            vf_dist_epoch /= total_updates

            aux_train_stats = {
                "match_loss": match_loss_epoch,
                "pred_loss": pred_loss_epoch,
                "pi_dist": pi_dist_epoch,
                "vf_dist": vf_dist_epoch,
            }
            train_stats.update(aux_train_stats)

        return train_stats
