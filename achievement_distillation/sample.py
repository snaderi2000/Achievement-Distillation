from typing import Dict
import numpy as np
import torch as th

from achievement_distillation.wrapper import VecPyTorch
from achievement_distillation.storage import RolloutStorage
from achievement_distillation.model.base import BaseModel


def sample_rollouts(
    venv: VecPyTorch,
    model: BaseModel,
    storage: RolloutStorage,
) -> Dict[str, np.ndarray]:
    # Set model to eval model
    model.eval()

    # Sample rollouts
    episode_rewards = []
    episode_lengths = []
    achievements = []
    successes = []

    # Track last-seen vitals to compute reward shaping bonuses
    last_vitals = th.zeros(storage.nproc, 3, device=venv.device)
    has_prev_vitals = th.zeros(storage.nproc, dtype=th.bool, device=venv.device)

    for step in range(storage.nstep):
        # Pass through model
        inputs = storage.get_inputs(step)
        outputs = model.act(**inputs)
        actions = outputs["actions"]

        # Step environment
        obs, rewards, dones, infos = venv.step(actions)
        outputs["obs"] = obs
        outputs["rewards"] = rewards
        outputs["masks"] = 1.0 - dones
        outputs["successes"] = infos["successes"]

        # Reward shaping: when hunger/thirst was low (<4) and improves, grant +0.75
        current_vitals = infos.get("vitals")
        if current_vitals is not None:
            prev_available = has_prev_vitals.unsqueeze(-1).expand(-1, 2)
            low_mask = (last_vitals[:, :2] < 4) & prev_available
            vital_delta = (current_vitals[:, :2] - last_vitals[:, :2]).clamp(min=0)
            bonus = 0.75 * (vital_delta * low_mask.float()).sum(dim=-1, keepdim=True)
            outputs["rewards"] = outputs["rewards"] + bonus
            last_vitals = current_vitals.clone()
            done_mask = dones.squeeze(-1).bool()
            has_prev_vitals[:] = True
            has_prev_vitals[done_mask] = False
            last_vitals[done_mask] = 0

        # Update storage
        storage.insert(**outputs, model=model)

        # Update stats
        for i, done in enumerate(dones):
            if done:
                # Episode lengths
                episode_length = infos["episode_lengths"][i].cpu().numpy()
                episode_lengths.append(episode_length)

                # Episode rewards
                episode_reward = infos["episode_rewards"][i].cpu().numpy()
                episode_rewards.append(episode_reward)

                # Achievements
                achievement = infos["achievements"][i].cpu().numpy()
                achievements.append(achievement)

                # Successes
                success = infos["successes"][i].cpu().numpy()
                successes.append(success)

    # Pass through model
    inputs = storage.get_inputs(step=-1)
    outputs = model.act(**inputs)
    vpreds = outputs["vpreds"]

    # Update storage
    storage.vpreds[-1].copy_(vpreds)

    # Stack stats
    episode_lengths = np.stack(episode_lengths, axis=0).astype(np.int32)
    episode_rewards = np.stack(episode_rewards, axis=0).astype(np.float32)
    achievements = np.stack(achievements, axis=0).astype(np.int32)
    successes = np.stack(successes, axis=0).astype(np.int32)

    # Define rollout stats
    rollout_stats = {
        "episode_lengths": episode_lengths,
        "episode_rewards": episode_rewards,
        "achievements": achievements,
        "successes": successes,
    }

    return rollout_stats
