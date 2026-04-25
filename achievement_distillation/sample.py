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
    progress_bonus_beta: float = 0.0,
    death_penalty: float = 0.0,
) -> Dict[str, np.ndarray]:
    # Set model to eval model
    model.eval()

    # Sample rollouts
    episode_rewards = []
    episode_lengths = []
    achievements = []
    successes = []
    intrinsic_reward_sums = []
 


    for step in range(storage.nstep):
        # Pass through model
        inputs = storage.get_inputs(step)
        outputs = model.act(**inputs)
        actions = outputs["actions"]

        # Step environment
        obs, rewards, dones, infos = venv.step(actions)

        
        # We check if 'vitals' key exists to maintain backward compatibility.
        if "vitals" in infos and "health" in infos:
            # 1. Get Inventory Vitals [Batch, 3] (Food, Drink, Energy)
            inv_vitals = infos["vitals"] 
            
            # 2. Get Health [Batch] and reshape to [Batch, 1]
            health = infos["health"].unsqueeze(-1)
            
            # 3. Concatenate to shape [Batch, 4] -> [Food, Drink, Energy, Health]
            all_vitals = th.cat([inv_vitals, health], dim=-1)
            
            # 4. Add to outputs. Storage will only save this if 'insert' accepts it.
            outputs["vitals"] = all_vitals

        intrinsic_rewards = th.zeros_like(rewards)
        if progress_bonus_beta > 0:
            progress_counts = infos["successes"].sum(dim=-1).long().clamp(
                min=0,
                max=storage.progress_bonus_counts.shape[0] - 1,
            )
            storage.progress_bonus_counts[progress_counts] += 1.0
            intrinsic_rewards = progress_bonus_beta / th.sqrt(
                storage.progress_bonus_counts[progress_counts].unsqueeze(-1)
            )

        death_rewards = th.zeros_like(rewards)
        if death_penalty != 0.0 and "health" in infos:
            died = (dones > 0.5) & (infos["health"] <= 0)
            if died.any():
                death_rewards = died.float().unsqueeze(-1) * death_penalty

        outputs["obs"] = obs
        outputs["rewards"] = rewards + intrinsic_rewards + death_rewards
        outputs["masks"] = 1.0 - dones
        outputs["successes"] = infos["successes"]
        outputs["episode_lengths"] = infos["episode_lengths"]
        intrinsic_reward_sums.append((intrinsic_rewards + death_rewards).mean().item())


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
        "intrinsic_reward_mean": float(np.mean(intrinsic_reward_sums)) if intrinsic_reward_sums else 0.0,
    }

    return rollout_stats
