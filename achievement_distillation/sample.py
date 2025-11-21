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


        # Reward shaping 

        
        # 1. Get separate tensors and concat to [Health, Food, Drink, Energy]
        food_drink_energy = infos["vitals"]      
        health = infos["health"].unsqueeze(1)    
        all_vitals = th.cat([health, food_drink_energy], dim=1) 

        # 2. ISOLATE the 3 we care about: Health(0), Food(1), Drink(2)
        #    We ignore Energy(3).
        current_vitals = all_vitals[:, :3]

        # 3. Calculate logic using matching shapes (N, 3)
        prev_available = has_prev_vitals.unsqueeze(-1).expand(-1, 3)
        
        # Check if stats were low previously
        low_mask = (last_vitals < 4) & prev_available
        
        # Calculate improvement
        vital_delta = (current_vitals - last_vitals).clamp(min=0)

        # 4. Resurrection fix & Bonus Calc
        valid_step_mask = outputs["masks"].expand(-1, 3)
        
        # Apply the valid mask to the delta
        relevant_delta = vital_delta * valid_step_mask
        
        # Calculate bonus
        bonus = 0.75 * (relevant_delta * low_mask.float()).sum(dim=-1, keepdim=True)

        # Apply bonus to rewards
        outputs["rewards"] = outputs["rewards"] + bonus

        # DEBUG: Verify shapes and values
        if step % 100 == 0:
            print(f"Step {step} Debug:")
            print(f"  Full Vitals Shape: {all_vitals.shape} (Expect N, 4)")
            print(f"  Tracked Vitals Shape: {current_vitals.shape} (Expect N, 3)")
            print(f"  Sample Vitals (H/F/D): {current_vitals[0].cpu().numpy()}")
            if bonus.sum() > 0:
                print(f"  Bonus Triggered! Value: {bonus[0].item()}")

        # Update state
        last_vitals = current_vitals.clone()
        has_prev_vitals[:] = True
        
        # Reset state for done agents
        done_mask = dones.squeeze(-1).bool()
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
