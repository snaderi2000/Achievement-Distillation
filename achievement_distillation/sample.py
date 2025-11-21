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

        
        # 1. Prepare Masks (This part remains largely the same)
        valid_step_mask = outputs["masks"].expand(-1, 3) # Fixes Resurrection Bug
        low_mask = (last_vitals < 7) & has_prev_vitals.unsqueeze(-1) # Simplified low_mask (no need to expand prev_available)

        # 2. Identify the Vitals: [Food, Drink, Energy]
        food_low_mask = low_mask[:, 0].float()  # Index 0
        drink_low_mask = low_mask[:, 1].float() # Index 1

        # 3. Identify the ACTION that triggered the delta (since we removed delta)
        # We assume that a change > 0 means the agent performed the appropriate action (Do, Eat Plant, etc.).
        # We use vital_delta > 0 as a proxy for the action being successful.
        vital_delta = (current_vitals - last_vitals).clamp(min=0)
        food_increased = (vital_delta[:, 0] > 0).float()
        drink_increased = (vital_delta[:, 1] > 0).float()

        # 4. CALCULATE BONUS: Multiplies the fixed reward by the conditions (Low AND Increased AND Valid Step)

        # A. Drink Bonus: +2.00 if Drink is low AND Drink increased AND it's a valid step
        drink_bonus = 2.00 * drink_low_mask * drink_increased * valid_step_mask[:, 1]

        # B. Food Bonus: +10.00 if Food is low AND Food increased AND it's a valid step
        food_bonus = 10.00 * food_low_mask * food_increased * valid_step_mask[:, 0]

        # 5. Sum the Bonuses for the total step reward
        bonus = (drink_bonus + food_bonus).sum(dim=-1, keepdim=True)

        # Apply bonus to rewards
        outputs["rewards"] = outputs["rewards"] + bonus

        # DEBUG: Verify shapes and values
        if step % 100 == 0:
            print(f"Step {step} Debug:")
            print(f"  Full Vitals Shape: {all_vitals.shape} (Expect N, 4)")
            print(f"  Tracked Vitals Shape: {current_vitals.shape} (Expect N, 3)")
            print(f"  Sample Vitals (H/F/D): {current_vitals[0].cpu().numpy()}")
        
        if bonus.sum() > 0:
            print(f"!!! BONUS TRIGGERED at Step {step} !!!")

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
