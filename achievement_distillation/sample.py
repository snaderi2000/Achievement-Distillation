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
        # 
        # --- REWARD SHAPING LOGIC START ---

        # 1. Prepare Masks and Deltas
        current_vitals = infos["vitals"]

        # The mask is 1.0 if the step is NOT a reset (Fixes Resurrection Bug)
        valid_step_mask = outputs["masks"].expand(-1, 3) 
        
        # Check if agent was below the threshold of 7 AND has history
        low_mask = (last_vitals < 7) & has_prev_vitals.unsqueeze(-1)
        
        # Calculate positive increase (Delta > 0)
        vital_delta = (current_vitals - last_vitals).clamp(min=0)

        # 2. Identify Increase and Low Vitals for Food/Drink (Indices 0 and 1)
        
        # Conditions (boolean/float masks)
        food_is_low = low_mask[:, 0].float()
        drink_is_low = low_mask[:, 1].float()
        
        food_increased = (vital_delta[:, 0] > 0).float()
        drink_increased = (vital_delta[:, 1] > 0).float()
        
        # Resurrection Fix Mask for Food/Drink dimensions
        valid_food_step = valid_step_mask[:, 0]
        valid_drink_step = valid_step_mask[:, 1]
        
        # 3. CALCULATE DIFFERENTIAL BONUS

        # A. Drink Bonus: +2.00 if Drink is low AND Drink increased AND it's a valid step
        drink_bonus = 2.00 * drink_is_low * drink_increased * valid_drink_step

        # B. Food Bonus: +10.00 if Food is low AND Food increased AND it's a valid step
        food_bonus = 10.00 * food_is_low * food_increased * valid_food_step

        # 4. Sum the Bonuses for the total step reward
        # The sum is across the two separate bonuses, then reshaped to (N, 1)
        bonus = (drink_bonus + food_bonus).sum(dim=-1, keepdim=True)

        # Apply bonus to rewards
        outputs["rewards"] = outputs["rewards"] + bonus

        # 5. DEBUGGING OUTPUTS (Corrected to avoid NameError and print relevant data)
        if step % 100 == 0:
            print(f"Step {step} Debug:")
            # Tracked Vitals is [Food, Drink, Energy]
            print(f"  Tracked Vitals Shape: {current_vitals.shape} (Expect N, 3)") 
            # Sample Vitals (F/D/E) for the first process
            print(f"  Sample Vitals (F/D/E): {current_vitals[0].cpu().numpy()}") 
            
        if bonus.sum().item() > 0:
            print(f"!!! BONUS TRIGGERED at Step {step} !!!")

        # 6. Update state (Reset logic)
        last_vitals = current_vitals.clone()
        has_prev_vitals[:] = True
        
        # Create the boolean mask for indexing (Fixes previous indexing error)
        done_mask = dones.squeeze(-1).bool()
        has_prev_vitals[done_mask] = False
        last_vitals[done_mask] = 0
        
        # --- REWARD SHAPING LOGIC END --- 

        
       

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
