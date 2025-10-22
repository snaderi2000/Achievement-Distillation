import argparse
import os
import sys
import random
import yaml
from functools import partial

import numpy as np
import torch as th
import torch.nn.functional as F

# Assuming your environment setup and imports are correct for Crafter and Stable Baselines
from crafter.env import Env
# from crafter.recorder import VideoRecorder # Optional
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

# Imports from your project structure
from achievement_distillation.model import * # Import necessary model classes
from achievement_distillation.wrapper import VecPyTorch
from achievement_distillation.constant import TASKS

# --- Part 1: Data Collection Function ---
def collect_data(args, config):
    """Loads a PPO model and collects episode data, matching eval.py setup."""

    print("--- Part 1: Loading Model & Collecting Data ---")

    # --- Setup ---
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    th.manual_seed(args.eval_seed)
    th.cuda.manual_seed_all(args.eval_seed)
    print(f"Set random seed for evaluation run: {args.eval_seed}")

    th.set_num_threads(1)
    cuda = th.cuda.is_available()
    device = th.device("cuda:0" if cuda else "cpu")
    print(f"Using device: {device}")

    # --- Load Your PPO Model ---
    run_name = f"{args.exp_name}-{args.timestamp}-s{args.train_seed:02}"
    ckpt_filename = f"agent-e{args.ckpt_epoch:03}.pt"
    ckpt_path = os.path.join("./models", run_name, ckpt_filename)

    venv = DummyVecEnv([lambda: Env(seed=args.eval_seed)])
    venv = VecPyTorch(venv, device=device)
    print(f"Crafter environment created and wrapped with seed: {args.eval_seed}")

    try:
        model_cls = getattr(sys.modules[__name__], config["model_cls"])
        model: BaseModel = model_cls(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            **config["model_kwargs"],
        )
        model.to(device)
        print(f"Model class {config['model_cls']} instantiated correctly.")
    except Exception as e:
        print(f"Error during model instantiation: {e}")
        sys.exit(1)

    if not os.path.exists(ckpt_path):
        print(f"Error: Model file not found at {ckpt_path}")
        sys.exit(1)
    try:
        state_dict = th.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model state_dict from: {ckpt_path}")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        sys.exit(1)

    model.eval()
    print("Model set to evaluation mode.")

    # --- Collect Dataset ---
    all_episodes = []
    hidsize = config.get("model_kwargs", {}).get("hidsize")
    if hidsize is None:
         print("Warning: 'hidsize' not found in config, using default 512.")
         hidsize = 512

    print(f"Starting data collection for {args.num_episodes} episodes...")
    for i in range(args.num_episodes):
        obs = venv.reset()
        states = th.zeros(1, hidsize).to(device)

        episode_obs = []
        episode_rewards = []
        episode_dones = []
        episode_achievements = []

        done = False
        step_count = 0
        while not done:
            with th.no_grad():
                outputs = model.act(obs, states=states)
                actions = outputs["actions"]
                if "next_states" in outputs:
                     states = outputs["next_states"]

            episode_obs.append(obs.squeeze(0).cpu())
            next_obs, rewards, dones, infos = venv.step(actions)

            episode_rewards.append(rewards.item())
            episode_dones.append(dones.item())
            if isinstance(infos, list) and len(infos) > 0 and 'achievements' in infos[0]:
                 episode_achievements.append(infos[0]['achievements'].copy())
            else:
                 # Ensure a placeholder of the correct shape if achievements are missing
                 episode_achievements.append(np.zeros(len(TASKS), dtype=int))

            obs = next_obs
            done = dones.item()
            step_count += 1

        if episode_obs:
            final_obs = obs.squeeze(0).cpu()
            episode_obs.append(final_obs)

            # Ensure achievements array has the correct length if episode ended early
            # Pad if necessary, although usually it should align with rewards/dones length
            ach_array = np.array(episode_achievements)
            if len(ach_array) < len(episode_rewards):
                 padding = np.zeros((len(episode_rewards) - len(ach_array), len(TASKS)), dtype=int)
                 ach_array = np.vstack((ach_array, padding))


            all_episodes.append({
                "observations": th.stack(episode_obs), # Length T+1
                "rewards": np.array(episode_rewards), # Length T
                "dones": np.array(episode_dones),     # Length T
                "achievements": ach_array # Length T, shape (T, 22)
            })
            print(f"Episode {i+1}/{args.num_episodes} finished. Length: {step_count} steps.")
        else:
             print(f"Episode {i+1}/{args.num_episodes} finished with 0 steps.")


    venv.close()
    print(f"\n--- Data Collection Complete ---")

    # Verification Print
    if all_episodes:
        print(f"Collected {len(all_episodes)} episodes.")
        first_ep = all_episodes[0]
        print("\nData structure for the first episode:")
        for key, value in first_ep.items():
             if isinstance(value, th.Tensor):
                 print(f"  '{key}': Tensor(shape={value.shape}, dtype={value.dtype}, device={value.device})")
             elif isinstance(value, np.ndarray):
                  print(f"  '{key}': numpy.ndarray(shape={value.shape}, dtype={value.dtype})")
             else:
                  print(f"  '{key}': type={type(value)}")
    else:
        print("Warning: No episodes were collected.")

    # --- Verify Achievements ---
    print("\n--- Verifying Achievements in Collected Data ---")
    total_achievements_unlocked = 0
    for i, episode in enumerate(all_episodes):
        print(f"Episode {i+1}:")
        episode_rewards = episode['rewards']
        print(f"  - Total reward: {np.sum(episode_rewards):.2f}")
        print(f"  - Number of positive reward steps: {np.sum(episode_rewards > 0)}")

        if 'achievements' in episode and episode['achievements'].shape[0] > 0:
            initial_ach = np.zeros((1, episode['achievements'].shape[1]))
            # diff tells us when an achievement is newly unlocked
            diff = np.diff(np.vstack([initial_ach, episode['achievements']]), axis=0)
            num_unlocked = np.sum(diff > 0)
            total_achievements_unlocked += num_unlocked
            print(f"  - Achievements unlocked in this episode: {num_unlocked}")
        else:
            print("  - 'achievements' key not found or empty.")

    if total_achievements_unlocked == 0:
        print("\nWARNING: No achievements were unlocked in any of the collected episodes.")
        print("The labeling process will result in all states having the default label (-1).")
        print("Consider using a better agent or collecting more episodes.")
    else:
        print(f"\nSUCCESS: A total of {total_achievements_unlocked} achievements were unlocked across all episodes.")

    return all_episodes, model, device # Return model, not just encoder

# --- Part 2: State Labeling Function ---
# Mapping from achievement name string to index (0-21)
TASK_TO_INDEX = {task: i for i, task in enumerate(TASKS)}

def label_states_with_next_achievement(all_episodes: list) -> list:
    """Labels each state in the collected episodes with the next achievement index."""
    print("\n--- Part 2: Labeling States ---")
    labeled_data = [] # Will store tuples of (observation_tensor, next_achievement_label)

    # Process each episode
    for ep_idx, episode in enumerate(all_episodes):
        print(f"\n[Debug] Processing Episode {ep_idx+1}...")
        observations = episode["observations"] # Shape (T+1, C, H, W)
        rewards = episode["rewards"]           # Shape (T,)
        achievements_over_time = episode["achievements"] # Shape (T, 22)

        episode_len = len(rewards) # Number of steps T
        if episode_len == 0:
            continue

        goal_steps_dict = {} # Map step_index -> achievement_index

        # Pad achievements_over_time with initial state (all zeros)
        initial_achievements = np.zeros((1, len(TASKS)), dtype=achievements_over_time.dtype)
        full_achievements = np.vstack((initial_achievements, achievements_over_time)) # Shape (T+1, 22)

        # Find where *any* achievement status changes from 0 to 1 (or just changes)
        diff = np.diff(full_achievements, axis=0) # Shape (T, 22), shows changes between steps
        print(f"[Debug] diff.sum() for episode {ep_idx+1}: {diff.sum()}")

        # Filter to find *newly* unlocked achievements (where diff is +1)
        newly_unlocked_indices = np.where(diff == 1) # Tuple: (array of rows, array of cols)
        print(f"[Debug] newly_unlocked_indices for episode {ep_idx+1}: {newly_unlocked_indices}")
        goal_steps_indices = newly_unlocked_indices[0] # Step index t (0 to T-1) where change occurred
        unlocked_achievement_indices = newly_unlocked_indices[1] # Which achievement (0-21) changed

        for step_idx, ach_idx in zip(goal_steps_indices, unlocked_achievement_indices):
            actual_step_of_unlock = step_idx + 1 # Index from 1 to T
            goal_steps_dict[actual_step_of_unlock] = ach_idx

        print(f"[Debug] goal_steps_dict for episode {ep_idx+1}: {goal_steps_dict}")
        sorted_goal_steps = sorted(goal_steps_dict.keys())

        # Label each state s_0 to s_T
        for t in range(episode_len + 1):
            next_achievement_label = -1 # Default: No future achievement

            found_next = False
            for goal_step in sorted_goal_steps:
                # Goal step is index 1 to T, representing unlock *after* step t-1
                # We need goal_step > t (current state index 0 to T)
                if goal_step > t:
                    next_achievement_label = goal_steps_dict[goal_step]
                    found_next = True
                    break

            if ep_idx == 0 and len(sorted_goal_steps) > 0: # Only print for first episode if it has achievements
                print(f"[Debug] t={t}, sorted_goal_steps={sorted_goal_steps}, next_achievement_label={next_achievement_label}")

            labeled_data.append((observations[t], next_achievement_label))

    print(f"--- Labeling Complete ---")
    print(f"Total labeled states: {len(labeled_data)}")

     # Verification Print
    if labeled_data:
        print("\nExample labeled data points (Observation Tensor, Next Achievement Label):")
        label_counts = {}
        for i in range(min(10, len(labeled_data))):
             obs_tensor, label = labeled_data[i]
             print(f"  Data point {i}: Obs shape={obs_tensor.shape}, Label={label}")
             label_counts[label] = label_counts.get(label, 0) + 1 # Count labels in sample

        print(f"\nLabel counts in full dataset:")
        full_label_counts = {}
        for _, label in labeled_data:
            full_label_counts[label] = full_label_counts.get(label, 0) + 1
        # Sort by label for readability
        sorted_counts = dict(sorted(full_label_counts.items()))
        print(f"  {sorted_counts}")

    return labeled_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name matching config")
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp of training run")
    parser.add_argument("--train_seed", type=int, required=True, help="Seed used during training")
    parser.add_argument("--ckpt_epoch", type=int, default=250, help="Epoch of checkpoint")
    parser.add_argument("--eval_seed", type=int, default=123, help="Seed for evaluation env")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect")
    args = parser.parse_args()

    # --- Load Config --- (Load once at the start)
    config_path = f"configs/{args.exp_name}.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Loaded config from: {config_path}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # --- Run Part 1: Data Collection ---
    collected_episodes, loaded_model, device = collect_data(args, config) # Pass config

    # --- Run Part 2: Labeling States ---
    labeled_state_data = label_states_with_next_achievement(collected_episodes)

    # --- Placeholder for Part 3 ---
    print("\n--- Placeholder for Part 3: Create Train/Test Splits ---")
    # Part 3 will take `labeled_state_data` and split it.
    # Part 4 will take the splits and the `loaded_model` (for its .encode method)
    # Part 5 will train the classifier.
    # Part 6 will evaluate the classifier.

    print("\nScript finished Parts 1 & 2.")