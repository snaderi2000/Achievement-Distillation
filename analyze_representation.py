import argparse
import os
import sys
import random
import yaml
from functools import partial

import numpy as np
import torch as th

# Assuming your environment setup and imports are correct for Crafter and Stable Baselines
from crafter.env import Env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
# Note: VecMonitor might add wrappers that change info dict, adjust if needed
# from stable_baselines3.common.vec_env.vec_monitor import VecMonitor

# Imports from your project structure
from achievement_distillation.model import * # Import necessary model classes (BaseModel, PPOBaselineModel, PPOADModel etc.)
from achievement_distillation.wrapper import VecPyTorch
from achievement_distillation.constant import TASKS # For achievement mapping later

def collect_data(args):
    """Loads a PPO model and collects episode data."""

    print("--- Part 1: Loading Model & Collecting Data ---")

    # --- 1. Setup (Borrow from train.py/eval.py) ---
    # Load config file
    config_path = f"configs/{args.config_name}.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Loaded config from: {config_path}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # Fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    th.cuda.manual_seed_all(args.seed)
    print(f"Set random seed: {args.seed}")

    # CUDA setting
    th.set_num_threads(1)
    cuda = th.cuda.is_available()
    device = th.device("cuda:0" if cuda else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Your PPO Model (Borrow from eval.py) ---
    # Create model instance
    try:
        model_cls = getattr(sys.modules[__name__], config["model_cls"])
        model: BaseModel = model_cls(
            observation_space=Env().observation_space, # Use base Env space for init
            action_space=Env().action_space,           # Use base Env space for init
            **config["model_kwargs"],
        )
        model.to(device)
        print(f"Model class {config['model_cls']} instantiated.")
    except AttributeError:
        print(f"Error: Model class {config['model_cls']} not found. Make sure it's imported.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key in config file: {e}")
        sys.exit(1)

    # Load the saved state dictionary
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        sys.exit(1)
    try:
        state_dict = th.load(args.model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model state_dict from: {args.model_path}")
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        sys.exit(1)

    # Set the model to evaluation mode
    model.eval()
    print("Model set to evaluation mode.")

    # Get reference to the encoder
    try:
        # Adjust 'encoder' if the attribute name is different in your model class
        encoder = model.encoder
        print("Encoder reference obtained.")
    except AttributeError:
        print("Error: Could not find 'encoder' attribute in the model. Check model definition.")
        sys.exit(1)


    # --- 3. Collect Dataset ---
    # Create a single Crafter environment (DummyVecEnv is simplest for one env)
    # Use a different seed for evaluation data collection if desired
    env_eval_seed = args.seed + 1000 # Example: offset seed
    venv = DummyVecEnv([lambda: Env(seed=env_eval_seed)])
    # venv = VecMonitor(venv) # Optional: if you need episode stats like VecMonitor provides
    venv = VecPyTorch(venv, device=device)
    print(f"Crafter environment created with seed: {env_eval_seed}")

    all_episodes = []
    hidsize = config.get("model_kwargs", {}).get("hidsize", 512) # Get hidsize from config

    print(f"Starting data collection for {args.num_episodes} episodes...")
    for i in range(args.num_episodes):
        obs = venv.reset()
        # Reset hidden state for RNNs if your model uses them (check model's act method)
        # Assuming model.act handles initial state if None is passed or has a reset mechanism
        states = th.zeros(1, hidsize).to(device) # Adjust if using VecEnv with >1 env

        # Store data for the current episode
        episode_obs = []
        episode_rewards = []
        episode_dones = []
        episode_achievements = [] # Store the full achievement vector

        done = False
        step_count = 0
        while not done:
            # Get action from the loaded model (no gradients needed)
            with th.no_grad():
                outputs = model.act(obs, states=states) # Pass states if model requires it
                actions = outputs["actions"]
                # Update hidden state if model is recurrent
                if "next_states" in outputs:
                     states = outputs["next_states"]

            # Store current observation *before* stepping
            # Move tensor to CPU and convert to numpy for easier storage if needed,
            # but keeping as tensors might be useful for later processing.
            episode_obs.append(obs.squeeze(0).cpu()) # Remove batch dim, move to CPU

            # Step the environment
            next_obs, rewards, dones, infos = venv.step(actions)

            # Store results for this step
            # Assuming VecPyTorch keeps rewards/dones as tensors
            episode_rewards.append(rewards.item()) # Get scalar value
            episode_dones.append(dones.item())     # Get scalar value

            # Extract achievements from the info dict
            # The structure might depend on wrappers (like VecMonitor)
            # Check what's inside infos[0] during a run
            if isinstance(infos, list) and len(infos) > 0 and 'achievements' in infos[0]:
                 episode_achievements.append(infos[0]['achievements'].copy()) # Store copy
            else:
                 # Fallback or error if achievements aren't found
                 # You might need to adjust this based on your env setup
                 print("Warning: Could not find 'achievements' in info dict.")
                 # Add a placeholder if needed, e.g., np.zeros(len(TASKS))
                 episode_achievements.append(np.zeros(len(TASKS), dtype=int))


            obs = next_obs
            done = dones.item()
            step_count += 1

        # Store the collected data for this episode
        if episode_obs: # Only store if episode had at least one step
            all_episodes.append({
                "observations": th.stack(episode_obs), # Stack list of tensors into one tensor
                "rewards": np.array(episode_rewards),
                "dones": np.array(episode_dones),
                "achievements": np.array(episode_achievements)
            })
        print(f"Episode {i+1}/{args.num_episodes} finished. Length: {step_count} steps.")

    venv.close()
    print(f"\n--- Data Collection Complete ---")

    # --- Verification Print ---
    if all_episodes:
        print(f"Collected {len(all_episodes)} episodes.")
        first_ep = all_episodes[0]
        print("\nData structure for the first episode:")
        for key, value in first_ep.items():
            if isinstance(value, th.Tensor):
                print(f"  '{key}': Tensor(shape={value.shape}, dtype={value.dtype})")
            elif isinstance(value, np.ndarray):
                 print(f"  '{key}': numpy.ndarray(shape={value.shape}, dtype={value.dtype})")
            else:
                 print(f"  '{key}': type={type(value)}")

        # Check obs tensor details (first episode, first step)
        first_obs = first_ep["observations"][0]
        print(f"\nExample Observation (first step of first ep):")
        print(f"  Shape: {first_obs.shape}")
        print(f"  Dtype: {first_obs.dtype}")
        print(f"  Min value: {first_obs.min().item():.2f}")
        print(f"  Max value: {first_obs.max().item():.2f}")
        print(f"  Device: {first_obs.device}") # Should be CPU after .cpu()

    else:
        print("Warning: No episodes were collected.")

    # --- Placeholder for Part 2 ---
    # Part 2: Label States (New Logic - Inspired by Buffer.get_goals)
    # Will go here... takes `all_episodes` as input.
    print("\n--- Placeholder for Part 2: Labeling States ---")

    # For now, just return the collected data
    return all_episodes, encoder, device, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved PPO model (.pt file)")
    parser.add_argument("--config_name", type=str, required=True,
                        help="Name of the config file (e.g., 'ppo_baseline') used for training the model")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for reproducibility")
    parser.add_argument("--num_episodes", type=int, default=10,
                        help="Number of episodes to collect for the dataset")
    args = parser.parse_args()

    # Run data collection
    collected_episodes, loaded_encoder, device, config = collect_data(args)

    # You can add more code here later to process `collected_episodes`
    print("\nScript finished Part 1.")