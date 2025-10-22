import argparse
import os
import sys
import random
import yaml
from functools import partial # Added for potential future use if needed

import numpy as np
import torch as th
import torch.nn.functional as F

# Assuming your environment setup and imports are correct for Crafter and Stable Baselines
from crafter.env import Env
# from crafter.recorder import VideoRecorder # Optional: remove if you don't need video
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

# Imports from your project structure
from achievement_distillation.model import * # Import necessary model classes
from achievement_distillation.wrapper import VecPyTorch
from achievement_distillation.constant import TASKS # For achievement mapping later

def collect_data(args):
    """Loads a PPO model and collects episode data."""

    print("--- Part 1: Loading Model & Collecting Data (Based on eval.py) ---")

    # --- 1. Setup ---
    # Load config file (using exp_name associated with the *training* run)
    config_path = f"configs/{args.exp_name}.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Loaded config from: {config_path}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # Fix random seed for data collection
    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    th.manual_seed(args.eval_seed)
    th.cuda.manual_seed_all(args.eval_seed)
    print(f"Set random seed for evaluation run: {args.eval_seed}")

    # CUDA setting
    th.set_num_threads(1)
    cuda = th.cuda.is_available()
    device = th.device("cuda:0" if cuda else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Your PPO Model ---
    # Define checkpoint path using training details
    run_name = f"{args.exp_name}-{args.timestamp}-s{args.train_seed:02}"
    # Use the specific checkpoint epoch provided, or default (like e250)
    ckpt_filename = f"agent-e{args.ckpt_epoch:03}.pt"
    ckpt_path = os.path.join("./models", run_name, ckpt_filename)

    # Create model instance
    try:
        model_cls = getattr(sys.modules[__name__], config["model_cls"])
        # Use base Env spaces for initialization before loading state_dict
        model: BaseModel = model_cls(
            observation_space=Env().observation_space,
            action_space=Env().action_space,
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
    if not os.path.exists(ckpt_path):
        print(f"Error: Model file not found at {ckpt_path}")
        sys.exit(1)
    try:
        # Load state_dict onto the correct device directly
        state_dict = th.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model state_dict from: {ckpt_path}")
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
    # Create environment
    # Note: No VideoRecorder needed unless you want videos alongside data
    venv = DummyVecEnv([lambda: Env(seed=args.eval_seed)])
    venv = VecPyTorch(venv, device=device)
    print(f"Crafter environment created with seed: {args.eval_seed}")

    all_episodes = []
    hidsize = config.get("model_kwargs", {}).get("hidsize", 512) # Get hidsize

    print(f"Starting data collection for {args.num_episodes} episodes...")
    for i in range(args.num_episodes):
        obs = venv.reset()
        # Reset hidden state for RNNs if needed
        states = th.zeros(1, hidsize).to(device) # Assuming 1 env

        # Store data for the current episode
        episode_obs = []
        episode_rewards = []
        episode_dones = []
        episode_achievements = []

        done = False
        step_count = 0
        while not done:
            # Get action from the loaded model
            with th.no_grad():
                outputs = model.act(obs, states=states)
                actions = outputs["actions"]
                # Update hidden state if model is recurrent
                if "next_states" in outputs:
                     states = outputs["next_states"]
                # Store latent state *before* step if needed, or recompute later
                # latents = outputs["latents"] # Optional: Store if needed later


            # --- Store data *before* stepping ---
            episode_obs.append(obs.squeeze(0).cpu()) # Store observation
            # --- End Store data ---

            # Step the environment
            next_obs, rewards, dones, infos = venv.step(actions)

            # --- Store results *after* stepping ---
            episode_rewards.append(rewards.item())
            episode_dones.append(dones.item())
            if isinstance(infos, list) and len(infos) > 0 and 'achievements' in infos[0]:
                 episode_achievements.append(infos[0]['achievements'].copy())
            else:
                 print("Warning: Could not find 'achievements' in info dict.")
                 episode_achievements.append(np.zeros(len(TASKS), dtype=int))
            # --- End Store results ---


            # Update states (This part from eval.py is for *its* specific state update,
            # which might differ from how states are handled during training/data collection.
            # We already update `states` from model.act if it's recurrent.
            # The original state update logic based on rewards seems specific to eval.py
            # and might not be needed/correct for just collecting data)
            # --- Original eval.py state update (commented out/removed) ---
            # if (rewards > 0.1).any():
            #     with th.no_grad():
            #         next_latents = model.encode(obs) # obs here is actually next_obs
            #     states = next_latents - latents
            #     states = F.normalize(states, dim=-1)
            # --- End Original eval.py state update ---

            obs = next_obs
            done = dones.item()
            step_count += 1

            # Safety break for very long episodes during testing
            # if step_count > 2000:
            #     print("Warning: Episode exceeded 2000 steps, breaking.")
            #     break


        # Store the collected data for this episode
        # Make sure we have data before trying to stack empty lists
        if episode_obs:
            # Need to store the *final* observation as well for completeness
            # The loop stores obs *before* step, so add the last `next_obs`
            final_obs = obs.squeeze(0).cpu()
            episode_obs.append(final_obs) # Add final observation

            # Ensure all lists have consistent length if needed, depends on analysis
            # Pad rewards, dones, achievements if necessary to match obs length + 1 ? Check requirement.
            # Usually we need obs[0...T], actions[0...T-1], rewards[1...T], dones[1...T]
            # Adjust storage based on exactly what Part 2 needs.
            # For now, store obs[0...T], rewards[0...T-1], dones[0...T-1], achievements[0...T-1]

            all_episodes.append({
                "observations": th.stack(episode_obs), # Length T+1
                "rewards": np.array(episode_rewards), # Length T
                "dones": np.array(episode_dones),     # Length T
                "achievements": np.array(episode_achievements) # Length T
            })
            print(f"Episode {i+1}/{args.num_episodes} finished. Length: {step_count} steps.")
        else:
             print(f"Episode {i+1}/{args.num_episodes} finished with 0 steps.")


    venv.close()
    print(f"\n--- Data Collection Complete ---")

    # --- Verification Print ---
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
        print(f"  Observations length: {len(first_ep['observations'])}")
        print(f"  Rewards length: {len(first_ep['rewards'])}")
        print(f"  Dones length: {len(first_ep['dones'])}")
        print(f"  Achievements length: {len(first_ep['achievements'])}")


    else:
        print("Warning: No episodes were collected.")

    # --- Placeholder for Part 2 ---
    # Part 2: Label States (New Logic - Inspired by Buffer.get_goals)
    # Will go here... takes `all_episodes` as input.
    print("\n--- Placeholder for Part 2: Labeling States ---")

    # Return collected data and necessary objects for next steps
    return all_episodes, encoder, device, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Args needed to find the model
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name matching the config file (e.g., ppo_baseline)")
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp of the training run")
    parser.add_argument("--train_seed", type=int, required=True, help="Seed used during training")
    parser.add_argument("--ckpt_epoch", type=int, default=250, help="Epoch number of the checkpoint to load (e.g., 250)")
    # Args for this data collection run
    parser.add_argument("--eval_seed", type=int, default=123, help="Seed for the evaluation environment")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect")
    args = parser.parse_args()

    # Run data collection
    collected_episodes, loaded_encoder, device, config = collect_data(args)

    print("\nScript finished Part 1.")