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

def collect_data(args):
    """Loads a PPO model and collects episode data, matching eval.py setup."""

    print("--- Part 1: Loading Model & Collecting Data (Corrected based on eval.py) ---")

    # --- 1. Setup ---
    config_path = f"configs/{args.exp_name}.yaml" # Use exp_name to load config
    try:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Loaded config from: {config_path}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    random.seed(args.eval_seed)
    np.random.seed(args.eval_seed)
    th.manual_seed(args.eval_seed)
    th.cuda.manual_seed_all(args.eval_seed)
    print(f"Set random seed for evaluation run: {args.eval_seed}")

    th.set_num_threads(1)
    cuda = th.cuda.is_available()
    device = th.device("cuda:0" if cuda else "cpu")
    print(f"Using device: {device}")

    # --- 2. Load Your PPO Model ---
    # Define checkpoint path using training details
    run_name = f"{args.exp_name}-{args.timestamp}-s{args.train_seed:02}"
    ckpt_filename = f"agent-e{args.ckpt_epoch:03}.pt"
    ckpt_path = os.path.join("./models", run_name, ckpt_filename)

    # --- Create environment *before* model instantiation (like eval.py) ---
    # Use the same seed logic as eval.py if needed, or stick to eval_seed
    venv = DummyVecEnv([lambda: Env(seed=args.eval_seed)])
    # Add wrappers exactly as needed for the model (matching training setup if possible)
    # VecPyTorch is crucial as it likely changes observation space
    venv = VecPyTorch(venv, device=device)
    print(f"Crafter environment created and wrapped with seed: {args.eval_seed}")

    # --- Create model instance *using wrapped venv spaces* (like eval.py) ---
    try:
        model_cls = getattr(sys.modules[__name__], config["model_cls"])
        # CRITICAL FIX: Use venv spaces, not Env() spaces
        model: BaseModel = model_cls(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            **config["model_kwargs"],
        )
        model.to(device)
        print(f"Model class {config['model_cls']} instantiated correctly using wrapped env spaces.")
        # Optional: Print the exact observation space used
        # print(f"  Obs Space: {venv.observation_space}")
    except AttributeError:
        print(f"Error: Model class {config['model_cls']} not found. Make sure it's imported.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Missing key in config file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during model instantiation: {e}")
        sys.exit(1)


    # --- Load the saved state dictionary (like eval.py) ---
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
        # Print model structure vs state_dict keys if helpful
        # print("Model state_dict keys:", model.state_dict().keys())
        # print("Checkpoint state_dict keys:", state_dict.keys())
        sys.exit(1)

    model.eval()
    print("Model set to evaluation mode.")

    try:
        encoder = model.encoder
        print("Encoder reference obtained.")
    except AttributeError:
        print("Error: Could not find 'encoder' attribute. Check model definition.")
        sys.exit(1)

    # --- 3. Collect Dataset ---
    all_episodes = []
    # Ensure hidsize retrieval is robust
    hidsize = config.get("model_kwargs", {}).get("hidsize")
    if hidsize is None:
         print("Warning: 'hidsize' not found in config, using default 512. Check your config.")
         hidsize = 512 # Or determine default dynamically if possible

    print(f"Starting data collection for {args.num_episodes} episodes...")
    for i in range(args.num_episodes):
        obs = venv.reset()
        states = th.zeros(1, hidsize).to(device) # Assuming 1 env

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
                 print("Warning: Could not find 'achievements' in info dict.")
                 episode_achievements.append(np.zeros(len(TASKS), dtype=int))

            obs = next_obs
            done = dones.item()
            step_count += 1

        if episode_obs:
            final_obs = obs.squeeze(0).cpu()
            episode_obs.append(final_obs)

            all_episodes.append({
                "observations": th.stack(episode_obs),
                "rewards": np.array(episode_rewards),
                "dones": np.array(episode_dones),
                "achievements": np.array(episode_achievements)
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


    # --- Placeholder for Part 2 ---
    print("\n--- Placeholder for Part 2: Labeling States ---")

    return all_episodes, encoder, device, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Args needed to find the model (Match eval.py structure where possible)
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name matching config")
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp of training run")
    parser.add_argument("--train_seed", type=int, required=True, help="Seed used during training") # Changed from default=0 to required
    parser.add_argument("--ckpt_epoch", type=int, default=250, help="Epoch of checkpoint")
    # Args for this data collection run
    parser.add_argument("--eval_seed", type=int, default=123, help="Seed for evaluation env")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect")
    args = parser.parse_args()

    # Run data collection
    collected_episodes, loaded_encoder, device, config = collect_data(args)

    print("\nScript finished Part 1.")