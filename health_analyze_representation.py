import argparse
import os
import sys
import random
import yaml
from functools import partial

import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Assuming your environment setup and imports are correct for Crafter and Stable Baselines
from crafter.env import Env
# from crafter.recorder import VideoRecorder # Optional
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

# Imports from your project structure
from achievement_distillation.model import * # Import necessary model classes
from achievement_distillation.wrapper import VecPyTorch
from achievement_distillation.constant import TASKS

# --- Part 1: Data Collection Function ---
def collect_data(args, use_expert=False):
    """Loads a specified model and collects episode data."""

    if use_expert:
        exp_name = args.expert_exp_name
        timestamp = args.expert_timestamp
        train_seed = args.expert_train_seed
        ckpt_epoch = args.expert_ckpt_epoch
        config_path = f"configs/{exp_name}.yaml"
        print("--- Part 1: Loading EXPERT Model & Collecting Data ---")
    else:
        exp_name = args.exp_name
        timestamp = args.timestamp
        train_seed = args.train_seed
        ckpt_epoch = args.ckpt_epoch
        config_path = f"configs/{exp_name}.yaml"
        print("--- Part 1: Loading Model & Collecting Data (Self-Evaluation) ---")

    # --- Load Config for the data collection model ---
    try:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Loaded config from: {config_path}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

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
    run_name = f"{exp_name}-{timestamp}-s{train_seed:02}"
    ckpt_filename = f"agent-e{ckpt_epoch:03}.pt"
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
        episode_health = []

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
            
            episode_health.append(infos['health'].item())
            episode_rewards.append(rewards.item())
            episode_dones.append(dones.item())
            # Corrected logic for extracting achievements
            if 'achievements' in infos:
                 # The wrapper returns a tensor, access the data for the first (and only) env
                 ach_tensor = infos['achievements'][0].cpu().numpy()
                 episode_achievements.append(ach_tensor.copy())
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
                "achievements": ach_array, # Length T, shape (T, 22)
                "health": np.array(episode_health), # Length T
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

    return all_episodes, device

# --- Part 2: State Labeling Function ---
def label_states_by_health_change(all_episodes: list) -> list:
    """Labels each state based on whether health decreases in the next state."""
    print("\n--- Part 2: Labeling States by Health Change ---")
    labeled_data = []  # Will store tuples of (observation_tensor, label)

    for ep_idx, episode in enumerate(all_episodes):
        observations = episode["observations"]  # Shape (T+1, C, H, W)
        health_over_time = episode["health"]      # Shape (T,)

        if len(health_over_time) == 0:
            continue

        # Prepend initial health (Crafter starts with 9)
        # health_over_time[t] is health at s_{t+1}
        # So full_health[t] is health at s_t
        initial_health = np.array([9.0])
        full_health = np.concatenate((initial_health, health_over_time)) # Shape (T+1,)

        # Label each state s_t based on health change at s_{t+1}
        for t in range(len(health_over_time)): # For s_0 to s_{T-1}
            current_health = full_health[t]
            next_health = full_health[t+1]

            # Label is 1 if health decreases, 0 otherwise
            label = 1 if next_health < current_health else 0
            labeled_data.append((observations[t], label))

    print(f"--- Labeling Complete ---")
    print(f"Total labeled states: {len(labeled_data)}")

    # Verification Print
    if labeled_data:
        labels = [label for _, label in labeled_data]
        label_counts = Counter(labels)
        print(f"\nLabel distribution:")
        print(f"  - 0 (No Decrease): {label_counts.get(0, 0)} states")
        print(f"  - 1 (Decrease):    {label_counts.get(1, 0)} states")

    return labeled_data


# --- Part 3: Create Train/Test Splits ---
def create_train_test_splits(labeled_data, train_size=50000, test_size=10000, random_state=22):
    """Subsamples the labeled data into fixed-size training and testing sets."""
    print("\n--- Part 3: Creating Train/Test Splits ---")

    if len(labeled_data) < train_size + test_size:
        print(f"Warning: Not enough data ({len(labeled_data)}) to create a train/test split of size {train_size}/{test_size}.")
        print("This may be due to a lack of health-decreasing events.")
        print("Exiting.")
        return None, None, None, None

    # Subsample the data as per the specified sizes
    random.seed(random_state)
    random.shuffle(labeled_data)
    
    train_samples = labeled_data[:train_size]
    test_samples = labeled_data[train_size : train_size + test_size]
    
    X_train_list, y_train_list = zip(*train_samples)
    X_test_list, y_test_list = zip(*test_samples)

    X_train = th.stack(X_train_list)
    y_train = th.tensor(y_train_list, dtype=th.long)
    X_test = th.stack(X_test_list)
    y_test = th.tensor(y_test_list, dtype=th.long)

    print(f"Data split into training and testing sets:")
    print(f"  - Training set size: {len(X_train)}")
    print(f"  - Testing set size:  {len(X_test)}")

    # Report label distribution in splits
    train_counts = Counter(y_train.numpy())
    test_counts = Counter(y_test.numpy())
    print("\nLabel distribution in splits:")
    print(f"  Training Set:")
    print(f"    - 0 (No Decrease): {train_counts.get(0, 0)}")
    print(f"    - 1 (Decrease):    {train_counts.get(1, 0)}")
    print(f"  Test Set:")
    print(f"    - 0 (No Decrease): {test_counts.get(0, 0)}")
    print(f"    - 1 (Decrease):    {test_counts.get(1, 0)}")
    
    return X_train, X_test, y_train, y_test

# --- Part 4: Extract Latent Representations ---
def extract_latent_vectors(model, data_loader, device):
    """Extracts latent vectors from the model's encoder for a given dataset."""
    model.eval()
    latent_vectors = []
    with th.no_grad():
        for observations_batch in data_loader:
            observations_batch = observations_batch[0].to(device)
            latents = model.encode(observations_batch)
            latent_vectors.append(latents.cpu())
    
    return th.cat(latent_vectors, dim=0)

# --- Part 5: Train and Evaluate Classifier ---
def train_and_evaluate_classifier(X_train_latents, y_train, X_test_latents, y_test, num_classes, device):
    """Trains and evaluates a linear classifier on the latent vectors."""
    print("\n--- Part 5: Training and Evaluating Classifier ---")
    
    # --- Setup ---
    input_dim = X_train_latents.shape[1]
    classifier = nn.Linear(input_dim, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(X_train_latents, y_train)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    test_dataset = TensorDataset(X_test_latents, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    # --- Training Loop ---
    num_epochs = 500 # Changed from 25 to 500 to match paper
    print(f"Training classifier for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        classifier.train()
        total_loss = 0
        for latents_batch, labels_batch in train_loader:
            latents_batch, labels_batch = latents_batch.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = classifier(latents_batch)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 50 == 0: # Log every 50 epochs
            print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")

    # --- Evaluation ---
    classifier.eval()
    all_preds = []
    all_labels = []
    all_confidences = []
    with th.no_grad():
        for latents_batch, labels_batch in test_loader:
            latents_batch, labels_batch = latents_batch.to(device), labels_batch.to(device)
            outputs = classifier(latents_batch)
            
            # Calculate confidence (softmax probability of the ground-truth class)
            probs = F.softmax(outputs, dim=1)
            gt_confidences = probs[range(len(labels_batch)), labels_batch].cpu().numpy()
            all_confidences.extend(gt_confidences)

            _, predicted = th.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Final Test Accuracy: {accuracy:.2f}%")

    # --- Detailed Report ---
    print("\n--- Classification Report ---")
    target_names = ['No-Decrease', 'Decrease']
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # --- Confusion Matrix ---
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print("Columns: Predicted, Rows: Actual")
    print(cm)


def load_analysis_model(exp_name, timestamp, train_seed, ckpt_epoch, device):
    """Loads the model whose representations will be analyzed."""
    print("\n--- Loading ANALYSIS Model for Representation Extraction ---")

    config_path = f"configs/{exp_name}.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        print(f"Loaded config from: {config_path}")
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    # A temporary venv is needed to get observation/action space for model instantiation
    temp_venv = VecPyTorch(DummyVecEnv([lambda: Env()]), device=device)
    
    try:
        model_cls = getattr(sys.modules[__name__], config["model_cls"])
        model: BaseModel = model_cls(
            observation_space=temp_venv.observation_space,
            action_space=temp_venv.action_space,
            **config["model_kwargs"],
        )
        model.to(device)
        print(f"Model class {config['model_cls']} instantiated for {exp_name}.")
    except Exception as e:
        print(f"Error during model instantiation for {exp_name}: {e}")
        sys.exit(1)
    finally:
        temp_venv.close()

    run_name = f"{exp_name}-{timestamp}-s{train_seed:02}"
    ckpt_filename = f"agent-e{ckpt_epoch:03}.pt"
    ckpt_path = os.path.join("./models", run_name, ckpt_filename)

    if not os.path.exists(ckpt_path):
        print(f"Error: Model file not found at {ckpt_path}")
        sys.exit(1)
    try:
        state_dict = th.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded ANALYSIS model state_dict from: {ckpt_path}")
    except Exception as e:
        print(f"Error loading state_dict for {exp_name}: {e}")
        sys.exit(1)

    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Analysis model arguments (the one whose representations we test)
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name of the model to ANALYZE.")
    parser.add_argument("--timestamp", type=str, required=True, help="Timestamp of the model to ANALYZE.")
    parser.add_argument("--train_seed", type=int, required=True, help="Seed of the model to ANALYZE.")
    parser.add_argument("--ckpt_epoch", type=int, default=250, help="Epoch of the model to ANALYZE.")

    # Expert model arguments (optional, for data collection)
    parser.add_argument("--expert_exp_name", type=str, default=None, help="If provided, use this model as the expert for data collection.")
    parser.add_argument("--expert_timestamp", type=str, default=None, help="Timestamp of the expert model.")
    parser.add_argument("--expert_train_seed", type=int, default=None, help="Seed of the expert model.")
    parser.add_argument("--expert_ckpt_epoch", type=int, default=250, help="Epoch of the expert model.")
    
    # General arguments
    parser.add_argument("--eval_seed", type=int, default=123, help="Seed for evaluation env")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to collect")
    args = parser.parse_args()

    # --- Run Part 1: Data Collection ---
    use_expert = bool(args.expert_exp_name)
    if use_expert:
        # Ensure all expert args are provided if expert_exp_name is set
        if not all([args.expert_timestamp, args.expert_train_seed is not None]):
            print("Error: If --expert_exp_name is used, you must provide --expert_timestamp and --expert_train_seed.")
            sys.exit(1)
    
    collected_episodes, device = collect_data(args, use_expert=use_expert)

    # --- Load the ANALYSIS Model ---
    analysis_model = load_analysis_model(
        args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device
    )

    # --- Run Part 2: Labeling States ---
    labeled_state_data = label_states_by_health_change(collected_episodes)

    # --- Run Part 3: Create Train/Test Splits ---
    X_train, X_test, y_train, y_test = create_train_test_splits(labeled_state_data)
    
    if X_train is not None:
        # --- Run Part 4: Extract Latent Representations ---
        print("\n--- Part 4: Extracting Latent Representations ---")
        # Create DataLoaders for extraction
        train_obs_dataset = TensorDataset(X_train)
        train_obs_loader = DataLoader(train_obs_dataset, batch_size=256)
        
        test_obs_dataset = TensorDataset(X_test)
        test_obs_loader = DataLoader(test_obs_dataset, batch_size=256)

        print("Extracting latents for training set...")
        X_train_latents = extract_latent_vectors(analysis_model, train_obs_loader, device)
        print("Extracting latents for testing set...")
        X_test_latents = extract_latent_vectors(analysis_model, test_obs_loader, device)

        print(f"Latent vector shapes:")
        print(f"  - Training latents: {X_train_latents.shape}")
        print(f"  - Testing latents:  {X_test_latents.shape}")

        # --- Run Part 5: Train and Evaluate Classifier ---
        num_classes = 2 # No-Decrease vs Decrease
        train_and_evaluate_classifier(
            X_train_latents, y_train, X_test_latents, y_test, num_classes, device
        )
    
    print("\nScript finished.")