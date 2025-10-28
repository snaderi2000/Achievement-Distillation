import argparse
import os
import sys
import random
import yaml
from functools import partial

import numpy as np
import pandas as pd
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

def extract_ach_vec_from_infos(infos, TASKS):
    # Supports: (a) list[dict] from VecEnv, (b) dict of tensors,
    # (c) achievements as dict of {name: int} or np array/tensor.
    if isinstance(infos, (list, tuple)):
        info0 = infos[0] if len(infos) > 0 else {}
    elif isinstance(infos, dict):
        # VecPyTorch can collate infos into dict of tensors
        if 'achievements' in infos:
            ach = infos['achievements']
            # ach could be tensor[N,22] or list/array; take first env
            if hasattr(ach, 'shape'):
                a0 = ach[0]
                return a0.detach().cpu().numpy() if hasattr(a0, 'detach') else np.asarray(a0)
            elif isinstance(ach, (list, tuple)):
                return np.asarray(ach[0])
        info0 = {k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in infos.items()}
    else:
        info0 = {}

    ach = info0.get('achievements', None)
    if ach is None:
        return None
    # If it's already a vector
    if isinstance(ach, (np.ndarray, list)) and (len(ach) == len(TASKS)):
        return np.asarray(ach, dtype=int)

    # If it's a dict of {task_name: 0/1}
    if isinstance(ach, dict):
        vec = np.zeros(len(TASKS), dtype=int)
        for i, t in enumerate(TASKS):
            vec[i] = int(bool(ach.get(t, 0)))
        return vec

    # If it's a tensor
    if hasattr(ach, 'shape'):
        arr = ach.detach().cpu().numpy() if hasattr(ach, 'detach') else np.asarray(ach)
        if arr.ndim == 1 and arr.shape[0] == len(TASKS):
            return arr.astype(int)
        if arr.ndim == 2:
            return arr[0].astype(int)
    return None

# --- Part 1: Data Collection Function ---
def collect_data(args, device, use_expert=False, model_config=None):
    """Loads a specified model and collects episode data."""

    if use_expert:
        exp_name = args.expert_exp_name
        timestamp = args.expert_timestamp
        train_seed = args.expert_train_seed
        ckpt_epoch = args.expert_ckpt_epoch
        # Correctly use the expert's config file
        config_path = f"configs/{args.expert_exp_name}.yaml"
        print("--- Part 1: Loading EXPERT Model & Collecting Data ---")
    else:
        exp_name = args.exp_name
        timestamp = args.timestamp
        train_seed = args.train_seed
        ckpt_epoch = args.ckpt_epoch
        # Use the analysis model's config for self-evaluation
        config_path = f"configs/{args.exp_name}.yaml"
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
        try:
            venv.env_method('seed', args.eval_seed + i)
        except Exception:
            pass
        obs = venv.reset()
        
        # Robustly handle recurrent states
        states = None
        if getattr(model, "use_memory", False): # Check if model is recurrent
            hidsize = config.get("model_kwargs", {}).get("hidsize", 512)
            states = th.zeros(1, hidsize, device=device)

        episode_obs = []
        episode_rewards = []
        episode_dones = []
        episode_achievements = []

        done = False
        step_count = 0
        while not done:
            with th.no_grad():
                outputs = model.act(obs, states=states) if states is not None else model.act(obs)
                actions = outputs["actions"]
                if states is not None and "next_states" in outputs:
                     states = outputs["next_states"]

            # Store observation as float32, but do not normalize here. The model's encoder handles it.
            episode_obs.append(obs.squeeze(0).cpu().to(dtype=th.float32))
            next_obs, rewards, dones, infos = venv.step(actions)

            # Robust achievement extraction
            ach_vec = extract_ach_vec_from_infos(infos, TASKS)
            if ach_vec is None:
                # Fallback if achievements are missing entirely for a step
                ach_vec = np.zeros(len(TASKS), dtype=int)
            episode_achievements.append(ach_vec)

            episode_rewards.append(rewards.item())
            episode_dones.append(dones.item())

            obs = next_obs
            done = dones.item()
            step_count += 1

        if episode_obs:
            final_obs = obs.squeeze(0).cpu().to(dtype=th.float32)
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

    return all_episodes

# --- Part 2: State Labeling Function ---
# Mapping from achievement name string to index (0-21)
TASK_TO_INDEX = {task: i for i, task in enumerate(TASKS)}

def label_states_with_next_achievement(all_episodes: list) -> list:
    """Labels each state in the collected episodes with the next achievement index."""
    print("\n--- Part 2: Labeling States ---")
    labeled_data = [] # Will store tuples of (observation_tensor, next_achievement_label)

    # Process each episode
    for ep_idx, episode in enumerate(all_episodes):
        observations = episode["observations"] # Shape (T+1, C, H, W)
        rewards = episode["rewards"]           # Shape (T,)
        achievements_over_time = episode["achievements"] # Shape (T, 22)

        episode_len = len(rewards) # Number of steps T
        if episode_len == 0:
            continue

        goal_steps_dict = {} # step -> list of ach_idx

        # Pad achievements_over_time with initial state (all zeros)
        initial_achievements = np.zeros((1, len(TASKS)), dtype=achievements_over_time.dtype)
        full_achievements = np.vstack((initial_achievements, achievements_over_time)) # Shape (T+1, 22)

        # Find where *any* achievement status changes from 0 to 1 (or just changes)
        diff = np.diff(full_achievements, axis=0) # Shape (T, 22), shows changes between steps

        # Filter to find *newly* unlocked achievements (where diff is +1)
        newly_unlocked_indices = np.where(diff == 1) # Tuple: (array of rows, array of cols)
        goal_steps_indices = newly_unlocked_indices[0] # Step index t (0 to T-1) where change occurred
        unlocked_achievement_indices = newly_unlocked_indices[1] # Which achievement (0-21) changed

        for step_idx, ach_idx in zip(goal_steps_indices, unlocked_achievement_indices):
            actual_step_of_unlock = step_idx + 1 # Index from 1 to T
            goal_steps_dict.setdefault(actual_step_of_unlock, []).append(int(ach_idx))

        sorted_goal_steps = sorted(goal_steps_dict.keys())

        # Label each state s_0 to s_T
        for t in range(episode_len + 1):
            next_achievement_label = -1 # Default: No future achievement

            for goal_step in sorted_goal_steps:
                # Goal step is index 1 to T, representing unlock *after* step t-1
                # We need goal_step > t (current state index 0 to T)
                if goal_step > t:
                    # pick the first achievement (lowest idx) at that earliest goal_step
                    next_achievement_label = sorted(goal_steps_dict[goal_step])[0]
                    break

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


# --- Part 3: Create Train/Test Splits ---
def create_train_test_splits(labeled_data, train_size=50000, test_size=10000, random_state=22):
    """Subsamples the labeled data into fixed-size training and testing sets."""
    print("\n--- Part 3: Creating Train/Test Splits ---")

    # Filter out data with label -1 (no future achievement)
    filtered_data = [item for item in labeled_data if item[1] != -1]
    
    if len(filtered_data) < train_size + test_size:
        print(f"Warning: Not enough data ({len(filtered_data)}) to create a train/test split of size {train_size}/{test_size}.")
        print("Using all available data with an 80/20 split instead.")
        # Fallback to percentage-based split if not enough data
        observations, labels = zip(*filtered_data)
        observations_tensor = th.stack(observations)
        labels_tensor = th.tensor(labels, dtype=th.long)
        
        # Check if stratification is possible
        label_counts = Counter(labels_tensor.numpy())
        min_class_count = min(label_counts.values())
        
        if min_class_count >= 2:
            print(f"Using stratified split (min class count: {min_class_count})")
            X_train, X_test, y_train, y_test = train_test_split(
                observations_tensor, labels_tensor,
                test_size=0.2, random_state=random_state, stratify=labels_tensor
            )
        else:
            print(f"Warning: Some classes have only {min_class_count} sample(s). Using non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                observations_tensor, labels_tensor,
                test_size=0.2, random_state=random_state, stratify=None
            )
    else:
        # Subsample the data and perform a stratified split
        random.seed(random_state)
        random.shuffle(filtered_data)
        
        # Ensure we have enough data for the full split
        total_required = train_size + test_size
        if len(filtered_data) < total_required:
            print(f"Error: Not enough filtered data ({len(filtered_data)}) for a {train_size}/{test_size} split.")
            return None, None, None, None

        sub_samples = filtered_data[:total_required]
        
        X_list, y_list = zip(*sub_samples)

        X_full = th.stack(X_list)
        y_full = th.tensor(y_list, dtype=th.long)

        # Check if stratification is possible with the subsampled data
        label_counts = Counter(y_full.numpy())
        min_class_count = min(label_counts.values())
        
        if min_class_count >= 2:
            print(f"Using stratified split (min class count: {min_class_count})")
            # Stratified split on the subsampled data
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full,
                train_size=train_size,
                test_size=test_size,
                random_state=random_state,
                stratify=y_full
            )
        else:
            print(f"Warning: Some classes have only {min_class_count} sample(s). Using non-stratified split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_full, y_full,
                train_size=train_size,
                test_size=test_size,
                random_state=random_state,
                stratify=None
            )

    print(f"Data split into training and testing sets:")
    print(f"  - Training set size: {len(X_train)}")
    print(f"  - Testing set size:  {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def balance_xy(X, y, max_per_class=None, min_required=5):
    y_np = y.cpu().numpy() if hasattr(y, 'cpu') else np.array(y)
    cls2idx = {}
    for i, c in enumerate(y_np):
        cls2idx.setdefault(c, []).append(i)
    # remove ultra-rare classes (optional): keep only classes with enough examples
    cls2idx = {c: idxs for c, idxs in cls2idx.items() if len(idxs) >= min_required}
    if not cls2idx:
        raise ValueError("No class has enough samples; collect more data.")

    min_count = min(len(idxs) for idxs in cls2idx.values())
    if max_per_class is not None:
        min_count = min(min_count, max_per_class)

    keep = []
    rng = np.random.default_rng(22)
    for c, idxs in cls2idx.items():
        sel = rng.choice(idxs, size=min_count, replace=False)
        keep.extend(sel.tolist())
    keep = np.array(keep)
    return X[keep], y[keep]

# --- Part 4: Extract Latent Representations ---
def extract_latent_vectors(model, data_loader, device, use_full_encoder=False, use_pre_relu=False): # Added use_pre_relu flag
    """Extracts latent vectors from the model's encoder for a given dataset.

    Args:
        model: The model to extract representations from
        data_loader: DataLoader with observations
        device: Device to use
        use_full_encoder: If True, uses model.encode() [1024-dim for PPO].
                          If False, uses model.enc() [256-dim for PPO, the ImpalaCNN output].
                          The paper likely uses False (ImpalaCNN output only).
        use_pre_relu: If True and use_full_encoder is False, extracts the pre-ReLU
                      features from the ImpalaCNN's final dense layer.
    """
    model.eval()
    latent_vectors = []
    with th.no_grad():
        for observations_batch in data_loader:
            observations_batch = observations_batch[0].to(device)
            if observations_batch.max() > 1.0:
                observations_batch = observations_batch / 255.0

            if use_full_encoder:
                # Full encoder: CNN + dense(256) + linear(1024)
                # Note: Pre-ReLU for this path is more complex, involving model.linear
                latents = model.encode(observations_batch)
                if use_pre_relu:
                     print("Warning: use_pre_relu with use_full_encoder=True not implemented easily, returning post-ReLU 1024-dim vector.")

            else:
                # ImpalaCNN only path
                if use_pre_relu:
                    # --- START MODIFICATION ---
                    # Manually compute pre-ReLU features from ImpalaCNN's dense layer
                    x = observations_batch
                    # Pass through convolutional stacks
                    for stack in model.enc.stacks:
                        x = stack(x)
                    # Flatten
                    x = x.reshape(x.size(0), -1)
                    # Apply ONLY the linear part of the final dense layer
                    # Assumes the dense layer module has an attribute '.layer' holding the nn.Linear
                    # If your model structure is different, you might need to adjust this line.
                    latents = model.enc.dense.layer(x)
                    # --- END MODIFICATION ---
                else:
                    # Default: Get post-ReLU features by calling the module directly
                    latents = model.enc(observations_batch) # This implicitly includes the ReLU

            if isinstance(latents, (tuple, list)):
                latents = latents[0]
            assert latents.ndim == 2, f"Unexpected latent shape {latents.shape}"
            latent_vectors.append(latents.cpu())

    return th.cat(latent_vectors, dim=0)

# --- Part 5: Train and Evaluate Classifier ---
def train_and_evaluate_classifier(X_train_latents, y_train, X_test_latents, y_test, num_classes, device, exp_name, num_epochs=500, random_state=42):
    """Trains and evaluates a linear classifier on the latent vectors."""
    print("\n--- Part 5: Training and Evaluating Classifier ---")
    
    # --- Setup ---
    input_dim = X_train_latents.shape[1]

    classifier = nn.Linear(input_dim, num_classes).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    train_dataset = TensorDataset(X_train_latents, y_train)
    g = th.Generator()
    g.manual_seed(random_state)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, generator=g)
    
    test_dataset = TensorDataset(X_test_latents, y_test)
    test_loader = DataLoader(test_dataset, batch_size=256)
    
    # --- Training Loop ---
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
            
            # Revert: Measure confidence on the ground-truth label, as per the paper's text.
            # This is the probability the model assigned to the correct answer.
            probs = F.softmax(outputs, dim=1)
            gt_confidences = probs[range(len(labels_batch)), labels_batch].cpu().numpy()
            all_confidences.extend(gt_confidences)

            _, predicted = th.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())
            
    accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
    median_confidence = np.median(all_confidences)
    
    print(f"\n--- Evaluation Complete ---")
    print(f"Final Test Accuracy: {accuracy:.2f}%")
    print(f"Median Confidence on Ground-Truth Labels: {median_confidence:.4f}")

    # --- Detailed Report ---
    print("\n--- Classification Report ---")
    # Get the names of all unique labels present in the test set
    unique_labels = np.unique(all_labels)
    target_names = [TASKS[i] for i in unique_labels]
    print(classification_report(all_labels, all_preds, target_names=target_names))

    # --- Confusion Matrix ---
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(all_labels, all_preds)
    print("Columns: Predicted, Rows: Actual")
    print(cm)

    # Per-class confidence check
    df = pd.DataFrame({"label": all_labels, "conf": all_confidences})
    mean_conf = df.groupby("label")["conf"].mean()
    print("\nMean confidence per class:")
    for i in mean_conf.index:
        print(f"  {TASKS[i]:20s}: {mean_conf[i]:.3f}")

    np.save(f"conf_{exp_name}.npy", np.array(all_confidences))
    print(f"Confidence values saved to: conf_{exp_name}.npy")

    return all_confidences


# --- Part 6: Visualize Confidence ---
def plot_confidence_density(conf_dict, save_path="confidence_comparison.png"):
    """
    conf_dict: {"PPO": np.ndarray, "Ours": np.ndarray}
    """

    plt.figure(figsize=(4, 3))

    # 20 bins from 0–1
    bins = np.linspace(0, 1, 21)

    for label, confs in conf_dict.items():
        color = "red" if label.lower() == "ppo" else "blue"
        alpha = 0.25 if label.lower() == "ppo" else 0.35

        # --- normalization style to match paper ---
        # Convert counts to probabilities, not area densities
        counts, _ = np.histogram(confs, bins=bins)
        probs = counts / counts.sum()  # normalize total area = 1
        plt.bar(
            bins[:-1],
            probs,
            width=np.diff(bins),
            align="edge",
            color=color,
            alpha=alpha,
            edgecolor=color,
            linewidth=1.0,
            label=label,
        )

    plt.xlabel("Confidence", fontsize=10)
    plt.ylabel("Density", fontsize=10)
    plt.xlim(0, 1)
    plt.ylim(0, 0.25)   # adjust to match figure scale
    plt.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved to {save_path}")


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
    parser.add_argument("--classifier_epochs", type=int, default=500, help="Number of epochs to train the linear classifier.")

    # Data handling arguments
    parser.add_argument("--output_dataset_path", type=str, default=None, help="If provided, save the collected and labeled dataset to this path.")
    parser.add_argument("--load_dataset_path", type=str, default=None, help="If provided, load a pre-existing labeled dataset from this path, skipping collection.")
    
    # Representation extraction arguments
    parser.add_argument("--use_full_encoder", action="store_true", help="If set, extract from full encoder (1024-dim). Otherwise, extract from ImpalaCNN only (256-dim). Paper likely uses ImpalaCNN only.")
    parser.add_argument("--use_pre_relu", action="store_true", help="If set and --use_full_encoder is NOT set, extract pre-ReLU features from ImpalaCNN.")
    
    args = parser.parse_args()

    # --- Argument Validation ---
    if args.output_dataset_path and args.load_dataset_path:
        print("Error: Cannot use --output_dataset_path and --load_dataset_path simultaneously.")
        sys.exit(1)

    # --- Global Setup ---
    cuda = th.cuda.is_available()
    device = th.device("cuda:0" if cuda else "cpu")
    print(f"--- Global Setup ---")
    print(f"Using device: {device}")


    # --- Part A: Data Generation or Loading ---
    if args.load_dataset_path:
        print(f"\n--- Loading Dataset from {args.load_dataset_path} ---")
        if not os.path.exists(args.load_dataset_path):
            print(f"Error: Dataset file not found at {args.load_dataset_path}")
            sys.exit(1)
        try:
            labeled_state_data = th.load(args.load_dataset_path)
            print("Dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            sys.exit(1)
    else:
        # --- Run Part 1: Data Collection ---
        use_expert = bool(args.expert_exp_name)
        if use_expert:
            # Ensure all expert args are provided if expert_exp_name is set
            if not all([args.expert_timestamp, args.expert_train_seed is not None]):
                print("Error: If --expert_exp_name is used, you must provide --expert_timestamp and --expert_train_seed.")
                sys.exit(1)
        
        # Determine the config for the data collection model
        collection_exp_name = args.expert_exp_name if use_expert else args.exp_name
        config_path = f"configs/{collection_exp_name}.yaml"
        try:
            with open(config_path, "r") as f:
                collection_config = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print(f"Error: Config file for data collection model not found at {config_path}")
            sys.exit(1)

        collected_episodes = collect_data(args, device, use_expert=use_expert, model_config=collection_config)

        # --- Run Part 2: Labeling States ---
        labeled_state_data = label_states_with_next_achievement(collected_episodes)

        if args.output_dataset_path:
            print(f"\n--- Saving Labeled Dataset to {args.output_dataset_path} ---")
            try:
                th.save(labeled_state_data, args.output_dataset_path)
                print("Dataset saved successfully.")
            except Exception as e:
                print(f"Error saving dataset: {e}")

    # --- Part B: Analysis (applies to both generated and loaded data) ---
    # --- Load the ANALYSIS Model ---
    analysis_model = load_analysis_model(
        args.exp_name, args.timestamp, args.train_seed, args.ckpt_epoch, device
    )

    # Freeze encoder (IMPORTANT)
    analysis_model.eval()
    for name, param in analysis_model.named_parameters():
        param.requires_grad = False
    print("✅ Encoder frozen. Trainable parameters should now be only classifier.")

    # Add This Debug Check
    print("---- Trainable Parameters After Freezing ----")
    for name, param in analysis_model.named_parameters():
        if param.requires_grad:
            print("❗ STILL TRAINABLE:", name)


    # --- Run Part 3: Create Train/Test Splits ---
    X_train, X_test, y_train, y_test = create_train_test_splits(labeled_state_data)
    
    if X_train is not None:
        X_train, y_train = balance_xy(X_train, y_train, max_per_class=1000, min_required=10)
        X_test,  y_test  = balance_xy(X_test,  y_test,  max_per_class=200,  min_required=10)
        print(X_train[0].max(), X_train[0].mean())
        # --- Dataset Sanity Check ---
        print("\n--- Dataset Sanity Check ---")
        label_counts = Counter(y_train.cpu().numpy())
        print(f"Unique labels: {len(label_counts)} / {len(TASKS)} possible")
        print(" labels:")
        print(label_counts.most_common(len(label_counts)))

        # --- Run Part 4: Extract Latent Representations ---
        print("\n--- Part 4: Extracting Latent Representations ---")
        # Create DataLoaders for extraction
        train_obs_dataset = TensorDataset(X_train)
        train_obs_loader = DataLoader(train_obs_dataset, batch_size=256)
        
        test_obs_dataset = TensorDataset(X_test)
        test_obs_loader = DataLoader(test_obs_dataset, batch_size=256)

        print(f"Extracting latents for training set (use_full_encoder={args.use_full_encoder})...")
        X_train_latents = extract_latent_vectors(analysis_model, train_obs_loader, device, use_full_encoder=args.use_full_encoder, use_pre_relu=args.use_pre_relu)
        print(f"Extracting latents for testing set (use_full_encoder={args.use_full_encoder})...")
        X_test_latents = extract_latent_vectors(analysis_model, test_obs_loader, device, use_full_encoder=args.use_full_encoder, use_pre_relu=args.use_pre_relu)

        print(f"Latent vector shapes:")
        print(f"  - Training latents: {X_train_latents.shape}")
        print(f"  - Testing latents:  {X_test_latents.shape}")

        # Diagnostic: Check latent vector statistics pre- and post-ReLU
        with th.no_grad():
            sample_batch = X_train[:32] # Use a small batch of observations
            x = sample_batch.to(device)
            
            # Check statistics at the final layer we're using
            if args.use_full_encoder:
                # Using full encoder: check the second linear layer
                enc_out = analysis_model.enc(x)
                if hasattr(analysis_model, 'linear') and hasattr(analysis_model.linear, 'layer'):
                    pre_relu = analysis_model.linear.layer(enc_out)
                    post_relu = F.relu(pre_relu)
                    layer_name = "second linear layer"
                else:
                    pre_relu = post_relu = enc_out
                    layer_name = "encoder output (no second layer)"
            else:
                # Using ImpalaCNN only: check the dense layer
                for stack in analysis_model.enc.stacks:
                    x = stack(x)
                x = x.reshape(x.size(0), -1)
                pre_relu = analysis_model.enc.dense.layer(x)
                post_relu = F.relu(pre_relu)
                layer_name = "ImpalaCNN dense layer"

            neg_frac = (pre_relu < 0).float().mean().item()
            zero_frac = (post_relu == 0).float().mean().item()
            print(f"\n--- Latent Vector Sanity Check ({layer_name}) ---")
            print(f"Fraction negative before ReLU: {neg_frac:.3f}")
            print(f"Fraction zero after ReLU: {zero_frac:.3f}")
            print(f"Mean pre-ReLU: {pre_relu.mean():.3f}, std: {pre_relu.std():.3f}")

        # --- Run Part 5: Train and Evaluate Classifier ---
        num_classes = len(TASKS) # 22 achievements
        confidences = train_and_evaluate_classifier(
            X_train_latents, y_train, X_test_latents, y_test, num_classes, device, args.exp_name, num_epochs=args.classifier_epochs, random_state=args.eval_seed
        )
        
        # --- Run Part 6: Visualize Confidence ---
        if confidences:
            plot_confidence_density(confidences)
    
