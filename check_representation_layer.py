"""
Diagnostic script to compare different encoder layer outputs.
This helps determine which layer to extract representations from.
"""

import argparse
import os
import sys
import yaml
import torch as th
from crafter.env import Env
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

from achievement_distillation.model import *
from achievement_distillation.wrapper import VecPyTorch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--timestamp", type=str, required=True) 
    parser.add_argument("--train_seed", type=int, required=True)
    parser.add_argument("--ckpt_epoch", type=int, default=250)
    args = parser.parse_args()
    
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    
    # Load config
    config_path = f"configs/{args.exp_name}.yaml"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Create environment
    temp_venv = VecPyTorch(DummyVecEnv([lambda: Env()]), device=device)
    
    # Load model
    model_cls = getattr(sys.modules[__name__], config["model_cls"])
    model = model_cls(
        observation_space=temp_venv.observation_space,
        action_space=temp_venv.action_space,
        **config["model_kwargs"],
    )
    model.to(device)
    
    # Load checkpoint
    run_name = f"{args.exp_name}-{args.timestamp}-s{args.train_seed:02}"
    ckpt_path = os.path.join("./models", run_name, f"agent-e{args.ckpt_epoch:03}.pt")
    state_dict = th.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"\n{'='*60}")
    print(f"Model: {args.exp_name}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"{'='*60}\n")
    
    # Get a sample observation
    obs = temp_venv.reset()
    
    with th.no_grad():
        # Extract at different layers
        print("Encoder Architecture:")
        print("-" * 60)
        
        # 1. After ImpalaCNN (before second linear layer)
        enc_output = model.enc(obs)
        print(f"1. After ImpalaCNN (model.enc):")
        print(f"   Shape: {enc_output.shape}")
        print(f"   This is the CNN + first dense layer (256-dim)")
        print()
        
        # 2. After full encoder (including second linear layer)
        full_enc_output = model.encode(obs)
        print(f"2. After full encoder (model.encode):")
        print(f"   Shape: {full_enc_output.shape}")
        print(f"   This is CNN + first dense + second linear (1024-dim)")
        print()
        
        print("-" * 60)
        print("\nRecommendation:")
        print("=" * 60)
        print("The paper likely uses the ImpalaCNN output (256-dim)")
        print("since they refer to 'encoder' and the second linear layer")
        print("is more of a projection to the hidden size for the policy/value heads.")
        print()
        print("Try modifying extract_latent_vectors to use:")
        print("    latents = model.enc(observations_batch)")
        print("instead of:")
        print("    latents = model.encode(observations_batch)")
        print("=" * 60)
    
    temp_venv.close()

if __name__ == "__main__":
    main()

