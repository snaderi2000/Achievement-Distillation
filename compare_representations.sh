#!/bin/bash

# Script to compare representations between PPO and PPO-AD
# Usage: ./compare_representations.sh <timestamp> <train_seed>

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <timestamp> <train_seed>"
    echo "Example: $0 debug 0"
    exit 1
fi

TIMESTAMP=$1
TRAIN_SEED=$2
NUM_EPISODES=500
EVAL_SEED=42

echo "=================================================="
echo "Representation Analysis Comparison"
echo "=================================================="
echo "Timestamp: $TIMESTAMP"
echo "Train Seed: $TRAIN_SEED"
echo "Num Episodes: $NUM_EPISODES"
echo "=================================================="
echo ""

# Check if models exist
PPO_MODEL="./models/ppo-${TIMESTAMP}-s$(printf "%02d" $TRAIN_SEED)/agent-e250.pt"
PPOAD_MODEL="./models/ppo_ad-${TIMESTAMP}-s$(printf "%02d" $TRAIN_SEED)/agent-e250.pt"

if [ ! -f "$PPO_MODEL" ]; then
    echo "ERROR: PPO model not found at $PPO_MODEL"
    exit 1
fi

if [ ! -f "$PPOAD_MODEL" ]; then
    echo "ERROR: PPO-AD model not found at $PPOAD_MODEL"
    exit 1
fi

echo "✓ Found PPO model at $PPO_MODEL"
echo "✓ Found PPO-AD model at $PPOAD_MODEL"
echo ""

# Step 1: Collect dataset using PPO-AD expert (only once)
DATASET_PATH="./dataset_ppo_ad_expert.pt"

if [ -f "$DATASET_PATH" ]; then
    echo "Dataset already exists at $DATASET_PATH, skipping collection..."
else
    echo "=================================================="
    echo "Step 1: Collecting dataset with PPO-AD expert"
    echo "=================================================="
    python analyze_representation.py \
        --exp_name ppo_ad \
        --timestamp "$TIMESTAMP" \
        --train_seed "$TRAIN_SEED" \
        --ckpt_epoch 250 \
        --num_episodes "$NUM_EPISODES" \
        --eval_seed "$EVAL_SEED" \
        --output_dataset_path "$DATASET_PATH"
    echo "✓ Dataset collected and saved"
    echo ""
fi

# Step 2: Analyze PPO representations (ImpalaCNN output, 256-dim)
echo "=================================================="
echo "Step 2: Analyzing PPO representations (256-dim)"
echo "=================================================="
python analyze_representation.py \
    --exp_name ppo \
    --timestamp "$TIMESTAMP" \
    --train_seed "$TRAIN_SEED" \
    --ckpt_epoch 250 \
    --eval_seed "$EVAL_SEED" \
    --load_dataset_path "$DATASET_PATH" \
    --classifier_epochs 500

echo ""
echo "✓ PPO analysis complete"
echo ""

# Step 3: Analyze PPO-AD representations (ImpalaCNN output, 256-dim)
echo "=================================================="
echo "Step 3: Analyzing PPO-AD representations (256-dim)"
echo "=================================================="
python analyze_representation.py \
    --exp_name ppo_ad \
    --timestamp "$TIMESTAMP" \
    --train_seed "$TRAIN_SEED" \
    --ckpt_epoch 250 \
    --eval_seed "$EVAL_SEED" \
    --load_dataset_path "$DATASET_PATH" \
    --classifier_epochs 500

echo ""
echo "✓ PPO-AD analysis complete"
echo ""

# Optional: Analyze with full encoder (1024-dim) for comparison
echo "=================================================="
echo "Step 4 (Optional): Analyzing with full encoder"
echo "=================================================="
read -p "Analyze with full encoder (1024-dim)? [y/N] " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Analyzing PPO with full encoder..."
    python analyze_representation.py \
        --exp_name ppo \
        --timestamp "$TIMESTAMP" \
        --train_seed "$TRAIN_SEED" \
        --ckpt_epoch 250 \
        --eval_seed "$EVAL_SEED" \
        --load_dataset_path "$DATASET_PATH" \
        --classifier_epochs 500 \
        --use_full_encoder
    
    echo ""
    echo "Analyzing PPO-AD with full encoder..."
    python analyze_representation.py \
        --exp_name ppo_ad \
        --timestamp "$TIMESTAMP" \
        --train_seed "$TRAIN_SEED" \
        --ckpt_epoch 250 \
        --eval_seed "$EVAL_SEED" \
        --load_dataset_path "$DATASET_PATH" \
        --classifier_epochs 500 \
        --use_full_encoder
fi

echo ""
echo "=================================================="
echo "Analysis Complete!"
echo "=================================================="
echo ""
echo "Check the outputs above to compare:"
echo "  - Test Accuracy"
echo "  - Median Confidence"
echo ""
echo "Expected (from paper):"
echo "  PPO:     ~44.9% accuracy, ~0.240 confidence"
echo "  PPO-AD:  Higher accuracy, higher confidence"
echo ""
echo "Confidence plots saved as confidence_density.png"
echo "Confidence values saved as conf_ppo.npy and conf_ppo_ad.npy"
echo "=================================================="

