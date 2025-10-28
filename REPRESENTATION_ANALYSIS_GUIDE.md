# Representation Analysis Guide

This guide explains how to correctly reproduce the representation analysis from the paper.

## The Paper's Methodology (from Appendix A)

1. **Train an expert policy**: Use PPO-AD with 1M environment steps (different seed from evaluation)
2. **Collect episodes**: Use the expert policy to collect 215,578 states
3. **Subsample data**: 50,000 states for training, 10,000 for testing
4. **Extract representations**: From the encoder of the model being analyzed
5. **Train linear classifier**: Predict next achievement from frozen latent representations
6. **Evaluate**: Measure accuracy and confidence on test set

## Expected Results from Paper

**PPO (baseline):**
- Test Accuracy: 44.9%
- Median Confidence: 0.240

**PPO-AD (their method):**
- Test Accuracy: (Not explicitly stated, but should be higher)
- Median Confidence: (Higher than 0.240 based on Figure 3)

## Your Results

You reported:
- Test Accuracy: 55.43%
- Median Confidence: 0.4135

**These are BETTER than the paper's PPO results!** This suggests one of two things:
1. You're analyzing PPO-AD instead of PPO
2. You're using a different encoder layer than the paper

## Critical Questions to Answer

### 1. Which model are you analyzing?

Check your command. You should be analyzing the **PPO baseline** (not PPO-AD).

**Correct command structure:**
```bash
python analyze_representation.py \
  --exp_name ppo \              # ← Analyze PPO
  --timestamp <your_timestamp> \
  --train_seed <your_seed> \
  --ckpt_epoch 250 \
  --expert_exp_name ppo_ad \    # ← Collect data with PPO-AD expert
  --expert_timestamp <expert_timestamp> \
  --expert_train_seed <expert_seed> \
  --expert_ckpt_epoch 250 \
  --num_episodes 500            # Collect enough episodes
```

### 2. Which encoder layer to use?

The architecture is:
```
ImpalaCNN: CNN stacks → Dense(256) + ReLU
                        ↓
PPOModel:              Linear(1024) + ReLU
                        ↓
                       Policy/Value Heads
```

**Two options:**
- `model.enc(obs)` → 256-dim (ImpalaCNN output)
- `model.encode(obs)` → 1024-dim (full encoder with second linear layer)

**The paper likely uses 256-dim** since they refer to "encoder" and the second linear layer is more of a projection for the heads.

## Running the Corrected Analysis

### Option 1: Using ImpalaCNN output (256-dim) - RECOMMENDED

```bash
python analyze_representation.py \
  --exp_name ppo \
  --timestamp debug \
  --train_seed 0 \
  --ckpt_epoch 250 \
  --expert_exp_name ppo_ad \
  --expert_timestamp debug \
  --expert_train_seed 0 \
  --expert_ckpt_epoch 250 \
  --num_episodes 500 \
  --eval_seed 42 \
  --classifier_epochs 500
  # Note: --use_full_encoder is NOT set, so uses ImpalaCNN (256-dim)
```

### Option 2: Using full encoder (1024-dim) - for comparison

```bash
python analyze_representation.py \
  --exp_name ppo \
  --timestamp debug \
  --train_seed 0 \
  --ckpt_epoch 250 \
  --expert_exp_name ppo_ad \
  --expert_timestamp debug \
  --expert_train_seed 0 \
  --expert_ckpt_epoch 250 \
  --num_episodes 500 \
  --eval_seed 42 \
  --classifier_epochs 500 \
  --use_full_encoder  # ← Uses 1024-dim
```

## Comparing PPO vs PPO-AD

To compare representations, run the analysis on both models:

### Analyze PPO
```bash
python analyze_representation.py \
  --exp_name ppo \
  --timestamp debug \
  --train_seed 0 \
  --ckpt_epoch 250 \
  --expert_exp_name ppo_ad \
  --expert_timestamp debug \
  --expert_train_seed 0 \
  --num_episodes 500
```

### Analyze PPO-AD
```bash
python analyze_representation.py \
  --exp_name ppo_ad \
  --timestamp debug \
  --train_seed 0 \
  --ckpt_epoch 250 \
  --expert_exp_name ppo_ad \
  --expert_timestamp debug \
  --expert_train_seed 0 \
  --num_episodes 500
```

## Diagnostic Tool

Use the diagnostic script to inspect the encoder layers:

```bash
python check_representation_layer.py \
  --exp_name ppo \
  --timestamp debug \
  --train_seed 0 \
  --ckpt_epoch 250
```

## Troubleshooting

### If accuracy is too high (>50%)
- Check you're analyzing `ppo` not `ppo_ad`
- Try using ImpalaCNN output only (without `--use_full_encoder`)
- Verify your PPO model was trained correctly (not with achievement distillation)

### If accuracy is too low (<40%)
- Check if enough achievements were unlocked during data collection
- Try collecting more episodes (`--num_episodes 1000`)
- Verify the expert policy is strong (should be PPO-AD trained to convergence)

### If confidence is too high (>0.3 for PPO)
- Same as "accuracy too high" above
- The paper shows PPO has low confidence (0.240), PPO-AD should be higher

## Understanding the Results

**Good PPO results should show:**
- Moderate accuracy (40-50%) - learns some structure but not perfectly
- Low confidence (0.2-0.3) - uncertain about predictions
- This motivates the need for PPO-AD!

**Good PPO-AD results should show:**
- Higher accuracy (50-60%+) - learns better representations
- Higher confidence (0.4+) - more certain about predictions
- This demonstrates PPO-AD learns achievement-aware representations!

