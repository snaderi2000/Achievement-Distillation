from typing import Dict
import numpy as np

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

    rnn_states = None
    if hasattr(model, "get_init_rnn_states"):
        rnn_states = model.get_init_rnn_states(storage.nproc).to(storage.device)

    for step in range(storage.nstep):
        # Pass through model
        inputs = storage.get_inputs(step)
        if rnn_states is not None:
            inputs["rnn_states"] = rnn_states
        outputs = model.act(**inputs)
        if rnn_states is not None:
            rnn_states = outputs.get("rnn_states", rnn_states)
        actions = outputs["actions"]

        # Step environment
        obs, rewards, dones, infos = venv.step(actions)
        outputs["obs"] = obs
        outputs["rewards"] = rewards
        outputs["masks"] = 1.0 - dones
        outputs["successes"] = infos["successes"]

        # Update storage
        storage.insert(**outputs, model=model, rnn_states=rnn_states)

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
    if rnn_states is not None:
        inputs["rnn_states"] = rnn_states
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
