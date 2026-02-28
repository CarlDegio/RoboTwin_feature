"""
PPO Rollout Buffer and Environment Interaction for ACT-RL.

Handles:
- RolloutBuffer: stores trajectories (obs, actions, rewards, dones, values, log_probs)
- Environment interaction: collect rollouts using chunk-level execution
- No temporal ensemble: execute full chunk as one RL action
- Sparse reward: +1.0 success, -1.0 failure (per episode)
"""

import os
import sys
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ChunkTransition:
    """Single chunk-level transition."""
    qpos: np.ndarray           # (state_dim,) normalized
    images: np.ndarray         # (num_cam, C, H, W)
    action: np.ndarray         # (chunk_size, state_dim) sampled action chunk
    reward: float              # sparse reward for this chunk
    done: bool                 # episode ended during this chunk
    value: float               # V(s) estimate
    log_prob: float            # log pi(a|s)


class RolloutBuffer:
    """
    Buffer for storing chunk-level PPO rollout data.
    Each entry corresponds to one chunk execution.
    """

    def __init__(self):
        self.transitions: List[ChunkTransition] = []

    def add(self, transition: ChunkTransition):
        self.transitions.append(transition)

    def clear(self):
        self.transitions = []

    def __len__(self):
        return len(self.transitions)

    def get_tensors(self, device="cuda"):
        """
        Convert buffer to tensors for PPO update.

        Returns dict with:
            qpos: (N, state_dim)
            images: (N, num_cam, C, H, W)
            actions: (N, chunk_size, state_dim)
            rewards: (N,)
            dones: (N,)
            values: (N,)
            log_probs: (N,)
        """
        qpos = torch.tensor(
            np.array([t.qpos for t in self.transitions]),
            dtype=torch.float32, device=device,
        )
        images = torch.tensor(
            np.array([t.images for t in self.transitions]),
            dtype=torch.float32, device=device,
        )
        actions = torch.tensor(
            np.array([t.action for t in self.transitions]),
            dtype=torch.float32, device=device,
        )
        rewards = torch.tensor(
            [t.reward for t in self.transitions],
            dtype=torch.float32, device=device,
        )
        dones = torch.tensor(
            [t.done for t in self.transitions],
            dtype=torch.float32, device=device,
        )
        values = torch.tensor(
            [t.value for t in self.transitions],
            dtype=torch.float32, device=device,
        )
        log_probs = torch.tensor(
            [t.log_prob for t in self.transitions],
            dtype=torch.float32, device=device,
        )
        return {
            "qpos": qpos,
            "images": images,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "values": values,
            "log_probs": log_probs,
        }


def encode_obs(observation):
    """
    Encode RoboTwin observation into format expected by ACT model.
    Same as deploy_policy.encode_obs.
    """
    import cv2
    head_cam = cv2.resize(
        observation["observation"]["head_camera"]["rgb"],
        (640, 480), interpolation=cv2.INTER_LINEAR,
    )
    left_cam = cv2.resize(
        observation["observation"]["left_camera"]["rgb"],
        (640, 480), interpolation=cv2.INTER_LINEAR,
    )
    right_cam = cv2.resize(
        observation["observation"]["right_camera"]["rgb"],
        (640, 480), interpolation=cv2.INTER_LINEAR,
    )
    head_cam = np.moveaxis(head_cam, -1, 0) / 255.0
    left_cam = np.moveaxis(left_cam, -1, 0) / 255.0
    right_cam = np.moveaxis(right_cam, -1, 0) / 255.0

    qpos = (
        observation["joint_action"]["left_arm"]
        + [observation["joint_action"]["left_gripper"]]
        + observation["joint_action"]["right_arm"]
        + [observation["joint_action"]["right_gripper"]]
    )
    return {
        "head_cam": head_cam,
        "left_cam": left_cam,
        "right_cam": right_cam,
        "qpos": np.array(qpos, dtype=np.float32),
    }


def obs_to_tensors(obs_dict, model, device="cuda"):
    """
    Convert encoded observation dict to model input tensors.

    Args:
        obs_dict: dict from encode_obs with head_cam, left_cam, right_cam, qpos
        model: ACTPPOModel (for pre_process)
        device: torch device

    Returns:
        qpos: (1, state_dim) normalized
        images: (1, num_cam, C, H, W)
    """
    qpos_np = obs_dict["qpos"]
    qpos_norm = model.pre_process(qpos_np)
    qpos = torch.from_numpy(qpos_norm).float().to(device).unsqueeze(0)

    images = np.stack([
        obs_dict["head_cam"],
        obs_dict["left_cam"],
        obs_dict["right_cam"],
    ], axis=0)  # (3, C, H, W)
    images = torch.from_numpy(images).float().to(device).unsqueeze(0)

    return qpos, images
