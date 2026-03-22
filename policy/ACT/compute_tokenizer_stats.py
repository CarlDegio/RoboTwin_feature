"""
compute_tokenizer_stats.py

Scans training episodes, computes delta actions, and saves per-dimension
Q1/Q99 statistics for the ActionTokenizer.

Usage:
    python compute_tokenizer_stats.py --dataset_dir <path> --num_episodes <N> --output_path <path> [--state_dim 14]
"""

import argparse
import json
import os

import h5py
import numpy as np


def compute_delta_actions(dataset_dir, num_episodes, state_dim=14):
    """
    Load all episodes and compute delta actions.
    delta[0] = action[0] - qpos
    delta[t] = action[t] - action[t-1]  for t > 0
    """
    all_deltas = []

    for ep_id in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f"episode_{ep_id}.hdf5")
        if not os.path.exists(dataset_path):
            print(f"Warning: {dataset_path} not found, skipping")
            continue

        with h5py.File(dataset_path, "r") as root:
            actions = root["/action"][:]  # (episode_len, action_dim)
            qpos = root["/observations/qpos"][0]  # (action_dim,) — initial qpos

        # Compute deltas
        deltas = np.zeros_like(actions)
        deltas[0] = actions[0] - qpos
        deltas[1:] = actions[1:] - actions[:-1]

        all_deltas.append(deltas)

    all_deltas = np.concatenate(all_deltas, axis=0)  # (total_steps, action_dim)
    return all_deltas


def main():
    parser = argparse.ArgumentParser(description="Compute tokenizer Q1/Q99 stats from training data")
    parser.add_argument("--dataset_dir", type=str, required=True, help="Path to HDF5 episode directory")
    parser.add_argument("--num_episodes", type=int, required=True, help="Number of episodes")
    parser.add_argument("--output_path", type=str, required=True, help="Output JSON path for stats")
    parser.add_argument("--state_dim", type=int, default=14, help="Action dimension")
    args = parser.parse_args()

    print(f"Computing delta action stats from {args.num_episodes} episodes in {args.dataset_dir}")
    all_deltas = compute_delta_actions(args.dataset_dir, args.num_episodes, args.state_dim)
    print(f"Total delta samples: {all_deltas.shape[0]}, action_dim: {all_deltas.shape[1]}")

    q01 = np.percentile(all_deltas, 1, axis=0).tolist()
    q99 = np.percentile(all_deltas, 99, axis=0).tolist()

    stats = {
        "q01": q01,
        "q99": q99,
        "num_samples": int(all_deltas.shape[0]),
        "action_dim": int(all_deltas.shape[1]),
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved tokenizer stats to {args.output_path}")
    print(f"Q01: {q01}")
    print(f"Q99: {q99}")


if __name__ == "__main__":
    main()
