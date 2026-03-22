"""
action_tokenizer.py

Discretizes continuous delta robot actions into N bins per dimension.
Standalone tokenizer (no LLM tokenizer dependency).
Inspired by OpenVLA's ActionTokenizer but adapted for ACT-DT.
"""

import json
import numpy as np
import torch


class ActionTokenizer:
    def __init__(self, n_bins=256, q01=None, q99=None, stats_path=None):
        """
        Args:
            n_bins: Number of discrete bins per action dimension.
            q01: np.array (action_dim,) — per-dimension 1st percentile of delta actions.
            q99: np.array (action_dim,) — per-dimension 99th percentile of delta actions.
            stats_path: Path to JSON file with q01/q99 stats. If provided, overrides q01/q99.
        """
        self.n_bins = n_bins

        if stats_path is not None:
            with open(stats_path, "r") as f:
                stats = json.load(f)
            q01 = np.array(stats["q01"], dtype=np.float32)
            q99 = np.array(stats["q99"], dtype=np.float32)

        assert q01 is not None and q99 is not None, "Must provide q01/q99 or stats_path"
        self.q01 = q01
        self.q99 = q99

        # 257 edges → 256 intervals → 256 bin centers
        self.bins = np.linspace(-1.0, 1.0, n_bins + 1)  # (n_bins+1,)
        self.bin_centers = ((self.bins[:-1] + self.bins[1:]) / 2.0).astype(np.float32)  # (n_bins,)

        # Torch version for GPU use
        self.bin_centers_torch = torch.from_numpy(self.bin_centers)  # (n_bins,)

    def normalize(self, delta_actions):
        """Normalize delta actions to [-1, 1] using Q99 boundaries."""
        normalized = 2.0 * (delta_actions - self.q01) / (self.q99 - self.q01 + 1e-8) - 1.0
        return np.clip(normalized, -1.0, 1.0)

    def unnormalize(self, normalized):
        """Unnormalize from [-1, 1] back to raw delta actions."""
        return (normalized + 1.0) / 2.0 * (self.q99 - self.q01 + 1e-8) + self.q01

    def encode(self, delta_actions):
        """
        Encode continuous delta actions to discrete token indices.

        Args:
            delta_actions: np.array (..., action_dim)
        Returns:
            np.array (..., action_dim) of int64 token indices in [0, n_bins-1]
        """
        normalized = self.normalize(delta_actions)
        # digitize returns indices in [1, n_bins+1], map to [0, n_bins-1]
        tokens = np.digitize(normalized, self.bins) - 1
        tokens = np.clip(tokens, 0, self.n_bins - 1)
        return tokens.astype(np.int64)

    def decode(self, tokens):
        """
        Decode discrete token indices to continuous delta actions.

        Args:
            tokens: np.array (..., action_dim) of int token indices in [0, n_bins-1]
        Returns:
            np.array (..., action_dim) of continuous delta actions
        """
        tokens = np.clip(tokens, 0, self.n_bins - 1)
        normalized = self.bin_centers[tokens]
        return self.unnormalize(normalized)

    def decode_torch(self, tokens):
        """
        Decode discrete token indices to continuous delta actions (torch version).

        Args:
            tokens: LongTensor (..., action_dim)
        Returns:
            FloatTensor (..., action_dim)
        """
        tokens = tokens.clamp(0, self.n_bins - 1)
        bin_centers = self.bin_centers_torch.to(tokens.device)
        normalized = bin_centers[tokens]
        # Unnormalize
        q01 = torch.from_numpy(self.q01).to(tokens.device).float()
        q99 = torch.from_numpy(self.q99).to(tokens.device).float()
        return (normalized + 1.0) / 2.0 * (q99 - q01 + 1e-8) + q01

    def soft_decode_torch(self, logits):
        """
        Soft-argmax decoding: convert logits to expected continuous delta actions.

        Args:
            logits: FloatTensor (..., n_bins)
        Returns:
            FloatTensor (...,) — expected normalized value per dimension
        """
        probs = torch.softmax(logits, dim=-1)  # (..., n_bins)
        bin_centers = self.bin_centers_torch.to(logits.device)  # (n_bins,)
        return (probs * bin_centers).sum(dim=-1)  # (...)
