"""
ACT-PPO Model: ACT policy adapted for PPO reinforcement learning fine-tuning.

Design:
- PPO-ACT (learnable): Load pretrained ACT weights, freeze only ResNet backbone,
  Transformer encoder/decoder + action_head are all trainable.
  Add learnable value_head and log_std_head.
  Output: mu (learnable), std (learnable), V (learnable)
- ACT-frozen (reference): Completely frozen DETRVAE copy, outputs mu_ref + fixed sigma_ref.
  Used only for KL divergence computation.
- KL(PPO-ACT || ACT-frozen) constrains the fine-tuned policy.
"""

import os
import copy
import torch
import torch.nn as nn
import numpy as np
import pickle
import torchvision.transforms as transforms
from argparse import Namespace

try:
    from detr.main import build_ACT_model_and_optimizer
except ImportError:
    from .detr.main import build_ACT_model_and_optimizer

try:
    from act_policy import ACTPolicy
except ImportError:
    from .act_policy import ACTPolicy


def _config_dict_to_namespace(config_dict):
    """Convert act_policy_config dict to argparse.Namespace for build_ACT_model_and_optimizer.
    Fills in default values for fields required by DETR builders."""
    defaults = {
        "lr": 5e-5,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "dilation": False,
        "masks": False,
        "position_embedding": "sine",
        "dropout": 0.1,
        "pre_norm": False,
        "enc_layers": 4,
        "dec_layers": 4,
        "nheads": 8,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "chunk_size": 50,
        "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
        "kl_weight": 10,
        "state_dim": 14,
        "weight_decay": 1e-4,
    }
    merged = {**defaults, **config_dict}
    return Namespace(**merged)


class ACTPPOModel(nn.Module):
    """
    ACT model wrapped for PPO fine-tuning.

    Architecture:
    - Frozen ResNet18 visual backbone (BatchNorm must stay in eval mode)
    - Learnable Transformer encoder/decoder + action_head (mu)
    - Learnable log_std_head: decoder hidden states -> per-step log-std (sigma)
    - Learnable value_head: pooled decoder hidden states -> scalar V(s)
    """

    def __init__(self, act_policy_config, ppo_config, RoboTwin_Config=None):
        super().__init__()

        self.chunk_size = act_policy_config["chunk_size"]
        self.state_dim = act_policy_config.get("state_dim", 14)
        self.hidden_dim = act_policy_config.get("hidden_dim", 512)

        # Build the ACT model (DETRVAE)
        # Convert config dict to Namespace to bypass argparse in build_ACT_model_and_optimizer
        act_args = _config_dict_to_namespace(act_policy_config)
        model, _ = build_ACT_model_and_optimizer(act_policy_config, act_args)
        self.act_model = model

        # Load pretrained weights if checkpoint provided
        ckpt_dir = act_policy_config.get("ckpt_dir", "")
        if ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location="cpu")
                # ACTPolicy wraps model as self.model, so keys have "model." prefix
                act_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        act_state_dict[k[len("model."):]] = v
                if act_state_dict:
                    self.act_model.load_state_dict(act_state_dict, strict=True)
                else:
                    # Try loading directly (in case checkpoint is raw DETRVAE)
                    self.act_model.load_state_dict(state_dict, strict=True)
                print(f"[ACT-PPO] Loaded pretrained ACT weights from {ckpt_path}")
            else:
                print(f"[ACT-PPO] Warning: checkpoint not found at {ckpt_path}")

        # Load dataset stats for normalization
        self.stats = None
        if ckpt_dir:
            stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
            if os.path.exists(stats_path):
                with open(stats_path, "rb") as f:
                    self.stats = pickle.load(f)
                print(f"[ACT-PPO] Loaded dataset stats from {stats_path}")

        # Only freeze the ResNet visual backbone (has BatchNorm, must stay eval)
        # Transformer encoder/decoder + action_head remain trainable
        if self.act_model.backbones is not None:
            for param in self.act_model.backbones.parameters():
                param.requires_grad = False
        # Also freeze input_proj for visual features (tied to frozen backbone)
        if hasattr(self.act_model, 'input_proj'):
            for param in self.act_model.input_proj.parameters():
                param.requires_grad = False

        # === New learnable heads ===
        # Value head: pool decoder hidden states -> V(s)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        # Log-std head: per-query log standard deviation
        # Maps decoder hidden states (chunk_size, hidden_dim) -> (chunk_size, state_dim)
        self.log_std_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.state_dim),
        )

        # Initialize log_std_head to output small std (close to deterministic)
        nn.init.constant_(self.log_std_head[-1].weight, 0.0)
        nn.init.constant_(self.log_std_head[-1].bias, np.log(0.01))

        # PPO config
        self.log_std_min = ppo_config.get("log_std_min", -5.0)
        self.log_std_max = ppo_config.get("log_std_max", 0.0)

        # Image normalization (same as ACTPolicy)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def train(self, mode=True):
        """Override train() to keep frozen ResNet backbone in eval mode.
        ResNet has BatchNorm that must stay in eval mode to preserve statistics."""
        super().train(mode)
        if self.act_model.backbones is not None:
            self.act_model.backbones.eval()  # Only keep backbone in eval
        return self

    def _get_decoder_hidden(self, qpos, image):
        """
        Run ACT forward pass to get decoder hidden states.
        Backbone runs without grad, transformer runs with grad.
        Returns: hs of shape (batch, chunk_size, hidden_dim)
        """
        image = self.normalize(image)
        bs, _ = qpos.shape

        # Inference mode: no VAE encoder, sample z=0
        mu = logvar = None
        latent_sample = torch.zeros(
            [bs, self.act_model.latent_dim], dtype=torch.float32
        ).to(qpos.device)
        latent_input = self.act_model.latent_out_proj(latent_sample)

        if self.act_model.backbones is not None:
            all_cam_features = []
            all_cam_pos = []
            # Backbone is frozen, run without grad
            with torch.no_grad():
                for cam_id, cam_name in enumerate(self.act_model.camera_names):
                    features, pos = self.act_model.backbones[0](image[:, cam_id])
                    features = features[0]
                    pos = pos[0]
                    all_cam_features.append(self.act_model.input_proj(features))
                    all_cam_pos.append(pos)

            # Detach backbone outputs so no grad flows back to frozen params
            all_cam_features = [f.detach() for f in all_cam_features]
            all_cam_pos = [p.detach() for p in all_cam_pos]

            # Transformer runs WITH grad (trainable)
            proprio_input = self.act_model.input_proj_robot_state(qpos)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.act_model.transformer(
                src, None, self.act_model.query_embed.weight,
                pos, latent_input, proprio_input,
                self.act_model.additional_pos_embed.weight,
            )[0]  # shape: (bs, chunk_size, hidden_dim)
        else:
            qpos_proj = self.act_model.input_proj_robot_state(qpos)
            env_state = self.act_model.input_proj_env_state(
                torch.zeros(bs, 7).to(qpos.device)
            )
            transformer_input = torch.cat([qpos_proj, env_state], axis=1)
            hs = self.act_model.transformer(
                transformer_input, None,
                self.act_model.query_embed.weight,
                self.act_model.pos.weight,
            )[0]

        # hs shape: (bs, chunk_size, hidden_dim) — [0] already removed the layer dim
        return hs

    def forward(self, qpos, image):
        """
        Forward pass for PPO.

        Args:
            qpos: (batch, state_dim) - normalized robot joint positions
            image: (batch, num_cam, C, H, W) - camera images

        Returns:
            mu: (batch, chunk_size, state_dim) - action means (trainable)
            std: (batch, chunk_size, state_dim) - action stds (trainable)
            value: (batch, 1) - state value estimate (trainable)
        """
        # Decoder hidden states (grad flows through transformer, not backbone)
        hs = self._get_decoder_hidden(qpos, image)

        # mu from trainable action head
        mu = self.act_model.action_head(hs)  # (bs, chunk_size, state_dim)

        # std from learnable log_std_head
        log_std = self.log_std_head(hs)  # (bs, chunk_size, state_dim)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)

        # value from learnable value_head (pool over chunk dimension)
        value_input = hs.mean(dim=1)  # (bs, hidden_dim)
        value = self.value_head(value_input)  # (bs, 1)

        return mu, std, value

    def get_action_and_value(self, qpos, image, deterministic=False):
        """
        Sample action from policy and compute log_prob + value.

        Returns:
            action: (batch, chunk_size, state_dim) - sampled action chunk
            log_prob: (batch,) - log probability of the sampled action
            value: (batch, 1) - state value
            mu: (batch, chunk_size, state_dim) - action means
            std: (batch, chunk_size, state_dim) - action stds
        """
        mu, std, value = self.forward(qpos, image)

        if deterministic:
            action = mu
        else:
            # Sample from diagonal Gaussian
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()

        # Compute log probability (sum over chunk_size and state_dim)
        log_prob = self._compute_log_prob(action, mu, std)

        return action, log_prob, value, mu, std

    def evaluate_actions(self, qpos, image, actions):
        """
        Evaluate given actions under current policy.
        Used during PPO update to compute new log_probs and values.

        Args:
            qpos: (batch, state_dim)
            image: (batch, num_cam, C, H, W)
            actions: (batch, chunk_size, state_dim)

        Returns:
            log_prob: (batch,)
            value: (batch, 1)
            entropy: (batch,)
        """
        mu, std, value = self.forward(qpos, image)
        log_prob = self._compute_log_prob(actions, mu, std)

        # Entropy of diagonal Gaussian
        dist = torch.distributions.Normal(mu, std)
        entropy = dist.entropy().sum(dim=-1).sum(dim=-1)  # sum over state_dim and chunk

        return log_prob, value, entropy

    @staticmethod
    def _compute_log_prob(actions, mu, std):
        """
        Compute log probability of actions under diagonal Gaussian.
        Sum over chunk_size and state_dim dimensions.

        Returns: (batch,) log probabilities
        """
        var = std ** 2
        log_prob = -0.5 * (
            ((actions - mu) ** 2) / var
            + torch.log(var)
            + np.log(2 * np.pi)
        )
        # Sum over state_dim and chunk_size
        return log_prob.mean(dim=-1).mean(dim=-1)  # (batch,)

    def get_trainable_params(self):
        """Return only the trainable parameters (everything except frozen backbone)."""
        return [p for p in self.parameters() if p.requires_grad]

    def pre_process(self, qpos_numpy):
        """Normalize qpos using dataset stats."""
        if self.stats is not None:
            return (qpos_numpy - self.stats["qpos_mean"]) / self.stats["qpos_std"]
        return qpos_numpy

    def post_process(self, action_tensor):
        """Denormalize actions using dataset stats."""
        if self.stats is not None:
            action_mean = torch.tensor(
                self.stats["action_mean"], dtype=torch.float32
            ).to(action_tensor.device)
            action_std = torch.tensor(
                self.stats["action_std"], dtype=torch.float32
            ).to(action_tensor.device)
            return action_tensor * action_std + action_mean
        return action_tensor


class ACTPPOReferenceModel(nn.Module):
    """
    Completely frozen ACT model for KL divergence computation.
    Independent copy of DETRVAE with all parameters frozen.
    Outputs mu_ref (from frozen action_head) + fixed sigma_ref.
    """

    def __init__(self, act_policy_config, ppo_config, RoboTwin_Config=None):
        """
        Build an independent frozen DETRVAE from the same pretrained checkpoint.

        Args:
            act_policy_config: same config used for ACTPPOModel
            ppo_config: PPO config dict (for ref_std)
            RoboTwin_Config: optional env config
        """
        super().__init__()
        self.ref_std = ppo_config.get("ref_std", 0.01)
        self.hidden_dim = act_policy_config.get("hidden_dim", 512)

        # Build a separate DETRVAE
        act_args = _config_dict_to_namespace(act_policy_config)
        model, _ = build_ACT_model_and_optimizer(act_policy_config, act_args)
        self.act_model = model

        # Load pretrained weights
        ckpt_dir = act_policy_config.get("ckpt_dir", "")
        if ckpt_dir:
            ckpt_path = os.path.join(ckpt_dir, "policy_last.ckpt")
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location="cpu")
                act_state_dict = {}
                for k, v in state_dict.items():
                    if k.startswith("model."):
                        act_state_dict[k[len("model."):]] = v
                if act_state_dict:
                    self.act_model.load_state_dict(act_state_dict, strict=True)
                else:
                    self.act_model.load_state_dict(state_dict, strict=True)
                print(f"[ACT-Ref] Loaded frozen reference weights from {ckpt_path}")

        # Freeze everything
        for param in self.act_model.parameters():
            param.requires_grad = False
        self.act_model.eval()

        # Image normalization (same as ACTPPOModel)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def train(self, mode=True):
        """Always stay in eval mode."""
        return self

    def get_distribution(self, qpos, image):
        """
        Get frozen reference policy distribution.

        Returns:
            ref_mu: (batch, chunk_size, state_dim) - frozen action means
            ref_std: (batch, chunk_size, state_dim) - fixed std
        """
        with torch.no_grad():
            image = self.normalize(image)
            bs, _ = qpos.shape

            latent_sample = torch.zeros(
                [bs, self.act_model.latent_dim], dtype=torch.float32
            ).to(qpos.device)
            latent_input = self.act_model.latent_out_proj(latent_sample)

            if self.act_model.backbones is not None:
                all_cam_features = []
                all_cam_pos = []
                for cam_id, cam_name in enumerate(self.act_model.camera_names):
                    features, pos = self.act_model.backbones[0](image[:, cam_id])
                    features = features[0]
                    pos = pos[0]
                    all_cam_features.append(self.act_model.input_proj(features))
                    all_cam_pos.append(pos)

                proprio_input = self.act_model.input_proj_robot_state(qpos)
                src = torch.cat(all_cam_features, axis=3)
                pos = torch.cat(all_cam_pos, axis=3)
                hs = self.act_model.transformer(
                    src, None, self.act_model.query_embed.weight,
                    pos, latent_input, proprio_input,
                    self.act_model.additional_pos_embed.weight,
                )[0]
            else:
                qpos_proj = self.act_model.input_proj_robot_state(qpos)
                env_state = self.act_model.input_proj_env_state(
                    torch.zeros(bs, 7).to(qpos.device)
                )
                transformer_input = torch.cat([qpos_proj, env_state], axis=1)
                hs = self.act_model.transformer(
                    transformer_input, None,
                    self.act_model.query_embed.weight,
                    self.act_model.pos.weight,
                )[0]

            # hs shape: (bs, chunk_size, hidden_dim) — [0] already removed the layer dim
            ref_mu = self.act_model.action_head(hs)
            ref_std = torch.full_like(ref_mu, self.ref_std)

        return ref_mu, ref_std
