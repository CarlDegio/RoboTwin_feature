import torch.nn as nn
import os
import torch
import numpy as np
import pickle
import json
from torch.nn import functional as F
import torchvision.transforms as transforms

try:
    from detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
    from action_tokenizer import ActionTokenizer
except:
    from .detr.main import (
        build_ACT_model_and_optimizer,
        build_CNNMLP_model_and_optimizer,
    )
    from .action_tokenizer import ActionTokenizer
import IPython

e = IPython.embed


class ACTPolicy(nn.Module):

    def __init__(self, args_override, RoboTwin_Config=None):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override, RoboTwin_Config)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override["kl_weight"]
        self.aux_weight = args_override.get("aux_weight", 0.5)
        self.n_bins = args_override.get("n_bins", 256)

        # Load tokenizer for soft-argmax auxiliary loss
        tokenizer_stats_path = args_override.get("tokenizer_stats_path", None)
        if tokenizer_stats_path is not None:
            self.tokenizer = ActionTokenizer(n_bins=self.n_bins, stats_path=tokenizer_stats_path)
            # Register bin centers as buffer for soft-argmax
            self.register_buffer("bin_centers", self.tokenizer.bin_centers_torch.clone())
        else:
            self.tokenizer = None
            self.register_buffer("bin_centers", torch.linspace(-1.0, 1.0, self.n_bins + 1)[:-1].add(1.0 / self.n_bins))

        print(f"KL Weight {self.kl_weight}, Aux Weight {self.aux_weight}, N_bins {self.n_bins}")

    def __call__(self, qpos, image, actions=None, is_pad=None, action_tokens=None, delta_actions=None):
        """
        Training:
            qpos: (batch, state_dim)
            image: (batch, num_cam, C, H, W)
            actions: (batch, seq, state_dim) — continuous delta actions (for CVAE encoder)
            is_pad: (batch, seq) — bool
            action_tokens: (batch, seq, state_dim) — LongTensor discrete targets
            delta_actions: (batch, seq, state_dim) — continuous delta actions (for aux loss, normalized)
        Inference:
            Only qpos and image provided. Returns decoded delta action logits.
        """
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)

        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            action_tokens = action_tokens[:, :self.model.num_queries]
            delta_actions = delta_actions[:, :self.model.num_queries]

            # Forward: CVAE encoder sees continuous delta actions, decoder outputs logits
            action_logits, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            # action_logits: (batch, num_queries, state_dim, n_bins)

            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            loss_dict = dict()

            # Cross-entropy loss on discrete tokens
            bs, seq, sdim, nbins = action_logits.shape
            # Reshape for CE: (batch*seq*state_dim, n_bins) vs (batch*seq*state_dim,)
            logits_flat = action_logits.reshape(-1, nbins)
            tokens_flat = action_tokens.reshape(-1)
            ce_all = F.cross_entropy(logits_flat, tokens_flat, reduction="none")
            # ce_all: (batch*seq*state_dim,) -> (batch, seq, state_dim)
            ce_all = ce_all.reshape(bs, seq, sdim)
            # Mask padded positions
            pad_mask = ~is_pad.unsqueeze(-1).expand_as(ce_all)  # (batch, seq, state_dim)
            ce_loss = (ce_all * pad_mask).sum() / pad_mask.sum().clamp(min=1)

            # Soft-argmax auxiliary L1 loss
            probs = F.softmax(action_logits, dim=-1)  # (batch, seq, state_dim, n_bins)
            bin_centers = self.bin_centers.to(action_logits.device)  # (n_bins,)
            soft_pred = (probs * bin_centers).sum(dim=-1)  # (batch, seq, state_dim)
            aux_l1 = F.l1_loss(soft_pred, delta_actions, reduction="none")
            aux_loss = (aux_l1 * pad_mask).sum() / pad_mask.sum().clamp(min=1)

            loss_dict["ce"] = ce_loss
            loss_dict["kl"] = total_kld[0]
            loss_dict["aux"] = aux_loss
            loss_dict["loss"] = ce_loss + self.kl_weight * loss_dict["kl"] + self.aux_weight * aux_loss
            return loss_dict
        else:  # inference time
            action_logits, (_, _) = self.model(qpos, image, env_state)
            # action_logits: (batch, num_queries, state_dim, n_bins)
            return action_logits

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):

    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model  # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None  # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict["mse"] = mse
            loss_dict["loss"] = loss_dict["mse"]
            return loss_dict
        else:  # inference time
            a_hat = self.model(qpos, image, env_state)  # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACT:
    """Inference wrapper for ACT-DT policy with discrete action tokens."""

    def __init__(self, usr_args, RoboTwin_Config):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load normalization stats for qpos
        ckpt_dir = usr_args["ckpt_dir"]
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        with open(stats_path, "rb") as f:
            stats = pickle.load(f)
        self.pre_process = lambda qpos: (qpos - stats["qpos_mean"]) / stats["qpos_std"]

        # Load tokenizer
        tokenizer_stats_path = usr_args.get("tokenizer_stats_path", os.path.join(ckpt_dir, "tokenizer_stats.json"))
        n_bins = usr_args.get("n_bins", 256)
        self.tokenizer = ActionTokenizer(n_bins=n_bins, stats_path=tokenizer_stats_path)

        # Build policy
        policy_config = {
            "lr": usr_args.get("lr", 5e-5),
            "num_queries": usr_args.get("chunk_size", 50),
            "kl_weight": usr_args.get("kl_weight", 10),
            "hidden_dim": usr_args.get("hidden_dim", 512),
            "dim_feedforward": usr_args.get("dim_feedforward", 3200),
            "lr_backbone": usr_args.get("lr_backbone", 1e-5),
            "backbone": usr_args.get("backbone", "resnet18"),
            "enc_layers": usr_args.get("enc_layers", 4),
            "dec_layers": usr_args.get("dec_layers", 4),
            "nheads": usr_args.get("nheads", 8),
            "camera_names": usr_args.get("camera_names", ["head_cam", "left_cam", "right_cam"]),
            "state_dim": usr_args.get("state_dim", 14),
            "n_bins": n_bins,
            "aux_weight": usr_args.get("aux_weight", 0.5),
            "tokenizer_stats_path": tokenizer_stats_path,
        }
        self.policy = ACTPolicy(policy_config, RoboTwin_Config)

        # Load checkpoint
        ckpt_name = usr_args.get("ckpt_name", "policy_best.ckpt")
        ckpt_path = os.path.join(ckpt_dir, ckpt_name)
        loading_status = self.policy.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        print(f"Loaded checkpoint from {ckpt_path}: {loading_status}")
        self.policy.to(self.device)
        self.policy.eval()

        self.state_dim = usr_args.get("state_dim", 14)
        self.num_queries = usr_args.get("chunk_size", 50)
        self.t = 0

    def get_action(self, obs):
        """
        Get actions from observation. Decodes full chunk of discrete tokens
        into absolute actions via delta accumulation.

        Returns: list of np.array actions (each shape (state_dim,))
        """
        # Normalize qpos
        qpos = np.array(obs["qpos"], dtype=np.float32)
        raw_qpos = qpos.copy()  # keep unnormalized for delta accumulation
        qpos_normalized = self.pre_process(qpos)
        qpos_tensor = torch.from_numpy(qpos_normalized).float().to(self.device).unsqueeze(0)

        # Prepare images
        curr_images = []
        camera_names = ["head_cam", "left_cam", "right_cam"]
        for cam_name in camera_names:
            curr_images.append(obs[cam_name])
        curr_image = np.stack(curr_images, axis=0)
        curr_image = torch.from_numpy(curr_image).float().to(self.device).unsqueeze(0)

        with torch.no_grad():
            action_logits = self.policy(qpos_tensor, curr_image)
            # action_logits: (1, num_queries, state_dim, n_bins)
            # Argmax decoding
            token_indices = action_logits.argmax(dim=-1)  # (1, num_queries, state_dim)
            token_indices = token_indices.squeeze(0)  # (num_queries, state_dim)

        # Decode tokens to continuous delta actions
        delta_actions = self.tokenizer.decode(token_indices.cpu().numpy())  # (num_queries, state_dim)

        # Accumulate deltas to get absolute actions
        # delta[0] = action[0] - qpos, so action[0] = qpos + delta[0]
        # delta[t] = action[t] - action[t-1], so action[t] = action[t-1] + delta[t]
        absolute_actions = np.zeros_like(delta_actions)
        absolute_actions[0] = raw_qpos + delta_actions[0]
        for t in range(1, len(delta_actions)):
            absolute_actions[t] = absolute_actions[t - 1] + delta_actions[t]

        # Return as list of actions
        actions = [absolute_actions[t] for t in range(len(absolute_actions))]
        return actions[:30]
