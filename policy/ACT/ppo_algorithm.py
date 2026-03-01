"""
PPO Algorithm Core for ACT-RL fine-tuning.

Components:
- GAE (Generalized Advantage Estimation)
- PPO clipped surrogate loss
- Value function loss
- KL divergence penalty (Gaussian KL between current and reference policy)
- Combined loss computation
"""

import torch
import numpy as np


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """
    Compute Generalized Advantage Estimation (GAE).

    For chunk-level PPO, each "step" corresponds to one chunk execution.
    rewards[i] is the cumulative reward for chunk i.

    Args:
        rewards: (T,) tensor of rewards per chunk
        values: (T+1,) tensor of value estimates (includes bootstrap value)
        dones: (T,) tensor of done flags
        gamma: discount factor
        gae_lambda: GAE smoothing parameter

    Returns:
        advantages: (T,) tensor
        returns: (T,) tensor (advantages + values[:-1])
    """
    T = len(rewards)
    advantages = torch.zeros(T, device=rewards.device)
    gae = 0.0

    for t in reversed(range(T)):
        next_non_terminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * values[t + 1] * next_non_terminal - values[t]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns


def compute_gaussian_kl(mu1, std1, mu2, std2):
    """
    Compute KL divergence KL(p1 || p2) between two diagonal Gaussians.
    KL(N(mu1,std1) || N(mu2,std2)) per element, then sum over dims.

    Args:
        mu1, std1: current policy distribution params (batch, chunk, state_dim)
        mu2, std2: reference policy distribution params (batch, chunk, state_dim)

    Returns:
        kl: (batch,) KL divergence summed over chunk and state_dim
    """
    var1 = std1 ** 2
    var2 = std2 ** 2
    kl_element = (
        torch.log(std2 / std1)
        + (var1 + (mu1 - mu2) ** 2) / (2 * var2)
        - 0.5
    )
    # Sum over state_dim and chunk_size
    return kl_element.mean(dim=-1).mean(dim=-1)  # (batch,)


def compute_ppo_loss(
    log_probs_new,
    log_probs_old,
    advantages,
    clip_ratio,
):
    """
    Compute PPO clipped surrogate loss.

    Args:
        log_probs_new: (batch,) new log probs
        log_probs_old: (batch,) old log probs (from rollout)
        advantages: (batch,) normalized advantages
        clip_ratio: PPO clip epsilon (e.g. 0.1)

    Returns:
        policy_loss: scalar
        clip_fraction: fraction of clipped ratios (for logging)
    """
    ratio = torch.exp(log_probs_new - log_probs_old)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)

    surr1 = -advantages * ratio
    surr2 = -advantages * clipped_ratio
    policy_loss = torch.max(surr1, surr2).mean()

    # Logging metrics
    with torch.no_grad():
        clip_fraction = ((ratio - 1.0).abs() > clip_ratio).float().mean()
        approx_kl = (log_probs_old - log_probs_new).mean()

    return policy_loss, clip_fraction, approx_kl


def compute_value_loss(values, returns, old_values=None, value_clip=None):
    """
    Compute value function loss with optional clipping.

    Args:
        values: (batch, 1) predicted values
        returns: (batch,) target returns
        old_values: (batch, 1) old value predictions (for clipping)
        value_clip: clip range for value function (None = no clipping)

    Returns:
        value_loss: scalar
    """
    values = values.squeeze(-1)
    if old_values is not None and value_clip is not None:
        old_values = old_values.squeeze(-1)
        v_clipped = old_values + (values - old_values).clamp(
            -value_clip, value_clip
        )
        loss1 = (values - returns) ** 2
        loss2 = (v_clipped - returns) ** 2
        value_loss = 0.5 * torch.max(loss1, loss2).mean()
    else:
        value_loss = 0.5 * ((values - returns) ** 2).mean()
    return value_loss


def compute_total_loss(
    model,
    ref_model,
    qpos,
    image,
    actions,
    old_log_probs,
    advantages,
    returns,
    old_values,
    ppo_config,
):
    """
    Compute combined PPO training loss.

    L = L_PPO + c1 * L_VF + beta * L_KL - c2 * entropy

    Args:
        model: ACTPPOModel
        ref_model: ACTPPOReferenceModel
        qpos, image: observation batch
        actions: (batch, chunk_size, state_dim) sampled actions
        old_log_probs: (batch,) log probs from rollout
        advantages: (batch,) normalized advantages
        returns: (batch,) target returns
        old_values: (batch, 1) old value predictions
        ppo_config: dict with hyperparameters

    Returns:
        total_loss: scalar
        loss_info: dict of loss components for logging
    """
    clip_ratio = ppo_config.get("clip_ratio", 0.1)
    vf_coef = ppo_config.get("vf_coef", 0.5)
    kl_beta = ppo_config.get("kl_beta", 0.05)
    entropy_coef = ppo_config.get("entropy_coef", 0.0)
    value_clip = ppo_config.get("value_clip", 0.2)

    # Single forward pass to get mu, std, value (avoids redundant computation)
    mu_cur, std_cur, values = model.forward(qpos, image)

    # Compute log_prob and entropy from the distribution
    new_log_probs = model._compute_log_prob(actions, mu_cur, std_cur)
    dist = torch.distributions.Normal(mu_cur, std_cur)
    entropy = dist.entropy().mean(dim=-1).mean(dim=-1)  # (batch,)

    # PPO clipped loss
    policy_loss, clip_frac, approx_kl = compute_ppo_loss(
        new_log_probs, old_log_probs, advantages, clip_ratio
    )

    # Value loss
    value_loss = compute_value_loss(values, returns, old_values, value_clip)

    # KL divergence penalty (current policy mu/std vs frozen reference mu_ref/std_ref)
    ref_mu, ref_std = ref_model.get_distribution(qpos, image)
    kl_loss = compute_gaussian_kl(mu_cur, std_cur, ref_mu, ref_std).mean()

    # Entropy bonus
    entropy_loss = entropy.mean()

    # Total loss
    total_loss = (
        policy_loss
        + vf_coef * value_loss
        + kl_beta * kl_loss
        - entropy_coef * entropy_loss
    )

    loss_info = {
        "total_loss": total_loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "kl_loss": kl_loss.item(),
        "entropy": entropy_loss.item(),
        "clip_fraction": clip_frac.item(),
        "approx_kl": approx_kl.item(),
    }

    return total_loss, loss_info
