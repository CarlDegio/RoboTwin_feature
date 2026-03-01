"""
ACT-PPO Training Script: PPO fine-tuning of pretrained ACT policy.

Usage:
    bash train_ppo.sh stack_bowls_two demo_clean 50 0 0

Design:
- Load pretrained ACT weights as PPO policy network
- Freeze visual encoder + ACT encoder-decoder
- Train only value_head + log_std_head
- Chunk-level PPO: execute full action chunk as one RL action
- Sparse reward: +1.0 success, -1.0 failure
- KL penalty against frozen reference ACT distribution
- GAE advantage estimation
"""

import os
import sys
import yaml
import torch
import numpy as np
import argparse
import importlib
import traceback
import time
import psutil
from copy import deepcopy
from argparse import Namespace
from collections import defaultdict

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../description/utils"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../script"))

from act_ppo_model import ACTPPOModel, ACTPPOReferenceModel
from ppo_algorithm import compute_gae, compute_total_loss
from ppo_rollout import RolloutBuffer, ChunkTransition, encode_obs, obs_to_tensors


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_ppo_config(ppo_config_path):
    with open(ppo_config_path, "r") as f:
        return yaml.safe_load(f)


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except Exception:
        raise SystemExit(f"No Task: {task_name}")


def setup_env_args(task_name, task_config_name, usr_args):
    """Load task config and setup environment arguments (mirrors eval_policy.py)."""
    from envs import CONFIGS_PATH

    with open(f"./task_config/{task_config_name}.yml", "r", encoding="utf-8") as f:
        args = yaml.safe_load(f)

    args["task_name"] = task_name
    args["task_config"] = task_config_name
    args["eval_mode"] = True
    args["render_freq"] = 0
    args["eval_video_save_dir"] = None

    # Embodiment config
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        _embodiment_types = yaml.safe_load(f)

    with open(CONFIGS_PATH + "_camera_config.yml", "r", encoding="utf-8") as f:
        _camera_config = yaml.safe_load(f)

    head_camera_type = args["camera"]["head_camera_type"]
    args["head_camera_h"] = _camera_config[head_camera_type]["h"]
    args["head_camera_w"] = _camera_config[head_camera_type]["w"]

    def get_embodiment_file(etype):
        return _embodiment_types[etype]["file_path"]

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False

    from script.eval_policy import get_embodiment_config
    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    # Merge user args
    for k, v in usr_args.items():
        if k not in args:
            args[k] = v

    return args


def collect_rollouts(
    model, TASK_ENV, env_args, ppo_config, rollout_buffer, seed_start, device="cuda",
):
    """
    Collect chunk-level rollouts by interacting with the RoboTwin environment.

    Each episode:
    1. Reset env
    2. Loop: observe -> sample action chunk -> execute full chunk -> record
    3. Assign sparse reward at episode end

    Returns:
        seed_next: next seed to use
        episode_stats: dict with success_rate, avg_return, etc.
    """
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

    num_episodes = ppo_config.get("num_episodes_per_iter", 4)
    max_chunks = ppo_config.get("max_chunks_per_episode", 60)
    success_reward = ppo_config.get("success_reward", 1.0)
    failure_reward = ppo_config.get("failure_reward", -1.0)
    chunk_size = model.chunk_size

    rollout_buffer.clear()
    now_seed = seed_start
    episodes_collected = 0
    successes = 0
    total_return = 0.0

    model.eval()

    while episodes_collected < num_episodes:
        # Try to setup a valid environment seed
        env_setup_ok = False
        for _ in range(50):  # max attempts to find valid seed
            try:
                TASK_ENV.setup_demo(
                    now_ep_num=episodes_collected, seed=now_seed,
                    is_test=True, **env_args,
                )
                episode_info = TASK_ENV.play_once()
                plan_ok = TASK_ENV.plan_success and TASK_ENV.check_success()
                TASK_ENV.close_env()

                if plan_ok:
                    env_setup_ok = True
                    break
                else:
                    now_seed += 1
            except (UnStableError, Exception) as e:
                print(f"    [Seed {now_seed}] Exception: {type(e).__name__}: {e}")
                TASK_ENV.close_env()
                now_seed += 1

        if not env_setup_ok:
            print(f"[Rollout] Could not find valid seed after 50 attempts, skipping")
            now_seed += 1
            continue

        # Setup env for actual rollout
        TASK_ENV.setup_demo(
            now_ep_num=episodes_collected, seed=now_seed,
            is_test=True, **env_args,
        )
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(
            env_args["task_name"], episode_info_list, 1,
        )
        instruction = np.random.choice(results[0]["unseen"])
        TASK_ENV.set_instruction(instruction=instruction)

        # Collect episode transitions
        episode_transitions = []
        done = False
        chunk_count = 0

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim and chunk_count < max_chunks:
            observation = TASK_ENV.get_obs()
            obs_dict = encode_obs(observation)
            qpos, images = obs_to_tensors(obs_dict, model, device)

            # Sample action chunk from policy
            with torch.no_grad():
                action, log_prob, value, mu, std = model.get_action_and_value(
                    qpos, images, deterministic=False,
                )

            # Denormalize action for environment execution
            action_denorm = model.post_process(action)  # (1, chunk_size, state_dim)
            action_np = action_denorm.squeeze(0).cpu().numpy()  # (chunk_size, state_dim)

            # Store transition (reward assigned later)
            trans = ChunkTransition(
                qpos=qpos.squeeze(0).cpu().numpy(),  # normalized qpos
                images=np.stack([
                    obs_dict["head_cam"], obs_dict["left_cam"], obs_dict["right_cam"],
                ], axis=0),
                action=action.squeeze(0).cpu().numpy(),  # normalized action
                reward=0.0,  # placeholder
                done=False,
                value=value.item(),
                log_prob=log_prob.item(),
            )
            episode_transitions.append(trans)

            # Execute full chunk in environment
            for step_i in range(chunk_size):
                TASK_ENV.take_action(action_np[step_i])
                if TASK_ENV.eval_success:
                    done = True
                    break
                if TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
                    done = True
                    break

            chunk_count += 1
            if done:
                break

        # Assign sparse reward to last transition
        if TASK_ENV.eval_success:
            episode_reward = success_reward
            successes += 1
        else:
            episode_reward = failure_reward

        if episode_transitions:
            episode_transitions[-1].reward = episode_reward
            episode_transitions[-1].done = True

            # Add all transitions to buffer
            for trans in episode_transitions:
                rollout_buffer.add(trans)

        total_return += episode_reward
        episodes_collected += 1
        TASK_ENV.close_env()
        now_seed += 1

        status = "Success" if TASK_ENV.eval_success else "Fail"
        print(
            f"  [Rollout] Episode {episodes_collected}/{num_episodes}: "
            f"{status}, chunks={chunk_count}, reward={episode_reward:.1f}"
        )

    stats = {
        "success_rate": successes / max(episodes_collected, 1),
        "avg_return": total_return / max(episodes_collected, 1),
        "episodes": episodes_collected,
        "total_chunks": len(rollout_buffer),
    }
    return now_seed, stats


def ppo_update(model, ref_model, optimizer, rollout_buffer, ppo_config, device="cuda"):
    """
    Perform PPO update epochs on collected rollout data.

    Steps:
    1. Convert buffer to tensors
    2. Compute GAE advantages and returns
    3. For each epoch, shuffle and split into minibatches
    4. Compute PPO loss and update

    Returns:
        update_info: dict of averaged loss metrics
    """
    gamma = ppo_config.get("gamma", 0.995)
    gae_lambda = ppo_config.get("gae_lambda", 0.95)
    update_epochs = ppo_config.get("update_epochs", 4)
    n_minibatches = ppo_config.get("n_minibatches", 4)
    max_grad_norm = ppo_config.get("max_grad_norm", 0.5)
    normalize_adv = ppo_config.get("normalize_advantages", True)

    data = rollout_buffer.get_tensors(device)
    N = len(data["rewards"])

    if N == 0:
        return {"total_loss": 0.0}

    # Compute bootstrap value for last state
    # Use 0 for terminal states (sparse reward, episode ends)
    values_with_bootstrap = torch.cat([
        data["values"],
        torch.zeros(1, device=device),  # bootstrap = 0 (episode always ends)
    ])

    # Compute GAE
    advantages, returns = compute_gae(
        data["rewards"], values_with_bootstrap, data["dones"],
        gamma, gae_lambda,
    )

    # Normalize advantages
    if normalize_adv and len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    old_values = data["values"].unsqueeze(-1)  # (N, 1)

    # PPO update epochs
    all_metrics = defaultdict(list)
    batch_size = max(N // n_minibatches, 1)

    for epoch in range(update_epochs):
        indices = torch.randperm(N, device=device)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            mb_idx = indices[start:end]

            mb_qpos = data["qpos"][mb_idx]
            mb_images = data["images"][mb_idx]
            mb_actions = data["actions"][mb_idx]
            mb_old_log_probs = data["log_probs"][mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_returns = returns[mb_idx]
            mb_old_values = old_values[mb_idx]

            total_loss, loss_info = compute_total_loss(
                model, ref_model,
                mb_qpos, mb_images, mb_actions,
                mb_old_log_probs, mb_advantages, mb_returns,
                mb_old_values, ppo_config,
            )

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.get_trainable_params(), max_grad_norm,
            )
            optimizer.step()

            for k, v in loss_info.items():
                all_metrics[k].append(v)

    # Average metrics
    update_info = {k: np.mean(v) for k, v in all_metrics.items()}
    return update_info


def evaluate_policy(model, TASK_ENV, env_args, ppo_config, seed_start, device="cuda"):
    """
    Evaluate current policy without exploration (deterministic).
    Returns success_rate over eval episodes.
    """
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

    num_eval = 20
    chunk_size = model.chunk_size
    max_chunks = ppo_config.get("max_chunks_per_episode", 60)
    now_seed = seed_start
    successes = 0
    evaluated = 0

    model.eval()

    for ep in range(num_eval):
        # Find valid seed
        env_ok = False
        for _ in range(20):
            try:
                TASK_ENV.setup_demo(
                    now_ep_num=ep, seed=now_seed, is_test=True, **env_args,
                )
                episode_info = TASK_ENV.play_once()
                plan_ok = TASK_ENV.plan_success and TASK_ENV.check_success()
                TASK_ENV.close_env()
                if plan_ok:
                    env_ok = True
                    break
                now_seed += 1
            except (UnStableError, Exception):
                TASK_ENV.close_env()
                now_seed += 1

        if not env_ok:
            now_seed += 1
            continue

        TASK_ENV.setup_demo(
            now_ep_num=ep, seed=now_seed, is_test=True, **env_args,
        )
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(
            env_args["task_name"], episode_info_list, 1,
        )
        instruction = np.random.choice(results[0]["unseen"])
        TASK_ENV.set_instruction(instruction=instruction)

        chunk_count = 0
        done = False
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim and chunk_count < max_chunks:
            observation = TASK_ENV.get_obs()
            obs_dict = encode_obs(observation)
            qpos, images = obs_to_tensors(obs_dict, model, device)

            with torch.no_grad():
                action, _, _, _, _ = model.get_action_and_value(
                    qpos, images, deterministic=True,
                )
            action_denorm = model.post_process(action)
            action_np = action_denorm.squeeze(0).cpu().numpy()

            for step_i in range(chunk_size):
                TASK_ENV.take_action(action_np[step_i])
                if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
                    done = True
                    break
            chunk_count += 1
            if done:
                break

        if TASK_ENV.eval_success:
            successes += 1
        evaluated += 1
        TASK_ENV.close_env()
        now_seed += 1

    success_rate = successes / max(evaluated, 1)
    print(f"  [Eval] {successes}/{evaluated} = {success_rate*100:.1f}%")
    return success_rate


def main(args):
    """Main PPO training loop."""
    task_name = args.task_name
    task_config_name = args.task_config
    expert_data_num = args.expert_data_num
    seed = args.seed
    gpu_id = args.gpu_id
    ppo_config_path = args.ppo_config

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    # Load PPO config
    ppo_config = load_ppo_config(ppo_config_path)
    ppo_config["device"] = device
    set_seed(ppo_config.get("seed", seed))

    # Must run from project root for task_config / env paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "../.."))
    os.chdir(project_root)

    from script.test_render import Sapien_TEST
    Sapien_TEST()

    # ACT policy config (matches train.sh / deploy_policy.yml)
    # Paths relative to project root
    act_ckpt_dir = f"policy/ACT/act_ckpt/act-{task_name}/{task_config_name}-{expert_data_num}"
    act_config = {
        "kl_weight": 10,
        "chunk_size": 50,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "lr": 5e-5,
        "lr_backbone": 1e-5,
        "backbone": "resnet18",
        "enc_layers": 4,
        "dec_layers": 4,
        "nheads": 8,
        "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
        "state_dim": 14,
        "ckpt_dir": act_ckpt_dir,
        "device": device,
    }

    # PPO output directory
    ppo_ckpt_dir = (
        f"policy/ACT/act_ckpt/act_ppo-{task_name}/{task_config_name}-{expert_data_num}"
    )
    os.makedirs(ppo_ckpt_dir, exist_ok=True)

    print("=" * 60)
    print(f"ACT-PPO Training: {task_name}")
    print(f"ACT checkpoint: {act_ckpt_dir}")
    print(f"PPO output: {ppo_ckpt_dir}")
    print("=" * 60)

    # Build ACT-PPO model
    model = ACTPPOModel(act_config, ppo_config)
    model.to(device)
    print(f"[ACT-PPO] Trainable params: "
          f"{sum(p.numel() for p in model.get_trainable_params())}")

    # Reference model (independent frozen ACT for KL)
    ref_model = ACTPPOReferenceModel(act_config, ppo_config)
    ref_model.to(device)

    # Optimizer (only trainable params)
    optimizer = torch.optim.Adam(
        model.get_trainable_params(),
        lr=ppo_config.get("lr", 3e-5),
    )

    # Setup environment (already chdir'd to project root above)
    TASK_ENV = class_decorator(task_name)
    env_args = setup_env_args(task_name, task_config_name, {
        "ckpt_setting": f"{task_config_name}-{expert_data_num}",
        "seed": seed,
    })
    # Remove 'seed' to avoid "multiple values" error when passing seed explicitly
    env_args.pop("seed", None)

    # Rollout buffer
    rollout_buffer = RolloutBuffer()

    # Training loop
    total_iters = ppo_config.get("total_iterations", 500)
    eval_freq = ppo_config.get("eval_freq", 10)
    save_freq = ppo_config.get("save_freq", 50)
    log_freq = ppo_config.get("log_freq", 1)

    now_seed = 100000 * (1 + seed)
    eval_seed = 200000 * (1 + seed)
    best_success_rate = 0.0

    # Log file
    log_path = os.path.join(ppo_ckpt_dir, "training_log.txt")
    log_file = open(log_path, "w")

    print(f"\n[ACT-PPO] Starting training for {total_iters} iterations\n")

    for iteration in range(1, total_iters + 1):
        iter_start = time.time()

        # === 1. Collect rollouts ===
        print(f"[Iter {iteration}/{total_iters}] Collecting rollouts...")
        now_seed, rollout_stats = collect_rollouts(
            model, TASK_ENV, env_args, ppo_config,
            rollout_buffer, now_seed, device,
        )

        if len(rollout_buffer) == 0:
            print(f"  No transitions collected, skipping update")
            continue

        # === 2. PPO update ===
        model.train()
        update_info = ppo_update(
            model, ref_model, optimizer,
            rollout_buffer, ppo_config, device,
        )
        model.eval()

        iter_time = time.time() - iter_start

        # === 3. Logging ===
        if iteration % log_freq == 0:
            # System memory
            mem = psutil.virtual_memory()
            mem_used_gb = mem.used / (1024 ** 3)
            mem_total_gb = mem.total / (1024 ** 3)
            # GPU memory
            gpu_mem_used_gb = torch.cuda.memory_reserved(device) / (1024 ** 3)
            gpu_mem_total_gb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 3)

            log_msg = (
                f"[Iter {iteration}] "
                f"success={rollout_stats['success_rate']:.2f} "
                f"return={rollout_stats['avg_return']:.2f} "
                f"chunks={rollout_stats['total_chunks']} "
                f"ploss={update_info.get('policy_loss', 0):.4f} "
                f"vloss={update_info.get('value_loss', 0):.4f} "
                f"kl={update_info.get('kl_loss', 0):.4f} "
                f"clip={update_info.get('clip_fraction', 0):.3f} "
                f"entropy={update_info.get('entropy', 0):.4f} "
                f"time={iter_time:.1f}s "
                f"RAM={mem_used_gb:.1f}/{mem_total_gb:.1f}GB "
                f"VRAM={gpu_mem_used_gb:.1f}/{gpu_mem_total_gb:.1f}GB"
            )
            print(log_msg)
            log_file.write(log_msg + "\n")
            log_file.flush()

        # === 4. Evaluation ===
        if iteration % eval_freq == 0:
            print(f"[Iter {iteration}] Evaluating...")
            eval_sr = evaluate_policy(
                model, TASK_ENV, env_args, ppo_config,
                eval_seed, device,
            )
            eval_msg = f"[Eval Iter {iteration}] success_rate={eval_sr:.2f}"
            print(eval_msg)
            log_file.write(eval_msg + "\n")
            log_file.flush()

            if eval_sr > best_success_rate:
                best_success_rate = eval_sr
                ckpt_path = os.path.join(ppo_ckpt_dir, "policy_best.ckpt")
                torch.save(model.state_dict(), ckpt_path)
                print(f"  New best! Saved to {ckpt_path}")

        # === 5. Checkpoint ===
        if iteration % save_freq == 0:
            ckpt_path = os.path.join(
                ppo_ckpt_dir, f"policy_iter_{iteration}.ckpt",
            )
            torch.save(model.state_dict(), ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    # Save final checkpoint
    final_path = os.path.join(ppo_ckpt_dir, "policy_last.ckpt")
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Best success rate: {best_success_rate:.2f}")
    print(f"Final checkpoint: {final_path}")

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ACT-PPO Training")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--task_config", type=str, required=True)
    parser.add_argument("--expert_data_num", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument(
        "--ppo_config", type=str,
        default=os.path.join(os.path.dirname(__file__), "ppo_config.yml"),
    )
    args = parser.parse_args()
    main(args)
