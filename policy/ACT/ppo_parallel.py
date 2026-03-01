"""
Parallel episode collection and evaluation for ACT-PPO.

Uses torch.multiprocessing with spawn context to run multiple episodes
concurrently. Each worker process creates its own model (inference-only)
and environment instance.

Usage:
    from ppo_parallel import parallel_collect_rollouts, parallel_evaluate_policy
"""

import os
import sys
import traceback
import numpy as np
import torch
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# Project paths (absolute, so workers can set up imports after spawn)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "../.."))

_SYS_PATHS = [
    _PROJECT_ROOT,
    os.path.join(_SCRIPT_DIR, ".."),
    _SCRIPT_DIR,
    os.path.join(_PROJECT_ROOT, "description/utils"),
    os.path.join(_PROJECT_ROOT, "script"),
]


# ---------------------------------------------------------------------------
# Worker initialisation helpers (called inside each subprocess)
# ---------------------------------------------------------------------------

def _worker_setup_env():
    """Set up sys.path, working directory, and SAPIEN rendering in a worker."""
    for p in _SYS_PATHS:
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(_PROJECT_ROOT)

    from script.test_render import Sapien_TEST
    Sapien_TEST()


def _worker_build_model(model_state_dict, act_config, ppo_config, device):
    """Create an ACTPPOModel in eval / no-grad mode and load weights."""
    from act_ppo_model import ACTPPOModel

    model = ACTPPOModel(act_config, ppo_config)
    model.load_state_dict(model_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _worker_build_env(task_name, task_config_name, env_args):
    """Create a TASK_ENV instance for this worker."""
    import importlib
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        TASK_ENV = env_class()
    except Exception:
        raise RuntimeError(f"Cannot instantiate env for task: {task_name}")
    return TASK_ENV


# ---------------------------------------------------------------------------
# Collect worker
# ---------------------------------------------------------------------------

def _worker_collect_episodes(args):
    """
    Collect rollout episodes in a worker process.

    Args (packed tuple):
        worker_id, model_state_dict, act_config, ppo_config, env_args,
        task_name, task_config_name, seed_start, num_episodes, device_str

    Returns:
        (transitions_list, stats_dict)
        transitions_list: list of dicts, each with keys
            qpos, images, action, reward, done, value, log_prob
        stats_dict: {successes, episodes, total_chunks, total_return}
    """
    (
        worker_id, model_state_dict, act_config, ppo_config, env_args,
        task_name, task_config_name, seed_start, num_episodes, device_str,
    ) = args

    try:
        _worker_setup_env()

        from ppo_rollout import encode_obs, obs_to_tensors
        from envs.utils.create_actor import UnStableError
        from generate_episode_instructions import generate_episode_descriptions

        device = torch.device(device_str)
        model = _worker_build_model(model_state_dict, act_config, ppo_config, device)
        TASK_ENV = _worker_build_env(task_name, task_config_name, env_args)

        max_chunks = ppo_config.get("max_chunks_per_episode", 20)
        success_reward = ppo_config.get("success_reward", 1.0)
        failure_reward = ppo_config.get("failure_reward", -1.0)
        chunk_size = model.chunk_size

        all_transitions = []
        now_seed = seed_start
        episodes_collected = 0
        successes = 0
        total_return = 0.0

        while episodes_collected < num_episodes:
            # Find valid seed
            env_setup_ok = False
            for _ in range(50):
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
                    print(f"    [Worker {worker_id}][Seed {now_seed}] {type(e).__name__}: {e}")
                    TASK_ENV.close_env()
                    now_seed += 1

            if not env_setup_ok:
                print(f"[Worker {worker_id}] Could not find valid seed after 50 attempts")
                now_seed += 1
                continue

            # Setup env for rollout
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

            # Collect transitions
            episode_transitions = []
            done = False
            chunk_count = 0

            while TASK_ENV.take_action_cnt < TASK_ENV.step_lim and chunk_count < max_chunks:
                observation = TASK_ENV.get_obs()
                obs_dict = encode_obs(observation)
                qpos, images = obs_to_tensors(obs_dict, model, device)

                with torch.no_grad():
                    action, log_prob, value, mu, std = model.get_action_and_value(
                        qpos, images, deterministic=False,
                    )

                action_denorm = model.post_process(action)
                action_np = action_denorm.squeeze(0).cpu().numpy()

                # Store as serializable dict
                trans = {
                    "qpos": qpos.squeeze(0).cpu().numpy(),
                    "images": np.stack([
                        obs_dict["head_cam"], obs_dict["left_cam"], obs_dict["right_cam"],
                    ], axis=0),
                    "action": action.squeeze(0).cpu().numpy(),
                    "reward": 0.0,
                    "done": False,
                    "value": value.item(),
                    "log_prob": log_prob.item(),
                }
                episode_transitions.append(trans)

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

            # Sparse reward
            if TASK_ENV.eval_success:
                episode_reward = success_reward
                successes += 1
            else:
                episode_reward = failure_reward

            if episode_transitions:
                episode_transitions[-1]["reward"] = episode_reward
                episode_transitions[-1]["done"] = True
                all_transitions.extend(episode_transitions)

            total_return += episode_reward
            episodes_collected += 1
            TASK_ENV.close_env()
            now_seed += 1

            status = "Success" if TASK_ENV.eval_success else "Fail"
            print(
                f"  [Worker {worker_id}] Episode {episodes_collected}/{num_episodes}: "
                f"{status}, chunks={chunk_count}, reward={episode_reward:.1f}"
            )

        stats = {
            "successes": successes,
            "episodes": episodes_collected,
            "total_chunks": len(all_transitions),
            "total_return": total_return,
        }
        return all_transitions, stats

    except Exception as e:
        print(f"[Worker {worker_id}] FATAL: {traceback.format_exc()}")
        return [], {"successes": 0, "episodes": 0, "total_chunks": 0, "total_return": 0.0}


# ---------------------------------------------------------------------------
# Eval worker
# ---------------------------------------------------------------------------

def _worker_eval_episodes(args):
    """
    Evaluate episodes in a worker process (deterministic inference).

    Args (packed tuple):
        worker_id, model_state_dict, act_config, ppo_config, env_args,
        task_name, task_config_name, seed_start, num_episodes, device_str

    Returns:
        (successes, evaluated_count)
    """
    (
        worker_id, model_state_dict, act_config, ppo_config, env_args,
        task_name, task_config_name, seed_start, num_episodes, device_str,
    ) = args

    try:
        _worker_setup_env()

        from ppo_rollout import encode_obs, obs_to_tensors
        from envs.utils.create_actor import UnStableError
        from generate_episode_instructions import generate_episode_descriptions

        device = torch.device(device_str)
        model = _worker_build_model(model_state_dict, act_config, ppo_config, device)
        TASK_ENV = _worker_build_env(task_name, task_config_name, env_args)

        chunk_size = model.chunk_size
        max_chunks = ppo_config.get("max_chunks_per_episode", 60)
        now_seed = seed_start
        successes = 0
        evaluated = 0

        for ep in range(num_episodes):
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

        print(f"  [Worker {worker_id}] Eval: {successes}/{evaluated}")
        return successes, evaluated

    except Exception as e:
        print(f"[Worker {worker_id}] FATAL: {traceback.format_exc()}")
        return 0, 0


# ---------------------------------------------------------------------------
# Orchestrators (called from main process)
# ---------------------------------------------------------------------------

def _divide_episodes(total, num_workers):
    """Divide total episodes among workers as evenly as possible."""
    base = total // num_workers
    remainder = total % num_workers
    counts = []
    for i in range(num_workers):
        counts.append(base + (1 if i < remainder else 0))
    return counts


def parallel_collect_rollouts(
    model, act_config, ppo_config, env_args,
    task_name, task_config_name,
    rollout_buffer, seed_start, device, num_workers,
):
    """
    Parallel version of collect_rollouts.

    Spawns num_workers processes, each collecting a subset of episodes.
    Merges all transitions into rollout_buffer.

    Returns:
        seed_next: next seed to use
        stats: dict with success_rate, avg_return, episodes, total_chunks
    """
    from ppo_rollout import ChunkTransition

    num_episodes = ppo_config.get("num_episodes_per_iter", 4)
    episode_counts = _divide_episodes(num_episodes, num_workers)

    # Prepare model state dict on CPU for pickling
    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    device_str = str(device)

    # Assign non-overlapping seed ranges (generous spacing to handle retries)
    seed_spacing = 1000  # each worker may try up to 50 seeds per episode
    worker_args = []
    for i, count in enumerate(episode_counts):
        if count == 0:
            continue
        worker_seed = seed_start + i * seed_spacing
        worker_args.append((
            i, model_state_dict, act_config, ppo_config, env_args,
            task_name, task_config_name, worker_seed, count, device_str,
        ))

    rollout_buffer.clear()

    # Spawn workers
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(worker_args)) as pool:
        results = pool.map(_worker_collect_episodes, worker_args)

    # Merge results
    total_successes = 0
    total_episodes = 0
    total_return = 0.0

    for transitions, worker_stats in results:
        for t in transitions:
            trans = ChunkTransition(
                qpos=t["qpos"],
                images=t["images"],
                action=t["action"],
                reward=t["reward"],
                done=t["done"],
                value=t["value"],
                log_prob=t["log_prob"],
            )
            rollout_buffer.add(trans)
        total_successes += worker_stats["successes"]
        total_episodes += worker_stats["episodes"]
        total_return += worker_stats["total_return"]

    # Advance seed past all workers' ranges
    seed_next = seed_start + num_workers * seed_spacing

    stats = {
        "success_rate": total_successes / max(total_episodes, 1),
        "avg_return": total_return / max(total_episodes, 1),
        "episodes": total_episodes,
        "total_chunks": len(rollout_buffer),
    }
    return seed_next, stats


def parallel_evaluate_policy(
    model, act_config, ppo_config, env_args,
    task_name, task_config_name,
    seed_start, device, num_workers,
):
    """
    Parallel version of evaluate_policy.

    Spawns num_workers processes, each evaluating a subset of episodes.
    Merges success counts to compute overall success rate.

    Returns:
        success_rate: float
    """
    num_eval = 20
    episode_counts = _divide_episodes(num_eval, num_workers)

    model_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    device_str = str(device)

    seed_spacing = 500
    worker_args = []
    for i, count in enumerate(episode_counts):
        if count == 0:
            continue
        worker_seed = seed_start + i * seed_spacing
        worker_args.append((
            i, model_state_dict, act_config, ppo_config, env_args,
            task_name, task_config_name, worker_seed, count, device_str,
        ))

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(worker_args)) as pool:
        results = pool.map(_worker_eval_episodes, worker_args)

    total_successes = 0
    total_evaluated = 0
    for successes, evaluated in results:
        total_successes += successes
        total_evaluated += evaluated

    success_rate = total_successes / max(total_evaluated, 1)
    print(f"  [Eval] {total_successes}/{total_evaluated} = {success_rate*100:.1f}%")
    return success_rate
