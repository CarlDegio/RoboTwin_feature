"""
Parallel episode collection and evaluation for ACT-PPO.

Supports two execution modes:

1) One-shot pool (backward compatible, default)
   - Uses torch.multiprocessing spawn context + Pool.map
   - Workers build their own model/env, run a single COLLECT/EVAL task, exit

2) Persistent worker pool (optional)
   - Uses explicit spawn Processes + Queues
   - Workers keep model/env alive across iterations
   - Main process can UPDATE_WEIGHTS then COLLECT/EVAL repeatedly

Usage:
    from ppo_parallel import parallel_collect_rollouts, parallel_evaluate_policy
"""

import gc
import os
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp


# ---------------------------------------------------------------------------
# Worker protocol
# ---------------------------------------------------------------------------

CMD_UPDATE_WEIGHTS = "UPDATE_WEIGHTS"
CMD_COLLECT = "COLLECT"
CMD_EVAL = "EVAL"
CMD_SHUTDOWN = "SHUTDOWN"


def _make_ok_response(worker_id: int, req_id: int, payload: Any) -> Dict[str, Any]:
    return {
        "worker_id": worker_id,
        "req_id": req_id,
        "ok": True,
        "payload": payload,
        "error": None,
        "traceback": None,
    }


def _make_err_response(worker_id: int, req_id: int, err: BaseException) -> Dict[str, Any]:
    return {
        "worker_id": worker_id,
        "req_id": req_id,
        "ok": False,
        "payload": None,
        "error": f"{type(err).__name__}: {err}",
        "traceback": traceback.format_exc(),
    }


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
# Reusable collect/eval logic (runs inside subprocesses)
# ---------------------------------------------------------------------------

def _collect_episodes_with_model_env(
    *,
    worker_id: int,
    model,
    TASK_ENV,
    ppo_config: Dict[str, Any],
    env_args: Dict[str, Any],
    seed_start: int,
    num_episodes: int,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], int]:
    """Collect episodes using an existing model + TASK_ENV.

    Returns:
        transitions: list[dict] (pickleable)
        stats: dict
        seed_next: int
    """
    from ppo_rollout import encode_obs, obs_to_tensors
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

    max_chunks = ppo_config.get("max_chunks_per_episode", 20)
    success_reward = ppo_config.get("success_reward", 1.0)
    failure_reward = ppo_config.get("failure_reward", -1.0)
    chunk_size = model.chunk_size

    all_transitions: List[Dict[str, Any]] = []
    now_seed = seed_start
    episodes_collected = 0
    successes = 0
    total_return = 0.0

    while episodes_collected < num_episodes:
        # Find valid seed
        env_setup_ok = False
        episode_info = None
        for _ in range(50):
            try:
                TASK_ENV.setup_demo(
                    now_ep_num=episodes_collected,
                    seed=now_seed,
                    is_test=True,
                    **env_args,
                )
                episode_info = TASK_ENV.play_once()
                plan_ok = TASK_ENV.plan_success and TASK_ENV.check_success()
                TASK_ENV.close_env()
                if plan_ok:
                    env_setup_ok = True
                    break
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
            now_ep_num=episodes_collected,
            seed=now_seed,
            is_test=True,
            **env_args,
        )
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(env_args["task_name"], episode_info_list, 1)
        instruction = np.random.choice(results[0]["unseen"])
        TASK_ENV.set_instruction(instruction=instruction)

        episode_transitions: List[Dict[str, Any]] = []
        done = False
        chunk_count = 0

        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim and chunk_count < max_chunks:
            observation = TASK_ENV.get_obs()
            obs_dict = encode_obs(observation)
            qpos, images = obs_to_tensors(obs_dict, model, device)

            with torch.no_grad():
                action, log_prob, value, _, _ = model.get_action_and_value(
                    qpos, images, deterministic=False
                )

            action_denorm = model.post_process(action)
            action_np = action_denorm.squeeze(0).cpu().numpy()

            trans = {
                "qpos": qpos.squeeze(0).cpu().numpy(),
                "images": np.stack(
                    [obs_dict["head_cam"], obs_dict["left_cam"], obs_dict["right_cam"]],
                    axis=0,
                ),
                "action": action.squeeze(0).cpu().numpy(),
                "reward": 0.0,
                "done": False,
                "value": float(value.item()),
                "log_prob": float(log_prob.item()),
            }
            episode_transitions.append(trans)

            for step_i in range(chunk_size):
                TASK_ENV.take_action(action_np[step_i])
                if TASK_ENV.eval_success or TASK_ENV.take_action_cnt >= TASK_ENV.step_lim:
                    done = True
                    break

            chunk_count += 1
            if done:
                break

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
    return all_transitions, stats, now_seed


def _eval_episodes_with_model_env(
    *,
    worker_id: int,
    model,
    TASK_ENV,
    ppo_config: Dict[str, Any],
    env_args: Dict[str, Any],
    seed_start: int,
    num_episodes: int,
    device: torch.device,
) -> Tuple[int, int, int]:
    """Evaluate episodes deterministically using an existing model + TASK_ENV.

    Returns:
        successes, evaluated, seed_next
    """
    from ppo_rollout import encode_obs, obs_to_tensors
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions

    chunk_size = model.chunk_size
    max_chunks = ppo_config.get("max_chunks_per_episode", 60)

    now_seed = seed_start
    successes = 0
    evaluated = 0

    for ep in range(num_episodes):
        env_ok = False
        episode_info = None
        for _ in range(20):
            try:
                TASK_ENV.setup_demo(now_ep_num=ep, seed=now_seed, is_test=True, **env_args)
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

        TASK_ENV.setup_demo(now_ep_num=ep, seed=now_seed, is_test=True, **env_args)
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(env_args["task_name"], episode_info_list, 1)
        instruction = np.random.choice(results[0]["unseen"])
        TASK_ENV.set_instruction(instruction=instruction)

        chunk_count = 0
        done = False
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim and chunk_count < max_chunks:
            observation = TASK_ENV.get_obs()
            obs_dict = encode_obs(observation)
            qpos, images = obs_to_tensors(obs_dict, model, device)

            with torch.no_grad():
                action, _, _, _, _ = model.get_action_and_value(qpos, images, deterministic=True)

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
    return successes, evaluated, now_seed


# ---------------------------------------------------------------------------
# Collect worker
# ---------------------------------------------------------------------------

def _worker_collect_episodes(args):
    """Backward-compatible one-shot collect worker."""
    (
        worker_id,
        model_state_dict,
        act_config,
        ppo_config,
        env_args,
        task_name,
        task_config_name,
        seed_start,
        num_episodes,
        device_str,
    ) = args

    TASK_ENV = None
    try:
        _worker_setup_env()

        device = torch.device(device_str)
        model = _worker_build_model(model_state_dict, act_config, ppo_config, device)
        TASK_ENV = _worker_build_env(task_name, task_config_name, env_args)

        transitions, stats, _ = _collect_episodes_with_model_env(
            worker_id=worker_id,
            model=model,
            TASK_ENV=TASK_ENV,
            ppo_config=ppo_config,
            env_args=env_args,
            seed_start=seed_start,
            num_episodes=num_episodes,
            device=device,
        )
        return transitions, stats

    except Exception:
        print(f"[Worker {worker_id}] FATAL: {traceback.format_exc()}")
        return [], {"successes": 0, "episodes": 0, "total_chunks": 0, "total_return": 0.0}

    finally:
        try:
            if TASK_ENV is not None:
                TASK_ENV.close_env()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Eval worker
# ---------------------------------------------------------------------------

def _worker_eval_episodes(args):
    """Backward-compatible one-shot eval worker."""
    (
        worker_id,
        model_state_dict,
        act_config,
        ppo_config,
        env_args,
        task_name,
        task_config_name,
        seed_start,
        num_episodes,
        device_str,
    ) = args

    TASK_ENV = None
    try:
        _worker_setup_env()

        device = torch.device(device_str)
        model = _worker_build_model(model_state_dict, act_config, ppo_config, device)
        TASK_ENV = _worker_build_env(task_name, task_config_name, env_args)

        successes, evaluated, _ = _eval_episodes_with_model_env(
            worker_id=worker_id,
            model=model,
            TASK_ENV=TASK_ENV,
            ppo_config=ppo_config,
            env_args=env_args,
            seed_start=seed_start,
            num_episodes=num_episodes,
            device=device,
        )
        return successes, evaluated

    except Exception:
        print(f"[Worker {worker_id}] FATAL: {traceback.format_exc()}")
        return 0, 0

    finally:
        try:
            if TASK_ENV is not None:
                TASK_ENV.close_env()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Persistent worker manager
# ---------------------------------------------------------------------------


def _persistent_worker_main(
    worker_id: int,
    in_q,
    out_q,
):
    """Worker main loop for persistent pool.

    Protocol:
      - receives command envelopes: {cmd, req_id, payload}
      - sends response envelopes: {worker_id, req_id, ok, payload|error|traceback}
    """
    TASK_ENV = None
    model = None
    device = None

    try:
        _worker_setup_env()

        while True:
            msg = in_q.get()
            cmd = msg.get("cmd")
            req_id = int(msg.get("req_id", -1))
            payload = msg.get("payload", {})

            try:
                if cmd == CMD_SHUTDOWN:
                    out_q.put(_make_ok_response(worker_id, req_id, {"status": "shutdown"}))
                    return

                if cmd == CMD_UPDATE_WEIGHTS:
                    # Lazily build env/model on first update
                    act_config = payload["act_config"]
                    ppo_config = payload["ppo_config"]
                    env_args = payload["env_args"]
                    task_name = payload["task_name"]
                    task_config_name = payload.get("task_config_name")
                    device_str = payload.get("device_str", "cpu")
                    model_state_dict = payload["model_state_dict"]

                    device = torch.device(device_str)

                    if model is None:
                        model = _worker_build_model(model_state_dict, act_config, ppo_config, device)
                    else:
                        model.load_state_dict(model_state_dict, strict=True)
                        model.to(device)
                        model.eval()

                    if TASK_ENV is None:
                        TASK_ENV = _worker_build_env(task_name, task_config_name, env_args)

                    out_q.put(_make_ok_response(worker_id, req_id, {"status": "weights_updated"}))
                    continue

                if model is None or TASK_ENV is None or device is None:
                    raise RuntimeError(
                        "Worker not initialized. Send UPDATE_WEIGHTS before COLLECT/EVAL."
                    )

                if cmd == CMD_COLLECT:
                    transitions, stats, seed_next = _collect_episodes_with_model_env(
                        worker_id=worker_id,
                        model=model,
                        TASK_ENV=TASK_ENV,
                        ppo_config=payload["ppo_config"],
                        env_args=payload["env_args"],
                        seed_start=int(payload["seed_start"]),
                        num_episodes=int(payload["num_episodes"]),
                        device=device,
                    )
                    out_q.put(
                        _make_ok_response(
                            worker_id,
                            req_id,
                            {"transitions": transitions, "stats": stats, "seed_next": seed_next},
                        )
                    )
                    continue

                if cmd == CMD_EVAL:
                    successes, evaluated, seed_next = _eval_episodes_with_model_env(
                        worker_id=worker_id,
                        model=model,
                        TASK_ENV=TASK_ENV,
                        ppo_config=payload["ppo_config"],
                        env_args=payload["env_args"],
                        seed_start=int(payload["seed_start"]),
                        num_episodes=int(payload["num_episodes"]),
                        device=device,
                    )
                    out_q.put(
                        _make_ok_response(
                            worker_id,
                            req_id,
                            {
                                "successes": int(successes),
                                "evaluated": int(evaluated),
                                "seed_next": int(seed_next),
                            },
                        )
                    )
                    continue

                raise ValueError(f"Unknown cmd: {cmd}")

            except Exception as e:
                out_q.put(_make_err_response(worker_id, req_id, e))

    except KeyboardInterrupt:
        # allow parent to terminate
        return

    except Exception as e:
        # If the worker loop itself crashed, try to notify parent once.
        try:
            out_q.put(_make_err_response(worker_id, -1, e))
        except Exception:
            pass

    finally:
        # Robust cleanup
        try:
            if TASK_ENV is not None:
                TASK_ENV.close_env()
        except Exception:
            pass

        try:
            del TASK_ENV
        except Exception:
            pass

        try:
            del model
        except Exception:
            pass

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        gc.collect()


@dataclass
class _WorkerHandle:
    worker_id: int
    proc: mp.Process
    in_q: Any
    out_q: Any


class PersistentWorkerPool:
    """Persistent worker manager using spawn + explicit Process/Queue."""

    def __init__(
        self,
        *,
        num_workers: int,
        act_config: Dict[str, Any],
        ppo_config: Dict[str, Any],
        env_args: Dict[str, Any],
        task_name: str,
        task_config_name: str,
        device_str: str,
        command_timeout_sec: float = 300.0,
        shutdown_timeout_sec: float = 30.0,
    ):
        self._ctx = mp.get_context("spawn")
        self._num_workers = int(num_workers)
        self._handles: List[_WorkerHandle] = []
        self._closed = False
        self._req_id = 0

        # Static payload shared across commands
        self._act_config = act_config
        self._ppo_config = ppo_config
        self._env_args = env_args
        self._task_name = task_name
        self._task_config_name = task_config_name
        self._device_str = device_str

        self._command_timeout_sec = float(command_timeout_sec)
        self._shutdown_timeout_sec = float(shutdown_timeout_sec)

        for wid in range(self._num_workers):
            in_q = self._ctx.Queue()
            out_q = self._ctx.Queue()
            proc = self._ctx.Process(
                target=_persistent_worker_main,
                args=(wid, in_q, out_q),
                daemon=True,
            )
            proc.start()
            self._handles.append(_WorkerHandle(worker_id=wid, proc=proc, in_q=in_q, out_q=out_q))

    def close(self):
        if self._closed:
            return
        self._closed = True

        # Best-effort graceful shutdown
        for h in self._handles:
            try:
                self._req_id += 1
                h.in_q.put({"cmd": CMD_SHUTDOWN, "req_id": self._req_id, "payload": {}})
            except Exception:
                pass

        deadline = time.time() + self._shutdown_timeout_sec
        for h in self._handles:
            remaining = max(0.0, deadline - time.time())
            try:
                h.proc.join(timeout=remaining)
            except Exception:
                pass

        # Force terminate stragglers
        for h in self._handles:
            try:
                if h.proc.is_alive():
                    h.proc.terminate()
            except Exception:
                pass

        for h in self._handles:
            try:
                h.proc.join(timeout=1.0)
            except Exception:
                pass

        # Close queues
        for h in self._handles:
            for q in (h.in_q, h.out_q):
                try:
                    q.close()
                except Exception:
                    pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _next_req_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def _broadcast(self, cmd: str, payload: Dict[str, Any]):
        req_id = self._next_req_id()
        for h in self._handles:
            h.in_q.put({"cmd": cmd, "req_id": req_id, "payload": payload})
        return req_id

    def _gather(
        self,
        *,
        req_id: int,
        timeout_sec: float,
    ) -> List[Dict[str, Any]]:
        responses: List[Dict[str, Any]] = []
        deadline = time.time() + timeout_sec
        for h in self._handles:
            remaining = max(0.0, deadline - time.time())
            resp = h.out_q.get(timeout=remaining)
            # Basic envelope validation
            if resp.get("req_id") != req_id:
                raise RuntimeError(
                    f"Mismatched req_id from worker {resp.get('worker_id')}: "
                    f"expected {req_id}, got {resp.get('req_id')}"
                )
            if not resp.get("ok", False):
                raise RuntimeError(
                    f"Worker {resp.get('worker_id')} error: {resp.get('error')}\n{resp.get('traceback')}"
                )
            responses.append(resp)
        return responses

    # --- public API ---

    def update_weights(self, *, model_state_dict: Dict[str, Any]):
        """Broadcast weights + config to all workers."""
        payload = {
            "model_state_dict": model_state_dict,
            "act_config": self._act_config,
            "ppo_config": self._ppo_config,
            "env_args": self._env_args,
            "task_name": self._task_name,
            "task_config_name": self._task_config_name,
            "device_str": self._device_str,
        }
        req_id = self._broadcast(CMD_UPDATE_WEIGHTS, payload)
        self._gather(req_id=req_id, timeout_sec=self._command_timeout_sec)

    def collect(
        self,
        *,
        worker_episode_counts: List[int],
        seed_start: int,
        seed_spacing: int,
    ) -> List[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """Run COLLECT on all workers (workers with 0 episodes return empty)."""
        req_id = self._next_req_id()
        for h in self._handles:
            count = int(worker_episode_counts[h.worker_id])
            worker_seed = int(seed_start + h.worker_id * seed_spacing)
            payload = {
                "ppo_config": self._ppo_config,
                "env_args": self._env_args,
                "seed_start": worker_seed,
                "num_episodes": count,
            }
            h.in_q.put({"cmd": CMD_COLLECT, "req_id": req_id, "payload": payload})

        responses = self._gather(req_id=req_id, timeout_sec=self._command_timeout_sec)
        results: List[Tuple[List[Dict[str, Any]], Dict[str, Any]]] = []
        for resp in responses:
            pl = resp["payload"]
            results.append((pl["transitions"], pl["stats"]))
        return results

    def eval(
        self,
        *,
        worker_episode_counts: List[int],
        seed_start: int,
        seed_spacing: int,
    ) -> List[Tuple[int, int]]:
        """Run EVAL on all workers (workers with 0 episodes return 0/0)."""
        req_id = self._next_req_id()
        for h in self._handles:
            count = int(worker_episode_counts[h.worker_id])
            worker_seed = int(seed_start + h.worker_id * seed_spacing)
            payload = {
                "ppo_config": self._ppo_config,
                "env_args": self._env_args,
                "seed_start": worker_seed,
                "num_episodes": count,
            }
            h.in_q.put({"cmd": CMD_EVAL, "req_id": req_id, "payload": payload})

        responses = self._gather(req_id=req_id, timeout_sec=self._command_timeout_sec)
        results: List[Tuple[int, int]] = []
        for resp in responses:
            pl = resp["payload"]
            results.append((int(pl["successes"]), int(pl["evaluated"])))
        return results



# ---------------------------------------------------------------------------
# Legacy helpers below
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
    model,
    act_config,
    ppo_config,
    env_args,
    task_name,
    task_config_name,
    rollout_buffer,
    seed_start,
    device,
    num_workers,
    runtime_pool=None,
):
    """
    Parallel version of collect_rollouts.

    Backward compatible: if runtime_pool is None, uses one-shot spawn Pool.
    If runtime_pool is provided, uses persistent worker processes.

    Returns:
        seed_next: next seed to use
        stats: dict with success_rate, avg_return, episodes, total_chunks
    """
    from ppo_rollout import ChunkTransition

    num_episodes = ppo_config.get("num_episodes_per_iter", 4)
    episode_counts = _divide_episodes(num_episodes, num_workers)

    model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    device_str = str(device)

    # Assign non-overlapping seed ranges (generous spacing to handle retries)
    seed_spacing = 1000  # each worker may try up to 50 seeds per episode
    worker_args = []
    for i, count in enumerate(episode_counts):
        if count == 0:
            continue
        worker_seed = seed_start + i * seed_spacing
        worker_args.append(
            (
                i,
                model_state_dict,
                act_config,
                ppo_config,
                env_args,
                task_name,
                task_config_name,
                worker_seed,
                count,
                device_str,
            )
        )

    rollout_buffer.clear()

    if runtime_pool is None:
        # One-shot pool behavior
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            results = pool.map(_worker_collect_episodes, worker_args)
    else:
        # Persistent pool behavior (weights should already be broadcast)
        results = runtime_pool.collect(
            worker_episode_counts=episode_counts,
            seed_start=seed_start,
            seed_spacing=seed_spacing,
        )

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

    seed_next = seed_start + num_workers * seed_spacing

    stats = {
        "success_rate": total_successes / max(total_episodes, 1),
        "avg_return": total_return / max(total_episodes, 1),
        "episodes": total_episodes,
        "total_chunks": len(rollout_buffer),
    }
    return seed_next, stats


def parallel_evaluate_policy(
    model,
    act_config,
    ppo_config,
    env_args,
    task_name,
    task_config_name,
    seed_start,
    device,
    num_workers,
    runtime_pool=None,
):
    """
    Parallel version of evaluate_policy.

    Backward compatible: if runtime_pool is None, uses one-shot spawn Pool.
    If runtime_pool is provided, uses persistent worker processes.

    Returns:
        success_rate: float
    """
    num_eval = 20
    episode_counts = _divide_episodes(num_eval, num_workers)

    model_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    device_str = str(device)

    seed_spacing = 500
    worker_args = []
    for i, count in enumerate(episode_counts):
        if count == 0:
            continue
        worker_seed = seed_start + i * seed_spacing
        worker_args.append(
            (
                i,
                model_state_dict,
                act_config,
                ppo_config,
                env_args,
                task_name,
                task_config_name,
                worker_seed,
                count,
                device_str,
            )
        )

    if runtime_pool is None:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(worker_args)) as pool:
            results = pool.map(_worker_eval_episodes, worker_args)
    else:
        results = runtime_pool.eval(
            worker_episode_counts=episode_counts,
            seed_start=seed_start,
            seed_spacing=seed_spacing,
        )

    total_successes = 0
    total_evaluated = 0
    for successes, evaluated in results:
        total_successes += successes
        total_evaluated += evaluated

    success_rate = total_successes / max(total_evaluated, 1)
    print(f"  [Eval] {total_successes}/{total_evaluated} = {success_rate*100:.1f}%")
    return success_rate
