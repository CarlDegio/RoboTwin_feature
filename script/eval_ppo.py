"""
ACT-PPO Evaluation Script: Evaluate PPO-finetuned ACT policy.

Usage:
    bash eval_ppo.sh stack_bowls_two demo_clean demo_clean 50 0 0
"""

import os
import sys
import yaml
import torch
import numpy as np
import argparse
import importlib
import time
from argparse import Namespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from policy.ACT.act_ppo_model import ACTPPOModel
from policy.ACT.ppo_rollout import encode_obs, obs_to_tensors


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    env_class = getattr(envs_module, task_name)
    return env_class()


def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda:0"

    # ACT config (same as training)
    act_config = {
        "kl_weight": 10, "chunk_size": 50, "hidden_dim": 512,
        "dim_feedforward": 3200, "lr": 5e-5, "lr_backbone": 1e-5,
        "backbone": "resnet18", "enc_layers": 4, "dec_layers": 4,
        "nheads": 8,
        "camera_names": ["cam_high", "cam_right_wrist", "cam_left_wrist"],
        "state_dim": 14, "ckpt_dir": args.act_ckpt_dir, "device": device,
    }
    ppo_config = {"log_std_min": -5.0, "log_std_max": 0.0}

    # Build model and load PPO weights
    model = ACTPPOModel(act_config, ppo_config)
    ppo_ckpt = os.path.join(args.ckpt_dir, "policy_best.ckpt")
    if not os.path.exists(ppo_ckpt):
        ppo_ckpt = os.path.join(args.ckpt_dir, "policy_last.ckpt")
    state_dict = torch.load(ppo_ckpt, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    print(f"Loaded PPO checkpoint: {ppo_ckpt}")

    # Setup environment
    os.chdir(os.path.join(os.path.dirname(__file__), "../.."))
    from test_render import Sapien_TEST
    Sapien_TEST()

    from policy.ACT.train_ppo import setup_env_args
    TASK_ENV = class_decorator(args.task_name)
    env_args = setup_env_args(args.task_name, args.task_config, {
        "ckpt_setting": args.task_config,
        "seed": args.seed,
    })

    # Evaluation loop
    from envs.utils.create_actor import UnStableError
    from script.generate_episode_instructions import generate_episode_descriptions

    chunk_size = model.chunk_size
    max_chunks = 20
    test_num = 30
    now_seed = 100000 * (1 + args.seed)
    successes = 0
    evaluated = 0

    while evaluated < test_num:
        # Find valid seed
        env_ok = False
        for _ in range(30):
            try:
                TASK_ENV.setup_demo(
                    now_ep_num=evaluated, seed=now_seed,
                    is_test=True, **env_args,
                )
                episode_info = TASK_ENV.play_once()
                plan_ok = TASK_ENV.plan_success and TASK_ENV.check_success()
                TASK_ENV.close_env()
                if plan_ok:
                    env_ok = True
                    print("eval env ok!")
                    break
                now_seed += 1
            except (UnStableError, Exception):
                TASK_ENV.close_env()
                now_seed += 1
                print("eval env error!")

        if not env_ok:
            now_seed += 1
            continue

        # Setup env for policy rollout
        TASK_ENV.setup_demo(
            now_ep_num=evaluated, seed=now_seed,
            is_test=True, **env_args,
        )
        episode_info_list = [episode_info["info"]]
        results = generate_episode_descriptions(
            env_args["task_name"], episode_info_list, 1,
        )
        instruction = np.random.choice(results[0]["unseen"])
        TASK_ENV.set_instruction(instruction=instruction)

        # Execute policy (deterministic)
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
            print(f"\033[92mEpisode {evaluated+1}: Success\033[0m")
        else:
            print(f"\033[91mEpisode {evaluated+1}: Fail\033[0m")

        evaluated += 1
        TASK_ENV.close_env()
        now_seed += 1

        print(
            f"Success rate: {successes}/{evaluated} = "
            f"{successes/evaluated*100:.1f}%"
        )

    # Final summary
    final_sr = successes / max(evaluated, 1)
    print(f"\n{'='*50}")
    print(f"Final: {successes}/{evaluated} = {final_sr*100:.1f}%")
    print(f"{'='*50}")

    # Save result
    result_path = os.path.join(args.ckpt_dir, "eval_result.txt")
    with open(result_path, "w") as f:
        f.write(f"Success rate: {final_sr}\n")
        f.write(f"Successes: {successes}/{evaluated}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("ACT-PPO Evaluation")
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--task_config", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--act_ckpt_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    main(args)
