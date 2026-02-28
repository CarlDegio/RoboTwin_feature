#!/bin/bash
# ACT-PPO Training Script
# Usage: bash train_ppo.sh stack_bowls_two demo_clean 50 0 0
#
# Args:
#   $1: task_name (e.g., stack_bowls_two)
#   $2: task_config (e.g., demo_clean)
#   $3: expert_data_num (e.g., 50)
#   $4: seed (e.g., 0)
#   $5: gpu_id (e.g., 0)

task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

export CUDA_VISIBLE_DEVICES=${gpu_id}

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python3 policy/ACT/train_ppo.py \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --expert_data_num ${expert_data_num} \
    --seed ${seed} \
    --gpu_id ${gpu_id} \
    --ppo_config policy/ACT/ppo_config.yml
