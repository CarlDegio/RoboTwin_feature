#!/bin/bash
# ACT-PPO Evaluation Script
# Usage: bash eval_ppo.sh stack_bowls_two demo_clean demo_clean 50 0 0
#
# Args:
#   $1: task_name
#   $2: task_config
#   $3: ckpt_setting
#   $4: expert_data_num
#   $5: seed
#   $6: gpu_id

task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}

export CUDA_VISIBLE_DEVICES=${gpu_id}

cd ../..

PYTHONWARNINGS=ignore::UserWarning \
python3 script/eval_ppo.py \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_dir policy/ACT/act_ckpt/act_ppo-${task_name}/${ckpt_setting}-${expert_data_num} \
    --act_ckpt_dir policy/ACT/act_ckpt/act-${task_name}/${ckpt_setting}-${expert_data_num} \
    --seed ${seed} \
    --gpu_id ${gpu_id}
