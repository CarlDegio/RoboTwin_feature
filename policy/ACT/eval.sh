#!/bin/bash

# == keep unchanged ==
policy_name=ACT
task_name=${1}
task_config=${2}
ckpt_setting=${3}
expert_data_num=${4}
seed=${5}
gpu_id=${6}
DEBUG=False

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

cd ../..

CKPT_DIR=policy/ACT/act_ckpt_token_action/act-${task_name}/${ckpt_setting}-${expert_data_num}

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/$policy_name/deploy_policy.yml \
    --overrides \
    --task_name ${task_name} \
    --task_config ${task_config} \
    --ckpt_setting ${ckpt_setting} \
    --ckpt_dir ${CKPT_DIR} \
    --seed ${seed} \
    --n_bins 256 \
    --tokenizer_stats_path ${CKPT_DIR}/tokenizer_stats.json
