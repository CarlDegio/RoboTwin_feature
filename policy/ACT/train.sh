#!/bin/bash
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}

DEBUG=False
save_ckpt=True

export CUDA_VISIBLE_DEVICES=${gpu_id}

# Step 1: Compute tokenizer stats (Q1/Q99 for delta actions)
CKPT_DIR=./act_ckpt_token_action/act-${task_name}/${task_config}-${expert_data_num}
TOKENIZER_STATS=${CKPT_DIR}/tokenizer_stats.json
TASK_KEY=sim-${task_name}/${task_config}-${expert_data_num}
TASK_NAME=sim-${task_name}-${task_config}-${expert_data_num}

if [ ! -f "${TOKENIZER_STATS}" ]; then
    echo "Computing tokenizer stats..."
    mkdir -p ${CKPT_DIR}
    python3 compute_tokenizer_stats.py \
        --dataset_dir ./processed_data/${TASK_KEY} \
        --num_episodes ${expert_data_num} \
        --output_path ${TOKENIZER_STATS} \
        --state_dim 14
fi

# Step 2: Train ACT-DT
python3 imitate_episodes.py \
    --task_name ${TASK_NAME} \
    --ckpt_dir ${CKPT_DIR} \
    --policy_class ACT \
    --kl_weight 10 \
    --chunk_size 50 \
    --hidden_dim 512 \
    --batch_size 32 \
    --dim_feedforward 3200 \
    --num_epochs 6000 \
    --lr 5e-5 \
    --save_freq 2000 \
    --state_dim 14 \
    --seed ${seed} \
    --n_bins 256 \
    --aux_weight 0.5 \
    --tokenizer_stats_path ${TOKENIZER_STATS}
