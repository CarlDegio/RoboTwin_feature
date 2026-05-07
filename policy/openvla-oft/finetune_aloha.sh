# In experiments, global batch size of less than 16 will easily lead to unsuccessful training, where the training and validation 
# loss would not converge low enough, and the final policy would repeat one trajectory regardless of the visual and language inputs.
# export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200
# export NCCL_TIMEOUT=7200 
# export NCCL_SOCKET_IFNAME=lo
# export GLOO_SOCKET_IFNAME=lo
# export MASTER_ADDR=127.0.0.1
# export NCCL_P2P_LEVEL=0
torchrun --standalone --nnodes 1 --nproc-per-node 8 \
  vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /mnt/tensorflow_datasets/ \
  --dataset_name robotwin4stack_rand_aloha \
  --run_root_dir ckpts \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 1 \
  --grad_accumulation_steps 1 \
  --use_proprio True \
  --batch_size 8 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 50005 \
  --use_val_set True \
  --val_freq 1000 \
  --save_freq 25000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "carldegio" \
  --wandb_project "openvla-oft" \
  --run_id_override "discrete_1frame_prop_rand_delta_actions" \
  --run_id_note some_run_id_note \
  # --resume True \
  # --resume_step 60000 \
  # --resume_base_model_path openvla/openvla-7b \
  # --resume_checkpoint_path ckpts_离散token+prop效果尚可/Optional--60000_chkpt
