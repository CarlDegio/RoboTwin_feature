# In experiments, global batch size of less than 16 will easily lead to unsuccessful training, where the training and validation 
# loss would not converge low enough, and the final policy would repeat one trajectory regardless of the visual and language inputs.
torchrun --standalone --nnodes 1 --nproc-per-node 3 \
  vla-scripts/finetune.py \
  --vla_path openvla/openvla-7b \
  --data_root_dir /home/ubuntu/tensorflow_datasets/ \
  --dataset_name robotwin4stack_aloha \
  --run_root_dir ckpts \
  --use_l1_regression False \
  --use_diffusion False \
  --use_film True \
  --num_images_in_input 3 \
  --grad_accumulation_steps 1 \
  --use_proprio True \
  --batch_size 2 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 50000 \
  --max_steps 100005 \
  --use_val_set True \
  --val_freq 1000 \
  --save_freq 30000 \
  --save_latest_checkpoint_only False \
  --image_aug True \
  --lora_rank 32 \
  --wandb_entity "carldegio" \
  --wandb_project "openvla-oft" \
  --run_id_override "Optional" \
  --run_id_note some_run_id_note \
  ### example usage for resuming a training process
  # --resume True\
  # --resume_step 5000 \
  # --resume_base_model_path openvla/openvla-7b
