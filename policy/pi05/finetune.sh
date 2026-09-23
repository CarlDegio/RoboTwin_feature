train_config_name=$1
model_name=$2
gpu_use=$3

export OPENPI_DATA_HOME=/mnt/openpi_base
export HF_HUB_OFFLINE=1

export CUDA_VISIBLE_DEVICES=$gpu_use
echo $CUDA_VISIBLE_DEVICES
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 python scripts/train.py $train_config_name --exp-name=$model_name --overwrite
