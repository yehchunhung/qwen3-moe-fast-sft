#!/bin/sh
MODEL_PATH=""
DATA_PATH=""
CKPT_PREFIX=""
OUTPUT_PATH="$CKPT_PREFIX/ckpt"
DS_CONFIG_PATH=""

NUM_NODE=2
NUM_GPU=$(nvidia-smi --list-gpus | wc -l)
MASTER_ADDR=""
MASTER_PORT=""
NUM_STEP=100

export HF_HOME="$CKPT_PREFIX/cache"
export TRITON_CACHE_DIR="$CKPT_PREFIX/triton_cache_dir"
export TORCH_EXTENSIONS_DIR="$CKPT_PREFIX/torch_extensions_dir"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

PYTHONWARNINGS="ignore::FutureWarning" \
TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
torchrun \
    --nnodes=$NUM_NODE \
    --nproc-per-node=$NUM_GPU \
    --rdzv-id=5566 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
    qwen3_sft.py \
    --model_path $MODEL_PATH \
    --dataset_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --train_steps $NUM_STEP \
    --micro_batch_size 2 \
    --deepspeed_config $DS_CONFIG_PATH \
    --max_length 4096 \
    --use_liger_kernel