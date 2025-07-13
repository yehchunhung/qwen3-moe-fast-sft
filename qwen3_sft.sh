#!/bin/sh
MODEL_PATH=""
DATA_PATH=""
CKPT_PREFIX=""
OUTPUT_PATH="$CKPT_PREFIX/ckpt"
DS_CONFIG_PATH=""

NUM_GPU=$(nvidia-smi --list-gpus | wc -l)
NUM_STEP=50
MICRO_BATCH_SIZE=4
SEQ_LEN=8192

export HF_HOME="$CKPT_PREFIX/cache"
export TRITON_CACHE_DIR="$CKPT_PREFIX/triton_cache_dir"
export TORCH_EXTENSIONS_DIR="$CKPT_PREFIX/torch_extensions_dir"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING=true

PYTHONWARNINGS="ignore::FutureWarning" \
TORCH_NCCL_AVOID_RECORD_STREAMS=1 \
torchrun \
    --standalone \
    --nproc-per-node=$NUM_GPU \
    qwen3_sft.py \
    --model_path $MODEL_PATH \
    --dataset_path $DATA_PATH \
    --output_dir $OUTPUT_PATH \
    --train_steps $NUM_STEP \
    --micro_batch_size $MICRO_BATCH_SIZE \
    --deepspeed_config $DS_CONFIG_PATH \
    --max_length $SEQ_LEN \
    --use_fast_moe_block \
    --use_liger_efficiency_callback \
    --use_liger_kernel \
    --use_fp8 \
    --torch_compile
    # --fsdp "full_shard auto_wrap"