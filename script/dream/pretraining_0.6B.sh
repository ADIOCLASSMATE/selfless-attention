#!/bin/bash

export TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1

# Change to the project directory
cd "$(dirname "$0")/../.."

# Run the training with accelerate
uv run accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
    --main_process_port=8892 \
    pretrain/train_dream.py \
    config=configs/dream/pretraining_0.6B.yaml
