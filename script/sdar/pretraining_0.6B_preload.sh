#!/bin/bash

export TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1

cd "$(dirname "$0")/../.."

uv run accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
    --main_process_port=8899 \
    pretrain/train_sdar.py \
    config=configs/sdar/pretraining_0.6B_preload.yaml
