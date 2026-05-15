#!/bin/bash

export TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1

cd "$(dirname "$0")/../.."

uv run accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
    --main_process_port=8898 \
    pretrain/train_dream.py \
    config=configs/dream/pretraining_1B.yaml
