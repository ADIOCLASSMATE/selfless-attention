#!/bin/bash

export TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1

cd "$(dirname "$0")/../.."

accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
    --main_process_port=8897 \
    pretrain/train_llada.py \
    config=configs/llada/pretraining_1B.yaml
