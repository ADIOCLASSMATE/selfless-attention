#!/bin/bash

export TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1

# Change to the project directory
cd "$(dirname "$0")/../.."

# Run the training with accelerate
uv run accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
    --main_process_port=8889 \
    pretrain/train_xlnet.py \
    config=configs/xlnet/pretraining_0.6B_smoke.yaml
