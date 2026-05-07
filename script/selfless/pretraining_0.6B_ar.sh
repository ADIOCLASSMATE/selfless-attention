#!/bin/bash

export TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1

# Change to the project directory
cd /inspire/hdd/global_user/wanjiaxin-253108030048/code/selfless-attention

# Run the training with accelerate (using uv for environment management)
uv run accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
    --main_process_port=8890 \
    pretrain/train_selfless.py \
    config=configs/selfless/pretraining_0.6B_ar.yaml
