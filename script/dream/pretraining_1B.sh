#!/bin/bash

export TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1

cd /inspire/hdd/global_user/wanjiaxin-253108030048/code/selfless-attention

accelerate launch \
    --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
    --main_process_port=8898 \
    pretrain/train_dream.py \
    config=configs/dream/pretraining_1B.yaml
