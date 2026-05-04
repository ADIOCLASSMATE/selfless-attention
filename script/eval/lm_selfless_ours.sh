#!/bin/bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
SCRIPT_NAME="eval/selfless/eval_worker_selfless_ours.py"
CONFIG_PATH="./configs/eval/lm_eval_selfless_ours.yaml"
TASKS="lambada_openai,wikitext,hellaswag,copa,piqa,arc_easy,openbookqa,winogrande,boolq,sciq,truthfulqa_mc1,truthfulqa_mc2,gpqa_diamond_zeroshot,super-glue-lm-eval-v1"
cd /inspire/hdd/global_user/wanjiaxin-253108030048/code/selfless-attention
accelerate launch $SCRIPT_NAME \
    --model dllm \
    --model_args config_path=$CONFIG_PATH \
    --tasks $TASKS \
    --batch_size 1 \
    --output_path "./output_eval_final/selfless-all" \
    --log_samples
