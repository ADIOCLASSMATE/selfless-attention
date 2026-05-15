#!/bin/bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
SIZE=${1:-"250M"}
SCRIPT_NAME="eval/xlnet/eval_worker_xlnet.py"
CONFIG_PATH="./configs/xlnet/lm_eval_xlnet_${SIZE}.yaml"
TASKS="lambada_openai,wikitext,hellaswag,copa,piqa,arc_easy,openbookqa,winogrande,boolq,sciq,truthfulqa_mc1,truthfulqa_mc2,gpqa_diamond_zeroshot,super-glue-lm-eval-v1"
cd "$(dirname "$0")/../.."
uv run accelerate launch --num_processes 8 $SCRIPT_NAME \
    --model dllm \
    --model_args config_path=$CONFIG_PATH \
    --tasks $TASKS \
    --batch_size 1 \
    --output_path "./output_eval/xlnet-${SIZE}-50BT-ar+ar" \
    --log_samples
