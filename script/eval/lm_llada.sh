#!/bin/bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
SIZE=${1:-"250M"}
SCRIPT_NAME="eval/llada/eval_worker_llada.py"
CONFIG_PATH="./configs/llada/lm_eval_llada_${SIZE}.yaml"

# 核心逻辑与常识
# TASKS="lambada_openai,wikitext,hellaswag,copa,piqa,arc_easy,openbookqa,winogrande,boolq,sciq,truthfulqa_mc1,truthfulqa_mc2,gpqa_diamond_zeroshot,super-glue-lm-eval-v1"
TASKS="lambada_openai"
# TASKS="paloma_c4_en,paloma_falcon-refinedweb"

# ===========================================

# echo "Starting Evaluation..."
# echo "Model Config: $CONFIG_PATH"
# echo "Tasks: $TASKS"


uv run accelerate launch --num_processes 8 $SCRIPT_NAME \
    --model dllm \
    --model_args config_path=$CONFIG_PATH \
    --tasks $TASKS \
    --batch_size 1 \
    --output_path "./output_eval/llada-${SIZE}-50BT" \
    --log_samples

