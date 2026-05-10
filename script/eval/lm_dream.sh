#!/bin/bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
SCRIPT_NAME="eval/Dream/eval_worker_dream.py"
CONFIG_PATH="./configs/eval/lm_eval_dream.yaml"

# 核心逻辑与常识
TASKS="lambada_openai,wikitext,hellaswag,copa,piqa,arc_easy,openbookqa,winogrande,boolq,sciq,truthfulqa_mc1,truthfulqa_mc2,gpqa_diamond_zeroshot,super-glue-lm-eval-v1"
# TASKS="lambada_openai"

# ===========================================

# echo "Starting Evaluation..."
# echo "Model Config: $CONFIG_PATH"
# echo "Tasks: $TASKS"


uv run accelerate launch $SCRIPT_NAME \
    --model dllm \
    --model_args config_path=$CONFIG_PATH \
    --tasks $TASKS \
    --batch_size 1 \
    --output_path "./output_eval/dream-250M-50BT" \
    --log_samples
