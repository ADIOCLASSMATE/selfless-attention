#!/bin/bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
SIZE=${1:-"342M"}
VARIANT=${2:-""}  # optional: "preload"
SCRIPT_NAME="eval/xlnet/eval_worker_xlnet.py"
if [ -n "$VARIANT" ]; then
    CONFIG_PATH="./configs/xlnet/lm_eval_xlnet_${SIZE}_ar+ar_${VARIANT}.yaml"
    OUTPUT_NAME="xlnet-${SIZE}-50BT-ar+ar-${VARIANT}"
else
    CONFIG_PATH="./configs/xlnet/lm_eval_xlnet_${SIZE}_ar+ar.yaml"
    OUTPUT_NAME="xlnet-${SIZE}-50BT-ar+ar"
fi
# TASKS="lambada_openai,wikitext,hellaswag,copa,piqa,arc_easy,openbookqa,winogrande,boolq,sciq,truthfulqa_mc1,truthfulqa_mc2,gpqa_diamond_zeroshot,super-glue-lm-eval-v1,arc_challenge,paloma_c4_en,paloma_falcon-refinedweb,paloma_wikitext_103"
TASKS="hellaswag,copa,piqa,arc_easy,openbookqa,winogrande,boolq,sciq,truthfulqa_mc1,truthfulqa_mc2,gpqa_diamond_zeroshot,super-glue-lm-eval-v1,arc_challenge"
cd "$(dirname "$0")/../.."
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Missing eval config: $CONFIG_PATH" >&2
    exit 1
fi
MODEL_PATH=$(awk -F'"' '/^[[:space:]]*model_path:/ {print $2; exit}' "$CONFIG_PATH")
if [ -n "$MODEL_PATH" ] && [ ! -d "$MODEL_PATH" ]; then
    echo "Missing model path from config: $MODEL_PATH" >&2
    exit 1
fi
uv run accelerate launch --num_processes 8 $SCRIPT_NAME \
    --model dllm \
    --model_args config_path=$CONFIG_PATH \
    --tasks $TASKS \
    --batch_size 1 \
    --output_path "./output_eval/${OUTPUT_NAME}" \
    --log_samples
