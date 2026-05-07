#!/bin/bash
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
SCRIPT_NAME="eval/PPL/test_PPL_sdar.py"
TASKS="lambada_openai,wikitext,hellaswag,copa,piqa,arc_easy,openbookqa,winogrande,boolq,sciq,truthfulqa_mc1,truthfulqa_mc2,gpqa_diamond_zeroshot,super-glue-lm-eval-v1"
cd "$(dirname "$0")/../.."
accelerate launch $SCRIPT_NAME \
    --model dllm \
    --model_args config_path=configs/eval/lm_eval_sdar.yaml \
    --tasks $TASKS \
    --batch_size 1 \
    --output_path "./output_eval_final/sdar-all" \
    --log_samples
