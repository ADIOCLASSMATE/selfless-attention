#!/bin/bash
# =============================================================================
# Extended LR sweep: 8e-4 and 1.5e-3 for all 6 families (342M, beta1=0.8, 5000 steps)
# Complement to the core sweep (1e-4, 2e-4, 4e-4) already completed.
# =============================================================================

set -uo pipefail

export TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"

FAMILIES=(ar selfless xlnet llada dream sdar)
LRS=(8e-4 1.5e-3)
PORT=8950

train_py_for() { echo "pretrain/train_$1.py"; }
config_for()   { echo "configs/$1/pretraining_342M.yaml"; }

echo "=================================================================="
echo "Extended LR sweep: ${#FAMILIES[@]} families x ${#LRS[@]} LRs"
echo "LRs: ${LRS[*]}"
echo "=================================================================="

for fam in "${FAMILIES[@]}"; do
  tpy="$(train_py_for "$fam")"
  cfg="$(config_for "$fam")"
  for lr in "${LRS[@]}"; do
    proj="appendix-sweep__${fam}__342M__lr${lr}__b1-0.8"
    echo "------------------------------------------------------------------"
    echo "[$proj]  lr=$lr  beta1=0.8  steps=5000  port=$PORT"
    echo "  $tpy  config=$cfg"

    uv run accelerate launch \
      --config_file accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml \
      --main_process_port="$PORT" \
      "$tpy" \
      config="$cfg" \
      experiment.project="$proj" \
      optimizer.params.learning_rate="$lr" \
      optimizer.params.beta1=0.8 \
      training.max_train_steps=5000 \
      lr_scheduler.params.warmup_steps=40 \
      lr_scheduler.params.decay_steps=800 \
      experiment.save_hfmodel_every=999999 \
      experiment.generate_every=999999
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "!! [$proj] FAILED (exit $rc). Continuing to next run." >&2
      echo "$proj lr=$lr beta1=0.8 exit=$rc" >> appendix_sweep_failures.log
    fi
    PORT=$((PORT + 1))
  done
done

echo "=================================================================="
echo "DONE. 12 runs completed."
echo "=================================================================="
