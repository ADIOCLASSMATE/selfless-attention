#!/bin/bash
# =============================================================================
# Appendix hyperparameter sweep — fairness & robustness, NOT tuning-to-optimal.
#
# Answers two reviewer questions:
#   A1 (LR fairness): is a single shared LR near-optimal across ALL families,
#                     and does family ranking stay stable under LR perturbation?
#   A3 (beta1):       does beta1=0.8 (the non-standard value in the configs)
#                     differ materially from the standard 0.9?
#
# Design decisions (see handoff discussion):
#   * Scale = 342M ONLY. Sweeping 0.6B adds 9x cost and ZERO new argument.
#   * Short runs (default 5000 steps). We read the TREND in the WSD stable
#     region, not a converged number. warmup/decay are scaled DOWN with the
#     step budget so the stable (flat-LR) phase still dominates the short run
#     — otherwise the whole short run lives in warmup+decay and you see noise.
#   * CLI override via OmegaConf (get_config uses OmegaConf.from_cli()), so we
#     do NOT write N YAML files; we override learning_rate / beta1 / steps /
#     project on the command line.
#   * Each run gets a UNIQUE project name -> separate output dir -> no clobber.
#   * Runs are SERIAL: one run already saturates all 8 GPUs via DeepSpeed.
#
# Usage:
#   bash script/eval/run_appendix_sweep.sh                 # core set (21 runs)
#   FULL=1 bash script/eval/run_appendix_sweep.sh          # extended (36 runs)
#   DRY_RUN=1 bash script/eval/run_appendix_sweep.sh       # print, don't launch
#   STEPS=8000 bash script/eval/run_appendix_sweep.sh      # longer short-runs
#   ACCEL_CFG=accelerate_configs/1_node_4_gpus_deepspeed_zero2.yaml bash ...
# =============================================================================

set -uo pipefail

export TOKENIZERS_PARALLELISM=true
export HF_HUB_OFFLINE=1

# cd to repo root regardless of where this is called from
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$(dirname "$0")/../..")"

# ----------------------------- knobs -----------------------------------------
ACCEL_CFG="${ACCEL_CFG:-accelerate_configs/1_node_8_gpus_deepspeed_zero2.yaml}"
STEPS="${STEPS:-5000}"            # short-run budget
PORT_BASE="${PORT_BASE:-8900}"    # each run gets PORT_BASE + offset
DRY_RUN="${DRY_RUN:-0}"
FULL="${FULL:-0}"
SWEEP_TAG="${SWEEP_TAG:-appendix-sweep}"
PY_LAUNCH="${PY_LAUNCH:-uv run accelerate launch}"  # set PY_LAUNCH='accelerate launch' if not using uv

# Scale warmup/decay with the short budget so the stable phase still dominates.
# Original 50000-step schedule: warmup=400 (0.8%), decay=8000 (16%).
# Keep the same fractions: warmup = 0.8% * STEPS, decay = 16% * STEPS.
WARMUP=$(( STEPS * 8 / 1000 ));  [ "$WARMUP" -lt 50 ] && WARMUP=50
DECAY=$(( STEPS * 16 / 100 ))

# Families and which train_*.py + config each uses.
# All families share the SAME 342M config except model path / attention_task.
FAMILIES=(ar selfless xlnet llada dream sdar)

# LR grid. Core = 3 points around the current 2e-4. FULL = 5 points.
if [ "$FULL" = "1" ]; then
    LRS=(7e-5 1e-4 2e-4 4e-4 8e-4)
else
    LRS=(1e-4 2e-4 4e-4)
fi

# beta1 ablation: families to test the standard 0.9 against the current 0.8.
# Core = representative subset (one AR, one PLM, one DLM). FULL = all six.
if [ "$FULL" = "1" ]; then
    BETA1_FAMILIES=(ar selfless xlnet llada dream sdar)
else
    BETA1_FAMILIES=(ar selfless sdar)
fi
BETA1_ALT=0.9                     # the standard value to compare against 0.8
BETA1_REF_LR=2e-4                 # beta1 ablation runs at the shared LR only

# ----------------------------- helpers ----------------------------------------
port_counter=0   # incremented in the parent shell (do NOT wrap in $())

launch() {
    # args: train_py  base_config  project  lr  beta1
    local train_py="$1" cfg="$2" project="$3" lr="$4" beta1="$5"
    local port=$(( PORT_BASE + port_counter ))
    port_counter=$(( port_counter + 1 ))

    local cmd=( $PY_LAUNCH
        --config_file "$ACCEL_CFG"
        --main_process_port="$port"
        "$train_py"
        config="$cfg"
        experiment.project="$project"
        optimizer.params.learning_rate="$lr"
        optimizer.params.beta1="$beta1"
        training.max_train_steps="$STEPS"
        lr_scheduler.params.warmup_steps="$WARMUP"
        lr_scheduler.params.decay_steps="$DECAY"
        experiment.save_hfmodel_every=999999   # short run: skip HF export
        experiment.generate_every=999999
    )

    echo "------------------------------------------------------------------"
    echo "[$project]  lr=$lr  beta1=$beta1  steps=$STEPS  port=$port"
    echo "  ${cmd[*]}"
    if [ "$DRY_RUN" = "1" ]; then return 0; fi

    "${cmd[@]}"
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "!! [$project] FAILED (exit $rc). Continuing to next run." >&2
        echo "$project lr=$lr beta1=$beta1 exit=$rc" >> "appendix_sweep_failures.log"
    fi
    return 0
}

train_py_for() { echo "pretrain/train_$1.py"; }
config_for()   { echo "configs/$1/pretraining_342M.yaml"; }

# ----------------------------- A1: LR fairness --------------------------------
echo "=================================================================="
echo "A1  LR fairness sweep: ${#FAMILIES[@]} families x ${#LRS[@]} LRs"
echo "    (beta1 held at config default 0.8)"
echo "=================================================================="
for fam in "${FAMILIES[@]}"; do
    tpy="$(train_py_for "$fam")"
    cfg="$(config_for "$fam")"
    for lr in "${LRS[@]}"; do
        proj="${SWEEP_TAG}__${fam}__342M__lr${lr}__b1-0.8"
        launch "$tpy" "$cfg" "$proj" "$lr" "0.8"
    done
done

# ----------------------------- A3: beta1 ablation -----------------------------
echo "=================================================================="
echo "A3  beta1 ablation: ${#BETA1_FAMILIES[@]} families at beta1=$BETA1_ALT, lr=$BETA1_REF_LR"
echo "    (compare against the lr=$BETA1_REF_LR / beta1=0.8 run from A1)"
echo "=================================================================="
for fam in "${BETA1_FAMILIES[@]}"; do
    tpy="$(train_py_for "$fam")"
    cfg="$(config_for "$fam")"
    proj="${SWEEP_TAG}__${fam}__342M__lr${BETA1_REF_LR}__b1-${BETA1_ALT}"
    launch "$tpy" "$cfg" "$proj" "$BETA1_REF_LR" "$BETA1_ALT"
done

echo "=================================================================="
echo "DONE. Runs land in output/${SWEEP_TAG}__*  (wandb project: selfless-attention)"
echo "Failures (if any) logged to appendix_sweep_failures.log"
echo "Next: collect val loss / BPB per run and plot loss-vs-LR per family."
echo "=================================================================="
