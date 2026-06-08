#!/usr/bin/env bash
# run_consistency_probe.sh — 多 GPU 并行跑 consistency_probe，再聚合出图
#
# 用法:
#   bash script/eval/run_consistency_probe.sh
# 环境变量（可覆盖）:
#   PY            默认 "uv run python"，无 uv 用 "python"
#   OUT_DIR       输出目录（默认 output_consistency）
#   MAX_LEN       窗口长度（默认 512）
#   NUM_WINDOWS   采样窗口数（默认 64）
#   MODES         "ar,random"
#   GPUS          可用 GPU 列表（默认 "0 1 2 3"）
#   TEXT_FILE     本地纯文本语料（训练机无外网时用）；否则用 HF wikitext-2
#   EXTRA         透传给 consistency_probe.py 的额外参数（如 --max_instances 16 冒烟）
#
# CONFIGS: 把你 selfless / xlnet 各 checkpoint 的 lm_eval yaml 路径列进来。
#          注意 yaml.experiment.project 必须含 "selfless" 或 "xlnet"，否则会被拒绝。

set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"

PY=${PY:-"uv run python"}
OUT_DIR=${OUT_DIR:-output_consistency}
MAX_LEN=${MAX_LEN:-512}
NUM_WINDOWS=${NUM_WINDOWS:-64}
MODES=${MODES:-"ar,random"}
GPUS=${GPUS:-"0 1 2 3"}
TEXT_FILE=${TEXT_FILE:-""}
EXTRA=${EXTRA:-""}

mkdir -p "$OUT_DIR" logs_consistency

# ====== 在这里列出 configs（每行一个 yaml）======
CONFIGS=(
  "configs/0.6B_scratch/selfless.yaml"
  "configs/0.6B_scratch/xlnet.yaml"
  "configs/0.6B_preload/selfless.yaml"
  "configs/0.6B_preload/xlnet.yaml"
  "configs/342M_scratch/selfless.yaml"
  "configs/342M_scratch/xlnet.yaml"
)
# ===============================================

if [ -n "$TEXT_FILE" ]; then
  DATA_ARGS="--text_file $TEXT_FILE"
else
  DATA_ARGS="--hf_dataset wikitext --hf_config wikitext-2-raw-v1 --hf_split test"
fi

read -ra GPU_ARR <<< "$GPUS"
n_gpu=${#GPU_ARR[@]}
pids=()
i=0
fail=0

for cfg in "${CONFIGS[@]}"; do
  if [ ! -f "$cfg" ]; then
    echo "[skip] config 不存在: $cfg"; continue
  fi
  gpu=${GPU_ARR[$(( i % n_gpu ))]}
  tag=$(basename "$(dirname "$cfg")")_$(basename "$cfg" .yaml)
  echo "[launch] GPU $gpu <- $cfg ($tag)"
  CUDA_VISIBLE_DEVICES=$gpu $PY eval/consistency_probe.py \
      --config "$cfg" $DATA_ARGS \
      --max_len "$MAX_LEN" --num_windows "$NUM_WINDOWS" --modes "$MODES" \
      --out_dir "$OUT_DIR" $EXTRA \
      > "logs_consistency/${tag}.log" 2>&1 &
  pids+=($!)
  i=$(( i + 1 ))
  # 占满所有 GPU 就等一批（用具体退出码，避免 wait -n 误杀）
  if (( i % n_gpu == 0 )); then
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then fail=$(( fail + 1 )); fi
    done
    pids=()
  fi
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then fail=$(( fail + 1 )); fi
done

echo "[run] 完成，失败数=$fail（失败详见 logs_consistency/*.log）"

echo "[plot] 聚合出图 ..."
$PY plot_consistency.py --in_dir "$OUT_DIR"
echo "[plot] 见 $OUT_DIR/figs 与 $OUT_DIR/CONSISTENCY_REPORT.md"
