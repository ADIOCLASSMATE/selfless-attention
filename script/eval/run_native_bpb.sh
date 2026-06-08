#!/usr/bin/env bash
# ============================================================================
# Native BPB 评测（多 GPU 并行版）
#  342M / 0.6B / 0.6B-preload 全部 6 个族 = 18 个 run
#
# 用法：从仓库根目录运行
#   bash script/eval/run_native_bpb.sh
#   NUM_GPUS=4 bash script/eval/run_native_bpb.sh
#   GPU_OFFSET=2 bash script/eval/run_native_bpb.sh
#   LIMIT=2000 bash script/eval/run_native_bpb.sh
#   TEXT_FILE=data/wikitext2_test.txt bash script/eval/run_native_bpb.sh
#   MAX_LEN=1024 bash script/eval/run_native_bpb.sh
#   SEED=123 bash script/eval/run_native_bpb.sh
#
# 注意：
#  - selfless/xlnet 的 native BPB 使用 random 配置，不使用 unified 里的 _ar+ar 配置。
#  - 每个 config 输出独立日志 {LOG_DIR}/NN_config_name.log。
#  - 监控进度：tail -f {LOG_DIR}/*.log
# ============================================================================
set -e
cd "$(dirname "$0")"/../..

# ── 环境变量 ────────────────────────────────────────────────────────────────
GPU_OFFSET=${GPU_OFFSET:-0}
HF_DATASET=${HF_DATASET:-wikitext}
HF_CONFIG=${HF_CONFIG:-wikitext-2-raw-v1}
HF_SPLIT=${HF_SPLIT:-test}
LIMIT=${LIMIT:-}
TEXT_FILE=${TEXT_FILE:-}
MAX_LEN=${MAX_LEN:-}
SEED=${SEED:-42}
LOG_DIR=${LOG_DIR:-logs/native_bpb}

# ── 自动检测 GPU 数量 ───────────────────────────────────────────────────────
if [ -n "${NUM_GPUS:-}" ]; then
  N_GPU=$NUM_GPUS
else
  N_GPU=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)
  N_GPU=$((N_GPU - GPU_OFFSET))
fi
if [ "$N_GPU" -le 0 ]; then
  echo "ERROR: 没有可用 GPU (N_GPU=$N_GPU, GPU_OFFSET=$GPU_OFFSET)"
  nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || true
  exit 1
fi

# ── 构建语料参数 ─────────────────────────────────────────────────────────────
EXTRA_ARGS=()
[ -n "$LIMIT" ] && EXTRA_ARGS+=(--limit "$LIMIT")
[ -n "$MAX_LEN" ] && EXTRA_ARGS+=(--max_len "$MAX_LEN")
EXTRA_ARGS+=(--seed "$SEED")

if [ -n "$TEXT_FILE" ]; then
  CORPUS_ARGS=(--text_file "$TEXT_FILE")
  CORPUS_DESC="$TEXT_FILE"
else
  CORPUS_ARGS=(--hf_dataset "$HF_DATASET" --hf_config "$HF_CONFIG" --hf_split "$HF_SPLIT")
  CORPUS_DESC="$HF_DATASET/$HF_CONFIG/$HF_SPLIT"
  # 预热缓存：先单进程下载一次，避免多进程并发下载同一数据集时的锁冲突
  echo ">>> 预热 HF 数据集缓存（首次下载，已缓存则秒过）..."
  uv run python -c "from datasets import load_dataset; load_dataset('$HF_DATASET','$HF_CONFIG',split='$HF_SPLIT')" \
    || { echo "ERROR: 数据集预热失败（无外网？改用 TEXT_FILE=...）"; exit 1; }
fi

# Native 配置：PLM(selfless/xlnet) 必须用 random 配置；_ar+ar 只用于 unified L->R。
CONFIGS=(
  # ---- 342M (from-scratch) ----
  configs/ar/lm_eval_ar_342M.yaml
  configs/llada/lm_eval_llada_342M.yaml
  configs/dream/lm_eval_dream_342M.yaml
  configs/sdar/lm_eval_sdar_342M.yaml
  configs/selfless/lm_eval_selfless_342M.yaml
  configs/selfless/lm_eval_selfless_342M_ar+ar.yaml
  configs/xlnet/lm_eval_xlnet_342M.yaml
  configs/xlnet/lm_eval_xlnet_342M_ar+ar.yaml
  # ---- 0.6B (from-scratch) ----
  configs/ar/lm_eval_ar_0.6B.yaml
  configs/llada/lm_eval_llada_0.6B.yaml
  configs/dream/lm_eval_dream_0.6B.yaml
  configs/sdar/lm_eval_sdar_0.6B.yaml
  configs/selfless/lm_eval_selfless_0.6B.yaml
  configs/selfless/lm_eval_selfless_0.6B_ar+ar.yaml
  configs/xlnet/lm_eval_xlnet_0.6B.yaml
  configs/xlnet/lm_eval_xlnet_0.6B_ar+ar.yaml
  # ---- 0.6B-preload ----
  configs/ar/lm_eval_ar_0.6B_preload.yaml
  configs/llada/lm_eval_llada_0.6B_preload.yaml
  configs/dream/lm_eval_dream_0.6B_preload.yaml
  configs/sdar/lm_eval_sdar_0.6B_preload.yaml
  configs/selfless/lm_eval_selfless_0.6B_preload.yaml
  configs/selfless/lm_eval_selfless_0.6B_ar+ar_preload.yaml
  configs/xlnet/lm_eval_xlnet_0.6B_preload.yaml
  configs/xlnet/lm_eval_xlnet_0.6B_ar+ar_preload.yaml
)

for cfg in "${CONFIGS[@]}"; do
  [ -f "$cfg" ] || { echo "ERROR: config 不存在: $cfg"; exit 1; }
done

mkdir -p "$LOG_DIR"
echo ">>> GPUs: $N_GPU (offset=$GPU_OFFSET)  seed: $SEED  corpus: $CORPUS_DESC  ${LIMIT:+(limit=$LIMIT)} ${MAX_LEN:+(max_len=$MAX_LEN)}"
echo ">>> Total configs: ${#CONFIGS[@]}   Logs: $LOG_DIR/   Monitor: tail -f $LOG_DIR/*.log"

# ── 并行调度 ─────────────────────────────────────────────────────────────────
declare -A pid_gpu
declare -A pid_cfg
declare -A gpu_used
failed=0
launched=0
total=${#CONFIGS[@]}

reap_some() {
  local got=0
  local rc=0
  while [ "$got" -eq 0 ]; do
    for pid in "${!pid_gpu[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid"; rc=$?
        [ "$rc" -ne 0 ] && failed=$((failed + 1))
        if [ "$rc" -ne 0 ]; then
          printf "  \033[1;31m[FAIL rc=%d]\033[0m GPU %d  <-  %s\n" "$rc" "${pid_gpu[$pid]}" "$(basename "${pid_cfg[$pid]}" .yaml)"
        else
          printf "  \033[1;33m[done]\033[0m GPU %d  <-  %s\n" "${pid_gpu[$pid]}" "$(basename "${pid_cfg[$pid]}" .yaml)"
        fi
        unset "gpu_used[${pid_gpu[$pid]}]"
        unset "pid_gpu[$pid]"
        unset "pid_cfg[$pid]"
        got=1
      fi
    done
    [ "$got" -eq 0 ] && sleep 0.5
  done
}

set +e

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"

  while [ ${#pid_gpu[@]} -ge "$N_GPU" ]; do
    reap_some
  done

  gpu=-1
  for g in $(seq "$GPU_OFFSET" $((GPU_OFFSET + N_GPU - 1))); do
    if [ -z "${gpu_used[$g]:-}" ]; then gpu=$g; break; fi
  done
  [ "$gpu" -lt 0 ] && { echo "INTERNAL ERROR: 池里应有空位却找不到空闲 GPU"; exit 1; }

  cfg_name="$(basename "$cfg" .yaml)"
  printf -v idx_padded '%02d' "$i"
  log_file="$LOG_DIR/${idx_padded}_${cfg_name}.log"

  printf "  \033[1;32m[%2d/%2d]\033[0m GPU %d  ->  %s\n" $((launched + 1)) "$total" "$gpu" "$cfg"
  CUDA_VISIBLE_DEVICES="$gpu" uv run python eval/text_likelihood.py \
      --config "$cfg" \
      "${CORPUS_ARGS[@]}" \
      "${EXTRA_ARGS[@]}" \
      > "$log_file" 2>&1 &

  pid_gpu[$!]=$gpu
  pid_cfg[$!]="$cfg"
  gpu_used[$gpu]=1
  launched=$((launched + 1))
done

while [ ${#pid_gpu[@]} -gt 0 ]; do
  reap_some
done

set -e

echo ""
echo "================================================================"
echo ">>> 全部完成。成功: $((total - failed))/$total"
if [ "$failed" -gt 0 ]; then
  echo ">>> 失败: $failed/$total  （以下日志含报错）"
  grep -l "Traceback\|Error\|FAILED" "$LOG_DIR"/*.log 2>/dev/null || true
fi
echo ">>> 日志: $LOG_DIR/"
echo ">>> 汇总: uv run python collect_likelihood.py"
echo "================================================================"
[ "$failed" -gt 0 ] && exit 1 || exit 0
