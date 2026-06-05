#!/usr/bin/env bash
# ============================================================================
# 统一 L->R BPB 评测（多 GPU 并行版）
#  342M / 0.6B / 0.6B-preload 全部 6 个族 = 18 个 run
#
# 用法：从仓库根目录运行
#   bash run_unified_bpb.sh                 # 默认：GRAN=1，使用全部 GPU
#   GRAN=4 bash run_unified_bpb.sh          # GRAN=4，使用全部 GPU
#   NUM_GPUS=4 bash run_unified_bpb.sh      # 只用 4 张 GPU
#   GPU_OFFSET=2 bash run_unified_bpb.sh    # 从 GPU 2 开始用（如 GPU 0-1 被占）
#   LIMIT=2000 bash run_unified_bpb.sh      # sanity check：只取前 2000 行
#
# 并行策略：
#  - 自动检测可用 GPU 数量，按需分配 slot
#  - 任务不等分块 —— 快任务（ar/selfless/xlnet，秒级）和慢任务（llada/dream/sdar，O(L²)）
#    共用 GPU 池，快任务自动填补空闲 slot
#  - 每个 config 输出写入独立日志文件 {LOG_DIR}/NN_config_name.log
#  - 监控进度：tail -f {LOG_DIR}/*.log
# ============================================================================
set -e
cd "$(dirname "$0")"/../..   # 切到仓库根目录

# ── 环境变量 ────────────────────────────────────────────────────────────────
GRAN=${GRAN:-1}                       # 解码粒度（仅 DLM 生效；1=严格逐 token L->R）
GPU_OFFSET=${GPU_OFFSET:-0}           # 起始 GPU 编号
HF_DATASET=${HF_DATASET:-wikitext}
HF_CONFIG=${HF_CONFIG:-wikitext-2-raw-v1}
HF_SPLIT=${HF_SPLIT:-test}
LIMIT=${LIMIT:-}                      # 可选：只取前 N 行（sanity）
TEXT_FILE=${TEXT_FILE:-}              # 可选：无网络时用本地纯文本语料代替 HF
LOG_DIR=${LOG_DIR:-logs/unified_bpb}  # 日志输出目录

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
EXTRA=""
[ -n "$LIMIT" ] && EXTRA="$EXTRA --limit $LIMIT"
if [ -n "$TEXT_FILE" ]; then
  CORPUS="--text_file $TEXT_FILE"
else
  CORPUS="--hf_dataset $HF_DATASET --hf_config $HF_CONFIG --hf_split $HF_SPLIT"
  # 预热缓存：先单进程下载一次，避免 18 个进程并发下载同一数据集/tokenizer 时的锁冲突
  echo ">>> 预热 HF 数据集缓存（首次下载，已缓存则秒过）..."
  uv run python -c "from datasets import load_dataset; load_dataset('$HF_DATASET','$HF_CONFIG',split='$HF_SPLIT')" \
    || { echo "ERROR: 数据集预热失败（无外网？改用 TEXT_FILE=...）"; exit 1; }
fi

# 每个 checkpoint 一个 config（PLM 用 ar+ar；其 random 版是同一 checkpoint，统一 L->R 数相同）
CONFIGS=(
  # ---- 342M (from-scratch) ----
  configs/ar/lm_eval_ar_342M.yaml
  configs/llada/lm_eval_llada_342M.yaml
  configs/dream/lm_eval_dream_342M.yaml
  configs/sdar/lm_eval_sdar_342M.yaml
  configs/selfless/lm_eval_selfless_342M_ar+ar.yaml
  configs/xlnet/lm_eval_xlnet_342M_ar+ar.yaml
  # ---- 0.6B (from-scratch) ----
  configs/ar/lm_eval_ar_0.6B.yaml
  configs/llada/lm_eval_llada_0.6B.yaml
  configs/dream/lm_eval_dream_0.6B.yaml
  configs/sdar/lm_eval_sdar_0.6B.yaml
  configs/selfless/lm_eval_selfless_0.6B_ar+ar.yaml
  configs/xlnet/lm_eval_xlnet_0.6B_ar+ar.yaml
  # ---- 0.6B-preload ----
  configs/ar/lm_eval_ar_0.6B_preload.yaml
  configs/llada/lm_eval_llada_0.6B_preload.yaml
  configs/dream/lm_eval_dream_0.6B_preload.yaml
  configs/sdar/lm_eval_sdar_0.6B_preload.yaml
  configs/selfless/lm_eval_selfless_0.6B_ar+ar_preload.yaml
  configs/xlnet/lm_eval_xlnet_0.6B_ar+ar_preload.yaml
)

mkdir -p "$LOG_DIR"
echo ">>> GRAN=$GRAN  GPUs: $N_GPU (offset=$GPU_OFFSET)  corpus: ${TEXT_FILE:-$HF_DATASET/$HF_CONFIG/$HF_SPLIT}  ${LIMIT:+(limit=$LIMIT)}"
echo ">>> Total configs: ${#CONFIGS[@]}   Logs: $LOG_DIR/   Monitor: tail -f $LOG_DIR/*.log"

# ── 并行调度 ─────────────────────────────────────────────────────────────────
declare -A pid_gpu   # pid → GPU index
declare -A pid_cfg   # pid → config 路径（用于完成时正确打印名字）
declare -A gpu_used  # GPU index → 1 (busy)
failed=0
launched=0
total=${#CONFIGS[@]}

# 阻塞直到“至少回收掉一个”已完成任务。
# 关键：用 `wait "$pid"` 对【具体 pid】取其真实退出码，每个状态只收一次。
# 不能用 `wait -n` + 事后按 kill -0 批量清理 —— 那样会把不是本次 wait 收割的、
# 退出码从未被读到的任务也清掉，导致 failed 漏计（4 卡/8 卡结果会不一致）。
reap_some() {
  local got=0
  while [ "$got" -eq 0 ]; do
    for pid in "${!pid_gpu[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid"; local rc=$?
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
    [ "$got" -eq 0 ] && sleep 0.5   # 没有完成的，歇一下再轮询
  done
}

set +e  # 并行段禁用 set -e，手动收集失败

for i in "${!CONFIGS[@]}"; do
  cfg="${CONFIGS[$i]}"

  # 池满时，等任意一个完成再继续
  while [ ${#pid_gpu[@]} -ge "$N_GPU" ]; do
    reap_some
  done

  # 找一张空闲 GPU
  gpu=-1
  for g in $(seq "$GPU_OFFSET" $((GPU_OFFSET + N_GPU - 1))); do
    if [ -z "${gpu_used[$g]:-}" ]; then gpu=$g; break; fi
  done
  [ "$gpu" -lt 0 ] && { echo "INTERNAL ERROR: 池里应有空位却找不到空闲 GPU"; exit 1; }

  cfg_name="$(basename "$cfg" .yaml)"
  printf -v idx_padded '%02d' "$i"
  log_file="$LOG_DIR/${idx_padded}_${cfg_name}.log"

  printf "  \033[1;32m[%2d/%2d]\033[0m GPU %d  ->  %s\n" $((launched + 1)) "$total" "$gpu" "$cfg"
  CUDA_VISIBLE_DEVICES="$gpu" uv run python eval/unified_lr_bpb.py \
      --config "$cfg" \
      $CORPUS \
      --block_size "$GRAN" \
      $EXTRA \
      > "$log_file" 2>&1 &

  pid_gpu[$!]=$gpu
  pid_cfg[$!]="$cfg"
  gpu_used[$gpu]=1
  launched=$((launched + 1))
done

# 等待剩余任务全部完成
while [ ${#pid_gpu[@]} -gt 0 ]; do
  reap_some
done

set -e

# ── 结果汇总 ─────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo ">>> 全部完成。成功: $((total - failed))/$total"
if [ "$failed" -gt 0 ]; then
  echo ">>> 失败: $failed/$total  （以下日志含报错）"
  grep -l "Traceback\|Error\|FAILED" "$LOG_DIR"/*.log 2>/dev/null || true
fi
echo ">>> 日志: $LOG_DIR/"
echo ">>> 汇总: uv run python collect_unified_bpb.py"
echo "================================================================"
[ "$failed" -gt 0 ] && exit 1 || exit 0