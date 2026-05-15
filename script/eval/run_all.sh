#!/bin/bash
# 统一评测脚本：按模型类型和尺寸依次评测
# 用法:
#   bash script/eval/run_all.sh            # 评测所有类型 x 250M+0.6B
#   bash script/eval/run_all.sh 1B         # 评测所有类型 x 1B
#   bash script/eval/run_all.sh 250M 0.6B  # 指定多个尺寸

SCRIPT_DIR="$(dirname "$0")"

# 默认评测 250M 和 0.6B
if [ $# -eq 0 ]; then
    SIZES=("250M" "0.6B")
else
    SIZES=("$@")
fi

# 所有模型类型（对应 script/eval/lm_<type>.sh）
# TYPES=(
#     "ar"
#     "dream"
#     "llada"
#     "sdar"
#     "selfless"
#     "selfless_ar"
#     "xlnet"
#     "xlnet_ar"
# )
TYPES=(
    "ar"
    "dream"
    "llada"
    "sdar"
    "selfless"
    "xlnet"
)

echo "=========================================="
echo "  统一评测: ${#TYPES[@]} 类型 x ${#SIZES[@]} 尺寸"
echo "  SIZES: ${SIZES[*]}"
echo "  TYPES: ${TYPES[*]}"
echo "=========================================="

for size in "${SIZES[@]}"; do
    for type in "${TYPES[@]}"; do
        script="${SCRIPT_DIR}/lm_${type}.sh"
        echo ""
        echo ">>> [${size}] ${type} — $(date)"
        echo ">>> bash ${script} ${size}"
        bash "${script}" "${size}"
        echo "<<< [${size}] ${type} done — $(date)"
    done
done

echo ""
echo "=========================================="
echo "  全部完成 — $(date)"
echo "=========================================="
