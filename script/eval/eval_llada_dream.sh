cd "$(dirname "$0")/../.."
source .venv/bin/activate

bash script/eval/lm_llada.sh
bash script/eval/lm_dream.sh