#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
text_likelihood.py — 各族“理论 native” BPB（每个族自己的估计量）
=================================================================

设计：零重写、最大忠实。
  - 每个族的 native 估计量（AR 链式 / PLM permutation-ELBO / LLaDA·Dream 吸收态 ELBO /
    SDAR block-diffusion ELBO）已经实现在它各自的 eval worker 的 `loglikelihood_rolling` 里
    —— 那也正是 output_eval 里 lm_eval 报的 native bits_per_byte 的来源。
  - 本脚本直接 **实例化对应 worker、调它自己的 loglikelihood_rolling** 在我们的语料上跑，
    拿回整段 total loglik（nats），再用 **我们自己的 UTF-8 字节数** 换成 BPB。
  - 语料加载复用 unified_lr_bpb.load_corpus；worker 的 rolling 用 stride=max_len//2 滑窗
    （与 unified_lr_bpb 完全一致）→ native 与 unified 的 Δ 协议干净（消除原来 lm_eval 字节口径
    带来的 +0.04 偏移）。

输出：output_eval/{project}/native_bpb_{时间戳}.json，schema 与 unified_lr_bpb 对齐
（estimator="native"，值字段 native_bpb），供 collect / 可视化统一读取。

用法（仓库根目录，单卡）：
  CUDA_VISIBLE_DEVICES=0 python eval/text_likelihood.py \
      --config configs/llada/lm_eval_llada_0.6B.yaml \
      --hf_dataset wikitext --hf_config wikitext-2-raw-v1 --hf_split test
  # 或本地语料： --text_file data/paloma_c4_en_test.txt

注意（PLM 的 native = permutation-ELBO，用 random-mode 配置）：
  selfless/xlnet 的 native BPB 要用它们的 *random* 配置（如 lm_eval_selfless_0.6B.yaml，
  attention_task=random），而不是 unified 用的 _ar+ar 配置——两者同一份权重，区别只在 eval 模式。
  这样 native 才对应 PLM 的随机顺序 ELBO（= RESULTS.md 的 native 列）。

【务必验证】首次运行后，本脚本算出的 native BPB 应当（在我们字节口径下）与 lm_eval/RESULTS.md
的 native bits_per_byte 同序、近值。对不上就说明 worker 实例化或语料口径有出入，先排查再信任。
"""
import os
import sys
import json
import math
import argparse
import importlib.util
from types import SimpleNamespace
from datetime import datetime

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from omegaconf import OmegaConf
from eval.unified_lr_bpb import load_corpus, detect_family, LN2  # 复用同一套语料/字节口径

# family -> worker 模块文件（每个都定义 @register_model("dllm") class DLLMEvalHarness(LM)）
WORKER_FILE = {
    "ar":       "eval/ar/eval_worker_ar.py",
    "llada":    "eval/llada/eval_worker_llada.py",
    "dream":    "eval/dream/eval_worker_dream.py",
    "sdar":     "eval/sdar/eval_worker_sdar.py",
    "selfless": "eval/selfless/eval_worker_selfless.py",
    "xlnet":    "eval/xlnet/eval_worker_xlnet.py",
}


def load_worker_class(family):
    """按文件路径导入对应 worker 模块，取出 DLLMEvalHarness 类。
    一次只导入一个族，避免多个模块同时 @register_model('dllm') 冲突。"""
    path = os.path.join(ROOT, WORKER_FILE[family])
    spec = importlib.util.spec_from_file_location(f"eval_worker_{family}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.DLLMEvalHarness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="lm_eval yaml（同 worker；PLM 用 random 配置）")
    ap.add_argument("--text_file", default=None)
    ap.add_argument("--hf_dataset", default=None)
    ap.add_argument("--hf_config", default=None)
    ap.add_argument("--hf_split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    project = cfg.experiment.project
    family = detect_family(project)
    print(f"[text_likelihood] project={project} | family={family} | "
          f"mc_num={cfg.get('mc_num')} | max_len={cfg.get('max_len')}")

    # 实例化对应 worker（它内部用 accelerate.Accelerator() 单进程即可，无需 accelerate launch）
    WorkerCls = load_worker_class(family)
    worker = WorkerCls(config_path=args.config)

    # 语料（与 unified_lr_bpb 同一加载/字节口径）
    text, _ = load_corpus(args)
    num_bytes = len(text.encode("utf-8"))
    try:
        num_tokens = len(worker._encode(text))
    except Exception:
        num_tokens = len(worker.tokenizer(text, add_special_tokens=False)["input_ids"])
    print(f"[text_likelihood] corpus: {num_tokens} tokens, {num_bytes} bytes")

    # 直接调 worker 自己的 loglikelihood_rolling —— 它封装了该族的 native 估计量
    req = SimpleNamespace(args=(text,))
    results = worker.loglikelihood_rolling([req])     # -> [total_loglik_nats]（loglik，通常为负）
    total_ll = float(results[0])
    total_nll = -total_ll                              # 转成 NLL（nats）

    native_bpb = total_nll / (LN2 * num_bytes)
    token_ppl = math.exp(total_nll / max(1, num_tokens))   # 近似（分母用全部 token）

    result = {
        "project": project,
        "family": family,
        "estimator": "native",
        "attention_regime": f"native {family} estimator (own ELBO/chain-rule)",
        "mc_num": int(cfg.get("mc_num", 1)),
        "max_len": int(cfg.get("max_len", 2048)),
        "corpus_tokens": int(num_tokens),
        "corpus_bytes": int(num_bytes),
        "total_nll_nats": float(total_nll),
        "native_bpb": float(native_bpb),
        "native_token_ppl_approx": float(token_ppl),
        "source": args.text_file or f"{args.hf_dataset}/{args.hf_config}/{args.hf_split}",
        "timestamp": datetime.now().isoformat(),
    }

    print("\n==================== RESULT (native) ====================")
    print(f"  family       : {family}")
    print(f"  native BPB   : {native_bpb:.4f}")
    print(f"  token-PPL(~) : {token_ppl:.2f}")
    print("=========================================================\n")

    out_dir = os.path.join(cfg.get("output_path", "./output_eval"), project)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"native_bpb_{datetime.now():%Y%m%dT%H%M%S}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[text_likelihood] saved -> {out_path}")


if __name__ == "__main__":
    main()