#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把 lm_eval 的 paloma 任务（paloma_c4_en / paloma_wikitext_103 / paloma_falcon-refinedweb）
实际使用的文档 dump 成本地纯文本，供 eval/unified_lr_bpb.py 用 --text_file 读取。
目的：让统一 L->R 估计量与你的 native paloma BPB 用 *完全相同* 的语料/字节，从而可直接并排。

用法（仓库根目录）：
  uv run python dump_paloma_corpus.py --task paloma_c4_en --out data/paloma_c4_en_test.txt --max_docs 1000
  # 之后：
  TEXT_FILE=data/paloma_c4_en_test.txt GRAN=1 bash script/eval/run_unified_bpb.sh
"""
import argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="paloma_c4_en")
    ap.add_argument("--out", default="public/dataset/paloma_c4_en_test.txt")
    ap.add_argument("--max_docs", type=int, default=1000,
                    help="最多取多少个文档（C4 很大，限量即可得稳定 BPB；0=全量）")
    args = ap.parse_args()

    from lm_eval.tasks import TaskManager, get_task_dict
    td = get_task_dict([args.task], TaskManager())
    task = td[args.task]

    # 取文档：优先 test，退而求其次 validation
    if task.has_test_docs():
        docs = list(task.test_docs())
    elif task.has_validation_docs():
        docs = list(task.validation_docs())
    else:
        raise RuntimeError(f"{args.task} 没有 test/validation docs")

    if args.max_docs and len(docs) > args.max_docs:
        docs = docs[: args.max_docs]

    # 取每个文档的文本：paloma 是 rolling-perplexity 任务，doc_to_text 返回整段文本
    texts = []
    for d in docs:
        t = None
        try:
            t = task.doc_to_text(d)
        except Exception:
            pass
        if not t:
            t = d.get("text") if isinstance(d, dict) else None
        if t:
            texts.append(t)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    blob = "\n\n".join(texts)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(blob)
    print(f"[dump] task={args.task}  docs={len(texts)}  chars={len(blob)}  bytes={len(blob.encode('utf-8'))}")
    print(f"[dump] -> {args.out}")
    print(f"[dump] 跑统一估计量：TEXT_FILE={args.out} GRAN=1 bash script/eval/run_unified_bpb.sh")

if __name__ == "__main__":
    main()