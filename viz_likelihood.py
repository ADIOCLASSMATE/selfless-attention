#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
viz_likelihood.py — BPB 结果可视化（兼容现有 unified 结果 + native 结果）

读取 output_eval/likelihood_long.csv（由 collect_likelihood.py 生成）。
产出（output_eval/plots_likelihood/）：
  1) estimator_bars__<ds>__<scale>_<init>.png
     每 (dataset,scale,init)：x=family，分组柱=各 estimator(native/lr_g1/lr_g2/lr_g4)。
     直观看 native↔L→R 差异、以及 g 的并行解码税。
  2) parallel_tax__<ds>__<scale>_<init>.png
     DLM(sdar/llada/dream) 的 BPB 随解码粒度 g(1→2→4) 变化曲线（并行解码税）。
  3) bpb_vs_acc__<ds>__<scale>_<init>.png
     x=lr_g1 BPB，y=下游 11 任务平均 acc（从 output_eval 的 lm_eval harness 结果读取）。
     —— 新 C2 的核心图：公平估计量下 BPB 与下游 acc 的反相关。

用法（仓库根目录）：python viz_likelihood.py
依赖：matplotlib, numpy（缺失会给出提示并跳过绘图）。
"""
import os, glob, json, csv, re
from collections import defaultdict

ROOT = "./output_eval"
OUT = os.path.join(ROOT, "plots_likelihood")
FAM_ORDER = ["ar", "selfless", "xlnet", "sdar", "llada", "dream"]
DLM = {"sdar", "llada", "dream"}
EST_ORDER = ["native", "lr_g1", "lr_g2", "lr_g4"]

# 与 RESULTS.md 一致的下游任务/指标
ACC_TASKS = [("arc_easy", "acc_norm,none"), ("arc_challenge", "acc_norm,none"),
             ("hellaswag", "acc_norm,none"), ("piqa", "acc_norm,none"),
             ("sciq", "acc,none"), ("winogrande", "acc,none"),
             ("openbookqa", "acc_norm,none"), ("boolq", "acc,none"),
             ("copa", "acc,none"), ("wic", "acc,none"), ("sglue_rte", "acc,none")]


def parse(proj_key):
    fam = ("selfless" if "selfless" in proj_key else "xlnet" if "xlnet" in proj_key else
           "llada" if "llada" in proj_key else "dream" if "dream" in proj_key else
           "sdar" if "sdar" in proj_key else "ar")
    scale = "342M" if "342M" in proj_key else "0.6B"
    init = "preload" if "preload" in proj_key else "scratch"
    return scale, fam, init


def load_long():
    p = os.path.join(ROOT, "likelihood_long.csv")
    if not os.path.exists(p):
        raise SystemExit("缺 likelihood_long.csv，先跑 collect_likelihood.py")
    rows = []
    for r in csv.DictReader(open(p)):
        try:
            r["bpb"] = float(r["bpb"])
        except (ValueError, TypeError):
            continue
        rows.append(r)
    return rows


def load_downstream_acc():
    """从 output_eval 的 lm_eval harness 结果读 11 任务平均 acc。
    返回 {(scale,family,init): acc_avg}。PLM 优先用 random+random 目录。"""
    cand = defaultdict(dict)   # (scale,fam,init) -> {dirname: acc}
    for d in sorted(os.listdir(ROOT)):
        sub = os.path.join(ROOT, d)
        if not os.path.isdir(sub) or d.endswith("-lm-eval"):
            continue
        fs = glob.glob(os.path.join(sub, "*", "results_*.json"))
        if not fs:
            continue
        try:
            res = json.load(open(sorted(fs)[-1])).get("results", {})
        except Exception:
            continue
        vals = []
        for t, k in ACC_TASKS:
            v = res.get(t, {}).get(k)
            if isinstance(v, (int, float)):
                vals.append(v)
        if not vals:
            continue
        scale, fam, init = parse(d)
        cand[(scale, fam, init)][d] = sum(vals) / len(vals)

    acc = {}
    for key, dct in cand.items():
        fam = key[1]
        pick = None
        if fam in ("selfless", "xlnet"):
            for name in dct:
                if "random+random" in name or name.endswith("random"):
                    pick = name
                    break
        if pick is None:
            pick = max(dct, key=dct.get) if dct else None
        if pick:
            acc[key] = dct[pick]
    return acc


def _setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(OUT, exist_ok=True)
    return plt, np


def groups_present(rows):
    return sorted({(r["dataset"], r["scale"], r["init"]) for r in rows})


def plot_estimator_bars(rows, plt, np):
    for ds, scale, init in groups_present(rows):
        sel = [r for r in rows if r["dataset"] == ds and r["scale"] == scale and r["init"] == init]
        tab = defaultdict(dict)
        for r in sel:
            tab[r["family"]][r["estimator"]] = r["bpb"]
        fams = [f for f in FAM_ORDER if f in tab]
        ests = [e for e in EST_ORDER if any(e in tab[f] for f in fams)]
        if not fams or not ests:
            continue
        x = np.arange(len(fams)); w = 0.8 / max(1, len(ests))
        fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(fams)), 4.2))
        for i, e in enumerate(ests):
            ys = [tab[f].get(e, np.nan) for f in fams]
            ax.bar(x + i * w, ys, w, label=e)
        ax.set_xticks(x + w * (len(ests) - 1) / 2)
        ax.set_xticklabels(fams)
        ax.set_ylabel("BPB (lower = better)")
        ax.set_title(f"{ds} | {scale} ({init}) — native vs unified L→R")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = os.path.join(OUT, f"estimator_bars__{ds}__{scale}_{init}.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print("  ->", p)


def plot_parallel_tax(rows, plt, np):
    for ds, scale, init in groups_present(rows):
        fig, ax = plt.subplots(figsize=(5.2, 4))
        any_line = False
        for fam in ["sdar", "llada", "dream"]:
            pts = []
            for g in (1, 2, 4):
                m = [r for r in rows if r["dataset"] == ds and r["scale"] == scale
                     and r["init"] == init and r["family"] == fam and r["estimator"] == f"lr_g{g}"]
                if m:
                    pts.append((g, m[0]["bpb"]))
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, "o-", label=fam); any_line = True
        if not any_line:
            plt.close(fig); continue
        ax.set_xticks([1, 2, 4]); ax.set_xlabel("decode granularity g")
        ax.set_ylabel("unified L→R BPB")
        ax.set_title(f"{ds} | {scale} ({init}) — parallel-decoding tax")
        ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
        p = os.path.join(OUT, f"parallel_tax__{ds}__{scale}_{init}.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print("  ->", p)


def plot_bpb_vs_acc(rows, acc, plt, np):
    if not acc:
        print("  [skip] 没读到下游 acc（output_eval 里没有 harness results_*.json）")
        return
    for ds, scale, init in groups_present(rows):
        pts = []
        for fam in FAM_ORDER:
            m = [r for r in rows if r["dataset"] == ds and r["scale"] == scale
                 and r["init"] == init and r["family"] == fam and r["estimator"] == "lr_g1"]
            a = acc.get((scale, fam, init))
            if m and a is not None:
                pts.append((fam, m[0]["bpb"], a))
        if len(pts) < 3:
            continue
        fig, ax = plt.subplots(figsize=(5.2, 4.4))
        for fam, b, a in pts:
            ax.scatter(b, a, s=60)
            ax.annotate(fam, (b, a), textcoords="offset points", xytext=(5, 4), fontsize=8)
        # 相关系数
        bs = np.array([p[1] for p in pts]); accs = np.array([p[2] for p in pts])
        r = float(np.corrcoef(bs, accs)[0, 1]) if len(pts) > 2 else float("nan")
        ax.set_xlabel("unified L→R BPB (g=1, lower=better)")
        ax.set_ylabel("downstream acc avg (higher=better)")
        ax.set_title(f"{ds} | {scale} ({init}) — BPB vs acc  (r={r:+.2f})")
        ax.grid(alpha=0.3); fig.tight_layout()
        p = os.path.join(OUT, f"bpb_vs_acc__{ds}__{scale}_{init}.png")
        fig.savefig(p, dpi=150); plt.close(fig)
        print("  ->", p)


def main():
    rows = load_long()
    try:
        plt, np = _setup()
    except ImportError:
        raise SystemExit("需要 matplotlib + numpy：pip install matplotlib numpy")
    acc = load_downstream_acc()
    print("estimator_bars:");  plot_estimator_bars(rows, plt, np)
    print("parallel_tax:");    plot_parallel_tax(rows, plt, np)
    print("bpb_vs_acc:");      plot_bpb_vs_acc(rows, acc, plt, np)
    print(f"\n>>> 图已存到 {OUT}/")


if __name__ == "__main__":
    main()