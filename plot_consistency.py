#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
plot_consistency.py — 聚合 consistency_probe 的多 run 输出并出图

读取 --in_dir 下所有 *_instances.csv / *_layerwise.csv / *_summary.json，
横跨 family（selfless/xlnet）与 checkpoint（project）画图，并写一个 markdown 汇总。

产图（PNG，写到 --in_dir/figs）：
  01_headline_kl_bar          各 project × family 的末层 KL(q||c) 柱状（log y）—— 主图
  02_kl_vs_context_ar         ar 模式 KL vs 上下文长度（按 family）
  03_kl_vs_context_random     random 模式 KL vs 上下文数（按 family）
  04_kl_vs_layer              逐层 logit-lens KL（按 family）—— 深度累积
  05_top1_agree_vs_context    top-1 一致率 vs 上下文长度
  06_kl_hist                  每实例 KL 分布（selfless 紧贴 0 / xlnet 散开）
  07_nll_scatter              query-nll vs content-nll 散点（对角线参照）
  08_js_bar                   JS 散度柱状（对称指标）
  09_ar_vs_random_bar         同 family 下 ar vs random 的 KL（对角线效应与顺序无关）
  10_layer_heatmap            family × layer 的 KL 热图
  11_leak_control_bar         content 流：真 token 目标位 nll≈0（泄露）vs masked 目标位
  12_logitcos_vs_context      logit 余弦相似度 vs 上下文长度

用法
    python plot_consistency.py --in_dir output_consistency
"""

import os
import sys
import csv
import json
import glob
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAM_COLOR = {"selfless": "#2a7", "xlnet": "#c44"}
FAM_ORDER = ["selfless", "xlnet"]


def load_instances(in_dir):
    rows = []
    for p in glob.glob(os.path.join(in_dir, "*_instances.csv")):
        with open(p) as f:
            for r in csv.DictReader(f):
                for k in ("kl_qc", "kl_cq", "js", "top1_agree", "logit_l2",
                          "logit_cos", "nll_q", "nll_c", "nll_absdiff"):
                    r[k] = float(r[k])
                r["context_size"] = int(r["context_size"])
                rows.append(r)
    return rows


def load_layerwise(in_dir):
    rows = []
    for p in glob.glob(os.path.join(in_dir, "*_layerwise.csv")):
        with open(p) as f:
            for r in csv.DictReader(f):
                r["layer"] = int(r["layer"])
                r["kl_qc_mean"] = float(r["kl_qc_mean"])
                r["kl_qc_median"] = float(r["kl_qc_median"])
                r["top1_agree_mean"] = float(r["top1_agree_mean"])
                rows.append(r)
    return rows


def load_summaries(in_dir):
    out = []
    for p in glob.glob(os.path.join(in_dir, "*_summary.json")):
        with open(p) as f:
            out.append(json.load(f))
    return out


def _safe_log(arr, floor=1e-8):
    return np.maximum(np.array(arr, dtype=float), floor)


def fig_headline(rows, out):
    # 末层 KL(q||c) 均值，分 (project, family)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["project"], r["family"])].append(r["kl_qc"])
    projects = sorted({k[0] for k in agg})
    fams = [f for f in FAM_ORDER if any(k[1] == f for k in agg)]
    x = np.arange(len(projects))
    w = 0.8 / max(1, len(fams))
    plt.figure(figsize=(max(6, 1.6 * len(projects)), 4.2))
    for i, fam in enumerate(fams):
        vals = [np.mean(agg.get((p, fam), [np.nan])) for p in projects]
        plt.bar(x + i * w, _safe_log(vals), w, label=fam, color=FAM_COLOR.get(fam))
    plt.yscale("log")
    plt.xticks(x + w * (len(fams) - 1) / 2, projects, rotation=20, ha="right", fontsize=8)
    plt.ylabel("mean KL(query ‖ content)  [nats, log]")
    plt.title("Single-stream consistency: content-readout vs query-readout (lower=more consistent)")
    plt.legend()
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_kl_vs_context(rows, mode, out):
    sub = [r for r in rows if r["mode"] == mode]
    if not sub:
        return
    plt.figure(figsize=(6, 4.2))
    for fam in FAM_ORDER:
        fam_rows = [r for r in sub if r["family"] == fam]
        if not fam_rows:
            continue
        byc = defaultdict(list)
        for r in fam_rows:
            byc[r["context_size"]].append(r["kl_qc"])
        cs = sorted(byc)
        means = [np.mean(byc[c]) for c in cs]
        plt.plot(cs, _safe_log(means), "o-", label=fam, color=FAM_COLOR.get(fam))
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("# context (non-mask) tokens"); plt.ylabel("mean KL(q‖c) [log]")
    plt.title(f"KL vs context size — {mode} mode")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_kl_vs_layer(lw, out):
    if not lw:
        return
    plt.figure(figsize=(6, 4.2))
    for fam in FAM_ORDER:
        fr = [r for r in lw if r["family"] == fam]
        if not fr:
            continue
        byl = defaultdict(list)
        for r in fr:
            byl[r["layer"]].append(r["kl_qc_mean"])
        ls = sorted(byl)
        means = [np.mean(byl[l]) for l in ls]
        plt.plot(ls, _safe_log(means), "o-", label=fam, color=FAM_COLOR.get(fam))
    plt.yscale("log")
    plt.xlabel("layer (logit-lens)"); plt.ylabel("mean KL(q‖c) [log]")
    plt.title("Where divergence accumulates across depth")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_top1_vs_context(rows, out):
    sub = [r for r in rows if r["mode"] == "ar"]
    if not sub:
        sub = rows
    plt.figure(figsize=(6, 4.2))
    for fam in FAM_ORDER:
        fam_rows = [r for r in sub if r["family"] == fam]
        if not fam_rows:
            continue
        byc = defaultdict(list)
        for r in fam_rows:
            byc[r["context_size"]].append(r["top1_agree"])
        cs = sorted(byc)
        means = [100 * np.mean(byc[c]) for c in cs]
        plt.plot(cs, means, "o-", label=fam, color=FAM_COLOR.get(fam))
    plt.xscale("log")
    plt.xlabel("# context tokens"); plt.ylabel("top-1 agreement (%)")
    plt.title("Argmax agreement: content vs query readout")
    plt.ylim(0, 101); plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_kl_hist(rows, out):
    plt.figure(figsize=(6, 4.2))
    for fam in FAM_ORDER:
        vals = [r["kl_qc"] for r in rows if r["family"] == fam]
        if not vals:
            continue
        plt.hist(np.log10(_safe_log(vals)), bins=50, alpha=0.5,
                 label=fam, color=FAM_COLOR.get(fam))
    plt.xlabel("log10 KL(q‖c)"); plt.ylabel("# instances")
    plt.title("Per-instance divergence distribution")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_nll_scatter(rows, out):
    plt.figure(figsize=(5, 5))
    for fam in FAM_ORDER:
        fr = [r for r in rows if r["family"] == fam]
        if not fr:
            continue
        idx = np.random.default_rng(0).choice(len(fr), min(2000, len(fr)), replace=False)
        q = [fr[i]["nll_q"] for i in idx]
        c = [fr[i]["nll_c"] for i in idx]
        plt.scatter(q, c, s=4, alpha=0.3, label=fam, color=FAM_COLOR.get(fam))
    lim = plt.xlim()
    hi = max(plt.xlim()[1], plt.ylim()[1])
    plt.plot([0, hi], [0, hi], "k--", lw=1)
    plt.xlabel("query-stream NLL(x_i)"); plt.ylabel("content-stream NLL(x_i)")
    plt.title("Token NLL: content vs query (on diagonal = consistent)")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_js_bar(rows, out):
    agg = defaultdict(list)
    for r in rows:
        agg[(r["project"], r["family"])].append(r["js"])
    projects = sorted({k[0] for k in agg})
    fams = [f for f in FAM_ORDER if any(k[1] == f for k in agg)]
    x = np.arange(len(projects)); w = 0.8 / max(1, len(fams))
    plt.figure(figsize=(max(6, 1.6 * len(projects)), 4.2))
    for i, fam in enumerate(fams):
        vals = [np.mean(agg.get((p, fam), [np.nan])) for p in projects]
        plt.bar(x + i * w, _safe_log(vals), w, label=fam, color=FAM_COLOR.get(fam))
    plt.yscale("log")
    plt.xticks(x + w * (len(fams) - 1) / 2, projects, rotation=20, ha="right", fontsize=8)
    plt.ylabel("mean JS divergence [log]"); plt.title("Symmetric divergence (JS)")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_ar_vs_random(rows, out):
    fams = [f for f in FAM_ORDER if any(r["family"] == f for r in rows)]
    modes = ["ar", "random"]
    x = np.arange(len(fams)); w = 0.35
    plt.figure(figsize=(6, 4.2))
    for j, mode in enumerate(modes):
        vals = []
        for fam in fams:
            v = [r["kl_qc"] for r in rows if r["family"] == fam and r["mode"] == mode]
            vals.append(np.mean(v) if v else np.nan)
        plt.bar(x + j * w, _safe_log(vals), w, label=mode)
    plt.yscale("log")
    plt.xticks(x + w / 2, fams); plt.ylabel("mean KL(q‖c) [log]")
    plt.title("Diagonal effect is order-independent (ar vs random)")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_layer_heatmap(lw, out):
    if not lw:
        return
    fams = [f for f in FAM_ORDER if any(r["family"] == f for r in lw)]
    layers = sorted({r["layer"] for r in lw})
    M = np.full((len(fams), len(layers)), np.nan)
    for i, fam in enumerate(fams):
        for j, l in enumerate(layers):
            vals = [r["kl_qc_mean"] for r in lw if r["family"] == fam and r["layer"] == l]
            if vals:
                M[i, j] = np.log10(max(np.mean(vals), 1e-8))
    plt.figure(figsize=(max(6, 0.25 * len(layers)), 2.5))
    im = plt.imshow(M, aspect="auto", cmap="magma")
    plt.colorbar(im, label="log10 KL(q‖c)")
    plt.yticks(range(len(fams)), fams); plt.xlabel("layer")
    plt.title("KL by family × layer (logit-lens)")
    plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_leak(summaries, out):
    fams, masked, real = [], [], []
    for s in summaries:
        lc = s.get("leak_control")
        if not lc:
            continue
        fams.append(f"{s['family']}\n{s['project'].split('_')[-1]}")
        masked.append(max(lc.get("x0_nll_masked_target", np.nan), 1e-8))
        real.append(max(lc.get("x0_nll_real_target", np.nan), 1e-8))
    if not fams:
        return
    x = np.arange(len(fams)); w = 0.38
    plt.figure(figsize=(max(6, 1.2 * len(fams)), 4.2))
    plt.bar(x, real, w, label="real token at target (leak)", color="#888")
    plt.bar(x + w, masked, w, label="masked target (true predict)", color="#48a")
    plt.yscale("log")
    plt.xticks(x + w / 2, fams, fontsize=8)
    plt.ylabel("content-stream NLL(x_i) [log]")
    plt.title("Prop.1 leakage: content stream trivially recovers x_i when target is real")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def fig_logitcos(rows, out):
    sub = [r for r in rows if r["mode"] == "ar"] or rows
    plt.figure(figsize=(6, 4.2))
    for fam in FAM_ORDER:
        fr = [r for r in sub if r["family"] == fam]
        if not fr:
            continue
        byc = defaultdict(list)
        for r in fr:
            byc[r["context_size"]].append(r["logit_cos"])
        cs = sorted(byc)
        means = [np.mean(byc[c]) for c in cs]
        plt.plot(cs, means, "o-", label=fam, color=FAM_COLOR.get(fam))
    plt.xscale("log")
    plt.xlabel("# context tokens"); plt.ylabel("cosine(query-logits, content-logits)")
    plt.title("Logit cosine similarity vs context")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()


def write_markdown(rows, summaries, in_dir, fig_dir):
    agg = defaultdict(list)
    for r in rows:
        agg[(r["project"], r["family"], r["mode"])].append(r["kl_qc"])
    lines = ["# Consistency / Leakage Probe — 汇总\n",
             "## 末层 KL(query ‖ content)（越低越一致）\n",
             "| project | family | mode | n | KL mean | KL median | top1-agree |",
             "|---|---|---|---|---|---|---|"]
    t1 = defaultdict(list)
    for r in rows:
        t1[(r["project"], r["family"], r["mode"])].append(r["top1_agree"])
    for k in sorted(agg):
        v = np.array(agg[k])
        lines.append(f"| {k[0]} | {k[1]} | {k[2]} | {len(v)} | "
                     f"{v.mean():.3e} | {np.median(v):.3e} | "
                     f"{100*np.mean(t1[k]):.1f}% |")
    lines.append("\n## Leak control（content 流目标位 nll）\n")
    lines.append("| project | family | masked-target nll | real-target nll (leak) |")
    lines.append("|---|---|---|---|")
    for s in summaries:
        lc = s.get("leak_control") or {}
        lines.append(f"| {s['project']} | {s['family']} | "
                     f"{lc.get('x0_nll_masked_target', float('nan')):.3e} | "
                     f"{lc.get('x0_nll_real_target', float('nan')):.3e} |")
    lines.append("\n## API 自检（hook vs return_both_streams 末层 max|Δ|）\n")
    for s in summaries:
        c = s.get("dual_stream_api_check")
        if c:
            lines.append(f"- {s['project']} ({s['family']}): "
                         f"xT={c['xt_max_abs_diff']:.2e}, x0={c['x0_max_abs_diff']:.2e}")
    lines.append("\n## 图\n")
    for png in sorted(glob.glob(os.path.join(fig_dir, "*.png"))):
        rel = os.path.relpath(png, in_dir)
        lines.append(f"![{os.path.basename(png)}]({rel})")
    with open(os.path.join(in_dir, "CONSISTENCY_REPORT.md"), "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="output_consistency")
    args = ap.parse_args()

    rows = load_instances(args.in_dir)
    lw = load_layerwise(args.in_dir)
    summaries = load_summaries(args.in_dir)
    if not rows:
        print(f"[plot] {args.in_dir} 下没有 *_instances.csv"); sys.exit(1)
    fig_dir = os.path.join(args.in_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)
    j = lambda n: os.path.join(fig_dir, n)

    fig_headline(rows, j("01_headline_kl_bar.png"))
    fig_kl_vs_context(rows, "ar", j("02_kl_vs_context_ar.png"))
    fig_kl_vs_context(rows, "random", j("03_kl_vs_context_random.png"))
    fig_kl_vs_layer(lw, j("04_kl_vs_layer.png"))
    fig_top1_vs_context(rows, j("05_top1_agree_vs_context.png"))
    fig_kl_hist(rows, j("06_kl_hist.png"))
    fig_nll_scatter(rows, j("07_nll_scatter.png"))
    fig_js_bar(rows, j("08_js_bar.png"))
    fig_ar_vs_random(rows, j("09_ar_vs_random_bar.png"))
    fig_layer_heatmap(lw, j("10_layer_heatmap.png"))
    fig_leak(summaries, j("11_leak_control_bar.png"))
    fig_logitcos(rows, j("12_logitcos_vs_context.png"))

    write_markdown(rows, summaries, args.in_dir, fig_dir)
    print(f"[plot] 图与 CONSISTENCY_REPORT.md 已写到 {args.in_dir}")


if __name__ == "__main__":
    main()
