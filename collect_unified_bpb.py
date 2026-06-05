#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
汇总 output_eval/*/unified_lr_bpb_*.json  ->  对比表 + markdown + csv。

分组维度：(dataset, project, decode_granularity)。同一组取时间戳最新的一份。
  - dataset：从 JSON 的 source 字段推断（如 wikitext-2-raw-v1 / paloma_c4_en / 本地文件名），
    支持多数据集分节出表（WikiText / C4 / ...）。
  - granularity：g=1/2/4 并排（DLM 的并行解码税曲线）；ar/plm 为 None。
合并 native BPB（写死在 NATIVE_BPB，取自 RESULTS.md；PLM 用 random-mode headline 值），
算 Δ = unified - native，并给出 native vs unified 排序翻转。

输出：控制台表、output_eval/unified_lr_bpb_table.md、output_eval/unified_lr_bpb_summary.csv。
用法（仓库根目录）：python collect_unified_bpb.py
"""
import os, glob, json, re, csv

ROOT = "./output_eval"

# native BPB（各族原生估计量；RESULTS.md。PLM 为 random-mode headline）。key:(scale,init)->{family:bpb}
NATIVE_BPB = {
    ("0.6B", "scratch"): {"ar": 0.823, "selfless": 0.971, "xlnet": 0.963,
                          "sdar": 0.904, "llada": 0.931, "dream": 0.936},
    ("0.6B", "preload"): {"ar": 0.771, "selfless": 0.930, "xlnet": 0.930,
                          "sdar": 0.846, "llada": 0.881, "dream": 0.882},
    ("342M", "scratch"): {"ar": 0.860, "selfless": 1.009, "xlnet": 1.004,
                          "sdar": 0.936, "llada": 0.963, "dream": 0.969},
}

FAM_ORDER = ["ar", "selfless", "xlnet", "sdar", "llada", "dream"]


def parse(proj_key):
    fam = ("selfless" if "selfless" in proj_key else "xlnet" if "xlnet" in proj_key else
           "llada" if "llada" in proj_key else "dream" if "dream" in proj_key else
           "sdar" if "sdar" in proj_key else "ar")
    scale = "342M" if "342M" in proj_key else "0.6B"
    init = "preload" if "preload" in proj_key else "scratch"
    return scale, fam, init


def dataset_label(source):
    """从 source 字段推断简短数据集名。
    HF 形如 'wikitext/wikitext-2-raw-v1/test' / 'paloma_c4_en/None/test'；本地为文件路径。"""
    if not source:
        return "unknown"
    s = str(source).strip()
    base = os.path.basename(s)
    if "/" in s and "." in base:                 # 像文件路径
        return os.path.splitext(base)[0]
    parts = s.split("/")
    ds = parts[0]
    cfg = parts[1] if len(parts) > 1 else ""
    return cfg if cfg and cfg.lower() != "none" else ds


def collect():
    """返回 {(dataset, proj_key, g): record}，同组取时间戳最新。"""
    by_key = {}
    for d in sorted(os.listdir(ROOT)):
        sub = os.path.join(ROOT, d)
        if not os.path.isdir(sub):
            continue
        fs = glob.glob(os.path.join(sub, "unified_lr_bpb_*.json"))
        if not fs:
            continue
        proj_key = re.sub(r"-lm-eval$", "", d)
        buckets = {}
        for f in fs:
            try:
                r = json.load(open(f))
            except Exception:
                continue
            ds = dataset_label(r.get("source"))
            g = r.get("decode_granularity")
            buckets.setdefault((ds, g), []).append((os.path.basename(f), f, r))
        for (ds, g), items in buckets.items():
            _, f, _ = max(items, key=lambda x: x[0])     # 文件名含时间戳，字典序即时间序
            by_key[(ds, proj_key, g)] = json.load(open(f))
    return by_key


def fam_sort_key(item):
    (_ds, proj_key, g), _ = item
    _, fam, _ = parse(proj_key)
    gnum = g if isinstance(g, int) else -1               # ar/plm(None) 排该族首位
    return (FAM_ORDER.index(fam), gnum)


def main():
    by_key = collect()
    if not by_key:
        print("没找到 unified_lr_bpb_*.json，先跑 run_unified_bpb.sh")
        raise SystemExit

    datasets = sorted({k[0] for k in by_key})
    groups = [("342M", "scratch"), ("0.6B", "scratch"), ("0.6B", "preload")]
    md_lines = ["# Unified Left-to-Right BPB", ""]
    csv_rows = []

    for ds in datasets:
        print(f"\n{'#'*72}\n# dataset: {ds}\n{'#'*72}")
        md_lines += [f"# dataset: {ds}", ""]

        for scale, init in groups:
            sub = {k: v for k, v in by_key.items()
                   if k[0] == ds and parse(k[1])[0] == scale and parse(k[1])[2] == init}
            if not sub:
                continue
            nat = NATIVE_BPB.get((scale, init), {})

            title = f"{scale} ({init})"
            print(f"\n{'='*64}\n{title}\n{'='*64}")
            print(f"{'family':<10}{'g':>3}{'native':>9}{'unified L→R':>13}{'Δ':>8}{'strict':>8}")
            md_lines += [f"## {ds} — {title}", "",
                         "| family | g | native BPB | unified L→R | Δ | strict |",
                         "|---|---|---|---|---|---|"]

            # 排序翻转：每族取最严格的 unified 值（DLM 取最小 g）
            uni_headline = {}
            for (_d, proj_key, g), r in sub.items():
                _, fam, _ = parse(proj_key)
                gkey = g if isinstance(g, int) else 0
                cur = uni_headline.get(fam)
                if cur is None or gkey < cur[0]:
                    uni_headline[fam] = (gkey, r["unified_lr_bpb"])

            for (_d, proj_key, g), r in sorted(sub.items(), key=fam_sort_key):
                _, fam, _ = parse(proj_key)
                u = r["unified_lr_bpb"]
                nb = nat.get(fam)
                gstr = str(g) if isinstance(g, int) else "-"
                nbstr = f"{nb:.3f}" if nb is not None else "-"
                dstr = f"{u - nb:+.3f}" if nb is not None else "-"
                strict = "yes" if r.get("strict_left_to_right") else "no"
                print(f"{fam:<10}{gstr:>3}{nbstr:>9}{u:>13.3f}{dstr:>8}{strict:>8}")
                md_lines.append(f"| {fam} | {gstr} | {nbstr} | {u:.3f} | {dstr} | {strict} |")
                csv_rows.append([ds, proj_key, scale, fam, init, g,
                                 r.get("strict_left_to_right"),
                                 nb if nb is not None else "", u,
                                 (u - nb) if nb is not None else "",
                                 r.get("attention_regime"), r.get("tokens_scored"),
                                 r.get("corpus_bytes")])

            present = [f for f in FAM_ORDER if f in uni_headline]
            uni_rank = sorted(present, key=lambda f: uni_headline[f][1])
            print("  unified 排序(低→高): " + " < ".join(uni_rank))
            md_lines.append("")
            if nat:
                nat_present = [f for f in FAM_ORDER if f in nat]
                nat_rank = sorted(nat_present, key=lambda f: nat[f])
                print("  native  排序(低→高): " + " < ".join(nat_rank))
                md_lines += [f"- native 排序(低→高): `{' < '.join(nat_rank)}`",
                             f"- unified 排序(低→高): `{' < '.join(uni_rank)}`", ""]

    md_path = os.path.join(ROOT, "unified_lr_bpb_table.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    csv_path = os.path.join(ROOT, "unified_lr_bpb_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "project", "scale", "family", "init", "decode_granularity",
                    "strict_left_to_right", "native_bpb", "unified_lr_bpb", "delta",
                    "attention_regime", "tokens_scored", "corpus_bytes"])
        w.writerows(csv_rows)

    print(f"\n>>> markdown -> {md_path}")
    print(f">>> csv      -> {csv_path}")


if __name__ == "__main__":
    main()