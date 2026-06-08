#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_likelihood.py — 统一收集 native + unified L→R 两类 BPB 结果，估计量作为一等维度。

读取（兼容现有结果）：
  - output_eval/*/native_bpb_*.json        (text_likelihood.py) -> estimator="native"
  - output_eval/*/unified_lr_bpb_*.json     (unified_lr_bpb.py)  -> estimator="lr_g{g}"
分组：(dataset, project, estimator) 取时间戳最新。
合并 native 也无需写死 —— native 现在由 text_likelihood.py 在同一字节口径下算出。

输出：
  - output_eval/likelihood_long.csv   (长表：每行一个 dataset×model×estimator 的 BPB；viz 用)
  - output_eval/likelihood_table.md   (按 dataset / scale-init 分节的对比表)
  - 控制台对比表 + 排序

用法（仓库根目录）：python collect_likelihood.py
"""
import os, glob, json, re, csv

ROOT = "./output_eval"
FAM_ORDER = ["ar", "selfless", "xlnet", "sdar", "llada", "dream"]


def parse(proj_key):
    fam = ("selfless" if "selfless" in proj_key else "xlnet" if "xlnet" in proj_key else
           "llada" if "llada" in proj_key else "dream" if "dream" in proj_key else
           "sdar" if "sdar" in proj_key else "ar")
    scale = "342M" if "342M" in proj_key else "0.6B"
    init = "preload" if "preload" in proj_key else "scratch"
    return scale, fam, init


def dataset_label(source):
    if not source:
        return "unknown"
    s = str(source).strip()
    base = os.path.basename(s)
    if "/" in s and "." in base:
        return os.path.splitext(base)[0]
    parts = s.split("/")
    return parts[1] if len(parts) > 1 and parts[1] and parts[1].lower() != "none" else parts[0]


def estimator_label(rec):
    if rec.get("estimator") == "native" or "native_bpb" in rec:
        return "native"
    g = rec.get("decode_granularity")
    if isinstance(g, int):
        return f"lr_g{g}"
    # AR/PLM：decode_granularity 为 None，但其 L→R 是精确逐 token = 严格 g1 等价，归入 lr_g1
    return "lr_g1" if rec.get("strict_left_to_right") else "lr"


def bpb_value(rec):
    if "native_bpb" in rec:
        return rec["native_bpb"]
    return rec.get("unified_lr_bpb")


def collect():
    """返回 list[row dict]，每行一个 (dataset, project, estimator) 的最新记录。"""
    by_key = {}
    for d in sorted(os.listdir(ROOT)):
        sub = os.path.join(ROOT, d)
        if not os.path.isdir(sub):
            continue
        proj_key = re.sub(r"-lm-eval$", "", d)
        for pat in ("native_bpb_*.json", "unified_lr_bpb_*.json"):
            for f in glob.glob(os.path.join(sub, pat)):
                try:
                    rec = json.load(open(f))
                except Exception:
                    continue
                ds = dataset_label(rec.get("source"))
                est = estimator_label(rec)
                key = (ds, proj_key, est)
                # 同组取文件名时间戳最新
                prev = by_key.get(key)
                if prev is None or os.path.basename(f) > prev[0]:
                    by_key[key] = (os.path.basename(f), rec)

    rows = []
    for (ds, proj_key, est), (_fn, rec) in by_key.items():
        scale, fam, init = parse(proj_key)
        rows.append({
            "dataset": ds, "project": proj_key, "scale": scale, "family": fam,
            "init": init, "estimator": est,
            "decode_granularity": rec.get("decode_granularity"),
            "bpb": bpb_value(rec),
            "strict": rec.get("strict_left_to_right"),
            "attention_regime": rec.get("attention_regime"),
            "corpus_bytes": rec.get("corpus_bytes"),
            "corpus_tokens": rec.get("corpus_tokens"),
            "source": rec.get("source"),
        })
    return rows


def est_sort(est):
    if est == "native":
        return (0, 0)
    m = re.match(r"lr_g(\d+)", est)
    return (1, int(m.group(1))) if m else (1, 0)


def main():
    rows = collect()
    if not rows:
        print("没找到 native_bpb_*.json / unified_lr_bpb_*.json，先跑 text_likelihood.py / unified_lr_bpb.py")
        raise SystemExit

    # 写长表 CSV（viz 用）
    csv_path = os.path.join(ROOT, "likelihood_long.csv")
    cols = ["dataset", "project", "scale", "family", "init", "estimator",
            "decode_granularity", "bpb", "strict", "attention_regime",
            "corpus_bytes", "corpus_tokens", "source"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["dataset"], r["scale"], r["init"],
                                             FAM_ORDER.index(r["family"]), est_sort(r["estimator"]))):
            w.writerow(r)

    # 控制台 + markdown：按 dataset / (scale,init) 分节，列=estimator
    datasets = sorted({r["dataset"] for r in rows})
    estimators_all = sorted({r["estimator"] for r in rows}, key=est_sort)
    md = ["# Likelihood (BPB) — native vs unified L→R", ""]
    groups = [("342M", "scratch"), ("0.6B", "scratch"), ("0.6B", "preload")]

    for ds in datasets:
        print(f"\n{'#'*72}\n# dataset: {ds}\n{'#'*72}")
        md += [f"# dataset: {ds}", ""]
        for scale, init in groups:
            sel = [r for r in rows if r["dataset"] == ds and r["scale"] == scale and r["init"] == init]
            if not sel:
                continue
            ests = [e for e in estimators_all if any(r["estimator"] == e for r in sel)]
            # family -> {estimator: bpb}
            tab = {}
            for r in sel:
                tab.setdefault(r["family"], {})[r["estimator"]] = r["bpb"]
            fams = [f for f in FAM_ORDER if f in tab]

            print(f"\n=== {scale} ({init}) ===")
            hdr = f"{'family':<10}" + "".join(f"{e:>11}" for e in ests)
            print(hdr)
            md += [f"## {ds} — {scale} ({init})", "",
                   "| family | " + " | ".join(ests) + " |",
                   "|---|" + "|".join(["---"] * len(ests)) + "|"]
            for fam in fams:
                cells = []
                for e in ests:
                    v = tab[fam].get(e)
                    cells.append(f"{v:.3f}" if isinstance(v, (int, float)) else "-")
                print(f"{fam:<10}" + "".join(f"{c:>11}" for c in cells))
                md.append(f"| {fam} | " + " | ".join(cells) + " |")
            # 排序（按各 estimator 列）
            md.append("")
            for e in ests:
                present = [f for f in fams if isinstance(tab[f].get(e), (int, float))]
                rank = sorted(present, key=lambda f: tab[f][e])
                line = f"  {e:>8} 排序(低→高): " + " < ".join(rank)
                print(line)
                md.append(f"- `{e}` 排序(低→高): `{' < '.join(rank)}`")
            md.append("")

    with open(os.path.join(ROOT, "likelihood_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"\n>>> 长表 CSV -> {csv_path}")
    print(f">>> markdown -> {os.path.join(ROOT, 'likelihood_table.md')}")


if __name__ == "__main__":
    main()