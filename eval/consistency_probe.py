#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
consistency_probe.py — 单流推理一致性 + 残差泄露 probe（验证 PAPER_PLAN §4 命题 1–3）

要回答的问题
------------
对一个 masked 目标位 i（上下文为真 token），比较两条读出路径在 i 处的 logits：
  - query / XT 流读出   : 训练 & likelihood 路径（XT 输入全 [MASK]，无对角线）。这是“正确”的预测分布。
  - content / X0 流读出 : 单流推理路径（X0 在目标位输入 [MASK]）。这是真正生成时读的流。

命题预测：
  - 命题 2/3（一致性，区分 XLNet/Selfless）：
      Selfless 两流用同一无对角线 mask → x0[i] ≡ xT[i]，KL≈0（数值级）。
      XLNet content 流 mask 含对角线（v_kv>=v_q）→ 目标位多 attend 自身 [MASK]，x0[i]≠xT[i]，KL>0。
  - 命题 1（残差泄露，两族都成立）：把目标位喂真 token（不 mask），content 流经残差直接泄露 x_i，
      x0 读出平凡命中 x_i（nll≈0）。这解释“训练期为何必须两条流”。见 --leak_control。

设计
----
* 每条窗口序列复用多次：
    - ar 模式：目标位 t 的上下文恰为 {0..t-1}，故 context_size == t；扫 t = 扫“非 mask 上下文数量”。
      mask 整个后缀 [t:]（等价于只 mask t，但更贴近真实生成状态：未来尚未生成）。
    - random 模式：随机揭示 k 个位置作为上下文（任意顺序，PLM 训练分布内），mask 其余；测任意顺序下的一致性。
* 取数：forward hook 抓每层 (X0, XT) 的 hidden，只在目标位 gather → 逐层 logit-lens 曲线；末层为精确读出。
* 仅对 {selfless, xlnet} 有意义（对角线差异）。其它族报错。

输出（写到 --out_dir）
    <run_id>_instances.csv   每个 probe 实例一行（末层指标）
    <run_id>_layerwise.csv   按 (mode, layer) 聚合的逐层指标
    <run_id>_summary.json    汇总 + 元数据 + leak control + dual-stream API 自检

用法
    PY=uv run python   # 或 python
    $PY eval/consistency_probe.py --config configs/xxx_selfless.yaml \
        --hf_dataset wikitext --hf_config wikitext-2-raw-v1 --hf_split test \
        --num_windows 64 --max_len 512 --out_dir output_consistency
冒烟：加 --max_instances 16
"""

import os
import sys
import json
import math
import argparse
import random
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# 仓库根目录
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from utils.utils import load_model_tokenizer, get_selfless_mask, get_xlnet_mask

LN2 = math.log(2.0)


# --------------------------------------------------------------------------- #
#  族识别（仅 selfless / xlnet）
# --------------------------------------------------------------------------- #
def detect_family(project: str) -> str:
    p = project.lower()
    if "xlnet" in p:
        return "xlnet"
    if "selfless" in p:
        return "selfless"
    raise ValueError(
        f"consistency_probe 仅支持 selfless / xlnet（对角线消融），但 project={project!r}"
    )


# --------------------------------------------------------------------------- #
#  v_sample 构造
# --------------------------------------------------------------------------- #
def ar_v_sample_row(L, device, eps=1e-3):
    """严格降序 [L]：pos0 最大、posL-1 最小，与 unified_lr_bpb._ar_v_sample 一致。
    位置 t 只 attend {0..t-1}（v_kv>v_q）。位置 0 attend 空（其 nll 不计、不作目标）。"""
    pos = torch.arange(L, device=device, dtype=torch.float32)
    if L > 1:
        v = 1 - eps - (1 - 2 * eps) * pos / (L - 1)
    else:
        v = torch.ones(1, device=device) * (1 - eps)
    return v  # [L]


# --------------------------------------------------------------------------- #
#  逐层流捕获器：hook 每个 decoder layer 输出 (X0_hidden, XT_hidden)，只在目标位 gather
# --------------------------------------------------------------------------- #
class StreamCapturer:
    def __init__(self, model):
        self.model = model
        self.layers = model.model.layers
        self.handles = []
        self.target_idx = None     # [B] long
        self.x0 = []               # list over layers of [B, H]
        self.xt = []

    def _make_hook(self):
        def hook(module, inputs, output):
            # output = (X0_hidden_states, XT_hidden_states)，均 [B, L, H]（XT 可能为 None）
            x0_h, xt_h = output[0], output[1]
            B = x0_h.shape[0]
            ar = torch.arange(B, device=x0_h.device)
            self.x0.append(x0_h[ar, self.target_idx].detach())
            if xt_h is not None:
                self.xt.append(xt_h[ar, self.target_idx].detach())
            else:
                self.xt.append(None)
        return hook

    def register(self):
        for layer in self.layers:
            self.handles.append(layer.register_forward_hook(self._make_hook()))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self, target_idx):
        self.target_idx = target_idx
        self.x0 = []
        self.xt = []


# --------------------------------------------------------------------------- #
#  把每层 gather 的 hidden 经 final-norm + lm_head 投到词表（logit-lens）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def hidden_to_logits(model, hidden):  # hidden [B, H] -> [B, V]
    h = model.model.norm(hidden)
    return model.lm_head(h).float()


# --------------------------------------------------------------------------- #
#  family-specific 前向（保证 XT 流被构造）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def forward_family(model, family, seq, v_sample, device):
    """seq [B,L] long; v_sample [B,L] float。触发 hook 捕获每层两流。返回 None（结果在 capturer）。"""
    L = seq.shape[-1]
    if family == "selfless":
        mask = get_selfless_mask(v_sample=v_sample, seq_len=L, device=device)
        # 必须 calculate_likelihood=True 才会构造 XT 流
        model(X0_input_ids=seq, attention_mask=mask, calculate_likelihood=True)
    elif family == "xlnet":
        qmask, kvmask = get_xlnet_mask(v_sample=v_sample, seq_len=L, device=device)
        model(X0_input_ids=seq, query_attention_mask=qmask, kv_attention_mask=kvmask)
    else:
        raise ValueError(family)


@torch.no_grad()
def dual_stream_logits(model, family, seq, v_sample, device, targets):
    """用模型自带的 return_both_streams API 取末层 (xT_logits, x0_logits) 在目标位的读出，
    用于和 hook 路径交叉校验。返回 (xt [B,V], x0 [B,V])。"""
    L = seq.shape[-1]
    B = seq.shape[0]
    ar = torch.arange(B, device=device)
    if family == "selfless":
        mask = get_selfless_mask(v_sample=v_sample, seq_len=L, device=device)
        out = model(X0_input_ids=seq, attention_mask=mask,
                    calculate_likelihood=True, return_both_streams=True)
    else:
        qmask, kvmask = get_xlnet_mask(v_sample=v_sample, seq_len=L, device=device)
        out = model(X0_input_ids=seq, query_attention_mask=qmask,
                    kv_attention_mask=kvmask, return_both_streams=True)
    xt = out.xT_logits[ar, targets].float()
    x0 = out.x0_logits[ar, targets].float()
    return xt, x0


# --------------------------------------------------------------------------- #
#  指标
# --------------------------------------------------------------------------- #
def pair_metrics(xt_logits, x0_logits, true_tokens):
    """xt_logits/x0_logits [B,V] (query=参照, content=被测)；true_tokens [B]。返回 dict of [B] numpy。"""
    lq = F.log_softmax(xt_logits, dim=-1)
    lc = F.log_softmax(x0_logits, dim=-1)
    pq = lq.exp()
    pc = lc.exp()
    kl_qc = (pq * (lq - lc)).sum(-1)
    kl_cq = (pc * (lc - lq)).sum(-1)
    m = 0.5 * (pq + pc)
    lm = (m + 1e-12).log()
    js = 0.5 * (pq * (lq - lm)).sum(-1) + 0.5 * (pc * (lc - lm)).sum(-1)
    top1_q = xt_logits.argmax(-1)
    top1_c = x0_logits.argmax(-1)
    top1_agree = (top1_q == top1_c).float()
    logit_l2 = (xt_logits - x0_logits).norm(dim=-1)
    logit_cos = F.cosine_similarity(xt_logits, x0_logits, dim=-1)
    idx = true_tokens.unsqueeze(-1)
    nll_q = -lq.gather(-1, idx).squeeze(-1)
    nll_c = -lc.gather(-1, idx).squeeze(-1)
    nll_absdiff = (nll_q - nll_c).abs()
    out = dict(kl_qc=kl_qc, kl_cq=kl_cq, js=js, top1_agree=top1_agree,
               logit_l2=logit_l2, logit_cos=logit_cos,
               nll_q=nll_q, nll_c=nll_c, nll_absdiff=nll_absdiff)
    return {k: v.detach().cpu().numpy() for k, v in out.items()}


# --------------------------------------------------------------------------- #
#  probe 实例构造
# --------------------------------------------------------------------------- #
def build_ar_instances(ids, context_sizes, mask_id, mask_target=True):
    """ids [L] long。对每个 t in context_sizes（即上下文长度，目标位=t），构造一条实例。
    mask 后缀 [t:]（mask_target=True）；leak 控制时 mask [t+1:]、保留真 t。返回 list of dict。"""
    L = ids.shape[0]
    out = []
    for t in context_sizes:
        if t < 1 or t >= L:
            continue
        row = ids.clone()
        if mask_target:
            row[t:] = mask_id            # 目标位及之后全部 mask（未来未生成）
        else:
            row[t + 1:] = mask_id        # 保留真 token 于 t —— leak 控制
        v = ar_v_sample_row(L, ids.device)
        out.append(dict(seq=row, v=v, target=t, ctx=t))
    return out


def build_random_instances(ids, context_sizes, n_per_k, mask_id, rng):
    """随机揭示 k 个位置作为上下文（任意顺序），mask 其余，预测一个随机目标位。
    context_size 记为实际被 attend 的位置数（= k + 锚点0）。"""
    L = ids.shape[0]
    device = ids.device
    out = []
    pool_all = list(range(1, L))
    for k in context_sizes:
        if k < 1 or k > L - 2:
            continue
        for _ in range(n_per_k):
            t = rng.randrange(1, L)
            pool = [j for j in pool_all if j != t]
            R = rng.sample(pool, min(k, len(pool)))
            v = torch.empty(L, device=device, dtype=torch.float32)
            # 位置0 作锚点（最高 v，恒被 attend），保留真 token，避免全空行问题
            v[0] = 1.0
            others = [j for j in range(1, L) if j not in R and j != t]
            v[t] = 0.40
            for j in R:
                v[j] = rng.uniform(0.50, 0.95)
            for j in others:
                v[j] = rng.uniform(0.02, 0.35)
            row = ids.clone()
            real = set(R) | {0}
            for j in range(L):
                if j not in real:           # t 与 others 一律 mask
                    row[j] = mask_id
            out.append(dict(seq=row, v=v, target=t, ctx=len(R) + 1))
    return out


# --------------------------------------------------------------------------- #
#  批处理执行（含逐层）
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_instances(model, family, instances, device, batch_size, capturer,
                  layerwise=True, true_tokens_from=None):
    """对一组实例执行前向，返回:
        rows: list of per-instance dict（末层指标 + 元数据）
        layer_kl: dict[layer] -> list of kl_qc（逐层 logit-lens，跨实例累计）
    true_tokens_from: 原始（未 mask）窗口 ids 的并行列表，用于取目标位真 token；
                      若 None 则用实例内 seq 在目标位的值（不可，已 mask）→ 必须提供。
    """
    rows = []
    n_layers = len(capturer.layers)
    layer_kl = {l: [] for l in range(n_layers)}
    layer_top1 = {l: [] for l in range(n_layers)}

    for s in range(0, len(instances), batch_size):
        chunk = instances[s:s + batch_size]
        B = len(chunk)
        L = chunk[0]["seq"].shape[0]
        seq = torch.stack([c["seq"] for c in chunk]).to(device)         # [B,L]
        v = torch.stack([c["v"] for c in chunk]).to(device)            # [B,L]
        targets = torch.tensor([c["target"] for c in chunk], device=device)
        true_tok = torch.tensor(
            [true_tokens_from[c["_widx"]][c["target"]] for c in chunk],
            device=device,
        )

        capturer.reset(targets)
        forward_family(model, family, seq, v, device)

        # 逐层捕获：x0[l] / xt[l] 均 [B,H]
        # 末层（=最后一层 hook 输出）作为精确读出
        xt_last_h = capturer.xt[-1]
        x0_last_h = capturer.x0[-1]
        if xt_last_h is None:
            raise RuntimeError("XT 流未被构造；selfless 需 calculate_likelihood=True。")
        xt_last = hidden_to_logits(model, xt_last_h)
        x0_last = hidden_to_logits(model, x0_last_h)

        # NaN 守卫（位置0 空行等异常）
        finite = torch.isfinite(xt_last).all(-1) & torch.isfinite(x0_last).all(-1)
        if not finite.all():
            bad = (~finite).sum().item()
            print(f"[warn] 丢弃 {bad} 个含 NaN/Inf 的实例（多为上下文为空的退化情形）")

        m = pair_metrics(xt_last, x0_last, true_tok)
        for b in range(B):
            if not bool(finite[b]):
                continue
            c = chunk[b]
            rows.append(dict(
                mode=c["mode"], window_idx=c["_widx"], target_pos=c["target"],
                context_size=c["ctx"],
                kl_qc=float(m["kl_qc"][b]), kl_cq=float(m["kl_cq"][b]),
                js=float(m["js"][b]), top1_agree=float(m["top1_agree"][b]),
                logit_l2=float(m["logit_l2"][b]), logit_cos=float(m["logit_cos"][b]),
                nll_q=float(m["nll_q"][b]), nll_c=float(m["nll_c"][b]),
                nll_absdiff=float(m["nll_absdiff"][b]),
            ))

        if layerwise:
            for l in range(n_layers):
                if capturer.xt[l] is None:
                    continue
                xt_l = hidden_to_logits(model, capturer.xt[l])
                x0_l = hidden_to_logits(model, capturer.x0[l])
                fl = torch.isfinite(xt_l).all(-1) & torch.isfinite(x0_l).all(-1)
                ml = pair_metrics(xt_l, x0_l, true_tok)
                for b in range(B):
                    if not bool(fl[b]):
                        continue
                    layer_kl[l].append(float(ml["kl_qc"][b]))
                    layer_top1[l].append(float(ml["top1_agree"][b]))

    return rows, layer_kl, layer_top1


# --------------------------------------------------------------------------- #
#  leak control（命题 1）：content 流在“真 token 目标位”是否平凡泄露
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_leak_control(model, family, windows, context_sizes, mask_id, device,
                     capturer, max_inst=256):
    """对 (masked 目标) 与 (真 token 目标) 两种情形，分别报告 content 流在目标位的 nll。
    预测：真 token 情形 nll≈0（残差泄露，两族都成立）；masked 情形 = 正常预测 nll。"""
    def collect(mask_target):
        insts = []
        for widx, ids in enumerate(windows):
            for c in build_ar_instances(ids, context_sizes, mask_id, mask_target=mask_target):
                c["mode"] = "leak_real" if not mask_target else "leak_masked"
                c["_widx"] = widx
                insts.append(c)
        insts = insts[:max_inst]
        nlls = []
        for s in range(0, len(insts), 32):
            chunk = insts[s:s + 32]
            B = len(chunk)
            L = chunk[0]["seq"].shape[0]
            seq = torch.stack([c["seq"] for c in chunk]).to(device)
            v = torch.stack([c["v"] for c in chunk]).to(device)
            targets = torch.tensor([c["target"] for c in chunk], device=device)
            true_tok = torch.tensor([windows[c["_widx"]][c["target"]] for c in chunk],
                                    device=device)
            capturer.reset(targets)
            forward_family(model, family, seq, v, device)
            x0 = hidden_to_logits(model, capturer.x0[-1])
            lc = F.log_softmax(x0, dim=-1)
            nll_c = -lc.gather(-1, true_tok.unsqueeze(-1)).squeeze(-1)
            nll_c = nll_c[torch.isfinite(nll_c)]
            nlls.extend(nll_c.cpu().tolist())
        return float(np.mean(nlls)) if nlls else float("nan")

    return dict(x0_nll_masked_target=collect(True),
                x0_nll_real_target=collect(False))


# --------------------------------------------------------------------------- #
#  语料 / 窗口
# --------------------------------------------------------------------------- #
def load_text(args):
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            return f.read()
    if args.hf_dataset:
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
        lines = [r for r in ds["text"] if r is not None]
        return "".join(lines) if "wikitext" in (args.hf_config or "") else "\n\n".join(lines)
    raise ValueError("必须提供 --text_file 或 --hf_dataset")


def make_windows(tokenizer, text, max_len, num_windows, rng):
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    N = ids.shape[0]
    starts = list(range(0, max(1, N - max_len), max_len))
    rng.shuffle(starts)
    starts = starts[:num_windows]
    return [ids[s:s + max_len].clone() for s in starts if (s + max_len) <= N]


# --------------------------------------------------------------------------- #
def parse_context_sizes(spec, max_len):
    if spec:
        cs = [int(x) for x in spec.split(",")]
    else:
        cs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 384]
    return [c for c in cs if 1 <= c < max_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="lm_eval yaml（同 unified_lr_bpb）")
    ap.add_argument("--text_file", default=None)
    ap.add_argument("--hf_dataset", default=None)
    ap.add_argument("--hf_config", default=None)
    ap.add_argument("--hf_split", default="test")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--num_windows", type=int, default=64)
    ap.add_argument("--context_sizes", default=None, help="逗号分隔；默认 1..384 的幂级")
    ap.add_argument("--modes", default="ar,random", help="ar / random / 两者")
    ap.add_argument("--random_per_k", type=int, default=4, help="random 模式每个 k 采样几个目标")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--no_layerwise", action="store_true")
    ap.add_argument("--leak_control", action="store_true", default=True)
    ap.add_argument("--dual_stream_check", action="store_true", default=True,
                    help="用 return_both_streams API 在小批上交叉校验 hook 读出")
    ap.add_argument("--max_instances", type=int, default=None, help="冒烟用：限制实例总数")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="output_consistency")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    project = cfg.experiment.project
    family = detect_family(project)
    print(f"[consistency_probe] project={project} | family={family} | max_len={args.max_len}")

    model, tokenizer = load_model_tokenizer(config=cfg)
    model.eval()
    device = next(model.parameters()).device
    if device.type == "cpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    mask_id = model.config.mask_token_id

    capturer = StreamCapturer(model)
    capturer.register()

    text = load_text(args)
    windows = make_windows(tokenizer, text, args.max_len, args.num_windows, rng)
    windows = [w.to(device) for w in windows]
    print(f"  windows={len(windows)} | mask_id={mask_id}")

    context_sizes = parse_context_sizes(args.context_sizes, args.max_len)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # ---- 构造所有实例 ----
    instances = []
    for widx, ids in enumerate(windows):
        if "ar" in modes:
            for c in build_ar_instances(ids, context_sizes, mask_id, mask_target=True):
                c["mode"] = "ar"; c["_widx"] = widx; instances.append(c)
        if "random" in modes:
            for c in build_random_instances(ids, context_sizes, args.random_per_k, mask_id, rng):
                c["mode"] = "random"; c["_widx"] = widx; instances.append(c)
    rng.shuffle(instances)
    if args.max_instances:
        instances = instances[:args.max_instances]
    print(f"  total probe instances = {len(instances)}")

    # ---- 主执行 ----
    rows, layer_kl, layer_top1 = run_instances(
        model, family, instances, device, args.batch_size, capturer,
        layerwise=not args.no_layerwise, true_tokens_from=windows,
    )

    # ---- dual-stream API 交叉校验（小批）----
    api_check = None
    if args.dual_stream_check and len(instances) > 0:
        chunk = instances[:min(8, len(instances))]
        B = len(chunk)
        seq = torch.stack([c["seq"] for c in chunk]).to(device)
        v = torch.stack([c["v"] for c in chunk]).to(device)
        targets = torch.tensor([c["target"] for c in chunk], device=device)
        capturer.reset(targets)
        forward_family(model, family, seq, v, device)
        xt_hook = hidden_to_logits(model, capturer.xt[-1])
        x0_hook = hidden_to_logits(model, capturer.x0[-1])
        xt_api, x0_api = dual_stream_logits(model, family, seq, v, device, targets)
        api_check = dict(
            xt_max_abs_diff=float((xt_hook - xt_api).abs().max().item()),
            x0_max_abs_diff=float((x0_hook - x0_api).abs().max().item()),
        )
        print(f"  [api-check] hook vs return_both_streams 末层 max|Δ|: {api_check}")

    # ---- leak control ----
    leak = None
    if args.leak_control:
        leak = run_leak_control(model, family, windows, context_sizes, mask_id, device, capturer)
        print(f"  [leak] content-流目标位 nll: {leak}")

    capturer.remove()

    # ---- 写盘 ----
    run_id = f"{project}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    inst_path = os.path.join(args.out_dir, f"{run_id}_instances.csv")
    with open(inst_path, "w") as f:
        cols = ["project", "family", "mode", "window_idx", "target_pos", "context_size",
                "kl_qc", "kl_cq", "js", "top1_agree", "logit_l2", "logit_cos",
                "nll_q", "nll_c", "nll_absdiff"]
        f.write(",".join(cols) + "\n")
        for r in rows:
            r2 = dict(project=project, family=family, **r)
            f.write(",".join(str(r2[c]) for c in cols) + "\n")

    lw_path = os.path.join(args.out_dir, f"{run_id}_layerwise.csv")
    with open(lw_path, "w") as f:
        f.write("project,family,layer,kl_qc_mean,kl_qc_median,top1_agree_mean,n\n")
        for l in sorted(layer_kl.keys()):
            vals = layer_kl[l]
            t1 = layer_top1[l]
            if not vals:
                continue
            f.write(f"{project},{family},{l},{np.mean(vals)},{np.median(vals)},"
                    f"{np.mean(t1) if t1 else float('nan')},{len(vals)}\n")

    # 汇总
    def agg(mode):
        sub = [r for r in rows if r["mode"] == mode]
        if not sub:
            return None
        kl = np.array([r["kl_qc"] for r in sub])
        return dict(n=len(sub), kl_qc_mean=float(kl.mean()), kl_qc_median=float(np.median(kl)),
                    kl_qc_p90=float(np.percentile(kl, 90)),
                    top1_agree=float(np.mean([r["top1_agree"] for r in sub])),
                    nll_absdiff_mean=float(np.mean([r["nll_absdiff"] for r in sub])))

    summary = dict(
        run_id=run_id, project=project, family=family,
        max_len=args.max_len, num_windows=len(windows),
        n_instances=len(rows), context_sizes=context_sizes, modes=modes,
        per_mode={m: agg(m) for m in modes},
        leak_control=leak, dual_stream_api_check=api_check,
        layerwise=[dict(layer=l, kl_qc_mean=float(np.mean(layer_kl[l])))
                   for l in sorted(layer_kl) if layer_kl[l]],
        timestamp=datetime.now().isoformat(),
    )
    sum_path = os.path.join(args.out_dir, f"{run_id}_summary.json")
    with open(sum_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[done] {family} | instances={len(rows)}")
    for m in modes:
        a = agg(m)
        if a:
            print(f"  mode={m:7s} KL(q||c) mean={a['kl_qc_mean']:.4e} "
                  f"median={a['kl_qc_median']:.4e} top1_agree={a['top1_agree']:.3f}")
    print(f"  写出: {inst_path}\n         {lw_path}\n         {sum_path}")


if __name__ == "__main__":
    main()
