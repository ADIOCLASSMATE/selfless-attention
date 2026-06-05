#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Left-to-Right (L->R) BPB estimator  —  EXPERIMENTS.md P0-#1
====================================================================

目的 / Purpose
--------------
把所有模型族（AR / PLM[selfless,xlnet] / DLM[llada,dream,sdar]）放到**同一个**
左到右 teacher-forced 估计量下，重新计算 WikiText / C4 的 BPB：

    对每个位置 i，计算  -log p(x_i | x_<i)   （只给左侧全确定上下文，右侧不可见）
    BPB = ( Σ_i -log p(x_i|x_<i) [nats] ) / ( ln(2) * 语料 UTF-8 字节数 )

各族的原生 BPB 报的是对 -log p(x) 的**不同界**：
  - AR        : 精确链式 NLL（其原生即 L->R，本脚本对它是恒等）
  - PLM       : E_σ 的随机顺序期望 NLL —— 用 ar-mode v_sample 即坍缩成 L->R
  - DLM       : 随机 mask 的 ELBO（双向、含 [MASK]）—— 本脚本强制改成左到右 block 半自回归

【关键】统一的是**信息集**（只给左侧上下文），不是 attention 形状。每个模型必须在它
**训练时的 attention 制度**下被查询，否则得到的概率无效（OOD）。各族制度：
  - ar              : causal，shift（logits[i]->token[i+1]），不涉及 mask token
  - selfless/xlnet  : ar-mode 两流，无 shift（logits[i]->token[i]），**绝不 attend mask token**
  - llada           : full bidirectional，无 shift，每次前向 attend 整条右侧 mask（训练即 full attn，in-dist）
  - dream           : full bidirectional，shift，attend 整条右侧 mask（同上）
  - sdar            : **block-causal**，attention 块大小固定=config.block_size(=4)；
                      解码粒度 g∈{1..4} 可选：g=1 严格逐 token L->R（与他族同档可比），
                      g=4 块并行（上界）。块内随机 masking 训练 => 块内逐 token 揭示 in-dist。
                      g 不能超过 attention 块大小(=4)。

预期结论：统一到 L->R 后，DLM 的 BPB 优势相对其原生估计量会显著变差/反转，
从而证明“DLM 低 BPB 是估计量假象，而非更好的语言建模”。

公平性保证：所有模型用**同一份语料、同一份 UTF-8 字节数**归一化（与 tokenizer 无关）。

用法 / Usage
-----------
    # 单卡
    python eval/unified_lr_bpb.py \
        --config configs/selfless/lm_eval_selfless_0.6B_ar+ar.yaml \
        --hf_dataset wikitext --hf_config wikitext-2-raw-v1 --hf_split test \
        --block_size 128

    # 用本地纯文本语料（无网络时）
    python eval/unified_lr_bpb.py \
        --config configs/llada/lm_eval_llada_0.6B.yaml \
        --text_file data/wikitext2_test.txt \
        --block_size 64

参数 / Args
----------
    --config       与现有 eval worker 相同的 lm_eval yaml（决定 model_path / 族）
    --hf_dataset / --hf_config / --hf_split   从 HF 拉语料（需联网）
    --text_file    本地纯文本语料（整文件视为一个 corpus），与 --hf_dataset 二选一
    --block_size B 仅对 DLM 生效：block 半自回归的块大小。
                   B=1 为**精确 L->R**（O(L) 次前向，最慢最准）；
                   B>1 为块并行近似（O(L/B) 次前向，是精确 L->R NLL 的上界）。
                   AR / PLM 不受影响（单次前向即精确）。
    --limit K      只评前 K 个 doc（快速 sanity）
    --max_len      覆盖 yaml 的 max_len（滑窗长度）

输出 / Output
------------
    打印 total_nll / tokens / BPB / word-PPL，并写
    {output_path}/{project}/unified_lr_bpb_{timestamp}.json

实现说明
--------
- 复用 utils.load_model_tokenizer 加载模型（沿用 yaml 的 model_path、from_scratch=false）。
- 滑窗逻辑与各 worker 的 loglikelihood_rolling 一致：stride = max_len // 2，
  非首窗的前 stride 个 token 作为左侧上下文（不计分），保证每个 token 只被计分一次。
- 绝对位置 0（无左侧上下文）一律跳过（AR 因 shift 本就不计 token0，这里对齐处理）。
"""

import os
import sys
import json
import math
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from omegaconf import OmegaConf
from utils.utils import (
    load_model_tokenizer,
    get_AR_attention_mask,
    get_full_attention_mask,
    get_selfless_mask,
    get_xlnet_mask,
)

LN2 = math.log(2.0)


# ----------------------------------------------------------------------------- #
#  族识别（与 utils.load_model_tokenizer 完全一致的优先级）
# ----------------------------------------------------------------------------- #
def detect_family(project: str) -> str:
    p = project.lower()
    if "sdar" in p:
        return "sdar"
    if "llada" in p:
        return "llada"
    if "dream" in p:
        return "dream"
    if "xlnet" in p:
        return "xlnet"
    if "selfless" in p:
        return "selfless"
    if "causal" in p or "ar" in p:
        return "ar"
    raise ValueError(f"Cannot detect family from project={project!r}")


# ----------------------------------------------------------------------------- #
#  统一 L->R 估计量：每个族返回一个 [L] 的 per-position NLL（nats），
#  约定 nll[i] = -log p(x_i | x_<i)。无法计分的位置填 NaN。
# ----------------------------------------------------------------------------- #
class LRAdapter:
    """family-specific 左到右 teacher-forced NLL。"""

    def __init__(self, model, tokenizer, family, device, block_size=128):
        self.model = model
        self.tok = tokenizer
        self.family = family
        self.device = device
        self.block_size = max(1, int(block_size))
        # 对 DLM（llada/dream）block_size = L->R 近似的并行粒度；
        # 对 sdar，这是“解码粒度 g”（块内每步并行预测几个 token），会 clamp 到 attention 块大小。
        self.decode_granularity = self.block_size
        self.mask_id = model.config.mask_token_id
        self.eps = 1e-3
        self._ar_attention_masks = {}
        # SDAR 原生 block_size（block diffusion 的块大小，训练时严格固定）。
        # SDAR 的 L->R 估计量必须用它自己的块大小与块因果 attention，不能用任意 block_size。
        self.native_block_size = int(getattr(model.config, "block_size", 0) or 0)

    # ---- PLM: 构造严格降序 v_sample（与 diffusion_utils.sample_v 的 ar 分支一致）---- #
    def _ar_v_sample(self, L):
        pos = torch.arange(L, device=self.device, dtype=torch.float32)
        if L > 1:
            v = 1 - self.eps - (1 - 2 * self.eps) * pos / (L - 1)  # pos0 最大, posL-1 最小
        else:
            v = torch.ones(1, device=self.device) * (1 - self.eps)
        return v.unsqueeze(0)  # [1, L]

    @torch.no_grad()
    def window_nll(self, ids):
        """
        ids: LongTensor [1, L]  —— 一个滑窗内的完整 token 序列。
        返回: FloatTensor [L]  —— nll[i] = -log p(x_i | x_<i)（nats），不可计分处为 NaN。
        """
        L = ids.shape[-1]
        fam = self.family

        if fam == "ar":
            return self._nll_ar(ids)
        elif fam in ("selfless", "xlnet"):
            return self._nll_plm(ids)
        elif fam == "sdar":
            # SDAR 用它原生的 block-causal attention（block_size=config.block_size），
            # 绝不能用 full attention（OOD）。见 _nll_sdar_block。
            return self._nll_sdar_block(ids)
        elif fam in ("llada", "dream"):
            # full-attention denoiser：训练即 full attention，eval 时 attend 整条右侧 mask 是 in-distribution。
            return self._nll_dlm_block(ids)
        else:
            raise ValueError(fam)

    # ---------- AR / causal：单次因果前向，shift 对齐 ---------- #
    def _nll_ar(self, ids):
        L = ids.shape[-1]
        # modeling_ar caches a 2048x2048 causal BlockMask by default. Rolling
        # evaluation ends with a shorter window, so provide an exact-size mask.
        if L not in self._ar_attention_masks:
            self._ar_attention_masks[L] = get_AR_attention_mask(L, device=self.device)
        out = self.model(
            input_ids=ids,
            attention_mask=self._ar_attention_masks[L],
            use_cache=False,
        )
        logits = out.logits  # [1, L, V]
        logp = F.log_softmax(logits.float(), dim=-1)
        nll = torch.full((L,), float("nan"), device=self.device)
        # logits[i] 预测 token[i+1]
        tgt = ids[0, 1:]                                   # [L-1]
        lp = logp[0, :-1, :].gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [L-1]
        nll[1:] = -lp
        return nll

    # ---------- PLM：单次两流前向 + ar-mode mask，无 shift ---------- #
    def _nll_plm(self, ids):
        L = ids.shape[-1]
        v = self._ar_v_sample(L)  # [1, L]，降序 => 位置 i 只 attend j<i
        if self.family == "selfless":
            mask = get_selfless_mask(v_sample=v, seq_len=L, device=self.device)
            out = self.model(X0_input_ids=ids, attention_mask=mask,
                             calculate_likelihood=True)
        else:  # xlnet：两条流不同 mask；模型恒用 XT(query) 流输出
            qmask, kvmask = get_xlnet_mask(v_sample=v, seq_len=L, device=self.device)
            out = self.model(X0_input_ids=ids,
                             query_attention_mask=qmask,
                             kv_attention_mask=kvmask)
        logits = out.logits  # [1, L, V]；logits[i] 直接预测 x_i（无 shift）
        logp = F.log_softmax(logits.float(), dim=-1)
        lp = logp[0].gather(-1, ids[0].unsqueeze(-1)).squeeze(-1)  # [L]
        nll = -lp
        nll[0] = float("nan")  # 位置0无左侧上下文（attend 空），跳过
        return nll

    # ---------- LLaDA / Dream：full-attention denoiser 的左到右 block 半自回归 ---------- #
    def _nll_dlm_block(self, ids):
        """
        仅用于 full-attention 训练的 denoiser（llada / dream）。
        对 block [bs, be)：揭示 [0,bs) 为真 token，mask [bs, L) 全部为 [MASK]，
        full bidirectional attention 前向（这两个模型训练即 full attention，故 in-distribution），
        读出 block 内位置的 logits（右侧全 mask => 无未来信息泄露）。
        block_size=1 时为精确 L->R；>1 为块并行近似（精确 NLL 的上界）。

        shift 约定：
          - llada : denoiser 原地预测，logits[p] 预测 token[p]（无 shift）
          - dream : 仿其 worker，喂 seq[:, :-1]，logits[p] 预测 token[p+1]（shift）

        注意：SDAR 不走这里——它训练用 block-causal（block_size=4）而非 full attention，见 _nll_sdar_block。
        """
        L = ids.shape[-1]
        nll = torch.full((L,), float("nan"), device=self.device)
        shift = (self.family == "dream")
        B = self.block_size

        # block 从位置 1 开始（位置 0 无左侧上下文，跳过，与 AR/PLM 对齐）
        bs = 1
        while bs < L:
            be = min(bs + B, L)
            seq = ids.clone()
            seq[:, bs:] = self.mask_id          # 当前块及右侧全部 mask；[0,bs) 保留真 token

            if not shift:
                amask = get_full_attention_mask(L, device=self.device)
                logits = self.model(input_ids=seq, attention_mask=amask).logits  # [1,L,V]
                logp = F.log_softmax(logits[0, bs:be, :].float(), dim=-1)         # [b,V]
                tgt = ids[0, bs:be]
                lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                nll[bs:be] = -lp
            else:
                # dream：模型吃 [:, :-1]，logits[i] 对齐 token[i+1]
                amask = get_full_attention_mask(L - 1, device=self.device)
                logits = self.model(input_ids=seq[:, :-1], attention_mask=amask).logits  # [1,L-1,V]
                # 预测 token p (bs<=p<be) 需读 logits[p-1]
                idx = torch.arange(bs, be, device=self.device) - 1
                logp = F.log_softmax(logits[0, idx, :].float(), dim=-1)
                tgt = ids[0, bs:be]
                lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                nll[bs:be] = -lp

            bs = be
        return nll

    # ---------- SDAR：用原生 block-causal attention 的块自回归 teacher-forced ---------- #
    def _block_causal_bool_mask(self, L, bs):
        """块因果 4D bool mask（True=attend），shape [1,1,L,L]：
        位置 q attend k 当且仅当 (k // bs) <= (q // bs)
        —— 块内双向、跨块只看更早的块。与 SDAR 训练的 block_diff_mask 在
        “xt 块看更早 x0 块(offset_block_causal) + 块内双向(block_diagonal)” 上等价。
        """
        idx = torch.arange(L, device=self.device)
        blk = idx // bs
        m = (blk[None, :] <= blk[:, None])      # [L,L]，True=attend
        return m[None, None, :, :]              # [1,1,L,L]

    def _nll_sdar_block(self, ids):
        """
        SDAR 的左到右估计量：区分两个互不相同的“块大小”——
          (1) attention 块大小 = config.block_size(=4)，固定。决定“谁能看见谁”的几何
              （块内双向 + 跨块因果），由 _block_causal_bool_mask 实现，不能改。
          (2) 解码粒度 g = self.decode_granularity ∈ {1,2,...,attn_block}。每步并行预测
              的 masked token 数。g=1 即**严格逐 token L->R**；g=attn_block 即整块并行。
              g 必须 <= attention 块大小（不能跨块并行，否则破坏块因果条件）。

        in-distribution 依据：SDAR 训练时块内是**随机比例** masking，故“块内左侧若干 token
        为 clean、其余 masked”是训练见过的配置 —— 因此可在固定的 4-token attention 块内
        逐 token 揭示（g=1）做严格 L->R，与 llada/dream 的 g=1 严格 L->R 同档可比。

        其它：model.eval()（train 模式会自走随机加噪 BD 前向、忽略我们的 mask）；
        无 shift（denoiser 原地预测 logits[p]->token[p]）。

        做法（逐 chunk）：在 attention 块 b=[B0,B0+attn) 内，按粒度 g 切 chunk [cs,ce)。
        预测该 chunk 时：揭示 [0,cs) 为真 token（含本块内更早 chunk）、把 [cs, B0+attn) 置
        [MASK]，前向区间取 [0, B0+attn)，block-causal mask 让该 chunk 只看
        “更早的真 token 块 + 本块(双向)”。读 logits[cs:ce] 的 CE。
        """
        L = ids.shape[-1]
        nll = torch.full((L,), float("nan"), device=self.device)
        attn = self.native_block_size if self.native_block_size > 0 else 4
        g = max(1, min(self.decode_granularity, attn))  # 解码粒度，clamp 到 [1, attn]

        B0 = 0
        while B0 < L:
            block_end = min(B0 + attn, L)        # 当前 attention 块 [B0, block_end)
            cs = max(B0, 1)                       # 块内 chunk 起点（绝对位置 0 跳过）
            while cs < block_end:
                ce = min(cs + g, block_end)       # 当前 chunk [cs, ce)
                seq = ids[:, :block_end].clone()
                seq[:, cs:block_end] = self.mask_id   # 揭示 [0,cs)，mask 本块剩余 [cs, block_end)
                amask = self._block_causal_bool_mask(block_end, attn)  # 固定 attn=4 的块因果
                logits = self.model(input_ids=seq, attention_mask=amask,
                                    use_cache=False).logits            # [1, block_end, V]
                logp = F.log_softmax(logits[0, cs:ce, :].float(), dim=-1)
                tgt = ids[0, cs:ce]
                lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
                nll[cs:ce] = -lp
                cs = ce
            B0 = block_end
        return nll


# ----------------------------------------------------------------------------- #
#  滑窗 rolling（与各 worker 的 loglikelihood_rolling 一致的覆盖方式）
# ----------------------------------------------------------------------------- #
@torch.no_grad()
def rolling_nll(adapter, all_ids, max_len, device):
    """
    all_ids: LongTensor [N] —— 整个语料的 token id。
    返回: (total_nll_nats, tokens_scored)

    覆盖方式：stride = max_len // 2。首窗计分 [0, max_len)，
    之后每窗的前 stride 个 token 作为左侧上下文（不计分），
    使得各窗计分区间恰好平铺 [0, N)，每个 token 只计一次。
    """
    N = all_ids.shape[0]
    stride = max(1, max_len // 2)
    total_nll = 0.0
    tokens = 0

    start = 0
    while start < N:
        end = min(start + max_len, N)
        window = all_ids[start:end].unsqueeze(0).to(device)  # [1, w]
        nll = adapter.window_nll(window)                     # [w]

        # 计分区间（窗内偏移）：首窗从 0 起；之后从 stride 起（前 stride 为上下文）
        ctx = 0 if start == 0 else stride
        if ctx >= nll.shape[0]:
            start += stride
            continue
        seg = nll[ctx:]
        finite = torch.isfinite(seg)
        total_nll += float(seg[finite].sum().item())
        tokens += int(finite.sum().item())

        if end == N:
            break
        start += stride

    return total_nll, tokens


# ----------------------------------------------------------------------------- #
#  语料加载
# ----------------------------------------------------------------------------- #
def load_corpus(args):
    """返回 (text, list_of_docs)。BPB 用 text 的 UTF-8 字节数归一化。"""
    if args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            text = f.read()
        docs = [text]
    elif args.hf_dataset:
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, args.hf_config, split=args.hf_split)
        # 取 'text' 列；wikitext 习惯按行拼接
        lines = [r for r in ds["text"] if r is not None]
        text = "".join(lines) if "wikitext" in (args.hf_config or "") else "\n\n".join(lines)
        docs = [text]
    else:
        raise ValueError("必须提供 --text_file 或 --hf_dataset")

    if args.limit:
        # 简单截断到前 limit 个换行段，便于 sanity
        parts = text.split("\n")
        text = "\n".join(parts[: args.limit])
        docs = [text]
    return text, docs


# ----------------------------------------------------------------------------- #
#  主流程
# ----------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="lm_eval yaml（同现有 worker）")
    ap.add_argument("--text_file", default=None)
    ap.add_argument("--hf_dataset", default=None)
    ap.add_argument("--hf_config", default=None)
    ap.add_argument("--hf_split", default="test")
    ap.add_argument("--block_size", type=int, default=128,
                    help="DLM 的解码粒度。llada/dream：L->R 块并行近似的块大小（1=精确严格 L->R）。"
                         "sdar：块内解码粒度 g，会 clamp 到 attention 块大小(=4)；g=1 严格逐 token L->R，"
                         "g=4 块并行（上界）。ar/plm 不受影响（单次前向即精确）。")
    ap.add_argument("--max_len", type=int, default=None, help="覆盖 yaml.max_len")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    project = cfg.experiment.project
    family = detect_family(project)
    max_len = args.max_len or int(cfg.get("max_len", 2048))
    print(f"[unified_lr_bpb] project={project} | family={family} | "
          f"max_len={max_len} | block_size={args.block_size}")

    # 加载模型 + tokenizer（沿用现有逻辑；from_scratch 应为 false）
    model, tokenizer = load_model_tokenizer(config=cfg)
    model.eval()
    device = next(model.parameters()).device
    if device.type == "cpu":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    adapter = LRAdapter(model, tokenizer, family, device, block_size=args.block_size)

    # --- SDAR：attention 块大小固定=config.block_size(=4)；--block_size 解释为“解码粒度 g”，
    #     clamp 到 [1, attn_block]。g=1 即严格逐 token L->R（推荐做公平 headline），g=4 块并行（上界）。---
    eff_block_size = args.block_size
    attn_regime = {
        "ar": "causal",
        "selfless": "ar-mode two-stream (no mask attn)",
        "xlnet": "ar-mode two-stream (no mask attn)",
        "llada": "full bidirectional (attends all right masks)",
        "dream": "full bidirectional (attends all right masks)",
        "sdar": "block-causal (attn_block=native), decode granularity g",
    }[family]
    if family == "sdar":
        nb = adapter.native_block_size or 4
        g = max(1, min(args.block_size, nb))
        if args.block_size > nb:
            print(f"[unified_lr_bpb][WARN] SDAR attention 块大小固定={nb}；解码粒度 g 不能超过它，"
                  f"已把 g 从 {args.block_size} clamp 到 {g}。"
                  f"做公平 headline 请用 --block_size 1（严格逐 token L->R）；g={nb} 是块并行上界。")
        adapter.decode_granularity = g
        eff_block_size = g
        attn_regime = f"block-causal (attn_block={nb}), decode granularity g={g}"
        # block-causal eval 路径需要 eval 模式（train 模式会自走随机加噪 BD 前向、忽略我们的 mask）
        model.eval()

    # 语料
    text, _ = load_corpus(args)
    num_bytes = len(text.encode("utf-8"))
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    all_ids = torch.tensor(ids, dtype=torch.long)
    print(f"[unified_lr_bpb] corpus: {len(ids)} tokens, {num_bytes} bytes")

    # 计算
    total_nll, tokens = rolling_nll(adapter, all_ids, max_len, device)

    bpb = total_nll / (LN2 * num_bytes)
    # word-PPL 这里用 token 级近似（如需严格 word ppl，可换成 lm_eval 的 word 计数）
    token_ppl = math.exp(total_nll / max(1, tokens))

    result = {
        "project": project,
        "family": family,
        "estimator": "unified_left_to_right",
        "attention_regime": attn_regime,
        # llada/dream: L->R 近似的并行粒度；sdar: 解码粒度 g（attention 块固定=native_block_size）
        "decode_granularity": eff_block_size if family in ("llada", "dream", "sdar") else None,
        "native_attn_block_size": adapter.native_block_size if family == "sdar" else None,
        "strict_left_to_right": (family in ("ar", "selfless", "xlnet"))
                                or (family in ("llada", "dream", "sdar") and eff_block_size == 1),
        "max_len": max_len,
        "corpus_tokens": int(len(ids)),
        "corpus_bytes": int(num_bytes),
        "tokens_scored": int(tokens),
        "total_nll_nats": float(total_nll),
        "unified_lr_bpb": float(bpb),
        "unified_lr_token_ppl": float(token_ppl),
        "source": args.text_file or f"{args.hf_dataset}/{args.hf_config}/{args.hf_split}",
        "timestamp": datetime.now().isoformat(),
    }

    print("\n==================== RESULT ====================")
    print(f"  family            : {family}")
    print(f"  attention regime  : {attn_regime}")
    if family in ("llada", "dream", "sdar"):
        print(f"  block_size        : {eff_block_size}")
    print(f"  unified L->R BPB  : {bpb:.4f}")
    print(f"  token-PPL         : {token_ppl:.2f}")
    print(f"  tokens scored     : {tokens} / {len(ids)}")
    print("================================================\n")

    out_dir = os.path.join(cfg.get("output_path", "./output_eval"), project)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"unified_lr_bpb_{datetime.now():%Y%m%dT%H%M%S}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"[unified_lr_bpb] saved -> {out_path}")


if __name__ == "__main__":
    main()
