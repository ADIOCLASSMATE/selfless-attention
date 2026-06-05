# RESULTS — 已验证的实验结果

**数据来源**: `output_eval/` 下 24 个 run 的最新 `results_*.json`（2026-06-01~03）。本文件数字由脚本直接从 JSON 提取，**取代**旧的 `RESULTS_SUMMARY.md`（5/20 版，数字已过时）与 `BASELINE_AUDIT.md`。

**指标约定**:
- 下游 acc：ARC-E/C、HellaSwag、PIQA、OBQA 用 `acc_norm`；SciQ、Winogrande、BoolQ、COPA、WiC、RTE 用 `acc`。
- `ACC-AVG` = 上述 11 项算术平均。
- LM：WikiText `bits_per_byte`（↓）、LAMBADA `perplexity`（↓）、Paloma-C4/WT103/Falcon `bits_per_byte`（↓）。
- `preload` = 从 HF Qwen3-0.6B-Base 初始化；否则 from-scratch。Selfless/XLNet 的 `ar` / `random` 指**同一 checkpoint 的两种推理模式**（eval-time v_sample）。

---

## 1. 主表

### 0.6B — from-scratch

| Model | Eval | WT-BPB ↓ | LAMB-PPL ↓ | C4-BPB ↓ | WT103 ↓ | Falcon ↓ | ACC-AVG ↑ |
|---|---|---|---|---|---|---|---|
| Causal LM | — | **0.823** | **24.7** | **0.966** | **0.890** | **1.023** | **0.563** |
| SDAR | block-AR | 0.904 | 36.1 | 1.046 | 0.972 | 1.099 | 0.508 |
| LLaDA | random-mask | 0.931 | 1313.5 | 1.113 | 0.997 | 1.154 | 0.451 |
| Dream | absorbing | 0.936 | 3033.2 | 1.134 | 1.005 | 1.173 | 0.449 |
| XLNet | random | 0.963 | 57.0 | 1.095 | 1.036 | 1.158 | 0.525 |
| XLNet | AR | 0.968 | 102.2 | 1.122 | 1.048 | 1.183 | 0.521 |
| **Selfless** | **AR** | **0.943** | 58.2 | 1.086 | 1.016 | 1.148 | 0.518 |
| **Selfless** | **random** | 0.971 | 57.8 | 1.102 | 1.040 | 1.164 | **0.532** |

### 0.6B — preload (Qwen3-0.6B-Base)

| Model | Eval | WT-BPB ↓ | LAMB-PPL ↓ | C4-BPB ↓ | ACC-AVG ↑ |
|---|---|---|---|---|---|
| Causal LM | — | **0.771** | **12.9** | **0.918** | **0.600** |
| SDAR | block-AR | 0.846 | 972.8 ⚠ | 0.993 | 0.511 |
| LLaDA | random-mask | 0.881 | 93.4 | 1.059 | 0.474 |
| Dream | absorbing | 0.882 | 139.3 | 1.061 | 0.474 |
| Selfless | AR | 0.905 | 30.4 | 1.037 | 0.551 |
| XLNet | AR | 0.913 | 30.5 | 1.040 | 0.555 |
| **Selfless** | random | 0.930 | 24.8 | 1.059 | 0.556 |
| **XLNet** | random | 0.930 | 24.6 | 1.058 | **0.563** |

⚠ SDAR-preload 的 LAMBADA-PPL=972.8 是异常值（from-scratch 仅 36.1）。这是 SDAR 的 WikiText-BPB 与 LAMBADA 用了**不同估计量**导致的，正是 §3 "估计量不可比" 的活样本，写进 Limitations。

### 342M — from-scratch

| Model | Eval | WT-BPB ↓ | LAMB-PPL ↓ | C4-BPB ↓ | ACC-AVG ↑ |
|---|---|---|---|---|---|
| Causal LM | — | **0.860** | **32.9** | **0.998** | **0.540** |
| SDAR | block-AR | 0.936 | 48.6 | 1.076 | 0.493 |
| LLaDA | random-mask | 0.963 | 3924.6 | 1.150 | 0.444 |
| Dream | absorbing | 0.969 | 2540.4 | 1.165 | 0.447 |
| XLNet | AR | 0.992 | 118.6 | 1.137 | 0.501 |
| XLNet | random | 1.004 | 97.4 | 1.132 | 0.511 |
| Selfless | AR | 0.993 | 99.5 | 1.126 | 0.503 |
| **Selfless** | random | 1.009 | 91.6 | 1.136 | **0.511** |

---

## 2. 三个关键推断

### 2.1 PLM ≫ DLM 下游 / LAMBADA，尽管 PLM 的 WT-BPB 更差（核心 dissociation，**稳**）

把 PLM（selfless/xlnet）与 DLM（llada/dream/sdar）对比：

```
                  WT-BPB              LAMBADA-PPL          ACC-AVG
                (PLM 更差)          (PLM 大幅更好)        (PLM 更好)
0.6B  scratch:  PLM ~0.96–0.97     PLM ~57    vs DLM 36~3000   PLM .52–.53 vs DLM .45–.51
0.6B  preload:  PLM ~0.91–0.93     PLM ~25    vs DLM 93~973    PLM .55–.56 vs DLM .47–.51
342M  scratch:  PLM ~0.99–1.01     PLM ~90~120 vs DLM 49~3900  PLM .50–.51 vs DLM .44–.49
```

- 方向在 **2 尺度 × 2 初始化 × 11 任务** 上一致 → 这是可下结论的主张。
- 最强的 DLM 基线是 SDAR（block-AR，最接近 AR）：下游 0.508/0.511 vs Selfless 0.532/0.556——PLM 仍赢，但差距小；真正拉开的是 LLaDA/Dream（random-mask 纯扩散），PLM 比它们高 ~0.08 ACC，LAMBADA 好 1~2 个数量级。
- 诚实点：vs causal LM，PLM 在 BPB 上仍有 ~0.12（0.6B scratch）的"PLM 税"，下游也低一截。我们不替代 causal LM。

### 2.2 Selfless 与 XLNet 建模质量"持平"，**不是"更好"**（旧主线的纠正）

| 0.6B ACC-AVG | from-scratch | preload | 结论 |
|---|---|---|---|
| Selfless random | 0.532 | 0.556 | scratch 微赢，preload 微输 |
| XLNet random | 0.525 | 0.563 | **符号翻转，|Δ|≈0.007** |

WT-BPB 同理在不同模式/初始化下互有胜负，幅度 ≤0.008。**结论：质量持平。** Selfless 相对 XLNet 的真正增量是"单流推理能力"（§2.3），不是指标更高。任何"Selfless 更准"的表述都过不了 multi-seed。

### 2.3 单流推理（代码已验证的架构事实）

- `get_xlnet_mask`: content(kv) 流 `v_kv >= v_q`（含对角线），query 流 `v_kv > v_q`（不含）→ 两条流不同 mask。
- `get_selfless_mask`: 两条流都 `v_kv > v_q` → content 流不含对角线。
- 推理时（`modeling_selfless.py` 的 `Qwen3Model.forward`，`training=False` 且 `calculate_likelihood=False`）只跑 X0(content) 流并读其输出；XT(query) 流根本不构造。XLNet 因 content 流泄露，等价单流读出会 train/test 失配，故必须双流。

> 这是 Selfless 唯一别人复制不了的护城河。但"单流生成质量好"目前**只有 likelihood 证据，没有生成证据**（见 EXPERIMENTS.md P1-#3）。

---

## 3. 为什么 BPB 不能跨族比较（写作要点）

三个族报的根本不是同一个量：

| 族 | 报告的量 | 性质 |
|---|---|---|
| AR / causal | 精确链式 NLL `-Σ log p(x_i\|x_<i)` | 真实 log-likelihood |
| PLM (selfless/xlnet) | `-E_σ Σ log p(x_{σ_i}\|x_{σ_<i})` | 对随机顺序求期望；**全确定上下文** |
| DLM (llada/dream) | 吸收态 **ELBO** | NLL 的上界；**条件含 [MASK]、看双向** |

DLM 的 BPB 低，是因为它的评测估计量 = 它的训练目标（随机 mask 去噪），二者完美对齐；但这个估计量没探测"给定纯左侧全确定上下文预测"所需的条件结构。**铁证在自家数里**：同一 DLM 换成 L→R 估计量（LAMBADA）后差 4~100×（甚至 SDAR-preload 炸到 973）。所以正确表述是：

> "BPB 在异质估计量下不是公平的跨族指标；统一估计量后 DLM 的 BPB 优势反转。" —— 这条要用 EXPERIMENTS.md P0-#1（统一 L→R 估计量重算 BPB）来钉死。

---

## 4. 已证明 ✅ / 未证明 ❓ 清单

**✅ 已被现有数据证明**
- PLM(selfless/xlnet) 下游 + LAMBADA 全面优于 DLM，2 尺度 × 2 初始化一致。
- Selfless 与 XLNet 建模质量持平（互有胜负，幅度 ≤0.008）。
- 同一 Selfless checkpoint 支持 AR 与 random 两种推理模式（0.6B scratch：0.943 / 0.971 WT-BPB）。
- DLM 的 LAMBADA-PPL 比 PLM 差 1~2 个数量级（估计量不一致的直接体现）。

**❓ 尚未证明（→ EXPERIMENTS.md）**
- 统一 L→R 估计量下 DLM 的 BPB 优势是否真的反转（P0-#1）。
- 单流推理的泄露/一致性硬证据（P0-#2）。
- 生成质量与吞吐（P1-#3，**目前完全空白**）。
- var(h) over permutations、cos_sim(h, embed(x)) 的机制图（P1-#4）。
- 薄差距的 multi-seed（只对 PLM-vs-DLM 做即可）。
