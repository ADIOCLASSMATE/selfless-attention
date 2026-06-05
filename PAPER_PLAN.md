# PAPER_PLAN — 故事、理论、写作大纲

**取代** 旧的 `PLAN.md` 与 `THRESHOLDS.md`。核心改动：主线从噪声驱动的 "Selfless>XLNet tradeoff" 换成数据驱动的 **"单流 PLM + BPB 不可比 insight"**。

---

## 1. Title（方向）

> **Single-Stream Permutation Language Modeling: Removing the Two-Stream Bottleneck Makes PLM a Practical Any-Order Generator**

备选副线标题：*The Diagonal Was Only Needed for Training: Single-Stream Inference for Permutation LMs.*

---

## 2. 一段话 pitch

PLM 自 XLNet(2019) 起被当作只能做理解的预训练目标，从未被证明能生成——根因是双流注意力让生成别扭。我们指出双流的真正约束来自 **content 流的残差泄露**：在残差 transformer 中，token 身份经输入 embedding 进入残差流，即便 attention 不看自己，content 表征 `h_i` 仍泄露 `x_i`，因而**训练期**必须用一条输入为 `[MASK]` 的 query 流来切断泄露。但我们证明：**推理期**未生成位置在 content 流里的输入本就是 `[MASK]`，残差不泄露，于是 content 流可直接坍缩成预测流。一行 mask 改动（两条流都去对角线，*Selfless Attention*）即解锁单流推理与生成；XLNet 因 content 流含对角线无法这样坍缩。由此 PLM 第一次能与 DLM 正面比较：我们发现 PLM 的 WikiText-BPB 劣于 DLM，但 LAMBADA 与 11 项下游全面胜出（2 尺度 × 2 初始化一致）。我们进一步证明这个 BPB 差距是**估计量假象**——三族报的是对 `-log p(x)` 的不同界，统一成 L→R 估计量后 DLM 优势反转。

---

## 3. 三个 Contribution

### C1 — 架构 + 理论（主菜）：单流推理

Selfless = 一行 mask 改动（content/query 两条流统一为严格 `v_kv > v_q`，剔除对角线）。据此给出**单流推理的可达性命题**（§4 形式化），说明：
- 为何 XLNet 推理结构上必须双流（content 流残差 + 对角线双重泄露 `x_i`，预测须走 query 流，而 query 流要 attend content 流 KV → 两条流共存）；
- 为何 Selfless 推理可单流（masked 位置 content 流输入 = `[MASK]`，其计算路径逐字节等价于训练 query 路径，前提是只 attend 已填充上下文）；
- 推论：AR-mode 生成与训练路径精确一致（质量高）；并行解码因引入训练中没有的 mask-to-mask attention 而失配（这正是 PLM 抗拒并行解码的精确机制）。

### C2 — 实证 + insight：PLM vs DLM 的 BPB-下游 dissociation

- PLM 原生 WT-BPB 劣于 DLM，但 LAMBADA-PPL 与下游 ACC 全面胜出（数据见 RESULTS.md §2.1，2 尺度 × 2 初始化稳定）。
- **insight**：BPB 跨族不可比。AR 报精确 NLL、PLM 报随机顺序期望 NLL、DLM 报 ELBO，是对 `-log p(x)` 的三种不同界。DLM 的低 BPB 反映"训练目标 = 评测估计量"的对齐，而非更好的语言建模。统一 L→R 估计量后优势反转（EXPERIMENTS P0-#1 证明）。

### C3 — 受控消融：质量持平 + 单流增量

同架构、同数据、同 schedule 下，**Selfless 与 XLNet 建模质量持平**（互有胜负 ≤0.008），但 Selfless 额外获得单流能力。明确写"持平 + 单流"，**不写"更好"**（margin 是噪声，见 RESULTS.md §2.2）。

---

## 4. 理论核心：单流推理可达性命题（放进 Method）

记两条流共享权重 `W_q, W_k, W_v` 与各层；位置 `i` 的输入 embedding 为 `e_i`，content 流 `[MASK]` 处输入为 `e_[M]`。注意力 DAG 由 `v_sample` 与 mask 决定。

**定义（泄露）**：位置 `i` 的表征 `h_i` 对 `x_i` *泄露*，若存在从 `e_i`（含 `x_i` 身份）到 `h_i` 的有向路径。

**命题 1（训练期双流必需）**：在含残差的 transformer 中，content 流（输入 `e_i` = 真 token）的 `h_i^{(l)}` 对任意 `l≥0` 均泄露 `x_i`（残差路径 `e_i → h_i^{(0)} → h_i^{(l)}`），与对角线是否存在无关。故用 content 流预测 `x_i` 不合法，须引入输入为 `e_[M]` 的 query 流切断残差路径。**两个模型（XLNet/Selfless）训练期都需双流。**

**命题 2（Selfless 推理期单流充分）**：推理时未生成位置 `i` 在 content 流的输入为 `e_[M]`（非 `x_i`），故残差路径不泄露 `x_i`。若 Selfless 的严格 mask 下位置 `i` 只 attend `v_kv > v_q[i]` 的**已填充**位置（block-AR 解码 + 恰当 sigma 即满足），则 content 流在 `i` 处的前向**逐层等价于**训练 query 流（同输入 `e_[M]`、同 mask、同权重、同 KV 来源）。故单流读出合法且与训练一致。

**命题 3（XLNet 推理期单流失配）**：XLNet content 流 mask 为 `v_kv >= v_q`（含对角线），masked 位置 `i` 会 attend 自身 `e_[M]`，使该路径混入训练中预测路径（query 流，无对角线）未出现过的项，故 content 流单流读出与训练分布失配。

**推论**：(a) AR-mode 单流生成与训练一致 → 高质量；(b) 并行/任意顺序解码引入 mask-to-mask attention（命题 2 前提被破坏）→ 受控失配，这是 PLM 难并行解码的机制级解释。

> 命题 1–3 用 EXPERIMENTS P0-#2（泄露 probe + train/infer 一致性）做经验验证：probe 从 content `h_i` 恢复 `x_i` 的准确率，XLNet 显著高于 Selfless；Selfless 的 XT-路径与 X0-at-mask-路径 likelihood 差近 0（AR-mode），XLNet 显著非 0。

---

## 5. 逐节写作大纲

**Abstract**：PLM 被弃用于生成 → 双流约束只在训练期必需 → Selfless 单流 → PLM 首次正面对比 DLM → 下游胜出但 BPB 差 → BPB 不可比、统一后反转。

**§1 Introduction**
1. PLM/XLNet 背景，双流为何让生成别扭，PLM 作为生成范式被搁置。
2. 架构问题：content 流对角线到底必需吗？我们从信息流角度回答。
3. 方法：一行 mask 改动 = Selfless；命题 1–3 概述。
4. 结果与 insight：单流可生成；PLM 下游胜 DLM；BPB 不可比。
5. Contributions C1/C2/C3。

**§2 Background**：PLM 损失；XLNet 双流 mask（`<` vs `<=`）；DLM 吸收态 ELBO。强调三族目标数学上不同。

**§3 Method: Selfless Attention**：mask 改动一行；两流前向；**命题 1–3 + 证明**；三种推理模式（ar / random / confidence-guided）。这一节是 C1，要厚。

**§4 Experiments**
- 4.1 Setup（50B FineWeb-Edu，342M/0.6B，scratch+preload，8×H200）。
- 4.2 主表（RESULTS.md §1）→ 支撑 C2/C3。
- 4.3 **统一估计量 BPB**（P0-#1）→ C2 的钉子，best paper 级别的图。
- 4.4 **泄露 probe + train/infer 一致性**（P0-#2）→ C1 经验验证。
- 4.5 机制图：var(h) over perms、cos_sim、对角线注意力质量（P1-#4）。
- 4.6 **生成质量 + 吞吐**（P1-#3）→ 兑现"生成"承诺；AR vs any-order 样例、generative PPL、parallel_rate。
- 4.7 （可选）infilling demo、1B 趋势。

**§5 Honest Positioning & Limitations**（见 §6）。

**§6 Related Work**（见 §7）。

**§7 Conclusion**：把"对角线"重新定位为"只服务训练期的工具"，单流 PLM 是与 DLM 并列的任意顺序生成范式。

---

## 6. 诚实定位（单独一节，提升可信度）

| 能力 | Causal LM | LLaDA/Dream | SDAR | XLNet | **Selfless** |
|---|---|---|---|---|---|
| L→R 生成质量 | **最佳** | 不能 | 好 | 中 | 中（≈XLNet） |
| 任意顺序生成 | 不能 | 能 | 受限(block) | 能 | **能** |
| 单流推理 | n/a | n/a | n/a | **不能** | **能** |
| 下游 zero-shot | 最佳 | 最差 | 中 | 好 | 好（≈XLNet） |
| LAMBADA(L→R) | 最佳 | 极差 | 差(估计量) | 好 | **好(非AR中最佳)** |

**我们赢在**：单流 + 任意顺序的组合（无人提供）；非 AR 模型中最好的下游与 LAMBADA。
**我们不赢**：vs causal LM 有 ~0.12 BPB 的 PLM 税；原生 random-mode BPB 劣于 DLM；当前无并行加速。
**我们不主张**：Selfless 比 XLNet 更准（持平）。

---

## 7. Related Work

- **XLNet (2019)**：原始双流；从未作为生成器评测。我们首次从信息流角度证明对角线只在训练期必需，并解锁单流生成。
- **LLaDA / Dream / MDLM**：随机 mask 训练，ELBO 估计量；任意顺序 BPB 好但不能 AR、下游弱、L→R 估计量下崩。我们互补而非替代。
- **SDAR**：block-AR 扩散，最接近 AR 的 DLM，是最强基线；但其 WT-BPB 与 LAMBADA 用不同估计量（preload LAMBADA=973 是症状）。
- **SUNDAE / Diffusion-LM**：迭代 refine / 连续空间，正交。

---

## 8. Reviewer Q&A（预演）

**Q：去掉对角线就一行改动，贡献在哪？**
A：贡献是指出这个被默认 6 年的"必需不对称"其实只在训练期必需，给出信息流命题与单流推理能力，并兑现生成。

**Q：你们 random-mode BPB 比所有 DLM 差。**
A：三族报的是 `-log p(x)` 的不同界，不可直接比。统一 L→R 估计量后 DLM 优势反转（§4.3）；同一 DLM 换 L→R 即 LAMBADA 差 1~2 个数量级。

**Q：Selfless 比 XLNet 好吗？**
A：建模质量持平（互有胜负 ≤0.008，跨初始化符号翻转）。增量是单流推理能力，不是指标。我们如实报告。

**Q：为什么不并行解码？**
A：命题 2 前提（只 attend 已填充上下文）在并行时被破坏，引入训练中没有的 mask-to-mask attention。这是机制级解释，并列为 future work（混合目标）。

**Q：估计量不同，比较公平吗？**
A：下游 zero-shot 全族同协议、直接可比；BPB 我们专门用 §4.3 统一估计量处理，并在 Limitations 披露 SDAR 的 LAMBADA/BPB 估计量不一致。

**Q：能外推到 1B 吗？**
A：dissociation 已在 342M 与 0.6B 一致；1B 为趋势确认（compute 允许则补，见 EXPERIMENTS P2）。

---

## 9. 图表清单（优先级）

1. **Teaser**：XLNet vs Selfless mask 对比（含/不含对角线），配单流 vs 双流推理示意。
2. **统一估计量 BPB**：原生估计量 vs 统一 L→R 估计量两组柱状，显示 DLM 优势反转（C2 钉子）。
3. **dissociation 散点**：x=WT-BPB，y=下游 ACC，PLM 与 DLM 分簇于反对角。
4. **LAMBADA log 柱状**：DLM 高出 1~2 个数量级。
5. **泄露 probe**：XLNet vs Selfless content 流恢复 `x_i` 准确率。
6. 机制：var(h) over perms / cos_sim 折线。
7. 生成：AR vs any-order 质量-吞吐散点。
