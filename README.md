# Selfless Attention — 项目文档索引

**最后更新**: 2026-06-04（基于 `output_eval/` 中 24 个真实 run 的重分析与重构）

---

## 一句话主张（thesis）

PLM（Permutation Language Modeling）自 2019 年起被认为只能做理解、不能做生成，根因是 XLNet 的双流注意力让生成变得别扭。我们指出：**双流的真正约束来自 content 流的残差泄露，而这个约束只在训练期必需**。一行 mask 改动（两条流都去掉对角线，即 *Selfless Attention*）即可在推理/生成时坍缩成单流。由此 PLM 第一次能与扩散语言模型（DLM）正面对比，并在下游任务上全面胜出——尽管它的原生 BPB 更差，而我们证明这个 BPB 差距是**估计量假象**，不是建模缺陷。

---

## 文档结构（共 4 个，替代原来的 7 个）

| 文档 | 作用 | 读它来回答 |
|---|---|---|
| **README.md**（本文件） | 索引、主张、现状 | 这项目在做什么？现在到哪一步？ |
| **PAPER_PLAN.md** | 故事线、三个 contribution、单流定理、逐节写作大纲、诚实定位、Related Work、Reviewer Q&A | paper 怎么写？卖点是什么？ |
| **RESULTS.md** | 基于真实 JSON 的全部指标表、已证明 / 未证明清单 | 数据到底说了什么？ |
| **EXPERIMENTS.md** | 按杠杆排序的待办实验 + 理由 | 接下来该跑什么？ |

阅读顺序：**README → RESULTS（先看清数据）→ PAPER_PLAN（再定故事）→ EXPERIMENTS（最后排活）**。

---

## 当前状态

- **训练**: 50B tokens FineWeb-Edu，Qwen3 架构，8×H200，全方法同 schedule。
- **尺度**: 342M（仅 from-scratch）+ 0.6B（from-scratch 与 preload 两种初始化）。`preload` = 从 HF Qwen3-0.6B-Base 权重初始化。
- **评测**: 24 个 run 已完成（LM: WikiText/LAMBADA/Paloma-C4/WT103/Falcon；下游 11 项）。MMLU 在此尺度无意义，已确认不评。
- **核心代码**: `models/modeling_model/modeling_{selfless,xlnet}.py`、`utils/utils.py` 的 `get_selfless_mask` / `get_xlnet_mask`、各族 `eval/.../eval_worker_*.py`。代码逻辑经审计无功能性 bug。

---

## 三大主张速览（详见 PAPER_PLAN.md）

1. **C1（架构 + 理论）** — Selfless 去掉两条流的对角线，解锁**单流推理与生成**；XLNet 因 content 流含对角线，推理时结构上必须维持两条流。有可证明的信息流命题支撑。
2. **C2（实证 + insight）** — PLM 在 WikiText-BPB 上劣于 DLM，但在 LAMBADA-PPL 与下游 11 项上大幅胜出。该 dissociation 在 **两个尺度、两种初始化下都成立**。BPB 不可跨族比较：统一估计量后 DLM 的优势反转。
3. **C3（受控消融）** — 在同架构下，Selfless 与 XLNet **建模质量持平**，但 Selfless 额外获得单流能力。**注意：不主张 Selfless 质量更好**（见下）。

---

## 相比旧文档的关键修正

- **删除"Selfless 下游优于 XLNet"主线**。真实数据：from-scratch 下 Selfless 微赢 0.007，preload 下 XLNet 反超 0.007，符号翻转且远小于 per-task stderr——是噪声。旧 PLAN.md 的 "expressiveness vs ordering-robustness tradeoff" 主线建立在这个噪声上，已废弃。
- **新主线** = 单流推理（C1）+ PLM vs DLM 的 BPB-下游 dissociation（C2）。这两条都稳。
- **"DLM 低 BPB = 过拟合" 改写为 "估计量不可比"**（见 RESULTS.md / PAPER_PLAN.md），后者 reviewer 驳不倒。
