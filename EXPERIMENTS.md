# EXPERIMENTS — 待办实验（按对故事的杠杆排序）

**取代** 旧的 `TODO.md`。原 TODO 里的 ARC-Challenge / Paloma-C4 / WT103 / Falcon 已完成（见 RESULTS.md），已从清单移除。MMLU 此尺度无意义，不做。

排序原则：能把"我看代码确信"或"我猜"升级成"可证伪证据"的实验优先。

---

## P0 — 缺了 paper 站不住

### #1 统一估计量的 BPB（杀手锏，C2 的钉子）

**做什么**：在同一份语料上，对所有族用**同一个 L→R teacher-forced NLL** 重算 BPB（统一信息集=只给左侧上下文，每个模型在自己训练的 attention 制度下查询）。
- AR：原生链式（shift）。
- PLM(selfless/xlnet)：ar-mode（严格降序 v_sample，无 shift），本就是 L→R。
- DLM(llada/dream)：full bidirectional，喂左侧真 token、右侧全 `[MASK]`，逐位/逐块预测。
- DLM(sdar)：**block-causal（attention 块=4 固定）**，解码粒度 g∈{1,2,4}；**绝不能用 full attention**（OOD）。g=1 严格逐 token L→R，g=4 块并行上界。

**实现**：已写好 `eval/unified_lr_bpb.py` + 并行 `run_unified_bpb.sh` + 汇总 `collect_unified_bpb.py`。

**状态（2026-06）✅ 主体已完成**：WikiText-2-raw-v1 test 上 18 个 run 跑完（DLM 为 **g=4 上界**）。结论三档（342M/0.6B-scratch/0.6B-preload）一致：排序彻底翻转。native 下 DLM 看似优于 PLM；统一 L→R 下 DLM 暴涨成最差。Δ 是 punchline：AR/PLM 的 Δ≈+0.02~0.06（估计量不变），DLM 的 Δ≈+0.55~0.58（强估计量依赖）。与 LAMBADA 灾难同序，交叉验证为真实效应。**C2 已被钉死。**

**剩余收尾**：
1. **DLM 跑 g=1** 出公平 headline（g=4 是上界；g=1 会把 DLM 往下拉但远不及 PLM，方向不变）。可选 g=1/2/4 画"并行解码税"曲线。
2. 把翻转表 + "estimator-invariance vs estimator-dependence" 写进 RESULTS.md §2/§3 与 PAPER_PLAN.md C2。
3. **附带量**：对每个 DLM 算 `native ELBO 与 L→R-NLL 之差`（即 Δ），精确度量"多依赖右侧上下文" = "低 BPB ≠ 好 LM"的可测定义。

**【补充实验】再加 C4 数据集（防 single-dataset 质疑，性价比高）**：
- 现结论只在 WikiText（百科域）一个域上。**强烈建议再补 `paloma_c4_en`（通用 web 域，与 wikitext 域差异大，且 native BPB 已有现成对照）**，跨域翻转一致即把主张从"一个数据集"升级成"跨域稳健"。
- WikiText-103 / RefinedWeb 可再加一个凑 3 域，但边际收益递减，时间紧可不做；**3 个以上属过度**，算力应留给 P0-#2 与 P1。
- 跑法：`--hf_dataset paloma_c4_en ...`（或 `--text_file`）。**C4 大，务必 `LIMIT` 截断**（5–10 万 token 的 BPB 已很稳；否则 DLM 在 g=1 是 O(L²) 会非常慢）。
- 可比性铁律：跨数据集必须保持同一 `max_len`、同一 stride(=max_len//2)、同一 UTF-8 字节归一化——用同一脚本跑即自动满足。
- `collect_unified_bpb.py` 已升级为**按 (dataset, project, granularity) 分组**，多数据集会分节出表。

**成本**：低（复用现有 forward；C4 用 LIMIT 控量）。**杠杆：最高。**

### #2 单流主张的硬证据：泄露 probe + train/infer 一致性（C1 验证）

**做什么（两个子实验）**：
- **泄露 probe**：冻结模型，从 masked 位置的 content-stream `h_i`（各层）训练 linear probe 恢复 `x_i`。比较 XLNet vs Selfless。**预测**：XLNet 准确率显著高 / 与 query 流不一致；Selfless content≈query。
- **一致性**：同序列同时用 (a) XT/query 路径（训练路径）与 (b) X0-at-mask 路径（推理路径）算 likelihood，报告差值。**预测**：Selfless AR-mode 近 0；XLNet 显著非 0。
**成本**：低（forward-only + 一个 linear probe）。验证命题 1–3。

---

## P1 — 让 insight 完整

### #3 生成质量 + 吞吐（目前完全空白，但 paper 标题写了"生成"）

**现状**：`output_eval/` 全是 likelihood/acc，**没有任何生成指标**。`generate()` / `speculative_generate()` 已存在且在记 `parallel_rate`，但没产出。
**做什么**：
- generative PPL（用更大 teacher 模型给 Selfless 生成文本打分）。
- AR-mode vs any-order-mode 样例 + 质量对比（vs SDAR / LLaDA）。
- throughput / `parallel_rate` 曲线（confidence-guided 解码）。
**为何重要**：兑现"单流生成"承诺；是与 SDAR（block-AR、不够灵活）拉开差距的地方。**不补这块，C1 只有 likelihood 证据。**
**成本**：中。

### #4 机制图：var(h) over permutations + cos_sim(h, embed(x)) + 对角线注意力质量

**做什么**：固定输入，多采样若干 permutation，测各层 `Var(h_i)`；测 `cos_sim(h_i, embed(x_i))`；测 XLNet content 流落在对角线上的注意力质量占比。
**用途**：解释 Selfless/XLNet 表征差异，作 §4.5 机制图；forward-only，便宜。
**注意**：这些图用来"解释"，不要再用来支撑"Selfless 更好"（那是噪声）。

---

## P2 — 加分项

### #5 multi-seed（只对要下结论的 claim）
对 **PLM vs DLM 的下游 gap** 跑 2–3 seed 确认显著（这是主张）。**不要**对 selfless-vs-xlnet 烧 compute——它已被定为"持平"，不是卖点。

### #6 1B 尺度
按 memory 里的几何一致配置（hidden=1280, layers=28, heads=20, kv=10, ffn=3840；与 0.6B 同 ffn/hidden=3×、GQA=2）重训 selfless/xlnet/baselines。dissociation 已在 2 尺度成立，1B 是趋势确认，非首投必需。先确认 0.6B 的 resume/LR schedule 与 xlnet 对齐（避免 confound）。

### #7 infilling / 迭代 refine demo
any-order 的天然杀手应用，把"单流任意顺序生成"落到一个有用 demo（held-out span 填空 vs LLaDA）。

---

## 一句话排期

先 **#1（统一估计量）+ #2（probe/一致性）** —— 这两个把 C1、C2 从断言变成证据，最便宜、杠杆最高；然后 **#3（生成）** 兑现标题；**#4** 配机制图；**#5–#7** 视时间与 compute 补。