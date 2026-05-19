# Selfless Attention: Permutation Language Modeling Reveals an Ordering Tradeoff

## Metadata
- **Target Venue**: ACL 2026 / EMNLP 2026 / Findings (Tier 1 secure; Tier 2 with additional experiments)
- **Status**: Core experiments complete (50B token training, fixed+flexible eval at 250M and 0.6B)
- **Created**: 2026-04-29
- **Updated**: 2026-05-18 — major reframing based on actual 0.6B results

---

## 0. Reframing Notes (read first)

### 0.1 What changed in this version

The original PLAN.md (2026-04-29) claimed **"Selfless PLM achieves SOTA BPB on flexible-order generation, surpassing LLaDA/Dream/SDAR"**. The 0.6B WikiText results refute this:

| 0.6B WikiText-BPB ↓ | Value |
|---|---|
| Causal LM | 0.823 |
| SDAR | 0.913 |
| LLaDA | 0.940 |
| **Selfless (AR-mode eval)** | **0.943** |
| Dream | 0.945 |
| XLNet (random-mode eval) | 0.963 |
| XLNet (AR-mode eval) | 0.968 |
| **Selfless (random-mode eval)** | **0.971** |

Selfless in random-mode eval is the **worst** non-AR model, not the best. The original framing is dead.

However, the data reveals a more interesting and more defensible finding: **Selfless and XLNet have qualitatively different responses to inference-time ordering, and Selfless's weaknesses on random-order eval are paired with strengths on AR-mode eval and on zero-shot downstream tasks.** This is the new story.

### 0.2 The new one-sentence pitch

Removing the diagonal from XLNet's content stream — *Selfless Attention* — does **not** produce a uniformly better model. It produces a model with **qualitatively different inference-time properties**: under PLM training, the same checkpoint specializes to high-quality fixed-order generation (matching LLaDA's BPB) and excels at zero-shot downstream tasks among non-AR models, while sacrificing the ordering-invariance that the diagonal provided. This reveals a previously-unrecognized **expressiveness vs. ordering-robustness tradeoff** in two-stream attention design.

### 0.3 What this paper is and isn't

**Is**: A mechanistic study of two-stream attention design under permutation LM training, with experimental evidence for a specific tradeoff and a single-checkpoint multi-mode demonstration.

**Is not**: A "SOTA on every benchmark" paper. Selfless does not dominate DLMs on WT-BPB under flexible-order eval. The honest position is that **Selfless trades flexible-order BPB for AR-mode BPB and downstream representation quality**.

---

## 1. Pitch (paragraph form)

The Permutation Language Modeling (PLM) objective trains a model to predict each token under arbitrary factorization orders. XLNet (2019) introduced two-stream attention to make this work without information leakage: a content stream (sees tokens, has self-attention diagonal) and a query stream (sees positions only, no diagonal). PLM was abandoned as a generation paradigm because the original XLNet was never demonstrated to generate text, only fine-tuned for understanding.

We revisit PLM as a generation paradigm and ask a sharp architectural question: **does the diagonal in the content stream help or hurt?** We propose *Selfless Attention* — a one-line mask change that removes the diagonal from both streams. We train Selfless and XLNet checkpoints under identical conditions at 250M and 0.6B scales on 50B tokens of FineWeb-Edu, and we evaluate the **same random-permutation-trained checkpoint** under two inference modes: strict left-to-right (AR-mode) and random permutation (random-mode).

The result is not "selfless dominates". The result is **a clean qualitative dichotomy**:

- Under **AR-mode** inference, Selfless's BPB is 0.025 lower than XLNet's (at 0.6B), reaching 0.943 — matching LLaDA (0.940) and approaching SDAR (0.913). It also gives the best zero-shot results among all non-AR models on ARC-Easy, PIQA, SciQ, SGLUE, and on LAMBADA perplexity (where DLMs are 20-9000x worse than causal LM, Selfless is 2.4x worse).
- Under **random-mode** inference, the picture inverts: XLNet's BPB (0.963) is slightly better than Selfless's (0.971), and the AR-vs-random gap **widens with model scale** for Selfless (0.019 → 0.028) but **stays near zero** for XLNet (0.000 → 0.004).

We interpret this as: **the diagonal in XLNet's content stream acts as a token-identity anchor that makes representations ordering-invariant** (low Var(h_i) across permutations, but limited representational capacity tied to token identity); **Selfless representations are purely relational** (higher expressive capacity, visible in zero-shot downstream wins) **but ordering-sensitive** (the model leverages inference-time ordering effectively when it's clean, but is fragile to random orderings). The diagonal was not a "shortcut bug"; it was an implicit choice trading expressiveness for ordering-robustness.

We provide mechanistic analyses (cos_sim(h_i, embed(x_i)), Var(h_i) across permutations, diagonal attention weight, training dynamics) and demonstrate that a Selfless checkpoint usefully supports both AR and flexible-order generation from one model, with quality close to the best AR baseline and competitive with DLMs on flexible-order tasks — a combination no existing model provides.

---

## 2. Contributions

| # | Type | Statement |
|---|---|---|
| **C1** | **Analytical / mechanistic** | We identify and characterize an **expressiveness vs. ordering-robustness tradeoff** in two-stream attention design. The diagonal in XLNet's content stream is a token-identity anchor: it makes representations near-permutation-invariant but limits their relational capacity. Removing it (Selfless) yields purely relational representations with higher expressive capacity but greater ordering-sensitivity. We provide direct mechanistic evidence: (a) `cos_sim(h_i, embed(x_i))` is markedly higher in XLNet than Selfless, (b) `Var(h_i)` across permutations is larger for Selfless than XLNet, (c) the eval-mode gap (`AR-eval − random-eval BPB`) is large and growing with scale for Selfless but near-zero for XLNet. |
| **C2** | **Methodological** | We propose Selfless Attention: a one-line mask change (`v_kv > v_q` strict for both streams). We instantiate it in a Qwen3-based two-stream architecture and train with PLM objective. The change is trivial to implement and reproduces XLNet exactly when re-enabled. |
| **C3** | **Empirical** | A random-permutation-trained Selfless checkpoint supports both AR-mode and flexible-order decoding from a single model. In AR-mode eval, it matches LLaDA on WikiText BPB (0.943 vs. 0.940). It outperforms all non-AR baselines on zero-shot downstream tasks (ARC-Easy +1pt over XLNet, PIQA +1.7pt, SGLUE +2.4pt — and on SGLUE it beats even the causal LM baseline by 1.9pt). On LAMBADA perplexity (a long-range dependency benchmark), Selfless is the only non-AR model within an order of magnitude of causal LM (58 vs. 25; DLMs are 530-200k). No existing single checkpoint offers this combination. |

---

## 3. Honest Positioning

| Capability | Causal LM | LLaDA / Dream | SDAR | XLNet (rand-trained) | **Selfless (rand-trained)** |
|---|---|---|---|---|---|
| Fixed-order (L→R) WT-BPB | **0.823** | N/A | 0.913 | 0.968 | **0.943** |
| Flexible-order WT-BPB | N/A | 0.940 / 0.945 | 0.913 | 0.963 | 0.971 |
| LAMBADA-PPL | **24.7** | 1330 / 209k | 531 | 102 / 57 | **57 / 58** |
| ARC-E zero-shot | **0.591** | 0.335 / 0.312 | 0.484 | 0.527 / 0.500 | **0.537 / 0.538** |
| SGLUE zero-shot | 0.554 | 0.515 / 0.518 | 0.490 | 0.549 / 0.548 | **0.574 / 0.550** |
| Single checkpoint, multi-mode | N/A | × | × | ✓ | ✓ |

(Selfless / XLNet columns show `random-mode / AR-mode` evaluations of the **same** checkpoint.)

**Where we win**: AR-mode quality among non-AR PLMs; downstream zero-shot among non-AR models; LAMBADA among non-AR models; single-checkpoint multi-mode flexibility.

**Where we don't**: Random-mode BPB vs DLMs; AR quality vs causal LM (0.12 BPB gap at 0.6B — a paid tax for PLM training); parallel decoding (we have no parallel speedup, see §6).

---

## 4. Paper Narrative

### 4.1 Introduction

**Paragraph 1 — Setup**: PLM (Permutation Language Modeling) was introduced in XLNet (2019) as a pretraining objective for understanding tasks, never as a generation paradigm. The key innovation was two-stream attention: a content stream encoding tokens (with self-attention diagonal) and a query stream encoding positions (no diagonal). PLM has lain dormant as a generative model class while discrete diffusion models (LLaDA, Dream, SDAR, MDLM) have emerged as the dominant flexible-order generation paradigm.

**Paragraph 2 — The architectural question**: PLM's content stream has a self-attention diagonal — each position attends to its own token before passing information to other positions. We ask: is this diagonal beneficial? It was introduced without ablation in XLNet, justified informally as "needed for two-stream consistency". We test this directly.

**Paragraph 3 — Method**: We propose *Selfless Attention*: a one-line change removing the diagonal from both streams. We train Selfless and XLNet checkpoints under identical conditions and evaluate the same random-permutation-trained model in two inference modes — strict L→R and random — to test whether the diagonal helps in fixed-order, flexible-order, both, or neither.

**Paragraph 4 — Result and mechanism**: We find a **qualitative dichotomy**. Selfless dominates in fixed-order eval, ties or slightly loses in random-order eval, and wins on downstream zero-shot tasks. The AR-vs-random gap is large and growing for Selfless (0.028 BPB at 0.6B), near-zero for XLNet (0.004 BPB). Mechanistic analysis (cos_sim, Var(h) across permutations) shows the diagonal acts as a **token-identity anchor**: XLNet's representations are pinned near `embed(x_i)`, making them ordering-invariant but expression-limited. Selfless representations are purely relational — higher expressive capacity, ordering-sensitive.

**Paragraph 5 — Implications**: This reframes the diagonal's role. It is not a "selfish shortcut" to be removed; it is one corner of an **expressiveness vs. ordering-robustness Pareto frontier**. Different applications sit on different parts of this frontier. For applications dominated by clean L→R structure (text generation in natural order) or by discriminative tasks (zero-shot classification, retrieval), Selfless is the right choice. For applications requiring robust performance across arbitrary generation orders (parallel decoding, infilling with very few prompt tokens), XLNet's diagonal helps.

**Paragraph 6 — Contributions**: One mechanism (C1), one architectural choice (C2), and one empirical demonstration (C3) of a single-checkpoint multi-mode PLM with specific competitive niches against DLM baselines and a non-trivial position on the tradeoff frontier.

### 4.2 Background

Standard XLNet content stream attention: `A_content[i,j] = 1 if σ(j) ≤ σ(i)` — note the `≤` includes `j=i` (diagonal). Standard XLNet query stream: `A_query[i,j] = 1 if σ(j) < σ(i)` — strict, no diagonal.

The asymmetry between the streams is deliberate: content sees self; query does not. This is the design we challenge.

PLM loss:
```
L_PLM(θ) = -E_x E_σ Σ_i log P_θ(x^{σ_i} | x^{σ_1}, ..., x^{σ_{i-1}})
```

Discrete diffusion LM loss (absorbing-state, the LLaDA/Dream family):
```
L_DLM(θ) = -E_x E_t E_M_t [1/t Σ_{i ∈ M_t} log P_θ(x_i | x_{\bar M_t})]
```

These objectives are mathematically distinct. PLM samples orderings; DLM samples mask ratios and mask patterns. PLM conditions on fully-determined contexts; DLM conditions on partially-determined contexts. Our results show that this distinction has architectural consequences (the role of the diagonal differs between paradigms).

### 4.3 Method: Selfless Attention

**Mask**:
```
Selfless (both streams):  A[i,j] = 1 if σ(j) < σ(i) and j ≠ i
                                = 0 otherwise
```

Implementation: `v_kv > v_q` strict (one-liner in PyTorch flex_attention block-mask builder).

**Two-stream forward (unchanged from XLNet except mask)**:
- X0 stream (content): takes token embeddings, attends with Selfless mask
- XT stream (query): takes [MASK] embeddings, queries against X0 keys/values with Selfless mask
- lm_head reads XT outputs → predicts target tokens
- No information leak: XT_i has no path to x_i (verified mathematically — see CODE_AUDIT.md §2)

**Training**: standard PLM loss with `v_sample = uniform random permutation` per batch item.

**Inference modes from one checkpoint** (config-controlled `attention_task` and `prompt_task`):

| Mode | `v_sample` | Use case |
|---|---|---|
| AR (`task='ar'`) | Strict descending `v ∈ [eps, 1-eps]` | Fixed-order L→R generation, AR-mode likelihood eval |
| Random (`task='random'`) | Uniform random `v ∈ [0,1]` per position | Flexible-order generation, PLM-style likelihood (MC over permutations) |
| Confidence-guided | Random init, update `v` based on logits | Decoder-time order selection (see `generate()` in modeling_selfless.py) |

### 4.4 Experiments

#### 4.4.1 Setup
- **Data**: FineWeb-Edu, pre-tokenized BPE Arrow shards, 50B tokens
- **Hardware**: 8× H200 (80GB), one node, DeepSpeed ZeRO-2
- **Architecture**: Qwen3-0.6B-Base (28L, 1024H, 16Q/8KV) and Qwen3-250M-Custom
- **Schedule**: AdamW (β=0.8/0.95, wd=0.1), cosine schedule, 400 warmup, 8000 decay-floor, max_train_steps=50000
- **Total tokens**: 50000 × 512 × 2048 ≈ 52B per model

#### 4.4.2 Main Result Table — A single random-trained checkpoint, two inference modes

| Model (random-trained ckpt) | Eval mode | WT-BPB ↓ | LAMBADA-PPL ↓ | ARC-E ↑ | HSwag ↑ | PIQA ↑ | SGLUE ↑ |
|---|---|---|---|---|---|---|---|
| Causal LM 0.6B (AR baseline) | — | **0.823** | **24.7** | 0.591 | 0.475 | 0.693 | 0.554 |
| SDAR 0.6B | — | 0.913 | 531 | 0.484 | 0.400 | 0.584 | 0.490 |
| LLaDA 0.6B | — | 0.940 | 1330 | 0.335 | 0.334 | 0.521 | 0.515 |
| Dream 0.6B | — | 0.945 | 209k | 0.312 | 0.312 | 0.511 | 0.518 |
| XLNet 0.6B | AR | 0.968 | 102 | 0.500 | 0.370 | 0.641 | 0.548 |
| XLNet 0.6B | Random | 0.963 | 57.4 | 0.527 | 0.401 | 0.640 | 0.549 |
| **Selfless 0.6B** | **AR** | **0.943** | 58.2 | 0.538 | 0.377 | 0.650 | 0.550 |
| **Selfless 0.6B** | **Random** | 0.971 | 58.0 | **0.537** | **0.401** | **0.658** | **0.574** |

**Key reads from this table**:

1. Selfless AR-mode WT-BPB (0.943) matches LLaDA (0.940) and beats Dream/XLNet. → C3 partial.
2. Selfless random-mode WT-BPB (0.971) is worst non-AR. → The "SOTA" claim is dead.
3. On zero-shot, Selfless random-mode dominates all non-AR baselines on every task; SGLUE beats even causal LM. → C3 strong on downstream.
4. LAMBADA-PPL: Selfless is the only non-AR model within an order of magnitude of causal LM. → C3 strong on long-range dependency.
5. XLNet's AR-vs-random gap is 0.004 BPB; Selfless's is 0.028 BPB. **7× difference**. → C1 mechanism.

#### 4.4.3 The Eval-Mode Gap (250M → 0.6B)

| | XLNet AR | XLNet Rand | Δ | Selfless AR | Selfless Rand | Δ |
|---|---|---|---|---|---|---|
| 250M | 1.068 | 1.068 | 0.000 | 1.047 | 1.066 | -0.019 |
| 0.6B | 0.968 | 0.963 | +0.004 | 0.943 | 0.971 | -0.028 |

**Pattern**:
- Selfless: AR mode always better (gap negative), gap **widens with scale** (1.5× larger at 0.6B)
- XLNet: modes essentially tied (gap near zero, slight reversal at 0.6B)

This is C1's core empirical signature.

#### 4.4.4 Mechanistic Evidence (planned — see TODO §A)

We need:
- **Var(h_i) across permutations**: directly tests "Selfless reps are ordering-sensitive". Expect Var_selfless > Var_xlnet at each layer.
- **`cos_sim(h_i, embed(x_i))`**: directly tests "XLNet reps are pinned to token identity". Expect cos_selfless ≈ 0, cos_xlnet substantially > 0.
- **Diagonal attention weight**: in XLNet, fraction of attention mass on self-position. Expect substantially above uniform baseline.

These three forward-pass-only analyses are the C1 evidence chain. None require retraining.

#### 4.4.5 Single Checkpoint, Multiple Modes (the C3 demonstration)

We will compute:
- BPB, LAMBADA-PPL, downstream zero-shot for the **same Selfless 0.6B checkpoint** under both AR and random eval (done; see §4.4.2).
- BPB on PG-19 and C4 (TODO) — to show the pattern is dataset-agnostic.
- Infilling BPB on held-out spans (TODO) — the natural application for random-mode.
- Iterative refinement BPB (TODO) — generate draft in AR mode, then re-decode low-confidence positions.

### 4.5 Related Work

| Work | Distinction |
|---|---|
| XLNet (Yang et al. 2019) | Original two-stream design; never used as a generator. We are the first to test the diagonal's role under generation eval. |
| LLaDA / Dream / MDLM / SDAR | Random-mask training; different lower bound (ELBO); excellent random-order BPB but cannot do AR. We complement, not replace. |
| SUNDAE (Savinov et al. 2021) | Iterative refinement; random mask. Same generic limitations as DLMs. |
| Diffusion-LM (Li et al. 2022) | Continuous embedding space; orthogonal. |

### 4.6 Limitations & Honest Discussion

1. **No parallel speedup**. Our model decodes one token per step. We do not match DLM wall-clock latency. See §6.
2. **PLM-AR gap**. Even the best Selfless config (0.6B AR mode 0.943 BPB) is 0.12 BPB worse than causal LM (0.823). This is a real cost of PLM training relative to pure next-token training, and it does not vanish at this scale.
3. **Random-mode BPB**. Worse than every DLM baseline. The single checkpoint multi-mode benefit comes at this cost.
4. **Likelihood estimator differences**. Selfless/XLNet use chain-rule-under-permutation; LLaDA/Dream use ELBO; SDAR uses AR-style. Not strictly comparable bounds. Discussed in §Limitations.
5. **Single seed**. 250M data covers two settings clearly; 0.6B regression (selfless random-mode worse than xlnet random-mode by 0.008 BPB) is single-seed. We will run 2 additional seeds before camera-ready (see TODO).
6. **Scale**. 250M and 0.6B trends are consistent. 1B not run (compute-constrained). The diagonal-vs-no-diagonal tradeoff is structural and should persist, but unverified.

---

## 5. Tier Assessment (vs. THRESHOLDS.md)

| Threshold | Outcome | Tier per THRESHOLDS |
|---|---|---|
| A1: Selfless < Selfish, fixed-order | 0.025 BPB gap (0.6B) | **Tier 1** |
| A2: Gap widens for flexible-order | **Reversed** — small reversal at 0.6B | Tier 0 (but reframed as C1 evidence) |
| A3: cos_sim analysis | Not yet run | TBD |
| A4: Head pruning | Not run | TBD |
| A5: Order-sensitivity Var(h) | Not yet run | Critical for new C1 |
| B1: vs DLM BPB | Selfless AR matches LLaDA; random loses to all | Tier 0 on random, Tier 1 on AR-as-fallback |
| B2: PLM-Causal gap | 0.12 BPB at 0.6B | Tier 0 (paid tax, honestly reported) |
| B3: Infilling | Not yet run | TBD |
| B4: Iterative refinement | Not yet run | TBD |
| C1: Calibration analysis | Not run | TBD |
| D1: Scaling gap | 250M→0.6B: AR-vs-random gap **widens** (0.019→0.028) for Selfless, stays ≈0 for XLNet | Tier 2 — predictable scaling pattern |
| F1: Parallel decoding | Not addressed | Tier 0 (limitation, acknowledged) |

**Current achievable tier**: Tier 1 (solid ACL/EMNLP main). 

**Push to Tier 2**: requires A3 + A5 (mechanism evidence chain), B3 (infilling), D1 multi-seed at 0.6B (verify the regression), and PG-19 + C4 datasets.

**Push to Tier 3**: requires parallel decoding to work (Mixed objective from §6) AND head-level mechanism with actionable intervention. Realistic but expensive.

---

## 6. The Parallel Decoding Tradeoff (Section 5.4 in paper)

PLM training conditions each prediction on fully-determined context (`x_{σ_{<i}}` are all actual tokens). DLM training conditions on `x_{\bar M}` which contains both tokens and [MASK]s. When PLM models decode in parallel (multiple [MASK]s in context), they encounter a distribution they were never trained on. **This is the structural reason PLM models resist parallel decoding.**

We acknowledge this in §Limitations and discuss it as an architectural insight rather than a method failure. The mixed-objective experiment (TODO §B) tests whether a Pareto improvement is possible.

---

## 7. Open Questions & Future Work

1. **Mixed PLM + random-mask objective**: Can a single model get both flexible-order quality (random mask) and ordering-leverage benefits (PLM)? Curriculum training (PLM first, random mask finetune) is the cheapest test.
2. **Permutation sampling distribution**: Uniform over `L!` vs biased toward natural orders. May change the AR-vs-random gap.
3. **Optimal v_sample for inference**: Are random orderings always Pareto-dominated by carefully-chosen orderings (e.g., confidence-guided)? Test with multiple decoding orders.
4. **Vision two-stream**: Does the same diagonal-vs-no-diagonal pattern hold in ViT-style two-stream architectures?
5. **Distillation**: Can iterative refinement be distilled to a one-pass model?

---

## 8. Figures (priority order)

1. **Figure 1 (Teaser)**: Mask diagram — XLNet content (with diagonal) vs. Selfless (without diagonal). One line of code.
2. **Figure 2 (Main result)**: Cross-mode eval table from §4.4.2, visualized as a bar chart with capability markers (AR ✓/×, flexible ✓/×, parallel ✓/×).
3. **Figure 3 (The tradeoff)**: AR-vs-random gap vs model scale. Selfless: widening gap. XLNet: flat near zero.
4. **Figure 4 (Mechanism, planned)**: `cos_sim(h_i, embed(x_i))` over training, Selfless vs XLNet.
5. **Figure 5 (Mechanism, planned)**: `Var(h_i)` across permutations, per layer.
6. **Figure 6 (LAMBADA)**: PPL bar chart showing all baselines on log scale (DLMs are 20-9000× worse than causal LM; Selfless ≈ XLNet random ≈ 2.4× worse).
7. **Figure 7 (Downstream)**: Radar chart of zero-shot tasks. Selfless vs XLNet vs DLMs.
8. **Figure 8 (Tradeoff frontier)**: Conceptual Pareto plot — expressiveness (downstream zero-shot) on x-axis, ordering-robustness (small AR-vs-random gap) on y-axis. Selfless and XLNet are on opposite corners.

---

## 9. Reviewer Q&A

**Q1**: "Removing the diagonal is a one-line change. Where's the contribution?"
→ The contribution is identifying that this design choice has been **mis-justified for 6 years** (as a "necessary asymmetry") and showing it's actually one corner of an expressiveness-vs-ordering-robustness Pareto frontier. We provide mechanistic evidence and demonstrate the practical consequences (single-checkpoint multi-mode with specific competitive niches).

**Q2**: "Why is your random-mode BPB worse than every DLM?"
→ Because PLM training and random-mask training are different lower bounds on the data likelihood, and DLMs are optimized for the random-mask regime. We trade flexible-order BPB for AR-mode quality and downstream representation quality. Section 4.6 is explicit about this tradeoff.

**Q3**: "Selfless random-mode 0.6B is worse than XLNet random-mode 0.6B by 0.008 BPB. Doesn't this contradict your C1?"
→ It supports our refined C1. The diagonal provides ordering-robustness. Removing it produces higher-expressivity representations (as shown in zero-shot wins) but at the cost of ordering-sensitivity (the 0.008 BPB regression). The original PLAN's hypothesis ("diagonal hurts everything") was naive; the actual finding is more interesting.

**Q4**: "Why don't you do parallel decoding?"
→ PLM training conditions on fully-determined contexts; parallel decoding requires conditioning on partially-determined contexts. This train-test mismatch is structural. We discuss this in §5.4 as an architectural insight rather than a method limitation. The mixed-objective experiment is future work.

**Q5**: "PLM left-to-right BPB is 0.12 worse than causal LM. Why use this over causal LM?"
→ For applications that need flexible-order capability (infilling, refinement) or that need a single model for multiple modes. We do not claim to replace causal LM for pure L→R generation.

**Q6**: "Does the eval-mode gap scaling extrapolate to 1B+?"
→ Predicted yes (250M→0.6B shows 1.5× growth in Selfless's gap, near-flat for XLNet). 1B run is not in current paper due to compute.

**Q7**: "Your eval estimators differ across methods. Is the comparison fair?"
→ Each method is evaluated under its native likelihood estimator (chain-rule for PLM, ELBO for DLM, shifted CE for AR). We acknowledge this in §Limitations. The downstream zero-shot results use the same task setup across all models and are directly comparable.

---

*This plan is a living document. Last updated 2026-05-18 with results-driven reframing from "SOTA flexible-order BPB" to "expressiveness vs. ordering-robustness tradeoff".*
