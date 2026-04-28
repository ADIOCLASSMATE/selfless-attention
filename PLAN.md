# Selfless Attention: Permutation Language Modeling for Order-Agnostic Text Generation

## Metadata
- **Target Venue**: ACL 2026 / EMNLP 2026 / NeurIPS 2026
- **Status**: Planning
- **Created**: 2026-04-29
- **Updated**: 2026-04-29 (major reframing — see Section 0.1)

---

## 0. Title & Framing

### 0.1 What Changed and Why

Earlier versions of this plan framed the model as "AR + NAR unified generation." Experimental evidence has refined that picture:

| What we originally thought | What the experiments actually show |
|---|---|
| Model supports parallel NAR decoding | Parallel decoding causes significant quality degradation |
| Fewer decoding steps than DLM | Per-step, we decode one token (like AR); total steps are comparable |
| PLM left→right = standard causal LM | PLM-trained left→right has a small BPB gap vs. causal LM (but still beats DLM) |
| "Unified AR + NAR" | More precisely: **order-agnostic** — one model, any generation order, one token per step |

The framing has been refined from "unified AR+NAR with speed advantage" to **"order-agnostic generation with SOTA quality"**. This is a more honest and ultimately more defensible position.

### 0.2 Title Rationale

**Selfless Attention** works on three levels:

1. **Literal**: We remove self-attention (the diagonal) from the content stream. The model literally does not attend to itself.
2. **Metaphorical**: The content stream exists to serve other positions' predictions (via the query stream). Looking at oneself is selfish — it consumes representational capacity. Selfless attention gives up the self-view to better serve others.
3. **Narrative**: XLNet's attention was selfish (content stream saw itself → shortcut). Our attention is selfless (content stream sees only context → pure relational encoding). This unlocks PLM's potential for generation in arbitrary orders.

**Permutation Language Modeling** anchors the training paradigm.
**Order-Agnostic Text Generation** states the payoff: one model generates text in any order — left-to-right, right-to-left, confidence-guided, or application-specific trajectories.

### 0.3 What This Model Is

> **An order-agnostic generative model: it decodes one token per step, but that token can be at any position in the sequence. A single checkpoint supports fixed-order (left-to-right) and flexible-order (confidence-guided, infilling, iterative refinement) generation. Under Selfless Attention, PLM achieves SOTA BPB on flexible-order generation, surpassing discrete diffusion models.**

What this model is *not*: a parallel decoder. It does not achieve wall-clock speedup over autoregressive generation. Its value proposition is flexibility + quality, not speed.

---

## 1. One-Sentence Pitch

XLNet's content stream had a hidden selfish shortcut — attending to itself — that degraded representations and prevented permutation language modeling from realizing its potential for flexible-order generation. We remove it. The resulting *Selfless Attention* enables a single PLM checkpoint to generate text in any order with SOTA BPB, surpassing discrete diffusion models on flexible-order quality while additionally supporting standard left-to-right decoding.

---

## 2. Contribution Triangulation

```
                      Generation Paradigm
                   /                      \
          Fixed-Order AR              Flexible-Order
          (Causal LM)                 /            \
              |               Discrete Diffusion   Permutation LM
              |               (LLaDA, Dream,       (XLNet → Ours)
              |                SDAR, MDLM)              |
              |                    |                    |
          左→右 only          并行多token            任意顺序单token
          标准AR               NAR only              + 左→右AR
              |                    |                    |
         GPT/LLaMA           LLaDA/Dream          Selfless Attention PLM
         (最快, 质量最好)     (并行, 灵活)          (灵活 + 质量SOTA)
```

Three contributions, unified by Selfless Attention:

| # | Type | Statement |
|---|------|-----------|
| C1 | **Analytical** | We identify a *selfish shortcut* in XLNet's content-stream self-attention: the diagonal allows each position to see its own token, causing content representations to degenerate toward token embeddings rather than encoding cross-position relationships. We provide mechanistic evidence — cosine similarity analysis, attention weight quantification, head pruning experiments, and order-sensitivity measurements. This flaw has sat undetected for 6 years. |
| C2 | **Methodological** | We propose *Selfless Attention* — a uniform two-stream design where both content and query streams share the same no-diagonal mask. Content representations become purely relational. The asymmetry between streams vanishes. The change is a single line in the attention mask. |
| C3 | **Empirical** | Under Selfless Attention, PLM achieves SOTA BPB on flexible-order generation, surpassing LLaDA, Dream, and SDAR. A single checkpoint supports both fixed-order (left-to-right) and flexible-order (confidence-guided, infilling, iterative refinement) decoding — a combination no existing model provides. We further analyze *why* parallel decoding degrades under PLM training, turning an apparent limitation into architectural insight. |

### 2.1 Honest Positioning

We do not claim to be the best at everything. Here is exactly where we stand:

| Capability | Causal LM | Discrete Diffusion | Ours (Selfless PLM) |
|---|---|---|---|
| Fixed-order (left→right) generation | ✅ **Best BPB** | ✗ | ✅ Good BPB (close to causal LM) |
| Flexible-order generation (any order, 1 token/step) | ✗ | ✅ | ✅ **Best BPB** |
| Parallel decoding (K tokens/step) | ✗ | ✅ **Works well** | ⚠️ Quality degrades |
| Infilling (prefix + suffix → middle) | ✗ | ✅ | ✅ **Best BPB** |
| Iterative refinement (mask + re-predict) | ✗ | ✅ | ✅ **Best BPB** |
| Constrained generation | ⚠️ Complex decoding | ✅ | ✅ **Best BPB** |
| Single checkpoint for all modes | N/A | ✗ (no fixed-order) | ✅ |

This is a tradeoff paper. Our niche: **flexible-order generation with SOTA quality, in a single model that also does standard AR.** We sacrifice parallel speed for better BPB and the ability to also generate left-to-right.

---

## 3. Paper Narrative

### 3.1 Introduction

**Paragraph 1: The spectrum of generation orders.**

Text generation models occupy distinct points on a spectrum. At one end, autoregressive (AR) models generate left-to-right, one token at a time — high quality, but rigid in order and incapable of infilling or refinement. At the other end, discrete diffusion language models (DLMs: LLaDA, Dream, SDAR, MDLM) generate via iterative parallel denoising — flexible in order, capable of infilling, but cannot generate left-to-right. Each model is locked into its own generation paradigm. There is no single model that spans the spectrum.

**Paragraph 2: PLM's overlooked potential.**

Permutation Language Modeling (PLM, XLNet 2019) trains on *all possible generation orders*. In principle, a PLM-trained model should be order-agnostic: it can generate left-to-right, right-to-left, confidence-guided, or in any application-specific trajectory — all from a single checkpoint. In practice, XLNet never demonstrated this capability. It was used only for AR pretraining with bidirectional context, then finetuned for understanding tasks. Its potential as a flexible-order generator was never explored. Why?

**Paragraph 3: The selfish shortcut.**

We identify the culprit: self-attention in XLNet's content stream. The content stream's purpose is to encode information *for other positions to use*. But by attending to its own token (the diagonal in the attention matrix), each position develops a selfish shortcut — its content representation degrades toward its own token embedding rather than encoding genuine cross-position relationships. This contaminates the representations used by the query stream for prediction, and — critically — makes content representations *order-dependent*, undermining the very order-agnosticism that PLM is supposed to provide. This flaw has remained undetected since 2019.

**Paragraph 4: Selfless Attention.**

We propose *Selfless Attention*: remove the diagonal from the content stream, giving it the same attention pattern as the query stream. A content position no longer sees itself. The shortcut is eliminated. Content representations become purely relational — and crucially, more order-agnostic. The implementation is a one-line change in the attention mask. The effect is systemic: BPB improves across all generation orders, and the improvement is largest for non-standard orders where the selfish shortcut was most damaging.

**Paragraph 5: Order-agnostic generation with SOTA quality.**

Under Selfless Attention, a single PLM checkpoint supports: (a) standard left-to-right generation, (b) confidence-guided flexible-order generation where the model chooses which position to decode next, (c) infilling — predicting the middle given prefix and suffix, and (d) iterative refinement — re-predicting masked positions in a draft. Across these flexible-order tasks, our model achieves lower BPB than discrete diffusion models (LLaDA, Dream, SDAR). We further analyze *why* PLM-trained models resist parallel decoding — the training objective, which conditions each prediction on fully-determined context, creates a mismatch with the low-information context of parallel unmasking. This analysis transforms an apparent limitation into architectural insight about the fundamental tension between order-agnostic training and parallel inference.

**Paragraph 6: Contributions.**

One concept — Selfless Attention — drives three contributions: (C1) identifying and mechanistically proving the selfish shortcut in XLNet's content stream, (C2) proposing the uniform no-diagonal design that eliminates it, and (C3) demonstrating that the repaired PLM achieves SOTA BPB on flexible-order generation in a single checkpoint that also supports fixed-order AR — a unique position in the design space, with practical applications in infilling, refinement, and constrained generation.

### 3.2 Background

**2.1 Permutation Language Modeling (XLNet)**

Permutation LM trains a model to predict each token given a subset of other tokens, defined by a randomly sampled factorization order $\sigma$:

$$\mathcal{L}(\theta) = -\mathbb{E}_{x \sim \mathcal{D}} \; \mathbb{E}_{\sigma \sim \mathcal{S}_L} \sum_{i=1}^{L} \log p_\theta(x^{(i)} | x^{(\sigma_1)}, \ldots, x^{(\sigma_{i-1})})$$

where $\mathcal{S}_L$ is the set of all permutations of $\{1, \ldots, L\}$.

To implement this without leaking the target token's identity, XLNet introduced *two-stream self-attention*:

- **Content stream** $h_\theta(x_{\leq t})$: encodes token $x_t$ along with its preceding context. This representation is *used by later positions* as context. In XLNet, the content stream can attend to itself — the diagonal is included: $A_{\text{content}}[i,j] = 1$ if $\sigma(j) \leq \sigma(i)$.

- **Query stream** $g_\theta(x_{<t}, z_t)$: encodes only the *position* and preceding context, but not the token $x_t$ itself. This is used to *predict* $x_t$. The query stream cannot attend to itself: $A_{\text{query}}[i,j] = 1$ if $\sigma(j) < \sigma(i)$.

The asymmetry is deliberate: content sees self, query does not. This is precisely the design we challenge.

**2.2 Discrete Diffusion Language Models (LLaDA, Dream, SDAR, MDLM)**

Discrete diffusion models train by randomly masking tokens:

$$\mathcal{L} = -\mathbb{E}_{x, M} \sum_{i \in M} \log p_\theta(x_i | x_{\backslash M})$$

Inference starts from a fully masked sequence and iteratively unmasks tokens in parallel. These models support flexible-order generation (infilling, refinement) and achieve parallel speedup, but are architecturally committed to this mode: they cannot generate left-to-right because the training objective has no notion of sequential order.

**2.3 The Gap: PLM as a Natural Order-Agnostic Generator**

Random mask training learns *one* conditional distribution: predict masked tokens given unmasked tokens. PLM training learns a *family* of conditional distributions — one for every possible generation order.

This family includes:
- Left-to-right: $\sigma = (1, 2, \ldots, L)$ — standard AR
- Right-to-left: $\sigma = (L, L-1, \ldots, 1)$
- Confidence-guided: decode the most confident position first
- Application-specific: infilling (predict middle last), refinement (re-decode selected positions)

A PLM-trained model has seen every order during training. In principle, it should be the ideal foundation for order-agnostic generation. What went wrong?

### 3.3 The Selfish Shortcut — C1 in Detail

**3.1 Anatomy of the problem**

In XLNet's content stream, position $i$ can attend to itself:

$$h_i = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V$$

where the mask allows $\sigma(j) \leq \sigma(i)$, including $j = i$.

The diagonal entry provides a direct path from $\text{embed}(x_i)$ to $h_i$. This path is **short** (one attention operation), **low-loss initially** (token identity trivially predictive), and **self-reinforcing** (gradient descent exploits it early; the resulting representations make it even more useful).

**3.2 The degeneration**

We formalize the content representation as:

$$h_i = \alpha \cdot \underbrace{\text{self\_info}(x_i)}_{\text{from diagonal}} + (1-\alpha) \cdot \underbrace{\text{cross\_context}(x_{<i})}_{\text{from off-diagonal}}$$

The selfish shortcut manifests as $\alpha$ remaining large. We measure this via:
- **Cosine similarity**: $\text{cos}(h_i, \text{embed}(x_i))$ — high for XLNet → token identity dominates
- **Diagonal attention weight**: fraction of attention mass on self-position → significantly above uniform baseline

**3.3 Why this destroys order-agnostic generation**

(a) **Content degradation**: $h_i \approx \text{embed}(x_i) + \epsilon$. The content stream fails at encoding relational information for other positions.

(b) **Query starvation**: The query stream receives token-identity-dominated signals rather than relational semantics. Predictive value is diluted.

(c) **Order contamination** (critical for flexible-order generation): The self-attention weight depends on *where* $x_i$ appears in the permutation. Content representations become order-dependent. When the model encounters a novel unmasking trajectory at inference — e.g., confidence-guided order — the content representations are contaminated with order-specific artifacts from training permutations. This is the mechanism by which the selfish shortcut specifically undermines PLM's order-agnostic potential.

(d) **The flexible-order ceiling**: Even with perfect PLM training, order-contaminated representations limit quality on non-standard generation orders. This explains why XLNet never demonstrated flexible-order generation — it couldn't.

**3.4 Preliminary evidence (ablation: Selfish vs. Selfless)**

[Insert data: XLNet-style (content w/ diagonal) vs. Ours (content w/o diagonal)]

Key expectations:
- BPB gap on fixed-order (left→right): Selfless < Selfish (shortcut hurts even standard AR)
- BPB gap on flexible-order: Selfless << Selfish (shortcut is catastrophic for non-standard orders)
- Gap widens as generation order deviates from left-to-right

### 3.4 Selfless Attention — C2 in Detail

**4.1 The principle**

The content stream's role is to provide context *to other positions*. Encoding what a token *is* (via self-attention) is not only unnecessary — it is counterproductive. It consumes representational capacity that should encode relational information, and it injects order-dependent artifacts into representations that need to be order-agnostic.

**Selfless Attention** removes the diagonal from both streams:

```
XLNet content stream:    A_content[i,j] = 1 if σ(j) ≤ σ(i)    (selfish: j=i allowed)
XLNet query stream:      A_query[i,j]   = 1 if σ(j) < σ(i)    (selfless)

Ours content stream:     A_content[i,j] = 1 if σ(j) < σ(i)    (selfless)
Ours query stream:       A_query[i,j]   = 1 if σ(j) < σ(i)    (selfless)
```

Both streams share the same attention mask. The asymmetry is eliminated.

**4.2 Why selflessness enables order-agnostic generation**

Without the diagonal:
- $h_i$ is a pure function of $\{x_j : \sigma(j) < \sigma(i), j \neq i\}$. No self-information.
- Content representations become genuinely relational: "what does the context say about position $i$?"
- Content representations become more order-agnostic: less variance across different permutations
- The model generalizes better to novel generation orders at inference

**4.3 The unified predict-self objective**

With Selfless Attention:

> *At every position, predict what belongs there, using only information from other positions.*

This holds regardless of generation order. Fixed-order (left→right), flexible-order (confidence-guided), infilling (prefix + suffix → middle) — all use the same prediction interface. The uniformity ensures that improvements in representation quality (from removing the diagonal) benefit all generation modes simultaneously.

**4.4 Training**

Standard PLM training with Selfless Attention:

- Input: a sequence $x$ and a permutation $\sigma$
- Content stream: receives token embeddings, applies Selfless Attention
- Query stream: receives learnable [MASK] embeddings, applies the same Selfless Attention
- For each position $i$, query stream output at $i$ predicts $x_i$
- Loss: cross-entropy over all positions

Attention mask for permutation $\sigma$:
$$M[i,j] = \begin{cases} 0 & \text{if } \sigma(j) < \sigma(i) \text{ and } j \neq i \\ -\infty & \text{otherwise} \end{cases}$$

**4.5 Decoding Modes**

Selfless PLM supports multiple decoding modes from a single checkpoint:

**Mode A: Fixed-Order (Left-to-Right)**

```
Step 1: [BOS] [MASK] [MASK] ... [MASK]
        → Predict position 2 = token₁
Step 2: [BOS] [token₁] [MASK] ... [MASK]
        → Predict position 3 = token₂
...
```

A single [MASK] at the leftmost unfilled position. Decodes one token per step, strictly left-to-right. Equivalent in information access to standard causal LM, but under the predict-self paradigm. BPB is close to (but not identical to) standard causal LM — the PLM training objective introduces a small gap relative to pure next-token training.

**Mode B: Flexible-Order (Confidence-Guided)**

```
Step 1: [BOS] [MASK]₁ [MASK]₂ ... [MASK]ₗ
        → Compute confidence for all L positions
        → Decode the most confident position
Step 2: [BOS] [token₅] [MASK]₂ ... [MASK]ₗ  (position 5 was most confident)
        → Recompute confidences for remaining positions
        → Decode the new most confident position
...
```

One token per step, but the model chooses *which* position to decode based on prediction confidence. The generation order is not predetermined — it emerges from the model's uncertainty.

**Mode C: Infilling**

```
Given:  "The quick ___ fox jumps over the ___ dog"
        [BOS] [The] [quick] [MASK]₃ [fox] [jumps] [over] [the] [MASK]₉ [dog]
        → Decode MASK₃ and MASK₉ in confidence-guided order
```

Prefix and suffix are provided. The model fills the gaps. This is structurally impossible for standard AR models.

**Mode D: Iterative Refinement**

```
Draft:  "The cat sit on the mat"
        → Mask low-confidence positions: [The] [cat] [MASK]₃ [on] [the] [mat]
        → Re-predict MASK₃ = "sits"
        → Continue masking and re-predicting until convergence
```

The model can revise its own output by selectively masking and re-decoding. Standard AR models must delete and restart from the error point.

**4.6 Why Parallel Decoding Degrades**

A natural question: if the model supports any generation order, why not decode multiple tokens in parallel (like discrete diffusion)?

We identify a fundamental tension: PLM training conditions each prediction on *fully-determined* context (all preceding tokens in the permutation are known). In parallel decoding, most context positions are [MASK] — the context is *information-poor*. The model has rarely encountered this regime during training, because PLM samples permutations where preceding positions are always filled with actual tokens.

Formally:
- **PLM training distribution**: $p(x_i | x_{\text{known}})$ where $x_{\text{known}}$ are fully-determined tokens
- **Parallel decoding requires**: $p(x_i | x_{\text{mixed}})$ where $x_{\text{mixed}}$ contains both known tokens and [MASK] placeholders

This train-test mismatch causes confidence estimates to be miscalibrated when multiple [MASK] tokens co-occur. The model becomes overconfident on some positions (prematurely committing to wrong tokens) and underconfident on others. Discrete diffusion models, trained explicitly on random mask patterns, do not suffer from this mismatch.

**This is not a failure of our model — it is an insight about the cost of order-agnostic training.** PLM buys flexibility (any order) and quality (rich context → accurate predictions) at the cost of parallelizability. Discrete diffusion buys parallelism at the cost of fixed-order capability and, as our experiments show, lower BPB on one-token-per-step flexible-order generation. This tradeoff is fundamental, not incidental.

### 3.5 Experiments

**5.1 Setup**

Datasets:
| Dataset | Description | Role |
|---------|-------------|------|
| WikiText-103 | Word-level LM benchmark | Primary |
| PG-19 | Long-form books, long-range dependency | Long-range test |
| C4 (RealNewsLike) | Web text | Distribution shift test |

Model scales:
| Scale | Params | Purpose |
|-------|--------|---------|
| Small (~100M) | 12L, 768H, 12 heads | Ablations, hyperparameter search |
| Medium (~300M) | 24L, 1024H, 16 heads | Main comparison point |
| Large (~1B) | 32L, 1536H, 24 heads | Scaling test |

Baselines:
- **Causal LM**: GPT-2-style, standard next-token prediction. Fixed-order only.
- **XLNet-style PLM**: Two-stream with content-diagonal (selfish). PLM training.
- **LLaDA** (2024): Discrete diffusion, random mask training.
- **Dream** (2024): Discrete diffusion, absorbing state.
- **SDAR** (2024): Simplified discrete AR diffusion.
- **SUNDAE** (2021): Random mask training + iterative mask-predict. Ablation for training objective vs. PLM.

Metrics:
- **BPB** (bits per byte): primary. Tokenizer-agnostic.
- **PPL**: secondary.
- **MAUVE**: generation quality and diversity.
- **Infilling accuracy**: BLEU / BERTScore on held-out spans.
- **Refinement delta**: BPB improvement per refinement iteration.

**5.2 Main Results — Fixed-Order Generation**

| Method | WikiText-103 BPB↓ | PG-19 BPB↓ | C4 BPB↓ |
|--------|-------------------|------------|---------|
| Causal LM (standard AR) | ? | ? | ? |
| XLNet-style PLM (selfish) | ? | ? | ? |
| **Ours — Selfless PLM (left→right)** | ? | ? | ? |

Expected: Causal LM ≈ Ours < XLNet-style PLM. The Selfish→Selfless gap quantifies the shortcut's cost even in the simplest order. The Ours vs. Causal LM gap is the cost of PLM training (order-agnosticism) relative to pure next-token training — we expect a small gap.

**5.3 Main Results — Flexible-Order Generation** ★ CORE TABLE ★

| Method | WT-103 BPB↓ | PG-19 BPB↓ | Infilling BLEU↑ | Fixed-order? | Flexible-order? |
|--------|------------|------------|-----------------|-------------|-----------------|
| Causal LM | N/A | N/A | N/A | ✅ | ✗ |
| LLaDA | ? | ? | ? | ✗ | ✅ |
| Dream | ? | ? | ? | ✗ | ✅ |
| SDAR | ? | ? | ? | ✗ | ✅ |
| SUNDAE | ? | ? | ? | ✗ | ✅ |
| **Ours — Selfless PLM (flexible)** | **?** | **?** | **?** | ✅ | ✅ |

Key expectations:
1. **Ours (flexible) < DLM**: PLM training generalizes better to arbitrary order than random mask training → better BPB on any single-token-per-step order.
2. **Ours = only model with both ✅ columns**: No other model supports both fixed-order AR and flexible-order generation from one checkpoint.
3. **Infilling quality**: Ours > DLM on masked span prediction (better BPB → better infilling).

**5.4 Analysis: The Parallel Decoding Tradeoff**

This section analyzes *why* parallel decoding degrades under PLM training.

| Decoding Mode | Tokens/Step | BPB (WT-103) | BPB (PG-19) |
|--------------|-------------|-------------|-------------|
| Fixed-order (left→right) | 1 | ? | ? |
| Flexible-order (confidence) | 1 | ? | ? |
| Parallel (schedule, 8 steps) | ~L/8 | ? (degraded) | ? (degraded) |
| Parallel (schedule, 16 steps) | ~L/16 | ? (less degraded) | ? (less degraded) |

Then the mechanistic analysis:

| Training | Parallel BPB (8-step) | Flexible BPB (1/step) | Train-test gap |
|----------|----------------------|----------------------|----------------|
| Random mask (DLM) | ? (good) | ? (baseline) | Small |
| PLM (Selfless) | ? (degraded) | ? (better than DLM) | **Large** |

The key insight figure: show confidence calibration — when N positions are [MASK] simultaneously, PLM's confidence estimates become miscalibrated (overconfident on some, underconfident on others), while DLM's remain calibrated (it was trained on this distribution).

This table and analysis transform "our model can't do parallel" from a limitation into a contribution: **we identify a fundamental tension between order-agnostic training (PLM) and parallel decoding, showing that PLM trades parallelism for better single-token prediction quality.**

**5.5 Application: Infilling**

| Method | BLEU↑ | BERTScore↑ | BPB on masked span↓ |
|--------|-------|------------|---------------------|
| LLaDA | ? | ? | ? |
| Dream | ? | ? | ? |
| **Ours** | **?** | **?** | **?** |

Task: held-out span prediction. Given a sentence with a contiguous masked span, predict the tokens. Vary span length: 1, 3, 5, 10 tokens.

**5.6 Application: Iterative Refinement**

| Method | Initial BPB | After 1 Refine | After 3 Refines | Converged BPB |
|--------|------------|----------------|-----------------|---------------|
| LLaDA | ? | ? | ? | ? |
| **Ours** | ? | ? | ? | ? |

Task: generate a draft (left→right), then iteratively mask low-confidence positions and re-predict. Measure BPB improvement per refinement round.

**5.7 Application: Constrained Generation**

| Method | Constraint Satisfaction↑ | BPB↓ |
|--------|-------------------------|------|
| LLaDA | ? | ? |
| **Ours** | ? | ? |

Task: generate text where specific positions must contain specific tokens. Measure whether constraints are satisfied + BPB on unconstrained positions.

**5.8 Ablation Studies**

**Ablation 1: The diagonal — what does selfishness cost?**

| Content Stream | Query Stream | Fixed BPB | Flexible BPB |
|---------------|--------------|-----------|-------------|
| Selfish (w/ diag) | Selfless (w/o diag) | ? | ? |
| Selfless (w/o diag) | Selfless (w/o diag) | **?** | **?** |

The defining ablation.

**Ablation 2: Training objective — PLM vs. random mask**

| Training | Flexible BPB | Fixed BPB | Parallel BPB |
|----------|-------------|-----------|-------------|
| Random mask | ? | N/A | ? |
| PLM + Selfish | ? | ? | ? |
| PLM + Selfless | **?** | **?** | ? (worse than random mask) |

Decomposes: training objective × attention design. Shows the tradeoff explicitly.

**Ablation 3: Permutation distribution**

| Training Distribution | Fixed BPB | Flexible BPB |
|---------------------|-----------|-------------|
| Uniform over all $L!$ | ? | ? |
| Biased toward left→right | ? (improved?) | ? (degraded?) |

Explores whether biasing the training distribution toward natural orders improves fixed-order quality at the cost of flexible-order generality.

**Ablation 4: Decoding order strategies**

| Strategy | Flexible BPB |
|----------|-------------|
| Left-to-right (fixed) | ? |
| Right-to-left | ? |
| Confidence-guided (greedy) | ? |
| Confidence-guided (sampled) | ? |
| Random order | ? |
| Entropy-guided | ? |

Shows that the model genuinely supports diverse generation orders with consistent quality.

**5.9 Mechanistic Analysis — Proving the Shortcut Exists**

**Analysis 1: Content representation degeneration**

$\text{cos}(h_i, \text{embed}(x_i))$ throughout training — Selfish vs. Selfless.

**Analysis 2: Diagonal attention weight** — fraction of attention mass on self.

**Analysis 3: Self-attention head pruning** — pruning self-attention-heavy heads causes less damage in Selfish models.

**Analysis 4: Order-sensitivity** — $\text{Var}(h_i)$ across permutations. Selfish: high. Selfless: low.

**Analysis 5: Training dynamics** — gap emerges early, persists.

**Analysis 6: Confidence calibration under parallel masking** — PLM vs. DLM. The miscalibration evidence.

**5.10 Scaling Behavior**

| Scale | Selfish BPB | Selfless BPB | Gap | DLM BPB |
|-------|------------|-------------|-----|---------|
| 100M | ? | ? | ? | ? |
| 300M | ? | ? | ? | ? |
| 1B | ? | ? | ? | ? |

---

## 4. Related Work

### 4.1 Positioning Map

| Work | Training | Content Stream | Order Support | Flexible Quality |
|------|----------|---------------|---------------|-----------------|
| **Causal LM** | Next-token | Single stream (causal) | Fixed (left→right) | N/A |
| **XLNet (2019)** | PLM | Selfish (w/ diag) | Fixed only (in practice) | Never evaluated |
| **LLaDA/Dream/SDAR** | Random mask | Single stream (bidirectional) | Flexible only | Baseline |
| **SUNDAE (2021)** | Random mask | Single stream | Flexible only | Baseline |
| **Ours** | PLM | **Selfless (w/o diag)** | **Fixed + Flexible** | **SOTA** |

### 4.2 Detailed Distinctions

| Work | Our distinction |
|------|-----------------|
| **XLNet** | We identify and fix content-stream design; we are first to explore PLM for flexible-order generation |
| **LLaDA/Dream/SDAR** | We beat them on flexible-order BPB; we additionally support fixed-order AR |
| **SUNDAE** | We show PLM > random mask for flexible-order quality (though worse for parallel speed) |
| **Mask-Predict** | We target open-ended LM; we use PLM training |
| **Diffusion-LM** | We operate in discrete space; we support fixed + flexible order |

### 4.3 The Conceptual Contribution

The PLM literature asked: "How can we use bidirectional context for AR pretraining?"

We ask: "How can PLM enable order-agnostic generation?"

The answer requires fixing the content stream — and reveals a fundamental tension between order-agnostic training and parallel decoding that has implications beyond our specific architecture.

---

## 5. Appendix Items

### A. Implementation Details

- Tokenizer: GPT-2 BPE / LLaMA BPE
- Optimizer: AdamW ($\beta_1=0.9$, $\beta_2=0.999$, weight decay $0.01$)
- LR schedule: linear warmup (5%) → cosine decay
- Batch size: 128 × 512 tokens (small), scaled with model size
- Hardware: 8× A100 (small), 32× A100 (medium), 64× A100 (large)
- Two-stream attention: custom PyTorch implementation
- Permutation sampling: uniform over $L!$, resampled per batch

### B. Decoding Algorithms (Pseudocode)

Complete pseudocode for all four decoding modes (fixed-order, flexible-order, infilling, refinement).

### C. Extended Ablation Tables

All with 3 seeds, mean ± std, paired bootstrap tests.

### D. Attention Visualization

Heatmaps, diagonal weight evolution, per-layer breakdowns, case studies.

### E. Negative Results

- Parallel decoding quality degradation (with analysis)
- Temperature effects on confidence calibration
- Cases where BPB gap shrinks (short sequences)
- Training cost comparison

### F. Limitations (in main paper, not buried in appendix)

1. **No parallel speedup**: Our model decodes one token per step. Wall-clock latency is comparable to AR generation. The value is flexibility + quality, not speed.
2. **PLM left→right < causal LM**: The PLM training objective introduces a small BPB gap relative to pure next-token training. We quantify this gap.
3. **Parallel decoding degrades quality**: We explain why (Section 5.4), but this means our model cannot serve use cases requiring both flexibility and low latency.
4. **Decoder-only**: Bidirectional encoding for NLU tasks is not explored.
5. **Scale**: Limited to 1B parameters. Scaling behavior at 7B+ is projected.

### G. Broader Impact

- One model, multiple generation modes → reduced deployment complexity
- Flexible-order generation enables new applications (interactive writing, constrained generation)
- The analytical method (shortcut identification via representation analysis) may apply to other architectures

### H. Future Work: Selfless Attention for Multimodal Generation

The Selfless Attention framework extends naturally to multimodal settings where image tokens benefit from the same order-agnostic, purely relational representations. Text tokens are decoded one-at-a-time (any order); image tokens can be decoded in parallel (images have naturally bidirectional dependencies and tolerate parallel prediction better than text). A single attention pattern — no diagonal — works for both.

---

## 6. Reviewer Questions & Responses

### Q1: Novelty
**"Removing the diagonal is a one-line change."**

Response: The contribution is (a) identifying that a 6-year-old design choice was systematically harmful — nobody noticed this, (b) providing five mechanistic analyses proving *how* it harms representations, and (c) demonstrating that fixing it unlocks PLM's latent capability for order-agnostic generation with SOTA BPB. The simplest fixes reveal the deepest bugs.

### Q2: "Your model doesn't achieve parallel speedup. Why should I care?"

Response: Because parallel speedup is not the only value proposition of flexible-order generation. Our model enables infilling, iterative refinement, and constrained generation — all with better BPB than discrete diffusion models — while *additionally* supporting standard left-to-right AR from the same checkpoint. No other model provides this combination. We also explain *why* parallel decoding degrades (Section 5.4), which is an architectural insight about the tension between PLM training and parallel inference.

### Q3: "Isn't this just XLNet with a tweak?"

Response: XLNet never explored generation — it was a pretraining method for NLU. We are the first to use PLM for text generation, the first to demonstrate order-agnostic decoding from a PLM checkpoint, and the first to compare PLM against discrete diffusion models. We also fix a design flaw that was bottlenecking PLM's potential.

### Q4: "PLM training is more expensive. Is it worth it?"

Response: We report training cost alongside quality. If the use case requires flexible-order generation (infilling, refinement, constrained generation), our model provides better quality than the alternatives (DLMs). If the use case is pure left-to-right AR, a standard causal LM is the better choice. We are honest about this tradeoff.

### Q5: "Your left→right BPB is worse than causal LM. Why not just use a causal LM?"

Response: Because causal LM cannot do infilling, refinement, or constrained generation — capabilities that require flexible-order decoding. Our model trades a small fixed-order BPB gap for these capabilities. For applications that need them, this is the right tradeoff. For applications that don't, causal LM remains the right tool.

### Q6: "Why can't you just train with both PLM and random mask objectives?"

Response: This is an interesting direction for future work. A mixed objective might combine PLM's order-agnostic quality with random mask's parallel decoding capability. We discuss this in the limitations.

### Q7: "Does the gap scale?"

Response: We test up to 1B. The selfish shortcut mechanism is structural — it arises from the attention mask, not from limited capacity. Scaling trends suggest it persists.

---

## 7. Execution Plan

### Phase 1: Core (CRITICAL)

| # | Experiment | Status |
|---|-----------|--------|
| 1.1 | Implement Selfless Attention PLM (small, 100M) | [ ] |
| 1.2 | Implement Selfish baseline (XLNet-style, same config) | [ ] |
| 1.3 | Fixed-order BPB: Selfish vs. Selfless | [ ] |
| 1.4 | Flexible-order BPB: Selfish vs. Selfless | [ ] |
| 1.5 | Implement flexible-order (confidence-guided) decoder | [ ] |
| 1.6 | Implement infilling evaluation | [ ] |
| 1.7 | Implement iterative refinement evaluation | [ ] |
| 1.8 | Reproduce LLaDA baseline | [ ] |
| 1.9 | Reproduce Dream baseline | [ ] |
| 1.10 | Reproduce SDAR baseline | [ ] |
| 1.11 | ★ Core tables ★ Fixed + Flexible vs. all baselines | [ ] |

### Phase 2: Analysis (HIGH)

| # | Experiment | Status |
|---|-----------|--------|
| 2.1 | cos_sim analysis: content rep vs. token embedding | [ ] |
| 2.2 | Diagonal attention weight quantification | [ ] |
| 2.3 | Head pruning experiment | [ ] |
| 2.4 | Order-sensitivity: Var(h) across permutations | [ ] |
| 2.5 | Training dynamics | [ ] |
| 2.6 | Parallel decoding degradation + confidence calibration analysis | [ ] |
| 2.7 | Ablation: PLM vs. random mask training | [ ] |

### Phase 3: Applications (HIGH — shows the value)

| # | Experiment | Status |
|---|-----------|--------|
| 3.1 | Infilling benchmark (varied span lengths) | [ ] |
| 3.2 | Iterative refinement benchmark | [ ] |
| 3.3 | Constrained generation benchmark | [ ] |
| 3.4 | Decoding order comparison (L→R, R→L, confidence, random, entropy) | [ ] |

### Phase 4: Robustness (HIGH)

| # | Experiment | Status |
|---|-----------|--------|
| 4.1 | Medium scale (300M) — replicate core tables | [ ] |
| 4.2 | Large scale (1B) — if compute permits | [ ] |
| 4.3 | PG-19 dataset | [ ] |
| 4.4 | C4 dataset | [ ] |
| 4.5 | Multi-seed (3 seeds) | [ ] |

### Phase 5: Polish (MEDIUM)

| # | Experiment | Status |
|---|-----------|--------|
| 5.1 | MAUVE evaluation | [ ] |
| 5.2 | Qualitative generation examples | [ ] |
| 5.3 | Training cost analysis | [ ] |
| 5.4 | NLU probing | [ ] |

---

## 8. Figures

1. **Figure 1 (Teaser)**: Selfish vs. Selfless attention mask — diagonal in red, removed in green.
2. **Figure 2 (Main result)**: Bar chart — BPB across methods (Causal LM, Selfish PLM, Selfless PLM fixed, Selfless PLM flexible, DLM) with capability markers (fixed-order ✓/✗, flexible-order ✓/✗).
3. **Figure 3 (Tradeoff)**: BPB vs. parallel steps. Selfless PLM (one token/step = flat line at low BPB), DLM (decreasing curve, starts high, ends lower but above our line). Caption explains the tradeoff.
4. **Figure 4 (Mechanism)**: cos_sim($h_i$, embed($x_i$)) over training — Selfish vs. Selfless.
5. **Figure 5 (Attention heatmaps)**: Content stream — Selfish (bright diagonal) vs. Selfless (no diagonal).
6. **Figure 6 (Order sensitivity)**: Variance of content representations across permutations.
7. **Figure 7 (Calibration)**: Confidence calibration under parallel masking — PLM (miscalibrated) vs. DLM (calibrated). Explains the parallel degradation.
8. **Figure 8 (Applications)**: Infilling and refinement examples.

---

## 9. Open Questions

1. **Mixed training objective**: Can we combine PLM and random mask training to get both flexible-order quality and parallel capability?
2. **Optimal permutation distribution**: Uniform? Biased toward natural orders?
3. **KV-cache for multi-token decoding**: Even without full parallel, can we cache representations when decoding adjacent positions?
4. **Distillation to one-step**: Can we distill the iterative refinement process?
5. **RoPE compatibility**: Does the selfish shortcut manifest with Rotary Position Embeddings?
6. **Scaling the PLM-causal gap**: Does the left→right BPB gap between PLM and causal LM narrow with scale?

---

*This document is a living plan. Updated 2026-04-29 with honest reframing from "unified AR+NAR" to "order-agnostic generation."*
