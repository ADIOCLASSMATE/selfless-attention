# Contribution Escalation: What Each Experimental Outcome Unlocks

This document maps experimental outcomes to contribution levels. Not all results are equal — some turn a solid paper into a standout paper.

---

## Tier Structure

| Tier | Venue Target | What It Means |
|------|-------------|---------------|
| **Tier 0 (Minimum Viable)** | Workshop / short paper | The core finding holds, scope is narrow |
| **Tier 1 (Solid Paper)** | ACL / EMNLP main | All three contributions are substantiated |
| **Tier 2 (Strong Paper)** | ACL / EMNLP oral contender | One claim transcends the immediate method |
| **Tier 3 (Standout)** | Best paper discussion | Multiple transcendent claims, opens a new sub-area |

---

## Experiment-to-Tier Map

### Axis A: The Selfish Shortcut (C1)

#### A1. Selfless vs. Selfish BPB gap (fixed-order left→right)

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| Gap exists but is small (≤0.02 BPB) | 0 | C1 is technically correct but practically marginal. Paper survives on C2+C3. |
| Gap is moderate (0.03–0.08 BPB) | 1 | C1 is solid. The shortcut is real and the fix matters. Baseline for a good paper. |
| Gap is large (≥0.10 BPB) | 2 | C1 becomes a headline finding: "a 6-year-old design bug cost XLNet X BPB." Reviewers will cite this. |

*How to measure BPB gap credibly*: report across 3 datasets, 3 seeds, with paired bootstrap confidence intervals.

#### A2. Selfless vs. Selfish BPB gap (flexible-order)

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| Gap is small or nonexistent | 0–1 | Weakens the "selfish shortcut specifically hurts flexible-order" argument. |
| Gap is moderate, larger than fixed-order gap | 1–2 | Supports the order-contamination theory. The shortcut is worse for non-standard orders. |
| Gap is large and **grows with permutation length** | 2–3 | Strong mechanistic evidence. The shortcut compounds with sequence length. This is a general principle, not a model-specific observation. |

#### A3. cos_sim(h_i, embed(x_i)) analysis

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| cos_sim is higher for Selfish than Selfless, but both decay over time | 1 | Confirms the shortcut exists but isn't permanent. |
| Selfish cos_sim stays high (≥0.4) even at convergence; Selfless is near zero | 2 | Shows the shortcut is a stable attractor — the model never outgrows it. Strong mechanistic evidence. |
| Selfish cos_sim correlates with BPB gap across layers/heads | 3 | You can point to specific attention heads and say "these heads are the problem." Opens the door to head-level interventions. |

#### A4. Head pruning experiment

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| Pruning self-attention-heavy heads causes less damage than random | 1 | Confirms these heads are serving the shortcut, not useful computation. |
| Pruning them *improves* BPB | 2 | These heads are actively harmful. Selfless Attention is not just removing a shortcut — it's removing a source of noise. |
| You can identify a small set of heads (≤5% of total) whose removal recovers most of the Selfish→Selfless gap | 3 | Practical takeaway: you don't even need to retrain. You can fix existing XLNet checkpoints by pruning a few heads. This is a *directly actionable* finding. |

#### A5. Order-sensitivity analysis

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| Var(h_i) across permutations: Selfish > Selfless | 1 | Confirms the order-contamination theory. |
| Var(h_i) correlates with BPB degradation on that specific order | 2 | Causal link between representation variance and generation quality. Strong mechanistic story. |
| Selfless representations are nearly permutation-invariant (Var ≈ 0) | 2–3 | Selfless Attention achieves what PLM was supposed to achieve all along: truly order-agnostic representations. This is a conceptual contribution beyond the method. |

---

### Axis B: Flexible-Order Generation Quality (C3)

#### B1. BPB comparison vs. Discrete Diffusion Models

This is the paper's anchor experiment. Without this, C3 doesn't exist.

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| Ours < DLM on some datasets but not all | 0 | Too weak. Not a reliable claim. |
| Ours < DLM on all datasets, gap is small (≤0.05 BPB) | 1 | C3 is valid but not exciting. "Comparable quality + more capabilities" is a reasonable but soft claim. |
| Ours < DLM on all datasets, gap is moderate (0.05–0.15 BPB) | 2 | C3 is strong. "Better quality + more capabilities" is a clear win. |
| Ours < DLM on all datasets, gap is large (≥0.15 BPB) | 2–3 | This would force the discrete diffusion community to pay attention. PLM is not just an alternative — it's better. |

*Critical detail*: You must use exactly the same decoding budget (same number of forward passes, not same number of "steps"). If DLM uses 128 steps of 1 forward pass each, and you use L steps (sequence length) of 1 forward pass each, state this clearly. If L < 128 for typical sequences, you're also faster — but don't oversell this.

#### B2. Gap between PLM left→right and Causal LM

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| Gap is large (≥0.10 BPB) | 0–1 | A liability. Reviewers will say "your fixed-order mode is clearly worse, why would anyone use this?" |
| Gap is moderate (0.03–0.08 BPB) | 1 | Acceptable. You're paying a small quality tax for flexible-order capability. Honest tradeoff. |
| Gap is small (≤0.02 BPB) | 2 | Excellent. "Essentially matches causal LM on its home turf, while also doing things causal LM cannot." |
| Gap is **zero or negative** (PLM matches/beats causal LM) | 3 | This would be a major finding: PLM training with Selfless Attention is *strictly better* than next-token prediction. The entire field's default training objective is suboptimal. |

*Why B2 could hit Tier 3*: The entire LLM industry uses next-token prediction because it's the default, not because anyone proved it's optimal. If PLM matches it on left→right while enabling flexible-order generation, you've made a case that next-token prediction is a local optimum we've been stuck in. This is a "change how people think" result.

#### B3. Infilling quality

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| Ours ≥ DLM on infilling | 0–1 | Not a differentiator. |
| Ours > DLM, gap is small | 1 | Supports the flexible-order quality claim but isn't a headline. |
| Ours > DLM, gap is large, and quality degrades gracefully with span length | 2 | The model genuinely understands bidirectional context better. Practical for text editing applications. |
| Ours > DLM **and** Ours ≈ human-level on short spans (≤5 tokens) | 3 | Infilling as a solved problem at practical scales. This is product-grade. |

#### B4. Iterative refinement

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| Refinement improves BPB but plateaus after 1–2 rounds | 1 | Works, but limited. |
| Refinement converges to near fixed-order BPB within 3–5 rounds | 2 | The model can self-correct to near-optimal quality. Practical for "generate fast, refine later" workflows. |
| Refinement improves over fixed-order BPB (draft from flexible-order → refine → better than either alone) | 3 | Self-refinement as a new capability: the model produces better output by iterating than by generating in one pass. This is an emergent property. |

---

### Axis C: Parallel Decoding Analysis (The Insight Contribution)

#### C1. Confidence calibration under parallel masking

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| PLM is miscalibrated, DLM is calibrated — you show this | 1 | Confirms the train-test mismatch theory. |
| You quantify the calibration error and show it correlates with parallel BPB degradation | 2 | The mechanism is not just plausible — it's measured. |
| You propose a simple correction (e.g., temperature scaling per mask ratio) that partially closes the gap | 2–3 | You not only diagnose the problem, you offer a partial fix. Even if it doesn't fully close the gap, it shows you understand the mechanism. |
| You show that the calibration error is predictable from the mask ratio alone | 3 | This is a general insight about PLM training: the degradation under parallel masking follows a predictable law. This could guide future work on hybrid training objectives. |

#### C2. Hybrid training (PLM + random mask)

This is optional — it's future work in the current plan. But if you have the compute:

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| Mixed objective improves parallel BPB but degrades flexible BPB | 2 | Confirms the tradeoff is fundamental. A single Pareto frontier governs both objectives. |
| Mixed objective improves parallel BPB with no degradation to flexible BPB | 3 | You've solved the tradeoff. The paper becomes "Selfless Attention + Mixed Training: The Best of Both Worlds." |
| There exists a mixing ratio that Pareto-dominates both pure PLM and pure random mask | 3 | Practical recipe: "use ratio X for Y% of training." Adoptable immediately. |

---

### Axis D: Scaling

#### D1. BPB gap vs. model size

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| Gap persists at ~300M | 1 | Baseline. |
| Gap persists at ~1B | 2 | The finding survives scale. Stronger claim. |
| Gap **widens** with scale | 3 | The selfish shortcut becomes *more* damaging at larger scales. This is a scaling law for a design flaw — extremely publishable. Suggests that large XLNet-style models would be leaving even more performance on the table. |
| Gap narrows and disappears at ~1B | 0–1 | Weakens C1. Large models learn to suppress the shortcut themselves. Selfless Attention is only useful at small scale. |

#### D2. PLM-Causal gap vs. model size

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| Gap persists | 1 | The PLM tax is real and doesn't vanish with scale. |
| Gap narrows | 2 | PLM asymptotically approaches causal LM. At large scale, you get flexible-order capability for free. |
| Gap disappears | 2–3 | PLM converges to causal LM at large scale. The default next-token objective is unnecessary at scale — PLM is strictly superior. |

---

### Axis F: Parallel Decoding (The Tier Elevator)

This is the single highest-leverage axis. Currently, the paper's largest limitation is "no parallel speedup." Overcoming it — even partially — changes the paper's category from "tradeoff paper" to "strict improvement paper."

#### F1. Parallel decoding quality vs. DLM

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| Parallel still degrades, no fix found | 1 | Baseline. Honest limitation, explained by calibration analysis. |
| Parallel degrades but you propose a correction that partially closes the gap | 2 | You not only diagnose the problem but offer a fix. Even a partial fix is a contribution. |
| Parallel quality **matches** DLM (same BPB at same step count) | 2–3 | You match DLM on its home turf while beating it on flexible BPB and adding fixed-order AR. DLM has no remaining advantage. |
| Parallel quality **beats** DLM (lower BPB at same step count) | 3 | DLM is strictly dominated. Your model is better at everything. |
| Parallel quality beats DLM **and** PLM left→right ≈ Causal LM | 3 | No model in existence can compete with you on any dimension. This is a new SOTA across the board. |

#### F2. How to potentially unlock parallel decoding

These are not in the current plan but are the most obvious paths to Tier 2–3:

| Approach | Cost | Risk | Potential Gain |
|----------|------|------|---------------|
| **Confidence calibration**: Apply temperature scaling per mask ratio (learned on validation set) | Low (post-hoc) | May not be enough — calibration error may be too large | Tier 1→2 if it works |
| **Mixed training**: PLM + random mask objective (e.g., 50/50 or curriculum) | Medium (retrain) | May degrade flexible BPB (tradeoff is fundamental) | Tier 1→2 if Pareto improvement; Tier 1→3 if no degradation |
| **Two-phase training**: PLM pretrain → random mask finetune | Medium | Flexible BPB may degrade in finetune phase | Tier 1→2 if finetune is short |
| **Consistency distillation**: Train student to replicate iterative refinement in fewer steps | High (complex pipeline) | Student may not match teacher quality | Tier 1→3 if one-step quality matches iterative |
| **Architecture change**: Add a small adapter trained on random mask while backbone is frozen | Low-Medium | Adapter may not have enough capacity | Tier 1→2 if adapter works |

#### F3. Why this axis is so powerful

The current competitive landscape:

```
                Fixed-Order BPB    Flexible BPB    Parallel Speed
Causal LM         ★★★ (best)         ✗                ✗
DLM (LLaDA)       ✗                  ★★               ★★★
Ours (current)    ★★☆ (good)         ★★★ (best)       ✗
Ours (parallel)   ★★☆ (good)         ★★★ (best)       ★★★ (or ★★)
```

Without parallel: you dominate one column (flexible BPB) and are competitive in another (fixed BPB). This is a niche.
With parallel: you lead or tie for the lead in ALL columns. This is not a niche — it's a new SOTA.

**The difference between Tier 1 and Tier 3 on this paper is, more than anything else, whether parallel decoding works.**


### Axis G: Cross-Domain Generalization (Bonus)

These are not in the current plan but could be high-impact additions.

#### E1. Selfish shortcut in vision transformers

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| You show the same cos_sim / diagonal-weight pattern in a ViT | 2 | The finding generalizes beyond language. Selfless Attention is a cross-domain principle. |
| You apply Selfless Attention to a ViT and get improvement | 3 | The paper transcends NLP. "We found a universal design flaw in two-stream attention architectures." |

#### E2. Selfish shortcut in encoder-decoder models

| Outcome | Tier | What It Unlocks |
|---------|------|-----------------|
| You show the same pattern in T5/BART's encoder self-attention | 2–3 | The encoder's job is also to provide context to the decoder. Self-attention in the encoder may be similarly wasteful. This generalizes the finding to the most common architecture in NLP. |

---

## Summary: What to Fight For

### Non-negotiable (paper dies without these)

1. Selfless < Selfish BPB on flexible-order, statistically significant
2. Ours (flexible) < at least one DLM baseline on BPB
3. cos_sim analysis shows measurable degradation in Selfish model
4. At least one dataset beyond WikiText-103

### High-impact (elevate from solid to strong)

1. PLM-Causal BPB gap ≤ 0.05 → "essentially matches causal LM"
2. Selfish-Selfless gap widens with permutation length
3. Head pruning: removing selfish heads *improves* BPB
4. Confidence calibration analysis quantifies the parallel decoding degradation
5. Scaling: gap persists or widens at 300M+
6. Parallel decoding quality matches DLM via calibration correction or mixed training

### Game-changers (best paper territory)

1. **Parallel decoding beats DLM on both quality and speed** (Axis F.1, Tier 3) — this is the single highest-impact result possible
2. PLM matches or beats Causal LM on left→right BPB — combined with #1, strictly dominates all models on all dimensions
3. Selfish-Selfless gap widens with model scale
4. You identify specific heads (≤5%) whose pruning recovers the gap
5. Refinement *exceeds* one-pass fixed-order BPB
6. Finding generalizes to vision or encoder-decoder models
7. Mixed training (PLM + random mask) achieves Pareto improvement over both pure objectives

---

## Decision Tree: What to Do Based on Results

```
Selfless < Selfish on flexible BPB?
  ├─ NO  → Pivot. The core premise is wrong.
  └─ YES → Continue.
            │
            Ours (flexible) < DLM?
              ├─ NO  → Paper becomes C1+C2 only (analysis paper). 
              │        Need very strong mechanistic evidence.
              └─ YES → Full paper viable.
                        │
                        Can we do parallel decoding?
                          ├─ NO (or poor quality) → Tier 1 baseline.
                          │     Paper is "order-agnostic with SOTA BPB, no speedup."
                          │     └─ PLM-Causal gap?
                          │           ├─ Large → Liability.
                          │           ├─ Moderate → Acceptable.
                          │           └─ Small → Strong. Tier 1→2 possible.
                          │
                          ├─ Partial (matches DLM quality) → Tier 2.
                          │     Paper is "matches DLM on speed, beats on BPB, 
                          │     plus fixed-order AR."
                          │
                          └─ YES (beats DLM on both quality and speed) → Tier 3.
                                Paper is "strictly dominates DLM on all dimensions."
                                └─ PLM ≈ Causal LM? → Tier 3 lock.
                        
                        Scaling: gap vs. model size?
                          ├─ Narrows → Mention, not headline.
                          ├─ Stable → Supports robustness.
                          └─ Widens → Headline. Extra tier boost when combined
                                       with parallel decoding success.
```

---

## Writing Strategy Based on Tier

### Tier 1 (Solid Paper) — Narrative

"We found a bug, fixed it, and showed PLM can do flexible-order generation with SOTA quality. Yes, left→right BPB is slightly worse than causal LM, and no, we can't do parallel decoding. But for flexible-order tasks, we're the best option, and we're the only model that does both fixed and flexible order."

### Tier 2 (Strong Paper) — Narrative

"We found a bug that cost XLNet significant performance for 6 years. Our fix not only recovers this loss but unlocks PLM's latent capability: a single model that generates in any order, with BPB matching causal LM on left→right and beating discrete diffusion on flexible order. We show *why* parallel decoding is incompatible with PLM training — a fundamental insight about the cost of order-agnosticism."

### Tier 3 (Standout) — Narrative

"The default training objective of every LLM — next-token prediction — may be suboptimal. We show that permutation language modeling with Selfless Attention matches or exceeds next-token prediction on standard AR generation, while additionally enabling flexible-order generation, infilling, and refinement that no causal LM can do. We further identify a universal design flaw in two-stream attention architectures and provide a one-line fix. The finding generalizes across modalities and scales. The field has been training language models wrong — not catastrophically wrong, but wrong enough to matter."

---

*This document should be revisited after each experimental phase. Mark outcomes as CONFIRMED / REJECTED / UNCERTAIN and update the projected tier.*
