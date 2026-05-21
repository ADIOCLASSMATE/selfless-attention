# TODO — Selfless Attention paper

**Last updated**: 2026-05-18
**Target submission**: ACL 2026 / EMNLP 2026
**Current tier (per THRESHOLDS.md)**: Tier 1 secure. Tier 2 reachable with §A + §B. Tier 3 requires §D.

---

## Status Legend
- [x] done
- [/] in progress
- [ ] todo
- [~] blocked / waiting
- [?] decision needed

---

## §A. Tier-2 lift: mechanism evidence chain (CRITICAL, ~1 week, no retraining)

These are the analyses that justify the **new** C1 claim ("expressiveness vs ordering-robustness tradeoff"). Without them, the new framing is hand-waving. With them, the paper is Tier 2.

- [ ] **A1. Var(h_i) across permutations** (highest priority)
  - For each of {Selfless 0.6B, XLNet 0.6B} random-trained:
    - Take 100 held-out sequences from WikiText valid set
    - For each sequence, sample 16 different permutations
    - Forward each through the model, collect hidden states at every layer
    - Compute `Var_i[h^(l)_i]` across the 16 permutations (variance over the permutation dimension)
    - Average over positions, then over sequences
  - **Hypothesis**: `Var_Selfless` > `Var_XLNet` at every layer; gap grows with depth
  - **Plot**: line plot, `Var(h)` vs layer index, two curves
  - Compute time: <2 hours on 1 GPU
  - Owner: ___

- [ ] **A2. `cos_sim(h_i, embed(x_i))` analysis**
  - Same 100 sequences, same models
  - For each layer, compute `cos_sim(h^(l)_i, embed(x_i))` averaged over positions
  - **Hypothesis**: XLNet's `cos_sim` >> Selfless's `cos_sim` at every layer; XLNet's stays high through depth (the "anchor"), Selfless's stays near zero
  - **Plot**: line plot, cos_sim vs layer index, two curves
  - Compute time: <1 hour
  - Owner: ___

- [ ] **A3. Diagonal attention weight quantification**
  - For XLNet only: collect attention weights from content stream
  - For each layer and each head, compute fraction of attention mass on diagonal position (`A[i,i]`)
  - **Hypothesis**: significantly above uniform baseline (`1/L`)
  - **Plot**: heatmap (layers × heads) of diagonal weight
  - Compute time: <1 hour
  - Owner: ___

- [ ] **A4. Eval-mode gap as a function of training step**
  - Take 5 intermediate checkpoints (e.g., steps 5k, 15k, 25k, 35k, 50k) for both Selfless and XLNet 0.6B
  - Run AR-mode and random-mode WT-BPB eval on each
  - **Hypothesis**: Selfless's AR-vs-random gap emerges early and persists; XLNet's stays near zero throughout
  - **Plot**: gap vs training step, two curves
  - Compute time: ~10 hours (10 eval runs × 1 hour each)
  - Owner: ___

---

## §B. Tier-2 lift: applications (CRITICAL, ~1-2 weeks)

C3 ("the model is useful for multiple modes") needs concrete applications, not just BPB on the same dataset.

- [ ] **B1. Infilling benchmark**
  - Held-out span prediction: take WikiText test, mask contiguous spans of length {1, 3, 5, 10} tokens
  - Compute BPB on masked span for: Selfless (random-mode), Selfless (AR-mode), LLaDA, Dream, SDAR
  - Pin random mask seed for fairness
  - **Hypothesis**: Selfless competitive on short spans, may degrade on longer spans
  - Compute time: ~4 hours (4 span lengths × 5 models × ~10 min)
  - Owner: ___

- [ ] **B2. Iterative refinement**
  - Generate draft L→R with Selfless (greedy temperature=0)
  - Mask 20% lowest-confidence positions, re-decode
  - Repeat 1, 2, 3, 5 iterations
  - Measure: BPB improvement per round vs the original draft BPB
  - **Hypothesis**: BPB monotonically decreases with refinement rounds; saturation point gives an interesting metric
  - Compute time: ~4 hours (rolling over a held-out set, multiple iterations)
  - Owner: ___

- [ ] **B3. Decoding order comparison**
  - Selfless 0.6B random-trained checkpoint, fixed test set
  - Decode under each strategy: L→R, R→L, confidence-greedy, confidence-sampled (top-k), random, entropy-min
  - Compute BPB / perplexity for each
  - **Hypothesis**: Confidence-guided > L→R > R→L > random; L→R ≈ entropy-min
  - Compute time: ~3 hours
  - Owner: ___

---

## §C. Tier-1 → Tier-2 lift: robustness (REQUIRED FOR REJECTION-PROOFING)

- [ ] **C1. Multi-seed for 0.6B**
  - 2 additional seeds for each of {Selfless rand-train, XLNet rand-train} at 0.6B
  - Same training data, same schedule, just different seed
  - Evaluate each at WT-BPB (both AR and random modes), LAMBADA-PPL, ARC-E, PIQA
  - Compute paired bootstrap CI for the AR-vs-random gap and the Selfless-vs-XLNet random-mode gap
  - **This determines whether the 0.6B random-mode regression (Selfless 0.971 vs XLNet 0.963) is real or noise**
  - Compute time: ~14 days at 0.6B × 2 seeds × 2 models on 8×H200 (the big-ticket compute item)
  - Owner: ___

- [ ] **C2. Second dataset for BPB**
  - Run wikitext-103 eval (already done) + PG-19 + C4 + WikiText-2 on all 0.6B models in both AR and random modes
  - Most baselines already have these datasets in `eval/PPL/`; just rerun and report
  - **Goal**: show all reported patterns are dataset-agnostic
  - Compute time: ~6 hours total
  - Owner: ___

- [ ] **C3. mc_num scaling / stderr**
  - For 0.6B random-mode eval on WT:
    - Run with mc_num ∈ {1, 4, 16, 32, 64, 128}
    - Plot BPB vs mc_num
    - At mc_num=32, compute stderr across the 32 permutation samples
  - **Goal**: prove the reported BPB is converged + give error bars
  - Compute time: ~3 hours
  - Owner: ___

- [ ] **C4. MMLU (zero-shot) eval**
  - Run MMLU on all 0.6B models (Causal LM, Selfless rand/AR, XLNet rand/AR, SDAR, LLaDA, Dream)
  - This is the single most widely reported pretraining benchmark; 57 subjects across STEM/humanities/social-science
  - **Goal**: show Selfless leads non-AR models on the most standard downstream benchmark; reviewers will ask for this
  - Compute time: ~8 hours (8 models × ~1 hour)
  - Owner: ___

- [ ] **C5. ARC-Challenge eval**
  - Run ARC-Challenge on all 0.6B models
  - ARC-Easy has a ceiling effect at 0.6B (Causal LM already at 0.788); ARC-C provides discrimination
  - **Goal**: verify Selfless's accuracy advantage holds on a harder science QA benchmark
  - Compute time: ~2 hours
  - Owner: ___

- [ ] **C6. Drop non-discriminating benchmarks**
  - Remove GPQA Diamond, TruthfulQA, OpenBookQA from the 0.6B/250M eval suite
  - These are too hard for small models (Causal LM scores: GPQA 0.283, TQA-MC1 0.215) or redundant (OpenBookQA ≈ ARC-E + SciQ)
  - Replace with MMLU + ARC-Challenge; net runtime change ≈ neutral
  - Owner: ___

---

## §D. Tier-2 → Tier-3 reach (OPTIONAL, ~3-4 weeks, BIG RISK/REWARD)

The single most powerful lift to Tier 3 is "parallel decoding works". The cheapest path:

- [ ] **D1. Mixed-objective finetune**
  - Start from `output/selfless-0.6B-50BT/hf_model-final`
  - Continue training for 5B more tokens (≈5000 steps) with:
    - 50% batches: standard PLM (random permutation)
    - 50% batches: random mask (LLaDA-style, sample t ∈ [0,1], mask each position with prob t)
  - Evaluate: AR-mode BPB, random-mode BPB, **parallel-mode BPB (decode with K tokens per step)** for K ∈ {1, 4, 8, 16}
  - **Hypothesis**: parallel BPB improves substantially; flexible-order BPB may degrade somewhat; AR-mode BPB approximately unchanged
  - **Tier-3 condition**: Pareto improvement (parallel works AND flexible-order BPB doesn't degrade beyond baseline)
  - Compute time: ~3 days finetune + 1 day eval
  - Owner: ___

- [ ] **D2. Calibration analysis (the C1 deepening)**
  - Take a random-trained Selfless 0.6B checkpoint
  - Construct contexts with varying masking ratios (5%, 25%, 50%, 75%, 95%)
  - For each ratio, compute the model's confidence (max-prob) on masked positions and compare to actual accuracy
  - **Hypothesis**: PLM-trained model is well-calibrated at low mask ratios (training distribution), miscalibrated at high mask ratios (parallel-decoding regime)
  - Plot reliability diagram per mask ratio
  - Compute time: ~4 hours
  - Owner: ___

- [ ] **D3. Head pruning experiment** (high-risk, high-reward)
  - For XLNet 0.6B random-trained: identify heads with highest diagonal attention weight (from A3)
  - Prune top-k% of such heads (set their output to zero); re-evaluate WT-BPB
  - **Tier-3 condition**: pruning recovers most of the Selfless-XLNet AR-mode gap, suggesting a small number of heads concentrate the shortcut
  - Compute time: ~6 hours (search over k)
  - Owner: ___

---

## §E. Code cleanups (BEFORE SUBMISSION)

- [ ] **E1. Refactor `model.train()` in eval workers**
  - Replace `self.model.train()` with an explicit flag `use_xt_stream=True`
  - Add `model.eval()` in inference path (proper convention)
  - Verify BPB on a quick sanity test (should be identical within fp16 noise)
  - File: `eval/{selfless,xlnet}/eval_worker_*.py` and `models/modeling_model/modeling_{selfless,xlnet}.py`

- [ ] **E2. Implement `generate_until` for selfless/xlnet eval workers**
  - For LAMBADA accuracy (currently 0 due to `NotImplementedError`)
  - Wire up the existing `generate()` method
  - File: `eval/selfless/eval_worker_selfless.py`, `eval/xlnet/eval_worker_xlnet.py`

- [ ] **E3. Fix first-window position-0 bias**
  - In `loglikelihood_rolling`, when `start_loc==0`, set `labels_window[:, 0] = -100`
  - This skips the "no-context" prediction at the first position, matching the AR baseline's behavior
  - File: same as E1

- [ ] **E4. Standardize output_path naming**
  - Either rename `output_eval/selfless-${SIZE}-50BT-ar+ar` → `selfless-${SIZE}-50BT-ar`
  - Or rename the script's `--output_path` to use `-ar+ar`
  - Pick one; current state has both formats in different runs (confusing)

- [ ] **E5. Verify resume_from_checkpoint did not break the schedule**
  - Plot wandb LR curve for selfless-0.6B vs xlnet-0.6B
  - Confirm both end at the same LR at step 50000 (i.e., same point on the cosine schedule)
  - If divergent: rerun the affected model (probably selfless-0.6B) from scratch
  - File: `configs/selfless/pretraining_0.6B.yaml` (the `resume_from_checkpoint` field)

- [ ] **E6. Report mc_num stderr**
  - Modify eval workers to also track per-permutation log-likelihoods
  - Report mean ± stderr in the final BPB output
  - File: `eval/{selfless,xlnet}/eval_worker_*.py`

---

## §F. Paper-writing tasks (RUN IN PARALLEL WITH §A–C)

- [ ] **F1. Rewrite Section 1 (Introduction)** per `PLAN.md §4.1`
- [ ] **F2. Rewrite Section 4 (Method)** dropping "selfish shortcut" framing
- [ ] **F3. Build Figure 1 (mask diagram)** — matplotlib, the "teaser"
- [ ] **F4. Build Figure 3 (AR-vs-random gap vs scale)** — only needs the 4 numbers
- [ ] **F5. Build Figure 6 (LAMBADA log-scale)** — visual punch
- [ ] **F6. Write §Limitations** — items from `PLAN.md §4.6` and `CODE_AUDIT.md §8`
- [ ] **F7. Write §Related Work** focusing on (a) why this differs from XLNet's original use, (b) why this complements rather than replaces DLMs
- [ ] **F8. Write §Discussion** unifying the new C1 framing with the empirical results

---

## §G. Stretch goals (POST-SUBMISSION OR REBUTTAL)

- [ ] **G1. Scale to 1B**: confirm gap-widens-with-scale pattern (`THRESHOLDS.md D1` Tier-3 condition)
- [ ] **G2. Vision Transformer test**: does the diagonal-vs-no-diagonal pattern hold in two-stream ViT? `THRESHOLDS.md E1`
- [ ] **G3. NLU probing**: lift Selfless representations and probe on GLUE — does the "higher expressiveness" claim transfer?
- [ ] **G4. Distillation**: distill iterative refinement to one pass; explore the speed/quality frontier

---

## Decision Points

- [?] **Should we add a unified eval (same estimator for all methods)?** — Improves cross-method fairness; ~1 week to implement; may shift baseline numbers
- [?] **Should we run the 1B experiments?** — Strongest scaling story; 2-3 weeks of compute; depends on availability
- [?] **Submission target — ACL or EMNLP?** — ACL has slightly earlier deadline; EMNLP allows more polish

---

## Critical Path to Submission (suggested 4-week plan)

**Week 1**: §A1-A4 + §E1, E2, E3, E5 (mechanism evidence + code cleanup, parallelizable)
**Week 2**: §C1 (multi-seed 0.6B, big compute) + §B1, B2, B3 (applications, smaller compute)
**Week 3**: §C2 (second datasets) + §C3 (mc_num scaling) + §F1-F8 (writing)
**Week 4**: §D1 if compute available (push to Tier 2/3) + final polish

---

## Notes
- The `0.6B random-mode regression` (Selfless 0.971 vs XLNet 0.963) is the **most paper-vulnerable result**. §C1 multi-seed is the highest-priority robustness task.
- The current mechanism story has **strong intuitive plausibility but no measurement** — §A1, A2 are non-negotiable.
- §E5 (LR schedule verification) is cheap and should be confirmed ASAP since it's a clean-or-dirty distinction with no in-between.
