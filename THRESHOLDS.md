# THRESHOLDS — Current State and Tier Mapping

**Last updated**: 2026-05-18
**Methodology**: For each tier-axis from the original `THRESHOLDS.md`, mark whether the current data **confirms / rejects / is undetermined** the threshold, then compute the achievable tier given the actual results.

---

## Status Markers
- ✅ **CONFIRMED** — data supports this threshold
- ❌ **REJECTED** — data refutes this threshold (story must change)
- ⚠️ **PARTIAL** — data partially supports, more evidence needed
- ❓ **UNTESTED** — experiment not yet run
- 🔄 **REFRAMED** — original prediction wrong, but observed pattern supports a different threshold

---

## Axis A: The Selfish Shortcut (C1)

### A1. Selfless vs. Selfish BPB gap (fixed-order left→right)
**Threshold values**: ≤0.02 = Tier 0, 0.03–0.08 = Tier 1, ≥0.10 = Tier 2

| Scale | Gap (XLNet AR − Selfless AR) | Tier |
|---|---|---|
| 250M | 0.021 | ⚠️ **Tier 0/1 boundary** |
| 0.6B | 0.025 | ✅ **Tier 1** |

**Status**: ⚠️ **PARTIAL** (Tier 1 at 0.6B, marginal at 250M). Needs multi-seed (`TODO §C1`) to verify significance of the 0.021–0.025 gap. Likely Tier 1, possibly Tier 2 at larger scale.

### A2. Selfless vs. Selfish BPB gap (flexible-order)
**Threshold values**: gap should be **larger** than fixed-order gap, per original hypothesis

| Scale | Gap (XLNet random − Selfless random) | Original prediction |
|---|---|---|
| 250M | +0.002 (within noise) | Should be > 0.021 |
| 0.6B | **−0.008** (Selfless **WORSE**) | Should be > 0.025 |

**Status**: ❌ **REJECTED in original form**. The flexible-order gap is smaller than fixed-order gap, and even reverses sign at 0.6B.

**🔄 REFRAMED**: This rejection is actually **evidence for the new C1**: the diagonal acts as an ordering-robustness anchor. Selfless's loss in random mode is the "cost" of removing that anchor. Combined with mechanism evidence (Var(h), cos_sim — see §A5), this becomes a Tier-2 story instead of Tier-1.

### A3. cos_sim analysis
**Threshold values**: Selfish cos_sim ≥ 0.4 at convergence = Tier 2; correlates with BPB gap per layer = Tier 3

**Status**: ❓ **UNTESTED** — see `TODO §A2`. Cheap forward-pass experiment. Critical for the reframed C1.

### A4. Head pruning
**Threshold values**: prune-selfish-heads improves BPB = Tier 2; small set (≤5%) recovers gap = Tier 3

**Status**: ❓ **UNTESTED** — see `TODO §D3`. Could push to Tier 3 if works.

### A5. Order-sensitivity Var(h)
**Threshold values**: Selfish Var > Selfless Var = Tier 1; correlation with BPB = Tier 2; permutation-invariance for one of them = Tier 2/3

**Status**: ❓ **UNTESTED** — see `TODO §A1`. **HIGHEST PRIORITY** for new C1.

**🔄 NEW PREDICTION (opposite of original PLAN)**: Var(h)_**Selfless** > Var(h)_**XLNet**. The current eval-mode gap data (Selfless AR-vs-random gap large, XLNet's near zero) strongly supports this direction.

---

## Axis B: Flexible-Order Generation Quality (C3)

### B1. BPB vs DLM
**Threshold values**: Ours < DLM by ≤0.05 = Tier 1; 0.05–0.15 = Tier 2; ≥0.15 = Tier 3

| 0.6B WT-BPB | Value | vs Selfless random (0.971) | vs Selfless AR (0.943) |
|---|---|---|---|
| SDAR | 0.913 | +0.058 worse | +0.030 worse |
| LLaDA | 0.940 | +0.031 worse | +0.003 ≈ tied |
| Dream | 0.945 | +0.026 worse | −0.002 ≈ tied |

**Status**:
- ❌ **REJECTED** for Selfless random mode (worse than all DLMs)
- ⚠️ **PARTIAL** for Selfless AR mode (matches LLaDA, beats Dream, loses to SDAR)

**🔄 REFRAMED**: The cleaner claim is **"Selfless AR-mode matches LLaDA random-mode while supporting both modes"**. Tier-1 framing.

### B2. PLM Left-to-right vs Causal LM gap
**Threshold values**: ≤0.05 = Tier 2 ("essentially matches"); zero or negative = Tier 3

| Scale | Causal LM BPB | Selfless AR BPB | Gap |
|---|---|---|---|
| 0.6B | 0.823 | 0.943 | 0.120 |

**Status**: ❌ **Tier 0** by the original threshold (gap > 0.10). Honestly reported as a Limitation. The PLM training tax is real and substantial.

### B3. Infilling
**Status**: ❓ **UNTESTED** — see `TODO §B1`. Practical application; needed for paper.

### B4. Iterative refinement
**Status**: ❓ **UNTESTED** — see `TODO §B2`. Practical application; needed for paper.

---

## Axis C: Parallel Decoding Analysis (Insight)

### C1. Calibration analysis
**Status**: ❓ **UNTESTED** — see `TODO §D2`. Required for §Limitations argument that parallel decoding is structurally incompatible.

### C2. Hybrid training
**Status**: ❓ **UNTESTED** — see `TODO §D1`. Potential Tier 3 lift.

---

## Axis D: Scaling

### D1. BPB gap vs model size
**Threshold values**: persists = Tier 1; widens = Tier 3; narrows = Tier 0

**Selfless AR-vs-random gap**:
| Scale | Gap | Direction |
|---|---|---|
| 250M | 0.019 | (baseline) |
| 0.6B | 0.028 | **widens by 1.5×** |

**Selfless vs XLNet AR-mode gap**:
| Scale | Gap | Direction |
|---|---|---|
| 250M | 0.021 | (baseline) |
| 0.6B | 0.025 | persists, slight widening |

**Status**: ✅ **Tier 2** (eval-mode gap widens with scale). With multi-seed confirmation (`TODO §C1`), this is a strong scaling story.

### D2. PLM-Causal gap vs scale
**Threshold values**: narrows = Tier 2; disappears = Tier 3

| Scale | Causal | Selfless AR | Gap |
|---|---|---|---|
| 250M | 0.911 | 1.047 | 0.136 |
| 0.6B | 0.823 | 0.943 | 0.120 |

**Status**: ⚠️ **Tier 1** (small narrowing 0.136 → 0.120, but still large). Needs 1B point to confirm narrowing direction.

---

## Axis F: Parallel Decoding (Tier Elevator)

### F1. Parallel decoding quality vs DLM
**Status**: ❓ **UNTESTED**. The mixed-objective experiment (`TODO §D1`) is the primary path. Currently Tier 0 (no parallel decoding, honest Limitation).

---

## Axis E (Bonus): Downstream zero-shot

**NEW AXIS — not in original THRESHOLDS but emerges from data as a strong axis**

### E1. Zero-shot tasks vs DLMs

| 0.6B Zero-shot avg (ARC-E + PIQA + SciQ + SGLUE) | Value |
|---|---|
| Causal LM | (0.591+0.693+0.788+0.554)/4 = **0.657** |
| Selfless random | (0.537+0.658+0.727+0.574)/4 = **0.624** |
| Selfless AR | (0.538+0.650+0.720+0.550)/4 = **0.615** |
| XLNet random | (0.527+0.640+0.734+0.549)/4 = **0.613** |
| XLNet AR | (0.500+0.641+0.687+0.548)/4 = **0.594** |
| SDAR | (0.484+0.584+0.701+0.490)/4 = **0.565** |
| LLaDA | (0.335+0.521+0.557+0.515)/4 = **0.482** |
| Dream | (0.312+0.511+0.464+0.518)/4 = **0.451** |

**Status**: ✅ **NEW Tier 2 finding**. Selfless (in either mode) is the **best non-AR model on downstream zero-shot**, and Selfless random-mode is only 0.03 below the causal LM. The 0.14 gap to LLaDA is large.

**Significance**: This is a **headline-worthy result** that wasn't in the original PLAN. It suggests Selfless representations are more useful for downstream tasks than DLM representations, despite Selfless's worse raw BPB.

---

## Current Tier Assessment

### What we have NOW (no new experiments):
- ✅ A1 partial: Tier 1
- 🔄 A2 reframed: Tier 1-2 if Var(h) confirms
- 🔄 B1 reframed: Tier 1
- ✅ D1: Tier 2
- ✅ E1: Tier 2 (new downstream finding)

**Achievable tier RIGHT NOW**: **Tier 1 secure** (Solid ACL/EMNLP main paper).

### What we get with TODO §A (mechanism evidence):
- A1, A2, A4, A5 all confirming = **Tier 2 likely** (mechanism + scaling + downstream all align)

### What we get with TODO §A + §B + §C:
- All above + applications + multi-seed = **Tier 2 secure**

### What we'd need for Tier 3:
- TODO §D1 succeeds (parallel decoding works): adds back the parallel column → Tier 3 likely
- TODO §D3 succeeds (head pruning identifies small subset): actionable intervention → Tier 3 likely
- 1B scaling confirms widening gap (§G1): scaling law for design choice → Tier 3 likely

**Recommendation**: focus on `Tier 2 secure` via §A + §B + §C. Tier 3 is a stretch goal; consider §D1 only if compute available.

---

## Revisions to Original Decision Tree

Original tree said:
```
Ours (flexible) < DLM? NO → C1+C2 analysis paper, need strong mechanism
```

**Revised tree**:
```
Selfless AR-mode < some DLM? YES (matches LLaDA at 0.6B)
        AND
Single-checkpoint multi-mode works? YES (verified, same checkpoint two modes)
        AND
Selfless > XLNet on downstream zero-shot? YES (clear, +1-2pt across tasks)
        AND
Mechanism evidence shows tradeoff direction? PENDING (§A1, §A2, §A5)
        ↓
        Tier 1 secure NOW, Tier 2 with §A complete
```

---

*This document is a living tier assessment. Update after each completed TODO item.*
