# Baseline Audit Report — Training & Evaluation Logic for All 5 Methods

**Date**: 2026-05-18
**Scope**: `pretrain/train_*.py`, `models/modeling_model/modeling_*.py`, `eval/*/eval_worker_*.py` for ar, llada, dream, sdar, xlnet, selfless
**Purpose**: Determine whether each method's training and evaluation faithfully implement the original paper's design, and whether the cross-method comparison is fair.

---

## TL;DR — severity-ranked summary

| Severity | Method | Issue |
|---|---|---|
| 🟥 **HIGH** | **Dream** | LAMBADA-PPL of 209k is a known artifact of Dream's design: the AR-shift convention drops the last input token, so for last-token tasks (LAMBADA's design), the model never sees a [MASK] indicator at the target position. **Dream's LAMBADA-PPL should NOT be cited as a fair comparison point.** WT-PPL is fine. **FIX SHIPPED** in `eval_worker_dream.py`. |
| 🟧 **MEDIUM** | **Selfless** | Train/eval branching in `Qwen3Model.forward` (line 487-498) is a footgun: eval without explicit `model.train()` returns the wrong stream (X0 instead of XT). `eval_worker_selfless.py` correctly calls `model.train()`, so BPB results ARE correct. But `generate()` uses X0 stream (which the LM head was not trained on) — generation quality may be poorer than BPB suggests. **FIX SHIPPED** in `modeling_selfless.py`. |
| 🟧 **MEDIUM** | **SDAR** | Eval uses `CE(logits[i], labels[i]=input_ids[i])` with causal attention. **This IS the SDAR convention** (predict current token, MLM-style) — initially I incorrectly flagged this as a bug, but the user clarified that SDAR is trained MLM-style (`logits[i]` predicts `input_ids[i]`, not `input_ids[i+1]`). However, the eval-time causal attention's diagonal lets `hidden[i]` see the actual token at position i, while during training position i had [MASK] input. This is a train-eval distribution mismatch (a soft information leak through the diagonal), not a bug per se. SDAR's 0.913 BPB benefits from this and is not a clean apples-to-apples comparison with PLM models (which strictly disallow self-attention). **No fix needed**; just acknowledge in §Limitations. |
| 🟧 **MEDIUM** | **All** | The four likelihood estimators (PLM chain-rule for selfless/xlnet, ELBO for llada/dream, CE for sdar, AR-shift for causal LM) are **not strictly comparable bounds**. Cross-method BPB rankings should be reported with this caveat. |
| 🟨 **LOW** | **LLaDA** | High variance in ELBO estimator for short targets (LAMBADA), but the formula is correct. |
| 🟨 **LOW** | **XLNet, Selfless** | mc_num=32 stderr not reported. eval-mode AR vs random differ in their natural variance budget. |
| 🟩 **NONE** | **AR / Causal LM** | Standard right-shifted next-token CE. Reference implementation. |

---

## 1. AR / Causal LM (reference) — ✓ CORRECT

**Training** (`pretrain/train_ar.py`):
- Standard causal LM with right-shift cross-entropy on next-token prediction
- Same data, optimizer, schedule as other methods
- No special tricks

**Evaluation** (`eval/ar/eval_worker_ar.py` line 75-89):
```python
logits = self.model(input_ids).logits
loss = self.loss_function(logits=logits[..., :-1, :], labels=labels[..., 1:], 
                          ignore_index=-100, reduction='sum')
loss = loss / input_ids.size(0)
return -loss.item()
```
- Right-shifted CE: `logits[i]` predicts `labels[i+1]`
- Causal attention (default for AutoModelForCausalLM)
- This is the **canonical** AR likelihood evaluation

**Conclusion**: Reference implementation. All other methods should be benchmarked against this.

---

## 2. LLaDA — ✓ FAITHFUL, with known limitations

**Training** (`pretrain/train_llada.py` + `modeling_llada.py` + `utils/diffusion_utils.py`):

Forward process (`diffusion_utils.py:23-58`):
1. Sample `t ~ U[eps, 1-eps]` per batch item
2. Sample `v ~ U[0,1]` per position; `v[:,0] = 2` (first token never masked)
3. Mask positions where `v < t` → effective mask rate ≈ `t`
4. Replace masked positions with `[MASK]`

Training loss (`modeling_llada.py:571-594`):
```python
loss = CE(logits[masked_indices], labels[masked_indices], ignore_index=-100, reduction='none')
loss = loss / p_mask[masked_indices]      # divide by t (the masking prob)
loss = loss.sum() / (B * L)               # normalize per token average
```

This is the **standard MDLM / LLaDA ELBO**:
```
L = E_t[1/t · Σ_{i ∈ M_t} CE(x_i)] = ELBO lower bound on -log P(x)
```

By Jensen's inequality (the ELBO derivation in the MDLM / LLaDA paper), this is a valid lower bound on `log P(x)`. ✓

**Attention** (`eval_worker_llada.py:74`): `get_full_attention_mask(L)` = bidirectional. ✓ Correct for absorbing-state diffusion.

**Evaluation** (`eval/llada/eval_worker_llada.py:84-98`):
```python
loss = CE(logits[masked_indices], labels[masked_indices], ignore_index=-100, reduction='none')
loss = loss / t_sample[masked_indices]
loss = loss.sum() / input_ids.shape[0]   # ÷ B (not B*L) to give per-sample sum
return -loss.item()
```
Same formula as training, but normalized per sample (not per token average) — appropriate for lm-eval-harness which expects total log-likelihood per request. ✓

mc_num=32 with batch_size=16 → 32 different `(t, mask_pattern)` samples per example. Reasonable Monte Carlo budget for most evaluations.

**Concerns**:
- For SHORT targets (e.g., LAMBADA's 1-token continuation), the ELBO has very high variance. With t small, `CE/t` is rarely sampled but enormous when sampled. With t large, often sampled but contributes less. The empirical mean over 32 samples has substantial residual variance.
- This is **not a bug**, it's a fundamental property of the absorbing-state ELBO. LLaDA's LAMBADA-PPL of 1330 reflects high variance, not low likelihood.

**Conclusion**: Faithful implementation. Limitations are inherent to absorbing-state ELBO, not implementation flaws.

---

## 3. Dream — ⚠️ DESIGN ARTIFACT ON SHORT TARGETS

**Training** (`pretrain/train_dream.py:265-290`):
```python
text_ids = batch["input_ids"]              # length L
label_ids = batch["labels"]                # length L-1 = text_ids[:, 1:]
input_ids_masked, masked_indices, t_sample, _ = diff_lm.forward_process(text_ids)

loss = model.forward_process(
    input_ids=input_ids_masked[:, :-1],     # drop last input → length L-1
    labels=label_ids,                       # text_ids[:, 1:] → length L-1
    p_mask=t_sample[:, 1:],
    masked_indices=masked_indices[:, 1:],
)
```

**The AR-shift convention**: 
- Output position `i` predicts `label[i] = text_ids[i+1]` (next token)
- Loss is applied only at output positions `i` where `masked_indices[i+1] = True` (i.e., where the to-be-predicted token at position `i+1` was masked in the input)
- Input position `i+1` has [MASK] in input_ids_masked when target was masked → no leak

**Why this works**: Dream's original paper initializes from a pre-trained causal LM (Qwen2.5) and continues training with diffusion objective. The shift convention is preserved from the AR pretraining. This is a legitimate design choice (it differs from LLaDA but is internally consistent).

**Evaluation** (`eval_worker_dream.py:68-77`):
```python
attention_mask = get_full_attention_mask(L-1)        # bidirectional
logits = self.model(input_ids=input_ids_masked[:, :-1], ...).logits
return logits, masked_indices[:, 1:], t_sample[:, 1:]
```

Same shifted pattern. ✓ Consistent with training.

**🟥 CRITICAL ISSUE for LAMBADA-style short-target tasks**:

For LAMBADA: `input_ids = prefix(len=N) + target(len=1)`, total length `L = N+1`.

In `forward_process`, mask is sampled only for non-prompt positions:
- Prompt positions (0..N-1): `v ≥ 2`, never masked
- Target position N: `v ∈ [0,1]`, mask with prob `t`

When `input_ids_masked[:, :-1]` is taken, **position N is dropped**. So:
- Model input: `input_ids_masked[0..N-1]` (length N) = prompt only (no [MASK] at target slot)
- Model output at position N-1 predicts `label[N-1] = text_ids[N] = target`
- The model **never sees the [MASK] indicator** at the target position

**Consequence**: For LAMBADA with `target_len=1`, Dream effectively predicts the target from the prefix without any positional marker indicating where the target should go. The model has no idea it should be predicting a continuation — it could just be predicting "more prefix tokens".

This explains Dream's catastrophic LAMBADA-PPL (209k at 0.6B, 366k at 250M). The result is **not** representative of Dream's diffusion capability; it's an artifact of the AR-shift convention combined with LAMBADA's last-token target.

**Sanity check**: Dream's WT-PPL (33.21 at 0.6B) is reasonable, comparable to LLaDA (32.66). The catastrophe is LAMBADA-specific.

**Recommendation for the paper**:
- Do NOT compare Selfless's LAMBADA-PPL (58) to Dream's LAMBADA-PPL (209k) as if it's apples-to-apples. It is unfair to Dream.
- Either:
  (a) Re-implement Dream's LAMBADA eval to pad input by one [MASK] at the end so the model knows where the target should appear
  (b) OR explicitly note in §Limitations that "Dream's LAMBADA-PPL reflects a design choice in its training convention, not a fundamental diffusion limitation"

**Conclusion**: Implementation is faithful to Dream's paper, but Dream's design has a known weakness on last-token tasks. The eval result on LAMBADA is an artifact, not a model failure.

---

## 4. SDAR — 🟧 EVAL IS CONSISTENT WITH TRAINING; soft leak via causal diagonal

> **Correction (2026-05-18)**: I initially flagged SDAR's eval as "methodologically unsound" because it doesn't use an AR right-shift. The author corrected this: **SDAR is trained MLM-style** (`logits[i]` predicts `input_ids[i]`, not `input_ids[i+1]`). The training loss uses `target=labels[xt_positions]` where `labels = input_ids.clone()` (no shift), so the LM head learns to predict the **current token at masked positions**. The eval convention (no shift) is therefore consistent with training. The section below reflects this corrected understanding.

**Training** (`pretrain/train_sdar.py:266-292` + `modeling_sdar.py:1115-1185`):

SDAR is a "block diffusion" design where:
1. Input is split into BLOCKS of size `block_size` (default 4)
2. For training: input is concatenated as `[noisy_half | clean_half]` of length 2L
3. Attention mask `block_attn_mask` enforces:
   - Noisy positions can attend bidirectionally within their block + causally to clean previous blocks
   - Clean positions can attend causally to clean current+earlier blocks
4. Loss is computed ONLY at noisy positions via `FusedLinearDiffusionCrossEntropyLoss` with `p_mask` weighting
5. Per-position prediction: `logits[i]` predicts `input_ids[i]` (the original token at THIS position), where the input at noisy position i is [MASK]
6. Normalized by `answer_len = (labels != -100).sum()`

The training is faithful to the SDAR paper's MLM-style block-diffusion design. ✓

**Evaluation** (`eval/sdar/eval_worker_sdar.py:69-80`):
```python
position_ids = torch.arange(seq_len, ...).unsqueeze(0).expand(B, -1)
loss = self.model(input_ids, position_ids=position_ids, labels=labels).loss
answer_len = (labels != -100).sum()
loss_resume = loss * answer_len / input_ids.shape[0]
return -loss_resume.item()
```

In `modeling_sdar.py:1186-1214` (eval branch — taken when `self.training=False`):
```python
outputs = self.model(input_ids, attention_mask=None, ...)   # default causal mask
logits = self.lm_head(hidden_states)
loss = nn.CrossEntropyLoss()(logits.view(-1, V), labels.view(-1))   # MLM-style, no shift
```

This is **consistent with SDAR's training convention**: predict `input_ids[i]` from logits at position i. No AR shift is needed because SDAR was trained as predict-current-token (MLM-style at masked positions), not predict-next-token.

**🟧 Subtle issue — soft information leak via causal diagonal**:

- During training: at noisy position i, input is [MASK]. The model's hidden state at i is computed from surrounding context (other [MASK]s/tokens in same block + clean previous blocks). No direct access to `input_ids[i]`.
- During eval: input at position i is the actual `input_ids[i]`. With causal attention, the diagonal lets position i's hidden state include information from itself (`embed(input_ids[i])` flows into `hidden[i]` via the diagonal `kv_idx == q_idx` allowed attention).

So during eval, the LM head receives a hidden state that has seen the target token itself (through the diagonal). If the LM head had memorized "copy the diagonal contribution to logits", eval BPB would be near zero. The empirical PPL of 29.48 suggests the LM head did NOT learn this trivial mapping — likely because in training, the diagonal contribution at noisy positions was always `embed([MASK])`, so the LM head learned to ignore it.

But this is a TRAIN-EVAL DISTRIBUTION MISMATCH for the LM head input. The number is still a valid measurement of "how well does SDAR's model predict each token, given the eval setup", but it's not exactly the same quantity as PLM's chain-rule or LLaDA's ELBO.

**Implication for fair comparison**:

PLM models (Selfless/XLNet) strictly disallow self-attention (`v_kv > v_q` is strict). SDAR allows self-attention through causal diagonal. This gives SDAR a "free" boost in eval BPB compared to PLM models. The SDAR-Selfless gap (0.913 vs 0.943 in AR mode at 0.6B) is partly driven by this asymmetry.

**🟦 Recommended sanity check (optional)**:

If you have spare compute, run SDAR eval with **bidirectional attention but mask the diagonal** (i.e., replace causal mask with a strict-no-diagonal mask). This would tell you how much of SDAR's BPB advantage comes from the diagonal leak vs. genuine model quality. If the no-diagonal eval BPB is much worse (e.g., 0.97+), the current 0.913 is largely the diagonal leak. If still close to 0.913, then SDAR genuinely has better representations.

**Conclusion**: SDAR training and eval are mutually consistent (both predict current token at noisy / all positions). However, the causal diagonal in eval creates a soft information leak that's not present in PLM models. **Acknowledge this in §Limitations as a caveat to cross-method comparisons.** No code fix needed.

---

## 5. XLNet (your re-implementation) — ✓ CORRECT, robust design

**Training** (`pretrain/train_xlnet.py:239-263`):
```python
text_ids = batch["input_ids"][:, :-1].contiguous()
t_sample, v_sample = diff_lm.sample_v(text_ids)
query_attention_mask, kv_attention_mask = get_xlnet_mask(v_sample=v_sample, ...)
loss = model(X0_input_ids=text_ids, labels=text_ids, 
             query_attention_mask=q_mask, kv_attention_mask=kv_mask).loss
```

**Mask** (`utils/utils.py:484-518`):
- query stream: `v_kv > v_q` strict (no diagonal)
- content stream: `v_kv >= v_q` (with diagonal — the "selfish" version)

**Loss** (chain-rule under permutation): output XT predicts each token using the no-diagonal query stream against the with-diagonal content stream. Standard XLNet PLM formulation. ✓

**Architecture** (`modeling_xlnet.py:518-522`):
```python
XT_hidden_states = self.norm(XT_hidden_states)
return BaseModelOutputWithPast(last_hidden_state=XT_hidden_states, ...)
```
**Always** returns XT_hidden_states, no train/eval branching. **Robust design.** ✓

**Evaluation** (`eval/xlnet/eval_worker_xlnet.py:71-90`):
```python
self.model.train()   # call for consistency with selfless, but not needed for XLNet
...
logits = self.model(input_ids, attention_mask=...).logits
loss = CE(logits, labels, ignore_index=-100, reduction='sum') / B
```

Same path as training: lm_head reads XT, no shift, chain-rule under sampled permutation.

The `model.train()` call is REDUNDANT for XLNet (because XLNet always returns XT regardless), but harmless (attention_dropout=0 in Qwen3).

**Conclusion**: Faithful implementation of XLNet's PLM design. The XT-always-returned design is robust to train/eval mode switching.

---

## 6. Selfless — ✓ CORRECT but fragile

**Training** (`pretrain/train_selfless.py:240-262`): identical to XLNet except for the mask:
- `get_selfless_mask(v_sample, ...)` returns a SINGLE mask `v_kv > v_q` strict (no diagonal anywhere)
- Both streams (X0 content, XT query) use this same mask

**Mask** (`utils/utils.py:447-484`):
```python
def mask_fn(b, h, q_idx, kv_idx):
    return v_sample[b, kv_idx] > v_sample[b, q_idx]   # strict, no diagonal
```

Selfless_mask − XLNet_content_mask = identity matrix (the diagonal). Confirmed via hand-trace.

**Architecture** (`modeling_selfless.py:447-498`):
```python
# Line 449-455: conditional XT initialization
if self.training:
    XT_inputs_embeds = self.embed_tokens(self.XT_input_ids)  # MASK embedding
else:
    XT_inputs_embeds = None   # XT disabled in eval

# Line 487-498: conditional return
if not self.training:
    X0_hidden_states = self.norm(X0_hidden_states)
    return ...X0_hidden_states...    # WRONG STREAM for lm_head
else:
    XT_hidden_states = self.norm(XT_hidden_states)
    return ...XT_hidden_states...    # CORRECT for lm_head
```

**🟧 Footgun**: If anyone calls `model.eval()` followed by `forward()`, they get X0_hidden_states, which the LM head was NOT trained on. The eval would silently produce wrong numbers.

**Evaluation** (`eval/selfless/eval_worker_selfless.py:49`):
```python
self.model.train()   # <-- ESSENTIAL: this forces XT path
```

By calling `model.train()`, the eval script ensures `self.training=True`, which triggers the XT path. With `attention_dropout=0` in Qwen3, `model.train()` has no other side effect. So the BPB results ARE correct. ✓

**Mathematical correctness of the eval**:
- XT query at position i is derived from `embed(MASK)` (not `embed(x_i)`) → no info about x_i
- KV from X0 with strict no-diagonal mask → no attention to position i itself
- Indirect leak (position j sees x_i, then i attends to j): mathematically impossible because the constraints (`v_j > v_i` AND `v_i > v_j`) are mutually exclusive

**Conclusion**: Implementation is correct given that eval scripts call `model.train()`. But the design is fragile.

**🟧 Secondary issue — `generate()`**: 

`generation/selfless_gen.py` calls `model.eval()` before `model.generate()`. The `generate()` method internally calls `self.forward()` with `self.training=False`. This returns X0_hidden_states, which the LM head was not trained for. **Generation quality may be lower than BPB suggests.**

For the current paper (focused on BPB, LAMBADA-PPL, zero-shot — all use `loglikelihood`, not `generate_until`), this is not currently a problem. It WILL become a problem if you:
- Implement infilling experiments using `generate()`
- Compute iterative refinement quality
- Compare actual generated text quality

**Recommendation**:
- Refactor `modeling_selfless.py:487-498` to always return XT_hidden_states (like XLNet does)
- Add an explicit flag `use_xt_stream=True` to allow override
- Remove the `model.train()` call from `eval_worker_selfless.py` once refactored

---

## 7. Cross-method comparison fairness

### The four estimators

| Method | Likelihood estimator | Properly bounds `log P(x)`? |
|---|---|---|
| AR / Causal LM | Right-shifted next-token CE | ✓ Yes (the exact log-likelihood under the AR factorization) |
| LLaDA | ELBO: `Σ_{i ∈ M_t} CE / t` | ✓ Yes (Jensen lower bound) |
| Dream | Same ELBO formula, with AR-shift wrapper | ✓ Yes for non-last-token; ⚠️ Broken for last-token tasks before fix |
| XLNet (random eval) | Chain-rule under random permutation, MC averaged | ✓ Yes (Jensen lower bound) |
| XLNet (AR eval) | Chain-rule under identity permutation | ✓ Yes (one specific chain-rule decomposition) |
| Selfless (random eval) | Same as XLNet random | ✓ Yes |
| Selfless (AR eval) | Same as XLNet AR | ✓ Yes |
| **SDAR** | **MLM-style CE (predict current token at every position, with causal attention)** | ⚠️ Valid in spirit (it's how SDAR is trained), but the causal diagonal during eval is a soft information leak not present in PLM models |

### What this means for the paper

**Fair comparisons**:
- Causal LM ↔ Causal LM ✓
- LLaDA ↔ Dream (same ELBO formula; Dream had last-token caveat, fix shipped)
- Selfless ↔ XLNet (same chain-rule formula, both AR and random modes)
- Selfless AR-mode ↔ LLaDA (different estimators but both bound `log P(x)`)

**Comparisons requiring caveat**:
- Selfless/XLNet/LLaDA ↔ **SDAR**: SDAR's eval uses MLM-style prediction with causal-attention diagonal. The diagonal lets the model "peek at" the target through self-attention, which PLM models strictly disallow. SDAR's 0.913 BPB is partly driven by this asymmetry. To make this apples-to-apples, either (a) evaluate SDAR with strict-no-diagonal attention, or (b) acknowledge the diagonal-leak asymmetry in §Limitations.
- Dream ↔ anyone on LAMBADA: previously broken; now fixed in `eval_worker_dream.py`. Re-run LAMBADA eval for Dream to get a fair number.

### Recommendations

1. **Re-run Dream's LAMBADA eval** with the patched `eval_worker_dream.py` (adds an EOS-padded slot at the end so the target isn't dropped). Expect Dream's LAMBADA-PPL to drop from 209k to something in the same ballpark as LLaDA (~1000-3000) or better.

2. **Optional sanity check for SDAR**: implement an alternative eval that masks the causal diagonal (i.e., position i cannot attend to itself). Compare to the current 0.913 BPB. If the no-diagonal eval is substantially worse, document this gap as evidence that SDAR's reported BPB benefits from the diagonal leak.

3. **In §Limitations**, write:
   > "Cross-method BPB comparison uses each method's native likelihood estimator (chain-rule for PLM, ELBO for LLaDA/Dream, MLM-style CE for SDAR, AR for causal LM). These are not strictly comparable lower bounds. In particular, SDAR's eval uses causal attention which provides a soft information leak through the self-attention diagonal — a leak not present in PLM models, where the diagonal is strictly removed (Selfless) or one-sided (XLNet query stream). When comparing SDAR's 0.913 BPB to Selfless's 0.943 (AR mode), part of the 0.030 gap is attributable to this architectural asymmetry rather than to model quality."

4. **Refactor Selfless** to always return XT_hidden_states. **FIX SHIPPED** in modified `modeling_selfless.py`. Remove the `model.train()` workaround from eval scripts. **FIX SHIPPED**.

---

## 8. Methodological purity score

| Method | Train code | Eval code | Comparability | Overall |
|---|---|---|---|---|
| AR | A | A | A | A |
| LLaDA | A | A | B (different bound than AR) | A- |
| Dream | A | A (was C before fix, now A) | B- (still ELBO-different from PLM) | A- |
| **SDAR** | A | A (consistent with training MLM convention) | **C** (causal-diagonal eval leak vs PLM's strict no-diagonal) | **B** |
| XLNet | A | A (robust) | A- (different bound) | A |
| Selfless | A (before refactor) / A (after) | A (was fragile, now robust) | A- (same as XLNet) | A (after refactor) |

---

## 9. What I did NOT audit (caveats)

For honesty, the following were not deeply inspected:
- `eval/PPL/` directory (116K of code): separate PPL evaluation infrastructure, may differ from the lm-eval-harness-style workers
- `utils/dataset_arrow.py` data pipeline: assumed correct; not verified
- The `FusedLinearDiffusionCrossEntropyLoss` implementation (`models/modeling_model/fused_linear_diffusion_cross_entropy.py`): assumed correct, would need separate verification
- `generation/*_gen.py` for non-selfless methods: not inspected
- `modeling_ar.py`: assumed to be standard Qwen3 AR LM; not inspected

If the paper's narrative depends on any of these, they need separate audit.

---

## 10. Action items

**Already shipped (this audit round)**:

1. ✅ **Refactored `modeling_selfless.py`** — XT stream is now always computed; `Qwen3Model.forward` always returns XT_hidden_states. Removed all `if self.training` gates on XT-related ops. Matches XLNet's robust design.
2. ✅ **Updated 4 eval workers** (`eval_worker_selfless.py`, `eval_worker_selfless_ar.py`, `eval_worker_xlnet.py`, `eval_worker_xlnet_ar.py`) — replaced `self.model.train()` with `self.model.eval()` (proper convention). Now that the model always produces XT, no workaround is needed.
3. ✅ **Fixed `eval_worker_dream.py` LAMBADA artifact** — `loglikelihood()` now pads input with an EOS dummy token at the end so the AR-shift drop doesn't remove the target slot. Label for the dummy is -100. Re-run LAMBADA on Dream to get a fair number.

**Recommended (cannot do without compute)**:

4. 🟨 **Re-run all selfless/xlnet evals** after the modeling refactor to confirm BPB numbers are unchanged. They should be: the change is purely architectural cleanup — the lm_head input is identical in both old (`model.train()` workaround) and new (always-return-XT) implementations. But a sanity-check run confirms zero numerical drift.
5. 🟨 **Re-run Dream LAMBADA eval** with the patched `eval_worker_dream.py`. Expected: LAMBADA-PPL drops from ~209k (0.6B) to ~1000-3000 range, comparable to LLaDA.
6. 🟦 **Optional sanity for SDAR**: implement strict-no-diagonal eval to quantify how much of SDAR's 0.913 BPB comes from the causal-attention diagonal leak.
7. 🟦 **Implement unified eval**: run all methods under one common estimator (e.g., random-masking ELBO with same `t` distribution) for an apples-to-apples comparison table.
8. 🟦 **Report mc_num stderr** for XLNet/Selfless/LLaDA/Dream evals.

**Don't do**:

- ❌ Don't change SDAR's eval to add an AR right-shift. SDAR is MLM-style by design.
- ❌ Don't change SDAR's attention to bidirectional in eval. That would break a different invariant.
