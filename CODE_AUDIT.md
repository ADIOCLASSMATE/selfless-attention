# Code Audit Report

**Date**: 2026-05-18
**Scope**: `pretrain/train_{selfless,xlnet}.py`, `eval/*/eval_worker_*.py`, `utils/{utils,diffusion_utils}.py`, `models/modeling_model/modeling_{selfless,xlnet}.py`

---

## TL;DR

**Training and evaluation logic are correct. No functional bugs.**

The selfless vs. xlnet comparison is a clean A/B test — identical training scripts, identical configs, identical optimizer/schedule/data, with the **only** difference being the attention mask construction. The evaluation is mathematically sound: no information leak, consistent train-eval forward path, well-defined Monte-Carlo PLM likelihood estimator.

Five cosmetic / reproducibility concerns are listed below — none invalidate current results, but they should be addressed before submission.

---

## 1. Mask semantics — verified correct

| Mask | Rule | Definition |
|---|---|---|
| `get_selfless_mask` | strict `v_kv > v_q` | no diagonal anywhere |
| `get_xlnet_mask` (content stream) | `v_kv >= v_q` | includes diagonal |
| `get_xlnet_mask` (query stream) | strict `v_kv > v_q` | no diagonal |

Hand-verified on `prompt_len=2, L=5` cases:

- **AR-mode `v_sample`**: `[3.0, 2.0, 1-eps, ..., eps]` (descending) → produces strict L→R causal attention pattern. Position 0 attends to nothing; position L-1 attends to all earlier positions; position i+1 sees position i exactly when it's its predecessor in the permutation.
- **Random-mode `v_sample`**: `[v0=2, random in [2,3] for prompt, random in [0,1] for target]` → produces a randomized lower-triangular-under-permutation pattern. Different per batch item (`torch.rand(b, l)` gives independent permutations for each example).
- **Selfless mask − XLNet content mask = identity matrix exactly** — confirming the only difference is the diagonal.

## 2. Train/eval consistency — verified, no information leak

`Qwen3Model.forward` (both `modeling_selfless.py` and `modeling_xlnet.py`) has a `if self.training` branch that controls which stream is returned:

```python
# In Qwen3Model.forward (modeling_selfless.py:487-498)
if not self.training:
    return X0_hidden_states  # would use content stream
else:
    return XT_hidden_states  # uses query stream (XT = [MASK] embeddings)
```

All Selfless/XLNet eval workers (`eval_worker_selfless.py:49`, `eval_worker_xlnet.py:53`, `..._ar.py`) call **`self.model.train()` before evaluation**. This routes the eval through the training branch, so the lm_head receives **XT (query stream) outputs** — exactly as during training.

### Why this is leak-free

**Direct leak**: XT query at position `i` comes from `[MASK]` embedding (not from `embed(x_i)`). XT attends to X0 keys/values, but the strict no-diagonal mask blocks attention to its own position. So `XT_i` is a function of `{X0_j : j with v_j > v_i, j ≠ i}` — no direct path to `x_i`.

**Indirect leak (via other positions)**: Could `x_i` leak through some other position's representation? Position `j` sees `x_i` iff `v_i > v_j` (in `j`'s context window). Position `i` attends to `j` iff `v_j > v_i`. **These two conditions are mutually exclusive.** So no path exists from `x_i` → `X0_j` → `XT_i`. Verified mathematically.

(For Qwen3, `attention_dropout=0`, so `model.train()` has no other side effect.)

## 3. Single-checkpoint multi-mode evaluation — verified

Confirmed via timestamps and config hashes:

- `output_eval/selfless-0.6B-50BT-ar+ar/` (2026-05-15 17:42, git `9163e9e`) and
- `output_eval/selfless-0.6B-50BT-random+random/` (2026-05-16 17:00, git `9163e9e`)

both reference `./configs/selfless/lm_eval_selfless_0.6B.yaml`, which loads `output/selfless-0.6B-50BT/hf_model-final` (the random-trained checkpoint). The two eval runs used the **same checkpoint** with different `attention_task` / `prompt_task` config settings (edited between runs).

This means **`ar+ar` is the random-trained Selfless checkpoint evaluated in AR mode** (NOT a separately AR-trained checkpoint, as I initially misread). The `single checkpoint, multiple modes` claim in PLAN.md §0.3 is empirically supported.

Same setup confirmed for `xlnet-0.6B-50BT-{ar+ar,random+random}` (both use `lm_eval_xlnet_0.6B.yaml`).

## 4. Likelihood estimator — correct chain-rule PLM

For one forward pass:

```python
loss = F.cross_entropy(logits.view(-1, V), labels.view(-1),
                       ignore_index=-100, reduction='sum')
loss = loss / input_ids.size(0)  # divides by batch size
ll = -loss.item()  # negative summed CE per sample
```

This computes `−Σ_i CE(x_i, logits_i)` for `i` in target positions = `Σ_i log P_θ(x_i | x_{σ_{<i}})` for the sampled permutation σ. Summed over 32 fresh permutations (mc_num=32, batch_size=16, inner loop runs 32/16=2 times with different `torch.rand` per batch item) and averaged → Monte-Carlo estimate of `E_σ[log P_σ(x | prompt)]`.

By Jensen's inequality, `E_σ[log P_σ(x)] ≤ log P(x)` — this is a valid lower bound.

**Comparison with LLaDA/Dream**: They use ELBO `E_t[1/t × Σ_{i∈M_t} CE(x_i)]`, also a lower bound but a different one. The bounds are not directly comparable in tightness across methods — **paper should note this caveat**.

**Comparison with AR baseline**: AR uses standard right-shift `logits[:-1, :]` predicting `labels[1:]`, scoring `L−1` tokens per non-overlapping window. Selfless eval (no shift) scores `L` tokens in the first window only (extra `log P(x_0 | ∅)` from learned prior). Effect is `~0.1%` on rolling BPB, negligible.

## 5. Random eval: mc_num and permutation sampling

`mc_num=32, batch_size=16` for random-mode 0.6B eval:

- Each example is **repeated 16 times** in the batch (`input_ids[None, :].repeat((batch_size, 1))`)
- Inner loop runs `mc_num // batch_size = 2` times
- Each forward pass calls `torch.rand(b=16, l=L)` → 16 independent permutations
- Total: 32 different permutations per example
- `ll_mean = np.mean(ll_list)` averages the 2 batch-mean log-likelihoods

This is a clean implementation of the MC PLM likelihood. Concerns:

- **No standard error reported** — need to compute and add to paper
- For wikitext_rolling with windows, each window gets fresh permutations; total noise across the test set may need separate stderr estimate
- For ar-mode `mc_num=1` is correct because `v_sample` is deterministic (line 86-88 of `diffusion_utils.py`)

## 6. Configs that need verification

`configs/selfless/pretraining_0.6B.yaml`:
```yaml
resume_from_checkpoint: "output/selfless-0.6B-50BT/checkpoint-34200"
```

`configs/xlnet/pretraining_0.6B.yaml`:
```yaml
resume_from_checkpoint: none
```

**Action needed**: Confirm that despite the resume, both runs trained for the **same total steps (50,000)** with the **same cosine LR schedule** ending at the same `min_lr_scale`. The resume mechanism (`accelerator.load_state` at line 178 of `train_selfless.py`) preserves optimizer state and scheduler step, so this is likely fine — but should be visually confirmed via the wandb LR curves.

## 7. Per-method eval pipeline differences

| Method | Eval estimator | Loss formula | Output stream |
|---|---|---|---|
| AR | Right-shifted next-token | `CE(logits[:-1], labels[1:])` | Causal LM hidden |
| Selfless | Chain-rule under random permutation | `CE(logits, labels, ignore=-100)`, mc_num=32 | XT (via `model.train()`) |
| XLNet | Chain-rule under random permutation | Same as Selfless | XT (via `model.train()`) |
| LLaDA, Dream | ELBO (absorbing-state diffusion) | `CE[masked] / t` summed | Bidirectional encoder |
| SDAR | AR-style with masked block | Standard CE with labels | Direct causal hidden |

**These estimators are not strictly comparable.** The paper must acknowledge this. Two ways to address:

1. **Add a unified eval**: run all methods under one common estimator (e.g., random masking with ratio 0.15, score on masked positions). This is an apples-to-apples comparison.
2. **Discuss in §Limitations**: note the estimator differences and that each method is evaluated under its natural metric.

## 8. Issues identified

| # | Issue | Severity | Effect on current results | Fix |
|---|---|---|---|---|
| 1 | `self.model.train()` in eval | Cosmetic | None functional | Refactor `if self.training` into explicit `use_xt_stream` flag; eval calls `model.eval()` |
| 2 | Different estimators across methods | Methodological | Cross-method ranking has uncertainty | Add §Limitations discussion + ideally a unified eval table |
| 3 | mc_num=32 stderr not reported | Reproducibility | ±?? confidence intervals not visible | Compute and report stderr |
| 4 | First-window bias against selfless | Negligible (~0.1%) | Selfless BPB slightly inflated | Add `start_loc==0` special case to skip position 0 |
| 5 | `resume_from_checkpoint` for selfless-0.6B but not xlnet-0.6B | Reproducibility | Probably no effect (verify LR curves) | Confirm wandb LR alignment |
| 6 | LAMBADA `generate_until = NotImplementedError` | Missing data | LAMBADA-acc shows 0 (PPL is OK) | Optional: implement `generate_until` for completeness |
| 7 | Path/naming inconsistencies between `lm_selfless_ar.sh` (`--output_path .../selfless-${SIZE}-50BT-ar`) and actual dirs (`selfless-${SIZE}-50BT-ar+ar`) | Cosmetic | None | Standardize on one naming convention |

None of these invalidate the current results. Issues 1, 3, 4, 5, 7 are pre-submission cleanups. Issue 2 needs §Limitations discussion. Issue 6 is optional polish.

## 9. What I did NOT audit

For honesty:
- `models/modeling_model/modeling_{llada,dream,sdar}.py` — only inspected eval workers, not the actual model implementations vs. their original papers
- `utils/{dataset_*,reward,math_eval}.py` — data pipeline correctness assumed
- `eval/PPL/` directory (116K, didn't unpack)
- `generation/{dream_gen,llada_gen,xlnet_gen}.py` — only looked at `selfless_gen.py`

If any of these are critical (e.g., if a reviewer might question whether LLaDA/Dream baselines are faithful reproductions of the original papers), those should be the next audit target.
