# Selfless Attention — Updated Research Documents

**Generated**: 2026-05-18
**Trigger**: Reanalysis after confirming "ar+ar" eval = random-trained checkpoint in AR mode (single checkpoint, two eval modes)

---

## What's in this bundle

| File | Purpose | Replaces |
|---|---|---|
| **PLAN.md** | Reframed research plan from "SOTA flexible-order BPB" to "expressiveness vs ordering-robustness tradeoff" | Old PLAN.md (2026-04-29) |
| **TODO.md** | Concrete actionable experiment list, ordered by priority and tier impact | (new) |
| **THRESHOLDS.md** | Tier mapping updated to actual current data (Tier 1 secure, Tier 2 reachable) | Old THRESHOLDS.md |
| **RESULTS_SUMMARY.md** | Master tables of all current eval results, derived quantities, what's proven and what's not | (new) |
| **CODE_AUDIT.md** | Audit of training and eval code — confirms no functional bugs, lists 5 cosmetic items + 1 config to verify | (new) |

---

## How to use this bundle

1. **Read RESULTS_SUMMARY.md first** — gives you the up-to-date picture of what your data actually shows.

2. **Read CODE_AUDIT.md** — confirms the code logic is sound and lists the small cleanups needed before submission.

3. **Read PLAN.md** — the new paper framing. Major changes:
   - The "SOTA flexible-order BPB" claim is gone (data doesn't support it).
   - The new C1 is the **expressiveness vs ordering-robustness tradeoff** — supported by the (AR − random) eval-mode gap data and the zero-shot dominance result.
   - The new C3 emphasizes the **single-checkpoint multi-mode** demonstration AND the **downstream zero-shot dominance among non-AR models**.
   - LAMBADA-PPL is featured as a "Selfless is the only non-AR model within an order of magnitude of causal LM" headline.

4. **Read TODO.md** — what to do next, week by week.

5. **Read THRESHOLDS.md** — to track your tier progression as experiments come in.

---

## The reframing in one sentence

Old: "Selfless removes a 6-year-old bug; PLM achieves SOTA on flexible-order generation."

New: "Selfless reveals a previously-unrecognized **expressiveness vs ordering-robustness tradeoff** in two-stream attention design; the same random-trained checkpoint excels at AR-mode generation and downstream zero-shot tasks (best among non-AR models), but loses some flexible-order BPB to DLMs — a tradeoff inherent to relational vs anchored representations."

---

## Critical actions BEFORE next step

These are cheap and should happen before any new experiment:

1. **Verify the resume_from_checkpoint did not break the LR schedule** for selfless-0.6B
   - Pull up the wandb LR curves for selfless-0.6B and xlnet-0.6B
   - Confirm they're aligned at step 50,000
   - If not aligned: this is a confound, you'll need to rerun selfless-0.6B from scratch

2. **Run the Var(h_i) across permutations experiment** (TODO §A1)
   - Without this, the new C1 framing is hand-waving
   - Cheap (<2 hours)
   - If `Var_Selfless > Var_XLNet`: C1 confirmed, story is solid
   - If not: the framing needs another revision

3. **Run a quick mc_num stderr check** (TODO §C3)
   - Specifically for the 0.6B random-mode regression (Selfless 0.971 vs XLNet 0.963)
   - If the stderr is > 0.008, the regression isn't significant single-seed
   - This determines whether you NEED multi-seed (expensive) or can proceed without

---

## Open question for you

The original PLAN.md had a clear "selfish shortcut bug" framing. The new framing is more nuanced ("tradeoff"). **Make sure you're comfortable with the nuance** before pitching this to advisors or collaborators — it's a softer claim than the original but it's a claim that the data actually supports.

Some people will prefer the original "bug we fixed" framing even if the data is weaker, because it's more clickable. I'd advise honesty over clickability — but it's your decision.

---

*Generated as part of code audit and replan session, 2026-05-18.*
