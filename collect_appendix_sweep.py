#!/usr/bin/env python3
"""
collect_appendix_sweep.py — assemble the appendix sweep into tables + a plot.

WHAT THIS DOES
  Pulls the final validation loss for each sweep run from wandb and produces:
    (1) appendix_lr_sweep.csv / .md  — per-family loss-vs-LR table  (answers A1)
    (2) appendix_beta1.csv / .md     — beta1=0.8 vs 0.9 per family  (answers A3)
    (3) appendix_lr_sweep.png        — loss-vs-LR curves, one line per family

CRITICAL CAVEAT (read before interpreting):
  The per-family training val loss is NOT comparable ACROSS families — AR logs a
  chain-rule NLL, DLM logs an ELBO/denoising loss, PLM logs both. These are
  different estimators (this is exactly the RESULTS.md §3 point). Therefore:
    * WITHIN a family, comparing val loss across LRs is valid -> use it to find
      each family's LR optimum and check 2e-4 is near it. THIS is what A1 needs.
    * ACROSS families, do NOT rank by this val loss. For the cross-family
      ranking-stability claim, re-run the UNIFIED L->R BPB estimator
      (eval/unified_lr_bpb.py) on the swept checkpoints — that is the only
      apples-to-apples number.

  Each family's loss key differs, handled below:
    ar              -> val/loss_ar
    selfless/xlnet  -> val/loss_ar   (AR-mode; the L->R-comparable head)
    llada/dream     -> val/loss_diff
    sdar            -> val/loss

USAGE
  pip install wandb pandas matplotlib --break-system-packages
  WANDB_ENTITY=<you> python collect_appendix_sweep.py --project selfless-attention \
      --tag appendix-sweep
  # offline machine: export the runs to CSV elsewhere and pass --from_csv runs.csv
"""
import argparse, re, sys
import pandas as pd

LOSS_KEY = {
    "ar": "val/loss_ar",
    "selfless": "val/loss_ar",
    "xlnet": "val/loss_ar",
    "llada": "val/loss_diff",
    "dream": "val/loss_diff",
    "sdar": "val/loss",
}
# run name format: {tag}__{family}__342M__lr{LR}__b1-{BETA1}
NAME_RE = re.compile(r"^(?P<tag>.+?)__(?P<fam>[a-z]+)__342M__lr(?P<lr>[0-9.eE+-]+)__b1-(?P<b1>[0-9.]+)$")


def pull_from_wandb(project, tag, entity):
    import wandb
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    rows = []
    for run in api.runs(path):
        m = NAME_RE.match(run.name or "")
        if not m or m.group("tag") != tag:
            continue
        fam = m.group("fam")
        key = LOSS_KEY.get(fam)
        if key is None:
            print(f"  ! unknown family {fam} in {run.name}, skipping", file=sys.stderr)
            continue
        # final logged value of the family's val-loss key
        hist = run.history(keys=[key], pandas=True)
        val = float(hist[key].dropna().iloc[-1]) if key in hist and not hist[key].dropna().empty else None
        rows.append(dict(family=fam, lr=float(m.group("lr")), beta1=float(m.group("b1")),
                         val_loss=val, run=run.name, state=run.state))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="selfless-attention")
    ap.add_argument("--tag", default="appendix-sweep")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--from_csv", default=None, help="skip wandb; load a CSV with family,lr,beta1,val_loss")
    args = ap.parse_args()

    if args.from_csv:
        df = pd.read_csv(args.from_csv)
    else:
        df = pull_from_wandb(args.project, args.tag, args.entity)

    if df.empty:
        print("No runs found. Check --tag / --entity / wandb login.", file=sys.stderr)
        sys.exit(1)

    fam_order = ["ar", "selfless", "xlnet", "sdar", "llada", "dream"]
    df["family"] = pd.Categorical(df["family"], categories=fam_order, ordered=True)

    # ---- A1: LR sweep (beta1 == 0.8 only) ------------------------------------
    a1 = df[df["beta1"] == 0.8].pivot_table(index="family", columns="lr",
                                            values="val_loss", observed=True)
    a1 = a1.sort_index()
    a1.to_csv("appendix_lr_sweep.csv")
    with open("appendix_lr_sweep.md", "w") as f:
        f.write("# A1 — LR fairness (per-family val loss; compare WITHIN row only)\n\n")
        f.write(a1.round(4).to_markdown())
        f.write("\n\n*Within each row, the minimum marks that family's LR optimum. "
                "Check it sits at/near 2e-4. Do NOT compare across rows — different "
                "families log different estimators (RESULTS §3).*\n")
    print("Wrote appendix_lr_sweep.{csv,md}")
    print(a1.round(4).to_string())

    # ---- A3: beta1 ablation (lr == 2e-4) -------------------------------------
    b = df[df["lr"] == 2e-4].pivot_table(index="family", columns="beta1",
                                         values="val_loss", observed=True)
    if 0.9 in b.columns and 0.8 in b.columns:
        b["delta(0.9-0.8)"] = b[0.9] - b[0.8]
    b.to_csv("appendix_beta1.csv")
    with open("appendix_beta1.md", "w") as f:
        f.write("# A3 — beta1 0.8 vs 0.9 at lr=2e-4 (compare WITHIN row only)\n\n")
        f.write(b.round(4).to_markdown())
        f.write("\n\n*Small |delta| supports 'beta1 in {0.8,0.9} has no material effect'.*\n")
    print("\nWrote appendix_beta1.{csv,md}")
    print(b.round(4).to_string())

    # ---- plot ----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        for fam in fam_order:
            if fam in a1.index:
                row = a1.loc[fam].dropna()
                ax.plot(row.index, row.values, marker="o", label=fam)
        ax.axvline(2e-4, ls="--", c="grey", lw=1, alpha=0.7)
        ax.set_xscale("log"); ax.set_xlabel("learning rate"); ax.set_ylabel("val loss (within-family only)")
        ax.set_title("Appendix A1: LR sweep per family (342M, short run)")
        ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig("appendix_lr_sweep.png", dpi=150)
        print("\nWrote appendix_lr_sweep.png")
    except Exception as e:
        print(f"\n(plot skipped: {e})", file=sys.stderr)


if __name__ == "__main__":
    main()
