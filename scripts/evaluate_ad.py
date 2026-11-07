"""Evaluate the trained autoencoders from the saved score files.

    python scripts/evaluate_ad.py            # after train_ad.py
    python scripts/evaluate_ad.py --smoke    # for the smoke-test outputs

Reads  reports/phase2/scores_*.csv
Writes reports/phase2/metrics_by_group.csv, metrics_summary.csv, selected_models.csv
       reports/figures/phase2_auc_by_group.png, phase2_scores_<model>.png

Reported per machine ID and model: ROC-AUC, partial AUC (max FPR 0.1), and on the TEST split a 95% bootstrap
interval plus a PROVISIONAL operating point (threshold = 95th percentile of that machine's normal VALIDATION
scores). The final thresholds are calibrated properly in Phase 3. Validation and test sets here contain equal
numbers of normal and abnormal clips, so precision is NOT what you would see in a plant where faults are rare.
Model selection uses validation AUC only; the test split is never used to choose anything.
"""
import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smart_listener import anomaly
from smart_listener.config import load_config, project_root


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate anomaly detectors")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--score", default="score_mean", choices=["score_mean", "score_max"],
                    help="which clip score to evaluate (score_mean is the pre-decided primary score)")
    args = ap.parse_args()
    cfg = load_config()
    root = project_root()
    rep = root / "reports" / ("phase2_smoke" if args.smoke else "phase2")
    figs = root / "reports" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    files = sorted(rep.glob("scores_*.csv"))
    if not files:
        print(f"No score files in {rep}. Run scripts/train_ad.py first.")
        return 1
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    sc = args.score
    rows = []
    for (model, group), g in df.groupby(["model", "group"]):
        thr = float(np.percentile(g[(g["split"] == "val") & (g["label"] == 0)][sc], 95))
        for split in ("val", "test"):
            h = g[g["split"] == split]
            if h["label"].nunique() < 2:
                continue
            auc, pauc = anomaly.auc_metrics(h["label"], h[sc])
            r = {"model": model, "group": group, "split": split, "n_normal": int((h["label"] == 0).sum()),
                 "n_abnormal": int((h["label"] == 1).sum()), "auc": auc, "pauc": pauc}
            if split == "test":
                lo, hi = anomaly.bootstrap_auc_ci(h["label"], h[sc])
                op = anomaly.operating_point(h["label"], h[sc], thr)
                r.update({"auc_ci_low": lo, "auc_ci_high": hi, "prov_threshold": thr, **{f"prov_{k}": v for k, v in op.items()}})
            rows.append(r)
    m = pd.DataFrame(rows)
    m.to_csv(rep / "metrics_by_group.csv", index=False)

    test = m[m["split"] == "test"]
    summ = (test.assign(machine_type=test["group"].str.split("_").str[0])
            .groupby(["model", "machine_type"])[["auc", "pauc", "prov_precision", "prov_recall", "prov_fpr"]].mean().reset_index())
    allm = test.groupby("model")[["auc", "pauc", "prov_precision", "prov_recall", "prov_fpr"]].mean().reset_index()
    allm.insert(1, "machine_type", "ALL")
    summ = pd.concat([summ, allm], ignore_index=True)
    summ.to_csv(rep / "metrics_summary.csv", index=False)

    val = m[m["split"] == "val"].pivot(index="group", columns="model", values="auc")
    models = list(val.columns)
    sel = val.idxmax(axis=1).rename("selected_model").reset_index()
    if "dense" in val.columns:  # ties go to the simpler model
        tie = val.nunique(axis=1) == 1
        sel.loc[tie.values, "selected_model"] = "dense"
    sel["rule"] = "highest validation ROC-AUC (ties: dense)"
    sel.to_csv(rep / "selected_models.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== TEST: ROC-AUC / pAUC per machine ID (95% bootstrap interval for AUC) ===")
    t = test.pivot(index="group", columns="model", values=["auc", "pauc"])
    ci = test.assign(ci=test.apply(lambda r: f"[{r.auc_ci_low:.3f}, {r.auc_ci_high:.3f}]", axis=1)).pivot(index="group", columns="model", values="ci")
    print(t.round(3).to_string())
    print("\n95% bootstrap intervals for test AUC:")
    print(ci.to_string())
    print("\n=== VALIDATION ROC-AUC (used for model selection) ===")
    print(val.round(3).to_string())
    print("\n=== SELECTED MODEL PER MACHINE ===")
    print(sel[["group", "selected_model"]].to_string(index=False))
    print("\n=== SUMMARY (mean over machine IDs, TEST split, provisional threshold = 95th pct of normal validation scores) ===")
    print(summ.round(3).to_string(index=False))
    print("\nNote: val/test are balanced 1:1, so precision here is optimistic compared with a plant where faults are rare.")

    # figure 1: AUC by group
    piv = test.pivot(index="group", columns="model", values="auc")
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(piv))
    w = 0.38
    for j, mname in enumerate(piv.columns):
        ax.bar(x + (j - (len(piv.columns) - 1) / 2) * w, piv[mname].values, w, label=mname)
    ax.set_xticks(x)
    ax.set_xticklabels(piv.index, rotation=30, ha="right")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_ylabel("test ROC-AUC")
    ax.set_title("Anomaly detection, test split (higher is better; 0.5 = chance)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figs / ("phase2_auc_by_group_smoke.png" if args.smoke else "phase2_auc_by_group.png"), dpi=140)
    plt.close(fig)

    # figure 2: score distributions per model
    for model, g in df.groupby("model"):
        groups = sorted(g["group"].unique())
        fig, axes = plt.subplots(2, 4, figsize=(14, 6), squeeze=False)
        for ax, grp in zip(axes.ravel(), groups):
            h = g[(g["group"] == grp) & (g["split"] == "test")]
            thr = float(np.percentile(g[(g["group"] == grp) & (g["split"] == "val") & (g["label"] == 0)][sc], 95))
            hi = max(h[sc].max(), thr) * 1.02
            bins = np.linspace(0, hi, 30)
            ax.hist(h[h["label"] == 0][sc], bins=bins, alpha=0.6, label="normal")
            ax.hist(h[h["label"] == 1][sc], bins=bins, alpha=0.6, label="abnormal")
            ax.axvline(thr, color="k", ls="--", lw=1)
            ax.set_title(grp, fontsize=9)
        for ax in axes.ravel()[len(groups):]:
            ax.axis("off")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f"{model}: test anomaly scores (dashed = provisional threshold)")
        fig.tight_layout()
        fig.savefig(figs / f"phase2_scores_{model}{'_smoke' if args.smoke else ''}.png", dpi=120)
        plt.close(fig)
    print(f"\nSaved tables in {rep.relative_to(root)} and figures in reports/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
