"""Calibrate the alert limits from the VALIDATION normal scores of the models selected in Phase 2.

    python scripts/calibrate_severity.py

Why this way: a pump has only 45-61 normal validation clips, too few to read a rare percentile (say the 99.5th) off
one machine directly. So the method splits the job in two:
  1. per machine: find its own normal level and spread (needs only a robust median, which 45 clips can give);
  2. across all 8 machines: express every validation-normal clip as a z-value and read the warning and alarm limits
     (the 95th and 99.5th percentile) from the pooled z-values (about 860 clips).
The assumption that the pooled z-values have a similar shape for every machine is CHECKED below on the TEST normal
clips: for each machine it prints how often normal clips cross each limit (target about 5% and 0.5%).
Test data is used here only for that check, never to set anything.

Writes config/severity_calibration.json, reports/phase3/calibration_table.csv, reports/figures/phase3_calibration.png
"""
import argparse
import sys
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smart_listener import severity as sv
from smart_listener.config import project_root


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate severity limits")
    ap.add_argument("--q-warn", type=float, default=0.95)
    ap.add_argument("--q-alarm", type=float, default=0.995)
    ap.add_argument("--score", default="score_mean", choices=["score_mean", "score_max"])
    args = ap.parse_args()
    root = project_root()
    rep2, rep3, figs = root / "reports" / "phase2", root / "reports" / "phase3", root / "reports" / "figures"
    rep3.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    df = sv.load_selected_scores(rep2, args.score)
    sel = pd.read_csv(rep2 / "selected_models.csv").set_index("group")["selected_model"].to_dict()

    machines = {}
    df["z"] = np.nan
    for g, d in df.groupby("group"):
        vn = d[(d["split"] == "val") & (d["label"] == 0)]["score"]
        c, s = sv.fit_center_scale(vn)
        machines[g] = {"model": sel[g], "center": c, "scale": s, "n_val_normal": int(len(vn))}
        df.loc[d.index, "z"] = sv.to_z(d["score"], c, s)
    pooled = df[(df["split"] == "val") & (df["label"] == 0)]["z"].to_numpy()
    z_warn, z_alarm = float(np.quantile(pooled, args.q_warn)), float(np.quantile(pooled, args.q_alarm))
    print(f"pooled validation-normal clips: {len(pooled)} | warning limit z={z_warn:.2f} (q={args.q_warn}) | alarm limit z={z_alarm:.2f} (q={args.q_alarm})")

    rows = []
    for g, d in df.groupby("group"):
        r = {"group": g, "model": sel[g], "n_val_normal": machines[g]["n_val_normal"]}
        for split in ("val", "test"):
            n, a = d[(d["split"] == split) & (d["label"] == 0)]["z"], d[(d["split"] == split) & (d["label"] == 1)]["z"]
            r[f"{split}_normal_over_warn"] = float((n > z_warn).mean())
            r[f"{split}_normal_over_alarm"] = float((n > z_alarm).mean())
            r[f"{split}_abnormal_over_warn"] = float((a > z_warn).mean())
            r[f"{split}_abnormal_over_alarm"] = float((a > z_alarm).mean())
            r[f"{split}_n_normal"] = int(len(n))
        rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(rep3 / "calibration_table.csv", index=False)
    pd.set_option("display.width", 220)
    show = t[["group", "n_val_normal", "test_n_normal", "test_normal_over_warn", "test_normal_over_alarm",
              "test_abnormal_over_warn", "test_abnormal_over_alarm"]].round(3)
    print("\nPer machine on the TEST split (clip level). 'normal_over_*' should be near 0.05 / 0.005; 'abnormal_over_*' is the share of faulty clips that cross the limit:")
    print(show.to_string(index=False))
    tot_n = int(t["test_n_normal"].sum())
    print(f"\nAll machines pooled, test normal clips: over warning limit {np.average(t['test_normal_over_warn'], weights=t['test_n_normal']):.3f}, "
          f"over alarm limit {np.average(t['test_normal_over_alarm'], weights=t['test_n_normal']):.3f} ({tot_n} clips)")

    cal = {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "score_column": args.score,
           "transform": "z = (ln(score) - center) / scale, per machine, fitted on normal VALIDATION clips",
           "machines": machines,
           "thresholds": {"q_warn": args.q_warn, "q_alarm": args.q_alarm, "z_warn": z_warn, "z_alarm": z_alarm,
                          "n_pooled": int(len(pooled))},
           "rule": dict(sv.DEFAULT_RULE),
           "note": "Operational severity. Not a failure probability, remaining life or fault diagnosis."}
    p = sv.save_calibration(cal)
    print(f"\nSaved {p.relative_to(root)}")

    groups = sorted(df["group"].unique())
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), squeeze=False)
    for ax, g in zip(axes.ravel(), groups):
        d = df[(df["group"] == g) & (df["split"] == "test")]
        hi = max(d["z"].max(), z_alarm) + 1
        bins = np.linspace(min(d["z"].min(), -3), hi, 40)
        ax.hist(d[d["label"] == 0]["z"], bins=bins, alpha=0.6, label="normal")
        ax.hist(d[d["label"] == 1]["z"], bins=bins, alpha=0.6, label="abnormal")
        ax.axvline(z_warn, color="orange", ls="--", lw=1.2)
        ax.axvline(z_alarm, color="red", ls="--", lw=1.2)
        ax.set_title(g, fontsize=9)
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Test clips in z-units (orange = warning limit, red = alarm limit)")
    fig.tight_layout()
    fig.savefig(figs / "phase3_calibration.png", dpi=120)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
