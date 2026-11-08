"""Sanity check: is the autoencoder doing more than a loudness meter?

    python scripts/sanity_level_baseline.py

For every machine this compares, on validation and test, the ROC-AUC of
  - the selected autoencoder's anomaly score, and
  - two trivial features that need no machine learning: the clip's average level (level_db) and its RMS loudness.
Trivial features are scored direction-free (max(AUC, 1 - AUC)), which is generous to them.
If a trivial feature matches the autoencoder on a machine, the autoencoder has not shown it adds anything there, and
the result must be reported that way. A loudness gap can also come from the recording session rather than the fault,
and this data cannot tell the two apart. Writes reports/phase3/sanity_level_baseline.csv.
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from smart_listener.config import load_config, project_root, resolve_path
from smart_listener.severity import load_selected_scores


def main() -> int:
    cfg, root = load_config(), project_root()
    rep2, rep3 = root / "reports" / "phase2", root / "reports" / "phase3"
    rep3.mkdir(parents=True, exist_ok=True)
    df = load_selected_scores(rep2)
    man = pd.read_csv(resolve_path(cfg, "manifest_dir") / "manifest.csv", keep_default_na=False)
    df = df.merge(man[["file_id", "level_db", "rms"]], on="file_id", how="left", validate="one_to_one")
    rows = []
    for g, d in df.groupby("group"):
        for split in ("val", "test"):
            h = d[d["split"] == split]
            if h["label"].nunique() < 2:
                continue
            ae = roc_auc_score(h["label"], h["score"])
            lv = roc_auc_score(h["label"], h["level_db"].astype(float))
            rm = roc_auc_score(h["label"], h["rms"].astype(float))
            nn = h[h["label"] == 0]
            rho = spearmanr(nn["score"], nn["level_db"].astype(float))[0]
            rows.append({"group": g, "split": split, "auc_autoencoder": ae, "auc_level_db_free": max(lv, 1 - lv),
                         "auc_rms_free": max(rm, 1 - rm), "level_direction": "abnormal louder" if lv > 0.5 else "abnormal quieter",
                         "spearman_score_vs_level_on_normal": rho})
    t = pd.DataFrame(rows)
    t.to_csv(rep3 / "sanity_level_baseline.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    flag = t[(t["split"] == "test") & (t["auc_level_db_free"] >= t["auc_autoencoder"] - 0.02)]
    print("\nMachines where a loudness feature alone is within 0.02 AUC of the autoencoder (test):")
    print(", ".join(flag["group"]) if len(flag) else "none")
    print("\nSaved reports/phase3/sanity_level_baseline.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
