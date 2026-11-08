"""Sanity check: is the autoencoder doing more than a loudness meter?

    python scripts/sanity_level_baseline.py

For every machine this compares, on validation and test, the ROC-AUC of
  - the selected autoencoder's anomaly score, and
  - two trivial features that need no machine learning: the clip's average level (level_db) and its RMS loudness.
Trivial features are scored direction-free (max(AUC, 1 - AUC)), which is generous to them.
If a trivial feature matches the autoencoder on a machine (within 0.02 AUC), the autoencoder has not shown it adds
anything there, and the result must be reported that way. The check uses the BETTER of the two loudness features. A loudness gap can also come from the recording session rather than the fault,
and this data cannot tell the two apart. Writes reports/phase3/sanity_level_baseline.csv.
"""
import sys

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
            lvf, rmf = max(lv, 1 - lv), max(rm, 1 - rm)
            rows.append({"group": g, "split": split, "auc_autoencoder": ae, "auc_level_db_free": lvf,
                         "auc_rms_free": rmf, "auc_loudness_best": max(lvf, rmf),
                         "level_direction": "abnormal louder" if lv > 0.5 else "abnormal quieter",
                         "spearman_score_vs_level_on_normal": rho})
    t = pd.DataFrame(rows)
    t.to_csv(rep3 / "sanity_level_baseline.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    close = t["auc_loudness_best"] >= t["auc_autoencoder"] - 0.02
    for split in ("test", "val"):
        names = t[(t["split"] == split) & close]["group"]
        print(f"\nMachines where the better loudness feature is within 0.02 AUC of the autoencoder or above it ({split}):")
        print(", ".join(names) if len(names) else "none")
    both = set(t[(t["split"] == "test") & close]["group"]) & set(t[(t["split"] == "val") & close]["group"])
    print("\nFlagged on BOTH validation and test (the autoencoder is not shown to add anything beyond loudness here):")
    print(", ".join(sorted(both)) if both else "none")
    print("\nSaved reports/phase3/sanity_level_baseline.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
