"""Model profiles: for each trained machine ID, which model is used and how well it did on held-out data.

Shown to the admin when registering a machine, so nobody is told a machine is "monitored" without seeing how
reliable the detector for it really is. Reads files written in Phases 2 and 3; missing files just leave fields empty.
"""
import json
from pathlib import Path
from typing import Optional

import pandas as pd

from smart_listener.config import project_root

LOUDNESS_MARGIN = 0.02   # same margin as scripts/sanity_level_baseline.py


def model_groups(cfg: dict) -> list:
    return [f"{t}_{i}" for t in cfg["dataset"]["machine_types"] for i in cfg["dataset"]["machine_ids"]]


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:  # a damaged report file must never break the application
        return None


def model_profiles(cfg: dict, root: Optional[Path] = None) -> list:
    root = Path(root) if root else project_root()
    cal = {}
    cal_path = root / "config" / "severity_calibration.json"
    if cal_path.exists():
        try:
            cal = json.loads(cal_path.read_text(encoding="utf-8"))
        except ValueError:
            cal = {}
    metrics = _read_csv(root / "reports" / "phase2" / "metrics_by_group.csv")
    sanity = _read_csv(root / "reports" / "phase3" / "sanity_level_baseline.csv")

    out = []
    for g in model_groups(cfg):
        m = cal.get("machines", {}).get(g, {})
        p = {"group": g, "machine_type": g.split("_")[0], "model": m.get("model"), "n_val_normal": m.get("n_val_normal"),
             "test_auc": None, "test_auc_low": None, "test_auc_high": None, "loudness_auc": None, "caution": None}
        if metrics is not None and m.get("model"):
            r = metrics[(metrics["group"] == g) & (metrics["model"] == m["model"]) & (metrics["split"] == "test")]
            if len(r):
                r = r.iloc[0]
                p["test_auc"] = round(float(r["auc"]), 3)
                if "auc_ci_low" in r and pd.notna(r["auc_ci_low"]):
                    p["test_auc_low"], p["test_auc_high"] = round(float(r["auc_ci_low"]), 3), round(float(r["auc_ci_high"]), 3)
        if sanity is not None:
            r = sanity[(sanity["group"] == g) & (sanity["split"] == "test")]
            if len(r):
                r = r.iloc[0]
                p["loudness_auc"] = round(float(max(r["auc_level_db_free"], r["auc_rms_free"])), 3)
        if p["test_auc"] is not None and p["loudness_auc"] is not None and p["loudness_auc"] >= p["test_auc"] - LOUDNESS_MARGIN:
            p["caution"] = (f"On this machine's held-out test data, simply measuring how loud the recording is separates "
                            f"normal from abnormal about as well as the model does (loudness ROC-AUC {p['loudness_auc']}, "
                            f"model {p['test_auc']}). The model is not shown to add anything beyond loudness here, "
                            f"so treat its alerts with extra care.")
        out.append(p)
    return out
