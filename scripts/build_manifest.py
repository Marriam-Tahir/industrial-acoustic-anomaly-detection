"""Build the manifest with labels and leakage-safe splits.  Run after audit_audio.py:

    python scripts/build_manifest.py

Writes data/manifests/manifest.csv (one row per usable file), excluded.csv, and manifest_summary.json.
Stops with an error if any leakage or labelling check fails.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from smart_listener.config import load_config, resolve_path
from smart_listener.manifest import (assign_anomaly_splits, assign_classifier_splits, scan_folder,
                                     validate_manifest)

# Numbers from Table 1 of the MIMII paper (normal, abnormal). For comparison only; the real files decide.
PAPER_COUNTS = {
    ("fan", "id_00"): (1011, 407), ("fan", "id_02"): (1016, 359), ("fan", "id_04"): (1033, 348), ("fan", "id_06"): (1015, 361),
    ("pump", "id_00"): (1006, 143), ("pump", "id_02"): (1005, 111), ("pump", "id_04"): (702, 100), ("pump", "id_06"): (1036, 102),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the dataset manifest and splits")
    ap.add_argument("--mono-dir")
    ap.add_argument("--no-audit", action="store_true", help="skip the audit requirement (not recommended)")
    args = ap.parse_args()

    cfg = load_config()
    root = Path(args.mono_dir) if args.mono_dir else resolve_path(cfg, "mono_dir")
    mdir = resolve_path(cfg, "manifest_dir")
    mdir.mkdir(parents=True, exist_ok=True)

    df, unexpected = scan_folder(root, cfg)
    if df.empty:
        print(f"No files found under {root}")
        return 1
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected files ignored, for example: {unexpected[:5]}")

    audit_path = mdir / "audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path, keep_default_na=False)
        cols = ["path", "sr", "channels", "duration_s", "peak", "rms", "level_db", "sha256", "exclude", "exclude_reason"]
        df = df.merge(audit[cols], on="path", how="left", validate="one_to_one")
        if df["exclude"].eq("").any():
            print("ERROR: audit.csv does not cover every file. Run scripts/audit_audio.py again.")
            return 1
        df["exclude"] = df["exclude"].astype(str).str.lower().eq("true")
        excluded = df[df["exclude"]]
        excluded.to_csv(mdir / "excluded.csv", index=False)
        df = df[~df["exclude"]].drop(columns=["exclude"]).reset_index(drop=True)
        print(f"Audit applied: {len(excluded)} files excluded (listed in excluded.csv), {len(df)} kept.")
        # keep the original file numbering so block splits still respect recording order
    elif args.no_audit:
        print("WARNING: no audit applied. Quality problems and duplicates will NOT be removed.")
    else:
        print(f"ERROR: {audit_path} not found. Run  python scripts/audit_audio.py  first (or use --no-audit).")
        return 1

    df = assign_anomaly_splits(df, cfg)
    df = assign_classifier_splits(df, cfg)
    problems = validate_manifest(df, cfg)
    if problems:
        print("\nLEAKAGE / LABEL CHECKS FAILED:")
        for p in problems:
            print("  -", p)
        return 2
    df.to_csv(mdir / "manifest.csv", index=False)

    print("\n=== FILES PER MACHINE AND LABEL (compared with the paper's table) ===")
    mismatches = 0
    for (mtype, mid), g in df.groupby(["machine_type", "machine_id"]):
        n, a = int((g["label"] == 0).sum()), int((g["label"] == 1).sum())
        ref = PAPER_COUNTS.get((mtype, mid))
        note = "" if ref is None or ref == (n, a) else f"   <-- paper says normal={ref[0]}, abnormal={ref[1]}"
        mismatches += bool(note)
        print(f"  {mtype:<5} {mid}: normal={n:<5} abnormal={a:<4}{note}")
    print("\n=== ANOMALY SPLITS (rows = machine, columns = split) ===")
    print(df.groupby(["group", "split_ad"]).size().unstack(fill_value=0).to_string())
    print("\n=== CLASSIFIER SPLITS (fan vs pump, split by whole machine) ===")
    print(df.groupby(["split_cls", "machine_type"]).size().unstack(fill_value=0).to_string())

    summary = {"files_in_manifest": int(len(df)), "count_differences_from_paper": mismatches,
               "strategy": cfg["split"]["anomaly"]["strategy"], "seed": cfg["split"]["seed"]}
    with open(mdir / "manifest_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nAll leakage and label checks PASSED. Saved {mdir / 'manifest.csv'}")
    if mismatches:
        print(f"NOTE: {mismatches} machine(s) differ from the paper's counts. Check excluded.csv; send me this table.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
