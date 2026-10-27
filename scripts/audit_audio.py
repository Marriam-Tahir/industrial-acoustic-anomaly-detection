"""Quality audit of every extracted WAV.  Run from the repository folder:

    python scripts/audit_audio.py

Writes data/manifests/audit.csv, audit_summary.json and one drift figure per machine in reports/figures/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from smart_listener import audit  # noqa: E402
from smart_listener.config import load_config, resolve_path  # noqa: E402
from smart_listener.manifest import scan_folder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit extracted MIMII WAV files")
    ap.add_argument("--mono-dir", help="folder with fan/ and pump/ (default from config)")
    args = ap.parse_args()

    cfg = load_config()
    root = Path(args.mono_dir) if args.mono_dir else resolve_path(cfg, "mono_dir")
    out_dir = resolve_path(cfg, "manifest_dir")
    fig_dir = resolve_path(cfg, "figures_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df, unexpected = scan_folder(root, cfg)
    if df.empty:
        print(f"No files found under {root}. Expected {root}/<fan|pump>/<id_xx>/<normal|abnormal>/*.wav")
        return 1
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected files were ignored, for example: {unexpected[:5]}")
    print(f"Auditing {len(df)} files under {root} ...")

    t0 = time.time()
    records = []
    for i, p in enumerate(df["path"], 1):
        records.append(audit.audit_file(root / p, cfg))
        if i % 500 == 0 or i == len(df):
            print(f"  {i}/{len(df)}  ({time.time() - t0:.0f} s)")
    res = audit.decide_exclusions(pd.concat([df, pd.DataFrame(records)], axis=1), cfg)
    res.to_csv(out_dir / "audit.csv", index=False)

    drift = audit.drift_table(res, cfg)
    for (mtype, mid), g in res[~res["exclude"] & res["level_db"].notna()].groupby(["machine_type", "machine_id"]):
        fig, ax = plt.subplots(figsize=(8, 3.2))
        for label, gl in g.groupby("label_name"):
            ax.scatter(gl["index_in_label"], gl["level_db"], s=6, label=label, alpha=0.7)
        ax.set_xlabel("file number within its label")
        ax.set_ylabel("mean log-mel level (dB)")
        ax.set_title(f"{mtype} {mid}: loudness against file number")
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / f"audit_level_{mtype}_{mid}.png", dpi=110)
        plt.close(fig)

    excl = res[res["exclude"]]
    summary = {
        "files_scanned": int(len(res)),
        "files_excluded": int(len(excl)),
        "excluded_by_reason": excl["exclude_reason"].str.split(":").str[0].value_counts().to_dict(),
        "warn_clipping_by_label": res[res["warn_clipping"]].groupby("label_name").size().to_dict(),
        "warn_silence_by_label": res[res["warn_silence"]].groupby("label_name").size().to_dict(),
        "drift": drift.to_dict(orient="records"),
    }
    with open(out_dir / "audit_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== AUDIT RESULT ===")
    print(f"Scanned {summary['files_scanned']} files, excluded {summary['files_excluded']}.")
    print("Excluded by reason:", summary["excluded_by_reason"] or "none")
    print("Warnings (kept in the data):  clipping:", summary["warn_clipping_by_label"] or "none",
          "| long silence:", summary["warn_silence_by_label"] or "none")
    if not drift.empty:
        print("\nLoudness vs file number (Spearman rho). A strong trend means recording order matters:")
        print(drift.round(3).to_string(index=False))
    print(f"\nSaved: {out_dir / 'audit.csv'}, {out_dir / 'audit_summary.json'}, figures in {fig_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
