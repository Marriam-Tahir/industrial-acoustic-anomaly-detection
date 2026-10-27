"""Does validation/test sound like training? Compares the average loudness of each split with training.

    python scripts/split_shift_report.py

Needs data/manifests/manifest.csv (from build_manifest.py). A shift above 0.5 means the split is measuring
'different recording conditions', not just 'normal vs abnormal'. No labels are used to tune anything.
"""
import sys

import pandas as pd

from smart_listener.config import load_config, resolve_path


def main() -> int:
    cfg = load_config()
    mdir = resolve_path(cfg, "manifest_dir")
    m = pd.read_csv(mdir / "manifest.csv", keep_default_na=False)
    m["level_db"] = pd.to_numeric(m["level_db"], errors="coerce")
    rows = []
    for group, g in m.groupby("group"):
        tr = g[(g.split_ad == "train")]["level_db"]
        row = {"group": group, "train_mean_db": round(tr.mean(), 2), "train_std_db": round(tr.std(), 2)}
        for split in ("val", "test"):
            for lab, name in ((0, "normal"), (1, "abnormal")):
                x = g[(g.split_ad == split) & (g.label == lab)]["level_db"]
                row[f"{split}_{name}_shift"] = round((x.mean() - tr.mean()) / tr.std(), 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(mdir / "split_shift.csv", index=False)
    print(f"Strategy: {cfg['split']['anomaly']['strategy']}")
    print("Shift = (split mean - train mean) / train std, in dB units of training spread.\n")
    print(out.to_string(index=False))
    worst = out[["val_normal_shift", "test_normal_shift"]].abs().to_numpy().max()
    print(f"\nLargest NORMAL-file shift: {worst:.2f}  ->", "OK (below 0.5)" if worst < 0.5 else "HIGH: normal test files differ from training")
    return 0


if __name__ == "__main__":
    sys.exit(main())
