"""Train normal-only autoencoders, one per machine ID, and score every clip.

    python scripts/train_ad.py --smoke                    # 1 machine, 2 epochs, writes to *_smoke folders
    python scripts/train_ad.py                            # all 8 machines, both models (use Colab for this)
    python scripts/train_ad.py --models conv --groups fan_id_00 pump_id_04
    python scripts/train_ad.py --force                    # retrain even if a checkpoint already exists

Writes:
    models/ad/<dense|conv>/<group>.pt                 checkpoint (weights + normalisation + settings)
    reports/phase2/scores_<model>_<group>.csv         one anomaly score per clip (train / val / test)
    reports/phase2/train_log_<model>_<group>.json     loss history and settings
"""
import argparse
import json
import sys
import time

import pandas as pd
import torch

from smart_listener import anomaly
from smart_listener.config import load_config, project_root

DEFAULTS = {"dense": {"batch": 512, "epochs": 40}, "conv": {"batch": 64, "epochs": 60}}


def main() -> int:
    ap = argparse.ArgumentParser(description="Train normal-only autoencoders")
    ap.add_argument("--groups", nargs="*", help="e.g. fan_id_00 pump_id_04 (default: all 8)")
    ap.add_argument("--models", nargs="*", default=list(anomaly.KINDS), choices=anomaly.KINDS)
    ap.add_argument("--epochs", type=int, help="override the maximum number of epochs")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="quick test: first machine, 2 epochs, 100 training clips")
    args = ap.parse_args()

    cfg = load_config()
    root = project_root()
    suffix = "_smoke" if args.smoke else ""
    models_dir = root / "models" / f"ad{suffix}"
    rep_dir = root / "reports" / f"phase2{suffix}"
    rep_dir.mkdir(parents=True, exist_ok=True)
    groups = args.groups or anomaly.list_groups(cfg)
    if args.smoke:
        groups = groups[:1]
    device = anomaly.pick_device(args.device)
    seed = cfg["split"]["seed"]
    print(f"device: {device} | groups: {groups} | models: {args.models} | outputs: {models_dir.relative_to(root)}")

    t_all = time.time()
    for group in groups:
        data = anomaly.load_group(cfg, group)
        rows = data["rows"]
        tr_idx = anomaly.training_rows(rows)                      # normal training clips only
        va_idx = anomaly.select_rows(rows, "val", normal_only=True)   # label-free early stopping
        if args.smoke:
            tr_idx = tr_idx[:100]
        for kind in args.models:
            ck_path = models_dir / kind / f"{group}.pt"
            if ck_path.exists() and not args.force:
                print(f"[skip] {group} {kind}: checkpoint exists (use --force to retrain)")
                continue
            t0 = time.time()
            print(f"[train] {group} {kind}: {len(tr_idx)} normal training clips, {len(va_idx)} normal validation clips")
            X_tr = anomaly.build_inputs(data, tr_idx, kind, cfg)
            X_va = anomaly.build_inputs(data, va_idx, kind, cfg)
            epochs = 2 if args.smoke else (args.epochs or DEFAULTS[kind]["epochs"])
            s = anomaly.group_seed(seed, group, kind)
            model = anomaly.make_model(kind, cfg)
            model, history, best_epoch = anomaly.fit_autoencoder(
                model, X_tr, X_va, epochs, DEFAULTS[kind]["batch"], args.lr, args.patience, device, s)
            del X_tr, X_va

            # score every train / val / test clip (buffer clips are never used)
            keep = rows[rows["split_ad"].isin(["train", "val", "test"])]
            scores = anomaly.score_rows(model, data, keep.index.to_numpy(), kind, cfg, device)
            out = pd.DataFrame({"group": group, "model": kind, "file_id": keep["file_id"].values,
                                "split": keep["split_ad"].values, "label": keep["label"].values,
                                "score_mean": scores[:, 0], "score_max": scores[:, 1]})
            out.to_csv(rep_dir / f"scores_{kind}_{group}.csv", index=False)

            settings = {"epochs_max": epochs, "batch": DEFAULTS[kind]["batch"], "lr": args.lr, "patience": args.patience,
                        "seed": s, "n_train_clips": int(len(tr_idx)), "n_val_normal_clips": int(len(va_idx)),
                        "best_epoch": best_epoch, "n_parameters": sum(p.numel() for p in model.parameters()),
                        "seconds": round(time.time() - t0, 1), "device": str(device),
                        "torch": torch.__version__, "config_fingerprint": data["fingerprint"]}
            anomaly.save_checkpoint(ck_path, model, kind, data, history, best_epoch, s, {"settings": settings})
            (rep_dir / f"train_log_{kind}_{group}.json").write_text(
                json.dumps({"settings": settings, "history": history}, indent=2), encoding="utf-8")
            print(f"[done] {group} {kind}: best epoch {best_epoch}, {settings['seconds']} s, checkpoint {ck_path.relative_to(root)}")
    print(f"\nTraining finished in {(time.time() - t_all) / 60:.1f} min. Next: python scripts/evaluate_ad.py"
          + (" --smoke" if args.smoke else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
