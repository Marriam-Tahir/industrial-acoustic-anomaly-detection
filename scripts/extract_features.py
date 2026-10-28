"""Turn every manifest file into a log-mel spectrogram and fit normalisation on TRAINING files only.

    python scripts/extract_features.py            # skips machines that are already done
    python scripts/extract_features.py --force    # rebuild everything

Writes, under data/processed/:
    logmel/<type>_<id>.npy          float32 array (n_files, 64, 313), NOT normalised
    logmel/<type>_<id>_files.csv    which file_id is in which row
    norm/<type>_<id>_anomaly.json   mean/std fitted on that machine's anomaly-TRAINING files
    norm/classifier.json            mean/std fitted on classifier-TRAINING files (both machine types)
    features_meta.json              settings fingerprint and library versions
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile

from smart_listener import audio
from smart_listener.config import config_fingerprint, load_config, resolve_path


def _save_stats(path: Path, stats: audio.RunningStats, fitted_on: str, file_ids, cfg: dict) -> None:
    mean, std = stats.finalize()
    payload = {"fitted_on": fitted_on, "n_files": stats.n_files, "n_frames": stats.n_frames,
               "n_mels": len(mean), "mean": mean.tolist(), "std": std.tolist(),
               "config_fingerprint": config_fingerprint(cfg),
               "file_ids_sha256": __import__("hashlib").sha256("\n".join(sorted(file_ids)).encode()).hexdigest()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract log-mel features")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--mono-dir")
    args = ap.parse_args()

    cfg = load_config()
    root = Path(args.mono_dir) if args.mono_dir else resolve_path(cfg, "mono_dir")
    mpath = resolve_path(cfg, "manifest_dir") / "manifest.csv"
    if not mpath.exists():
        print(f"{mpath} not found. Run scripts/build_manifest.py first.")
        return 1
    manifest = pd.read_csv(mpath, keep_default_na=False)
    out = resolve_path(cfg, "features_dir")
    (out / "logmel").mkdir(parents=True, exist_ok=True)

    n_mels, n_frames, n_samples = cfg["features"]["n_mels"], audio.expected_frames(cfg), audio.expected_samples(cfg)
    fingerprint = config_fingerprint(cfg)
    fixed = {"padded": 0, "trimmed": 0}
    t0 = time.time()

    for group, g in manifest.groupby("group", sort=True):
        g = g.sort_values(["label_name", "index_in_label"]).reset_index(drop=True)
        npy = out / "logmel" / f"{group}.npy"
        fcsv = out / "logmel" / f"{group}_files.csv"
        meta_path = out / "features_meta.json"
        cache_valid = False
        if meta_path.exists():
             with open(meta_path, encoding="utf-8") as f:
                  old_meta = json.load(f)
                  cache_valid = old_meta.get("config_fingerprint") == fingerprint
                  if npy.exists() and fcsv.exists() and not args.force and cache_valid:
                      old = pd.read_csv(fcsv)
                      if list(old["file_id"]) == list(g["file_id"]):
                          print(f"[skip] {group}: already done ({len(g)} files)")
                          continue
        arr = np.zeros((len(g), n_mels, n_frames), dtype=np.float32)
        for i, p in enumerate(g["path"]):
            y = audio.load_mono_wav(root / p, cfg)
            y, how = audio.fix_length(y, n_samples)
            if how != "ok":
                fixed[how] += 1
            spec = audio.log_mel(y, cfg)
            if spec.shape != (n_mels, n_frames):
                raise RuntimeError(f"{p}: unexpected spectrogram shape {spec.shape}")
            arr[i] = spec
        if not np.isfinite(arr).all():
            raise RuntimeError(f"{group}: NaN or infinite values in features")
        np.save(npy, arr)
        g[["file_id"]].to_csv(fcsv, index=False)
        print(f"[done] {group}: {len(g)} files, shape {arr.shape}, range {arr.min():.1f} to {arr.max():.1f} dB "
              f"({time.time() - t0:.0f} s)")

    # ---- normalisation statistics: TRAINING files only ----
    for group, g in manifest.groupby("group", sort=True):
        g = g.sort_values(["label_name", "index_in_label"]).reset_index(drop=True)
        arr = np.load(out / "logmel" / f"{group}.npy", mmap_mode="r")
        train = g[g["split_ad"] == "train"]
        assert (train["label"] == 0).all(), "anomaly training split must contain normal files only"
        st = audio.RunningStats(n_mels)
        for i in train.index:
            st.update(arr[i])
        _save_stats(out / "norm" / f"{group}_anomaly.json", st, "split_ad == train", train["file_id"], cfg)
    st = audio.RunningStats(n_mels)
    cls_ids = []
    for group, g in manifest.groupby("group", sort=True):
        g = g.sort_values(["label_name", "index_in_label"]).reset_index(drop=True)
        arr = np.load(out / "logmel" / f"{group}.npy", mmap_mode="r")
        for i in g.index[g["split_cls"] == "train"]:
            st.update(arr[i])
            cls_ids.append(g.loc[i, "file_id"])
    if st.n_files:
        _save_stats(out / "norm" / "classifier.json", st, "split_cls == train", cls_ids, cfg)

    meta = {"created_utc": datetime.now(timezone.utc).isoformat(), "config_fingerprint": fingerprint,
            "features": cfg["features"], "audio": cfg["audio"],
            "versions": {"numpy": np.__version__, "librosa": librosa.__version__, "soundfile": soundfile.__version__},
            "files_padded": fixed["padded"], "files_trimmed": fixed["trimmed"]}
    with open(out / "features_meta.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\nFeatures done. Config fingerprint: {fingerprint}. Padded: {fixed['padded']}, trimmed: {fixed['trimmed']}.")
    print(f"Normalisation statistics saved in {out / 'norm'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
