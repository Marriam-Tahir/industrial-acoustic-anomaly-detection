"""Quality checks for the extracted mono WAV files.

EXCLUDED (technical corruption only): unreadable, wrong sample rate, not mono, wrong length, completely
silent, exact duplicates (only the first copy is kept), and label conflicts (same audio, different label).
ONLY WARNED (never excluded): clipping, long silent stretches, unusual loudness. These can be genuine
anomalies, and removing them would make the abnormal test data artificially easy.
"""
import hashlib

import numpy as np
import pandas as pd
import soundfile as sf

from smart_listener import audio


def audit_file(path, cfg: dict) -> dict:
    """Measure one file. Never raises: problems are recorded in the result."""
    q = cfg["quality"]
    rec = {"readable": False, "sr": np.nan, "channels": np.nan, "frames": np.nan, "duration_s": np.nan,
           "peak": np.nan, "rms": np.nan, "dc_offset": np.nan, "clip_fraction": np.nan,
           "silence_fraction": np.nan, "level_db": np.nan, "sha256": "", "error": ""}
    try:
        info = sf.info(str(path))
        rec.update(sr=info.samplerate, channels=info.channels, frames=info.frames,
                   duration_s=round(info.frames / info.samplerate, 4))
        data, _ = sf.read(str(path), dtype="int16", always_2d=True)
    except Exception as exc:  # noqa: BLE001 - any read problem is a finding, not a crash
        rec["error"] = repr(exc)
        return rec
    rec["readable"] = True
    rec["sha256"] = hashlib.sha256(data.tobytes()).hexdigest()
    if data.shape[0] == 0:
        return rec
    x = data[:, 0].astype(np.float32) / 32768.0
    rec["peak"] = float(np.abs(x).max())
    rec["rms"] = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    rec["dc_offset"] = float(x.mean())
    rec["clip_fraction"] = float(np.mean(np.abs(x) >= q["clip_level"]))
    block = max(1, int(cfg["audio"]["sample_rate"] * q["silence_block_ms"] / 1000))
    n_blocks = len(x) // block
    if n_blocks:
        b = x[:n_blocks * block].reshape(n_blocks, block)
        rec["silence_fraction"] = float(np.mean(np.sqrt((b.astype(np.float64) ** 2).mean(axis=1)) < q["silence_block_rms"]))
    if info.samplerate == cfg["audio"]["sample_rate"] and len(x) >= cfg["features"]["n_fft"]:
        rec["level_db"] = float(audio.log_mel(x, cfg).mean())
    return rec


def decide_exclusions(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add columns exclude (bool), exclude_reason, and warning flags. df must be in manifest order."""
    a, q = cfg["audio"], cfg["quality"]
    out = df.copy()
    reason = pd.Series("", index=out.index, dtype=object)

    def mark(mask, text):
        nonlocal reason
        mask = mask & (reason == "")
        reason = reason.mask(mask, text)

    mark(~out["readable"].astype(bool), "unreadable")
    mark(out["sr"] != a["sample_rate"], "wrong_sample_rate")
    mark(out["channels"] != 1, "not_mono")
    mark((out["duration_s"] - a["expected_duration_s"]).abs() > a["duration_tolerance_s"], "wrong_duration")
    mark(out["peak"] == 0, "silent")

    ok = out[reason == ""]
    for sha, g in ok[ok["sha256"] != ""].groupby("sha256"):
        if len(g) < 2:
            continue
        if g["label"].nunique() > 1:
            reason.loc[g.index] = "label_conflict"
        else:
            keep = g.index[0]
            reason.loc[g.index[1:]] = f"duplicate_of:{out.loc[keep, 'file_id']}"

    out["exclude_reason"] = reason
    out["exclude"] = reason != ""
    out["warn_clipping"] = out["clip_fraction"] > q["clip_fraction_warn"]
    out["warn_silence"] = out["silence_fraction"] > q["silence_fraction_warn"]
    return out


def drift_table(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Does loudness change with the file number? A strong trend means neighbouring recordings are
    similar, which is exactly why we split in blocks with gaps."""
    from scipy.stats import spearmanr

    q = cfg["quality"]
    rows = []
    kept = df[~df["exclude"] & df["level_db"].notna()]
    for (mtype, mid, label), g in kept.groupby(["machine_type", "machine_id", "label_name"]):
        if len(g) < q["drift_min_files"]:
            continue
        rho, p = spearmanr(g["index_in_label"], g["level_db"])
        rows.append({"machine_type": mtype, "machine_id": mid, "label_name": label, "n_files": len(g),
                     "rho": float(rho), "p_value": float(p), "strong_trend": bool(abs(rho) >= q["drift_rho_warn"])})
    return pd.DataFrame(rows)
