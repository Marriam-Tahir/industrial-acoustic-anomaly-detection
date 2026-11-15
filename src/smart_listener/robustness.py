"""Phase 8: how the trained detectors cope with louder factory noise (MIMII 0 dB and -6 dB recordings).

The models were trained and calibrated on the 6 dB recordings only. MIMII also publishes the same machine recordings
mixed with more background noise (0 dB and -6 dB signal-to-noise ratio). Here the TEST clips of those versions are
scored with the existing models and the existing alert limits - nothing is retrained or re-tuned - and compared with
the same clips at 6 dB.

Clips are read straight from the downloaded zip files (or an extracted folder), so nothing needs to be unpacked.
Only clips that are in the TEST split of the Phase 1 manifest are used; training and validation clips are skipped.
"""
import io
import re
import zipfile
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import soundfile as sf

from smart_listener import anomaly, audio

PATTERN = re.compile(r"(fan|pump)/(id_\d\d)/(normal|abnormal)/([^/]+\.wav)$")


def snr_from_name(name: str) -> Optional[int]:
    m = re.search(r"(-?\d+)_dB_", Path(name).name)
    return int(m.group(1)) if m else None


def rel_key(name: str) -> Optional[str]:
    m = PATTERN.search(name.replace("\\", "/"))
    return "/".join(m.groups()) if m else None


def test_clips(manifest: pd.DataFrame, groups=None, split: str = "test") -> pd.DataFrame:
    m = manifest[(manifest["split_ad"] == split) & (manifest["exclude_reason"] == "")]
    if groups:
        m = m[m["group"].isin(groups)]
    return m.assign(key=m["path"].str.replace("\\", "/", regex=False))


def folder_counts(names) -> dict:
    """(type, id, label) -> number of wav files, to check that the noisy release has the same files."""
    out = {}
    for n in names:
        k = rel_key(n)
        if k:
            t, i, lab, _ = k.split("/")
            out[(t, i, lab)] = out.get((t, i, lab), 0) + 1
    return out


def manifest_counts(manifest: pd.DataFrame) -> dict:
    out = {}
    for (t, i, lab), g in manifest.groupby(["machine_type", "machine_id", "label_name"]):
        out[(t, i, lab)] = len(g)
    return out


def iter_source(source: Path, wanted: dict) -> Iterator:
    """Yield (manifest row, wav bytes) for every wanted clip found in a zip file or a folder."""
    source = Path(source)
    if source.is_dir():
        for p in source.rglob("*.wav"):
            k = rel_key(str(p.relative_to(source)))
            if k in wanted:
                yield wanted[k], p.read_bytes()
        return
    with zipfile.ZipFile(source) as z:
        for name in z.namelist():
            k = rel_key(name)
            if k in wanted:
                yield wanted[k], z.read(name)


def source_names(source: Path) -> list:
    source = Path(source)
    if source.is_dir():
        return [str(p.relative_to(source)) for p in source.rglob("*.wav")]
    with zipfile.ZipFile(source) as z:
        return z.namelist()


def load_clip(data: bytes, cfg: dict) -> np.ndarray:
    y, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    if sr != cfg["audio"]["sample_rate"]:
        raise ValueError(f"sample rate {sr} Hz, expected {cfg['audio']['sample_rate']} Hz")
    y = y[:, min(cfg["audio"]["channel_index"], y.shape[1] - 1)]
    y, _ = audio.fix_length(y, audio.expected_samples(cfg))
    return y


def score_rows(analyzer, items, cfg: dict, snr: int, progress=None) -> pd.DataFrame:
    rows = []
    for k, (row, data) in enumerate(items):
        y = load_clip(data, cfg) if isinstance(data, (bytes, bytearray)) else data
        r = analyzer.analyze_waveform(row["group"], y)
        rows.append({"snr_db": snr, "group": row["group"], "file_id": row["file_id"], "label": int(row["label"]),
                     "score": r.score, "z": r.z})
        if progress and (k + 1) % 100 == 0:
            progress(k + 1)
    return pd.DataFrame(rows)


def summarize(scores: pd.DataFrame, z_warn: float, z_alarm: float) -> pd.DataFrame:
    out = []
    for (snr, g), h in scores.groupby(["snr_db", "group"]):
        n, a = h[h["label"] == 0], h[h["label"] == 1]
        auc = anomaly.auc_metrics(h["label"], h["score"])[0] if h["label"].nunique() == 2 else np.nan
        out.append({"snr_db": snr, "group": g, "n_normal": len(n), "n_abnormal": len(a), "auc": auc,
                    "normal_above_warning_pct": 100 * float((n["z"] > z_warn).mean()) if len(n) else np.nan,
                    "normal_above_alarm_pct": 100 * float((n["z"] > z_alarm).mean()) if len(n) else np.nan,
                    "abnormal_above_alarm_pct": 100 * float((a["z"] > z_alarm).mean()) if len(a) else np.nan,
                    "normal_median_z": float(n["z"].median()) if len(n) else np.nan})
    df = pd.DataFrame(out).sort_values(["group", "snr_db"], ascending=[True, False])
    base = df[df["snr_db"] == 6].set_index("group")["auc"]
    df["auc_change_vs_6db"] = df.apply(lambda r: r["auc"] - base.get(r["group"], np.nan), axis=1)
    return df
