"""Phase 5: score audio with the trained models (the same maths as Phase 2 training and evaluation).

Two parts:
  prepare_upload(...)  turns an uploaded WAV file into 10-second windows the models can score, with warnings.
  Analyzer             loads a machine's checkpoint once, scores a 10-second window, and converts the score to
                       the calibrated z-value and clip level used by the severity engine (Phase 3).

Scoring is NOT re-implemented here. It reuses audio.log_mel, anomaly.clip_inputs and the saved model, so a clip
gets the same score it would get in evaluate_ad.py (scripts/verify_inference.py checks that on real clips).

torch is imported only when a model is first needed, so importing this module (for example to prepare a file)
stays cheap and the API can start without loading any model.
"""
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from smart_listener import audio, severity
from smart_listener.config import config_fingerprint, project_root

MIN_SECONDS_DEFAULT = 9.5     # shortest clip accepted: inside the dataset's duration tolerance (10 s +/- 0.5 s)
SILENT_RMS = 1e-4             # whole-clip RMS below this (full scale = 1.0) counts as silence


class AudioRejected(ValueError):
    """The uploaded file cannot be analysed. The message is written for the person who uploaded it."""


@dataclass
class PreparedAudio:
    windows: list                                   # list of float32 arrays, each exactly expected_samples long
    offsets_s: list                                 # start time of each window inside the file (seconds)
    warnings: list = field(default_factory=list)
    info: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------- upload preparation
def filename_hint(name: str) -> Optional[dict]:
    """Look for a machine type / id in a file name such as 'fan_id_06_00000123.wav' or 'pump-04.wav'.

    This is ONLY a convenience check against the machine the admin selected. It is not detection: the sound is never
    used to work out which machine it came from. Returns {'machine_type': ..., 'machine_id': ...} or None."""
    import re
    s = Path(name).name.lower()
    m = re.search(r"(fan|pump)[^a-z0-9]*(?:id)?[^a-z0-9]*(\d{2})(?!\d)", s)
    if not m:
        return None
    return {"machine_type": m.group(1), "machine_id": f"id_{m.group(2)}"}


def check_hint(hint: Optional[dict], model_group: str) -> Optional[str]:
    """A warning text if the file name names a different machine than the registered one, otherwise None."""
    if not hint:
        return None
    t, i = model_group.split("_", 1)
    if hint["machine_type"] != t or hint["machine_id"] != i:
        return (f"The file name looks like {hint['machine_type']} {hint['machine_id']}, but you selected "
                f"{model_group}. The file was analysed with the model of the machine you selected; check that "
                f"this is the right file.")
    return None


def prepare_upload(path, cfg: dict, max_seconds: float = 600.0, min_seconds: float = MIN_SECONDS_DEFAULT) -> PreparedAudio:
    """Read a WAV file and cut it into consecutive 10-second windows (no overlap).

    - A different sample rate is converted to the model's rate (a warning says so).
    - A multi-channel file uses channel `audio.channel_index` only, as in training (a warning says so).
    - A final piece shorter than the clip length is kept only if it is at least `min_seconds` long; it is then padded
      with silence like in Phase 1. Shorter leftovers are dropped (a warning says so).
    - Files shorter than `min_seconds` in total, unreadable files, silent files and files longer than `max_seconds`
      are refused.
    """
    p = Path(path)
    try:
        info = sf.info(str(p))
    except Exception as e:  # soundfile raises its own error type for non-audio files
        raise AudioRejected("This file could not be read as audio. Please upload a WAV file.") from e
    a = cfg["audio"]
    target_sr, n = a["sample_rate"], audio.expected_samples(cfg)
    warns = []
    if info.duration > max_seconds + 1e-6:
        raise AudioRejected(f"The file is {info.duration:.0f} s long; the limit is {max_seconds:.0f} s. Please upload a shorter file.")
    try:
        y, sr = sf.read(str(p), dtype="float32", always_2d=True)
    except Exception as e:
        raise AudioRejected("This file could not be read as audio. Please upload a WAV file.") from e
    if y.shape[1] > 1:
        ch = a["channel_index"]
        if ch >= y.shape[1]:
            raise AudioRejected(f"The file has {y.shape[1]} channels but channel {ch + 1} is needed.")
        warns.append(f"The file has {y.shape[1]} channels; only channel {ch + 1} was analysed (as in training).")
        y = y[:, ch]
    else:
        y = y[:, 0]
    if sr != target_sr:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        warns.append(f"The file's sample rate is {sr} Hz; it was converted to {target_sr} Hz. "
                     f"Scores for converted files are less reliable than for original recordings.")
    if not np.isfinite(y).all():
        raise AudioRejected("The file contains invalid sample values.")
    dur = len(y) / target_sr
    if dur < min_seconds:
        raise AudioRejected(f"The file is only {dur:.1f} s long. At least {min_seconds:.1f} s is needed "
                            f"(the models work on {a['expected_duration_s']:.0f}-second clips).")
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms < SILENT_RMS:
        raise AudioRejected("The recording is silent (or almost silent), so it cannot be analysed.")
    clipped = float(np.mean(np.abs(y) >= cfg["quality"]["clip_level"]))
    if clipped > cfg["quality"]["clip_fraction_warn"]:
        warns.append(f"{clipped * 100:.2f}% of the samples are clipped (too loud for the recording). "
                     f"Scores may be affected.")

    windows, offsets = [], []
    start = 0
    while start + n <= len(y):
        windows.append(np.ascontiguousarray(y[start:start + n]))
        offsets.append(start / target_sr)
        start += n
    rest = len(y) - start
    if rest > 0:
        if rest / target_sr >= min_seconds:
            piece, _ = audio.fix_length(y[start:], n)
            windows.append(np.ascontiguousarray(piece))
            offsets.append(start / target_sr)
            warns.append(f"The last {rest / target_sr:.1f} s were padded with silence to make a full window.")
        else:
            warns.append(f"The last {rest / target_sr:.1f} s were too short for a window and were not analysed.")
    return PreparedAudio(windows, offsets, warns,
                         {"duration_s": round(dur, 2), "source_sample_rate": int(sr), "n_windows": len(windows),
                          "rms": rms, "peak": float(np.max(np.abs(y)))})


# ----------------------------------------------------------------------------- scoring
@dataclass
class WindowResult:
    group: str
    model_kind: str
    score: float
    z: float
    z_warn: float
    z_alarm: float
    clip_level: str

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def clip_level(z: float, z_warn: float, z_alarm: float) -> str:
    """Where ONE window sits against the limits. (The state NORMAL/WATCH/WARNING/HIGH needs several windows.)"""
    if z > z_alarm:
        return "above_alarm_limit"
    if z > z_warn:
        return "above_warning_limit"
    return "within_normal"


class Analyzer:
    """Scores 10-second windows for any machine group that has a checkpoint. Thread-safe: models load once, and
    scoring is serialised with a lock, so several monitoring threads and uploads can share one Analyzer."""

    def __init__(self, cfg: dict, models_dir=None, calibration: Optional[dict] = None, device: str = "cpu"):
        self.cfg = cfg
        self.models_dir = Path(models_dir) if models_dir else project_root() / "models" / "ad"
        self.calibration = calibration if calibration is not None else severity.load_calibration()
        self.device = device
        self._models = {}
        self._lock = threading.RLock()

    # ---- what is available
    @property
    def thresholds(self) -> dict:
        t = self.calibration["thresholds"]
        return {"z_warn": float(t["z_warn"]), "z_alarm": float(t["z_alarm"])}

    @property
    def rule(self) -> dict:
        return dict(self.calibration.get("rule", severity.DEFAULT_RULE))

    def checkpoint_path(self, group: str) -> Path:
        kind = self.calibration["machines"][group]["model"]
        return self.models_dir / kind / f"{group}.pt"

    def available(self, group: str) -> tuple:
        """(True, '') or (False, reason). Does not load torch."""
        if group not in self.calibration.get("machines", {}):
            return False, f"no calibration for {group}"
        p = self.checkpoint_path(group)
        if not p.exists():
            return False, (f"the trained model file {p.name} is missing at {p.parent}. "
                           f"Train it (scripts/train_ad.py) or copy it from Colab.")
        return True, ""

    # ---- model access
    def load(self, group: str):
        """Return (model, kind, norm_mean, norm_std) for a group, loading it on first use."""
        with self._lock:
            if group in self._models:
                return self._models[group]
            ok, why = self.available(group)
            if not ok:
                raise FileNotFoundError(why)
            from smart_listener import anomaly
            path = self.checkpoint_path(group)
            model, ck = anomaly.load_checkpoint(path, self.cfg, self.device)
            expect = self.calibration["machines"][group]["model"]
            if ck["kind"] != expect:
                raise ValueError(f"{path.name} holds a {ck['kind']} model but the calibration expects {expect}")
            if ck["group"] != group:
                raise ValueError(f"{path.name} was trained for {ck['group']}, not {group}")
            self._models[group] = (model, ck["kind"], np.asarray(ck["norm_mean"], np.float32),
                                   np.asarray(ck["norm_std"], np.float32))
            return self._models[group]

    def spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Log-mel spectrogram of one 10-second window, exactly as Phase 1 feature extraction makes it."""
        n = audio.expected_samples(self.cfg)
        if len(y) != n:
            raise ValueError(f"a window must be exactly {n} samples, got {len(y)}")
        return audio.log_mel(np.asarray(y, dtype=np.float32), self.cfg)

    def score_spectrogram(self, group: str, spec: np.ndarray) -> float:
        import torch
        from smart_listener import anomaly
        model, kind, mean, std = self.load(group)
        x = anomaly.clip_inputs(spec, mean, std, kind, self.cfg)
        with self._lock, torch.no_grad():
            X = torch.from_numpy(x).to(self.device)
            err = ((model(X) - X) ** 2).flatten(1).mean(1).cpu().numpy()
        return float(err.mean())            # the "mean" score of Phase 2 (score_mean)

    def analyze_waveform(self, group: str, y: np.ndarray) -> WindowResult:
        spec = self.spectrogram(y)
        return self.result_from_score(group, self.score_spectrogram(group, spec))

    def result_from_score(self, group: str, score: float) -> WindowResult:
        m = self.calibration["machines"][group]
        z = float(severity.to_z(score, m["center"], m["scale"]))
        t = self.thresholds
        return WindowResult(group, m["model"], float(score), z, t["z_warn"], t["z_alarm"],
                            clip_level(z, t["z_warn"], t["z_alarm"]))

    def fingerprint_ok(self) -> bool:
        return True if not self.calibration.get("config_fingerprint") else \
            self.calibration["config_fingerprint"] == config_fingerprint(self.cfg)
