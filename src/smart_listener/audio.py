"""Audio loading and log-mel features.

This is the ONLY place where feature maths lives. Training, evaluation and the web app must all call
these functions, so the same sound always becomes the same numbers.
"""
import numpy as np
import soundfile as sf
import librosa


def expected_samples(cfg: dict) -> int:
    return int(round(cfg["audio"]["sample_rate"] * cfg["audio"]["expected_duration_s"]))


def expected_frames(cfg: dict) -> int:
    """Number of spectrogram frames for one clip (librosa pads the edges, hence the +1)."""
    return 1 + expected_samples(cfg) // cfg["features"]["hop_length"]


def load_mono_wav(path, cfg: dict) -> np.ndarray:
    """Read a mono WAV as float32 in [-1, 1]. Refuses wrong formats instead of silently fixing them."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != cfg["audio"]["sample_rate"]:
        raise ValueError(f"{path}: sample rate is {sr}, expected {cfg['audio']['sample_rate']}")
    if y.shape[1] != 1:
        raise ValueError(f"{path}: has {y.shape[1]} channels, expected 1 (mono)")
    return y[:, 0]


def fix_length(y: np.ndarray, n: int):
    """Pad with zeros or trim to exactly n samples. Returns (signal, 'ok' | 'padded' | 'trimmed')."""
    if len(y) == n:
        return y, "ok"
    if len(y) > n:
        return y[:n], "trimmed"
    return np.pad(y, (0, n - len(y))), "padded"


def log_mel(y: np.ndarray, cfg: dict) -> np.ndarray:
    """Log-mel spectrogram in absolute decibels, shape (n_mels, frames), float32.

    Deliberately NO per-clip scaling: top_db=None and ref=1.0 keep the true loudness, because a louder
    or quieter machine can itself be a sign of a problem."""
    f = cfg["features"]
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg["audio"]["sample_rate"], n_fft=f["n_fft"], hop_length=f["hop_length"],
        n_mels=f["n_mels"], fmin=f["fmin"], fmax=f["fmax"], power=f["power"], center=True)
    return librosa.power_to_db(mel, ref=1.0, amin=f["amin"], top_db=None).astype(np.float32)


class RunningStats:
    """Mean and standard deviation per mel band, computed file by file (so it fits in 8 GB of RAM)."""

    def __init__(self, n_mels: int):
        self.n_frames = 0
        self.n_files = 0
        self._sum = np.zeros(n_mels, dtype=np.float64)
        self._sumsq = np.zeros(n_mels, dtype=np.float64)

    def update(self, spec: np.ndarray) -> None:
        x = spec.astype(np.float64)
        self.n_frames += x.shape[1]
        self.n_files += 1
        self._sum += x.sum(axis=1)
        self._sumsq += (x ** 2).sum(axis=1)

    def finalize(self, min_std: float = 1e-6):
        if self.n_frames == 0:
            raise ValueError("No data was given to RunningStats")
        mean = self._sum / self.n_frames
        var = np.maximum(self._sumsq / self.n_frames - mean ** 2, 0.0)
        return mean, np.maximum(np.sqrt(var), min_std)


def apply_norm(spec: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardise each mel band with statistics that were fitted on TRAINING data only."""
    return ((spec - mean[:, None]) / std[:, None]).astype(np.float32)


def make_windows(spec: np.ndarray, window_frames: int, hop_frames: int) -> np.ndarray:
    """Cut one spectrogram (n_mels, frames) into overlapping windows -> (n_windows, n_mels, window_frames)."""
    n_frames = spec.shape[1]
    if n_frames < window_frames:
        raise ValueError(f"spectrogram has {n_frames} frames, fewer than the window size {window_frames}")
    starts = range(0, n_frames - window_frames + 1, hop_frames)
    return np.stack([spec[:, s:s + window_frames] for s in starts])
