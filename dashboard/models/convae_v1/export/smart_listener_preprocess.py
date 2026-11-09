"""
smart_listener_preprocess.py
============================
Single source of truth for Smart Listener audio preprocessing.

The Member-1 notebook uses this exact module to build the dataset, and the
API / Streamlit code must use it to preprocess new audio, so training data and
live data go through identical code.

Pipeline (order is fixed, parameters come from preprocessing_config.json):
  wav -> mono mix-down (mean of channels) -> resample to 16 kHz if needed
      -> peak-normalise to [-1, 1] -> fix to 10 s (trim / zero-pad)
      -> 19 windows of 1 s, hop 0.5 s
      -> power mel spectrogram (n_fft=1024, hop=512, n_mels=64, 0-8000 Hz)
      -> log(mel + 1e-6) -> per-window min-max to [0, 1] -> float32
Output shape: (19, 64, 32), dtype float32.

Only numpy and scipy are required (no librosa / soundfile), so the result does
not depend on the version of an audio library. The mel filterbank follows the
Slaney convention (same as librosa's default) and the STFT uses a periodic Hann
window with zero padding of n_fft//2 on both sides ("centered" frames).

Usage:
    from smart_listener_preprocess import Preprocessor
    pre = Preprocessor.from_json("preprocessing_config.json")
    x = pre.preprocess_file("some_clip.wav")        # (19, 64, 32) float32
"""
import hashlib
import io
import json
import math
import warnings
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import get_window, resample_poly

# Parameters that define the dataset. Changing any of them makes a different
# dataset, so they are hashed into a fingerprint stored in the config.
FROZEN_PARAMS = {
    "sample_rate": 16000,
    "mono_mixdown": "mean_of_channels",
    "peak_normalize": True,
    "clip_seconds": 10.0,
    "window_seconds": 1.0,
    "hop_seconds": 0.5,
    "n_windows": 19,
    "n_fft": 1024,
    "hop_length": 512,
    "n_mels": 64,
    "fmin": 0.0,
    "fmax": 8000.0,
    "power": 2.0,
    "log_eps": 1e-6,
    "window_scaling": "per_window_min_max_0_1",
    "stft_window": "hann_periodic",
    "stft_center": True,
    "stft_pad_mode": "constant_zeros",
    "mel_scale": "slaney",
    "mel_norm": "slaney",
    "output_dtype": "float32",
    "output_shape": [19, 64, 32],
    "implementation": "numpy-scipy-v1",
}


def params_fingerprint(params):
    """Short stable hash of the preprocessing parameters."""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def build_config(splits=None, extra=None):
    """Config dict as saved to preprocessing_config.json."""
    cfg = {
        "schema_version": 1,
        "fingerprint": params_fingerprint(FROZEN_PARAMS),
        "preprocessing": dict(FROZEN_PARAMS),
    }
    if splits is not None:
        cfg["splits"] = splits
    if extra:
        cfg.update(extra)
    return cfg


# ----------------------------------------------------------------------------
# WAV reading
# ----------------------------------------------------------------------------
def read_wav(source):
    """Read a wav from bytes or a path.

    Returns (samples, sample_rate) with samples float32 in [-1, 1] and shape
    (n_samples, n_channels). Supports 8/16/32-bit PCM and 32/64-bit float.
    Raises ValueError for anything that cannot be parsed.
    """
    if isinstance(source, (bytes, bytearray)):
        fh = io.BytesIO(source)
    else:
        fh = open(source, "rb")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # unknown RIFF chunks etc.
            sr, data = wavfile.read(fh)
    except Exception as first_error:
        # rare encodings scipy cannot parse (e.g. 24-bit): fall back to soundfile if it is installed
        try:
            import soundfile as sf
            fh.seek(0)
            data, sr = sf.read(fh, dtype="float32", always_2d=True)
        except Exception:
            raise ValueError(f"cannot read wav: {first_error}") from None
    finally:
        fh.close()
    if data.ndim == 1:
        data = data[:, None]
    kind = data.dtype
    if kind == np.int16:
        x = data.astype(np.float32) / 32768.0
    elif kind == np.int32:
        x = (data.astype(np.float64) / 2147483648.0).astype(np.float32)
    elif kind == np.uint8:
        x = (data.astype(np.float32) - 128.0) / 128.0
    elif kind in (np.float32, np.float64):
        x = data.astype(np.float32)
    else:
        raise ValueError(f"unsupported wav sample type {kind}")
    return x, int(sr)


def wav_stats(samples, sr):
    """(n_channels, n_samples, duration_s, peak) of a decoded wav."""
    n = samples.shape[0]
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    return samples.shape[1], n, n / sr if sr else 0.0, peak


# ----------------------------------------------------------------------------
# Mel filterbank (Slaney scale and normalisation)
# ----------------------------------------------------------------------------
def _hz_to_mel(f):
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mels = f / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(f >= min_log_hz,
                    min_log_mel + np.log(np.maximum(f, 1e-12) / min_log_hz) / logstep,
                    mels)


def _mel_to_hz(m):
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    freqs = f_sp * m
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    return np.where(m >= min_log_mel,
                    min_log_hz * np.exp(logstep * (m - min_log_mel)),
                    freqs)


def mel_edges_hz(sr, n_mels, fmin, fmax):
    return _mel_to_hz(np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2))


def mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    """(n_mels, 1 + n_fft//2) triangular filters, Slaney area-normalised."""
    mel_f = mel_edges_hz(sr, n_mels, fmin, fmax)
    fft_f = np.linspace(0.0, sr / 2.0, 1 + n_fft // 2)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fft_f)
    weights = np.zeros((n_mels, 1 + n_fft // 2))
    for i in range(n_mels):
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0.0, np.minimum(lower, upper))
    enorm = 2.0 / (mel_f[2:n_mels + 2] - mel_f[:n_mels])
    return weights * enorm[:, None]


# ----------------------------------------------------------------------------
class Preprocessor:
    """Builds the (19, 64, 32) float32 array for one recording."""

    def __init__(self, config):
        # accept the whole config dict or just the "preprocessing" part
        p = config.get("preprocessing", config)
        self.params = dict(p)
        self.sr = int(p["sample_rate"])
        self.clip_len = int(round(p["clip_seconds"] * self.sr))
        self.win_len = int(round(p["window_seconds"] * self.sr))
        self.win_hop = int(round(p["hop_seconds"] * self.sr))
        self.n_windows = int(p["n_windows"])
        expected = (self.clip_len - self.win_len) // self.win_hop + 1
        if expected != self.n_windows:
            raise ValueError(f"config n_windows={self.n_windows} but the clip/window/hop "
                             f"settings give {expected}")
        self.n_fft = int(p["n_fft"])
        self.hop = int(p["hop_length"])
        self.n_mels = int(p["n_mels"])
        self.power = float(p["power"])
        self.eps = float(p["log_eps"])
        self.n_frames = 1 + self.win_len // self.hop
        self.shape = (self.n_windows, self.n_mels, self.n_frames)
        if list(self.shape) != list(p["output_shape"]):
            raise ValueError(f"computed output shape {self.shape} != config "
                             f"output_shape {p['output_shape']}")
        self.fingerprint = params_fingerprint(self.params)
        self._fft_win = get_window("hann", self.n_fft, fftbins=True)
        self._fb = mel_filterbank(self.sr, self.n_fft, self.n_mels,
                                  float(p["fmin"]), float(p["fmax"]))
        self._starts = np.arange(self.n_windows) * self.win_hop

    # -- construction helpers -------------------------------------------------
    @classmethod
    def from_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def mel_center_freqs(self):
        """Centre frequency (Hz) of each mel band, for plotting axes."""
        p = self.params
        return mel_edges_hz(self.sr, self.n_mels, float(p["fmin"]), float(p["fmax"]))[1:-1]

    # -- pipeline steps -------------------------------------------------------
    def to_mono_16k(self, samples, sr):
        """samples: (n,) or (n, channels) float. Mean of channels, then resample."""
        x = np.asarray(samples, dtype=np.float64)
        if x.ndim == 2:
            x = x.mean(axis=1)
        if sr != self.sr:
            g = math.gcd(int(sr), self.sr)
            x = resample_poly(x, self.sr // g, int(sr) // g)
        return x

    def peak_normalize(self, x):
        peak = np.max(np.abs(x)) if x.size else 0.0
        return x / peak if peak > 0 else x

    def fix_length(self, x):
        if len(x) >= self.clip_len:
            return x[:self.clip_len]
        return np.pad(x, (0, self.clip_len - len(x)))

    def windows(self, x):
        idx = self._starts[:, None] + np.arange(self.win_len)[None, :]
        return x[idx]  # (n_windows, win_len)

    def power_mel(self, w):
        """w: (n_windows, win_len) -> mel power spectrogram (n_windows, n_mels, n_frames)."""
        pad = self.n_fft // 2
        wp = np.pad(w, ((0, 0), (pad, pad)))  # zeros, "centered" frames
        n_fr = 1 + w.shape[1] // self.hop
        idx = np.arange(n_fr)[:, None] * self.hop + np.arange(self.n_fft)[None, :]
        frames = wp[:, idx] * self._fft_win  # (n_win, n_fr, n_fft)
        spec = np.abs(np.fft.rfft(frames, n=self.n_fft, axis=-1)) ** self.power
        return np.einsum("mf,wtf->wmt", self._fb, spec)

    def mel_windows(self, w):
        """w: (n_windows, win_len) -> log-mel scaled per window, (n_windows, n_mels, n_frames)."""
        mel = self.power_mel(w)
        logmel = np.log(mel + self.eps)
        mn = logmel.min(axis=(1, 2), keepdims=True)
        mx = logmel.max(axis=(1, 2), keepdims=True)
        denom = mx - mn
        safe = np.where(denom > 0, denom, 1.0)
        out = np.where(denom > 0, (logmel - mn) / safe, 0.0)  # flat window -> zeros
        return out.astype(np.float32)

    # -- public API -----------------------------------------------------------
    def preprocess_samples(self, samples, sr):
        x = self.to_mono_16k(samples, sr)
        x = self.peak_normalize(x)
        x = self.fix_length(x)
        out = self.mel_windows(self.windows(x))
        assert out.shape == self.shape and out.dtype == np.float32
        return out

    def preprocess_wav_bytes(self, data):
        samples, sr = read_wav(data)
        return self.preprocess_samples(samples, sr)

    def preprocess_file(self, path):
        samples, sr = read_wav(str(path))
        return self.preprocess_samples(samples, sr)

    def mono_16k_for_plot(self, data):
        """Mono 16 kHz waveform (peak-normalised, fixed length) for waveform plots."""
        samples, sr = read_wav(data)
        return self.fix_length(self.peak_normalize(self.to_mono_16k(samples, sr)))
