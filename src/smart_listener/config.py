"""Load and check the single configuration file (config/audio_config.yaml)."""
import hashlib
import json
from pathlib import Path

import yaml

SECTIONS = ("dataset", "audio", "features", "quality", "split", "paths")


def project_root() -> Path:
    """Return the repository folder (the one that contains config/ and src/)."""
    return Path(__file__).resolve().parents[2]


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _pos_int(section: dict, key: str, where: str) -> None:
    v = section.get(key)
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
        raise ValueError(f"{where}.{key} must be a positive whole number, got {v!r}")


def _pos_num(section: dict, key: str, where: str) -> None:
    v = section.get(key)
    if not _is_number(v) or v <= 0:
        raise ValueError(f"{where}.{key} must be a positive number, got {v!r}")


def validate_config(cfg: dict) -> None:
    """Raise ValueError with a clear message if the configuration is unusable."""
    for name in SECTIONS:
        if not isinstance(cfg.get(name), dict):
            raise ValueError(f"Missing section '{name}' in the configuration file")

    a, f, q, s, d = cfg["audio"], cfg["features"], cfg["quality"], cfg["split"], cfg["dataset"]

    _pos_int(a, "sample_rate", "audio")
    if not isinstance(a.get("channel_index"), int) or a["channel_index"] < 0:
        raise ValueError("audio.channel_index must be 0 or a positive whole number")
    _pos_num(a, "expected_duration_s", "audio")
    if not _is_number(a.get("duration_tolerance_s")) or a["duration_tolerance_s"] < 0:
        raise ValueError("audio.duration_tolerance_s must be 0 or a positive number")

    for key in ("n_fft", "hop_length", "n_mels", "window_frames", "window_hop_frames"):
        _pos_int(f, key, "features")
    if f["hop_length"] > f["n_fft"]:
        raise ValueError("features.hop_length must not be larger than features.n_fft")
    if f["window_hop_frames"] > f["window_frames"]:
        raise ValueError("features.window_hop_frames must not be larger than features.window_frames")
    for key in ("power", "amin"):
        _pos_num(f, key, "features")
    if not _is_number(f.get("fmin")) or f["fmin"] < 0:
        raise ValueError("features.fmin must be 0 or a positive number")
    if not _is_number(f.get("fmax")) or f["fmax"] <= f["fmin"] or f["fmax"] > a["sample_rate"] / 2:
        raise ValueError("features.fmax must be above fmin and not above half the sample rate")
    if f.get("normalization") != "global_per_mel_band":
        raise ValueError("features.normalization must be 'global_per_mel_band'")

    for key in ("clip_level", "clip_fraction_warn", "silence_block_rms", "silence_fraction_warn", "drift_rho_warn"):
        _pos_num(q, key, "quality")
    _pos_int(q, "silence_block_ms", "quality")
    _pos_int(q, "drift_min_files", "quality")

    _pos_int(s, "seed", "split")
    an = s.get("anomaly")
    if not isinstance(an, dict):
        raise ValueError("Missing section 'split.anomaly'")
    if an.get("strategy") not in ("interleaved", "block", "random"):
        raise ValueError("split.anomaly.strategy must be 'interleaved', 'block' or 'random'")
    if not _is_number(an.get("abnormal_val_fraction")) or not 0 < an["abnormal_val_fraction"] < 1:
        raise ValueError("split.anomaly.abnormal_val_fraction must be between 0 and 1")
    if not isinstance(an.get("buffer_files"), int) or an["buffer_files"] < 0:
        raise ValueError("split.anomaly.buffer_files must be 0 or a positive whole number")
    _pos_int(an, "min_train_files", "split.anomaly")

    cl = s.get("classifier")
    if not isinstance(cl, dict):
        raise ValueError("Missing section 'split.classifier'")
    seen = []
    for key in ("train_ids", "val_ids", "test_ids"):
        ids = cl.get(key)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"split.classifier.{key} must be a non-empty list")
        seen += ids
    if len(seen) != len(set(seen)):
        raise ValueError("split.classifier: the same machine id appears in more than one split")
    unknown = set(seen) - set(d.get("machine_ids", []))
    if unknown:
        raise ValueError(f"split.classifier uses ids that are not in dataset.machine_ids: {sorted(unknown)}")

    for key in ("mono_dir", "manifest_dir", "features_dir", "figures_dir"):
        if not isinstance(cfg["paths"].get(key), str) or not cfg["paths"][key]:
            raise ValueError(f"paths.{key} must be a folder path (text)")


def load_config(path=None) -> dict:
    """Read the YAML configuration. Uses config/audio_config.yaml when no path is given."""
    path = Path(path) if path else project_root() / "config" / "audio_config.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} is empty or not a valid YAML mapping")
    validate_config(cfg)
    return cfg


def config_fingerprint(cfg: dict) -> str:
    """Short code that changes whenever any audio or feature setting changes.

    Saved next to every feature file. Training and the app compare it, so features made with
    different settings can never be mixed by accident."""
    payload = {"audio": cfg["audio"], "features": cfg["features"]}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def resolve_path(cfg: dict, key: str) -> Path:
    """Turn a relative folder from the config (for example 'mono_dir') into a full path."""
    return project_root() / cfg["paths"][key]
