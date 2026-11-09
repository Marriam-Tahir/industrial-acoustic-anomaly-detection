"""Real inference using the unchanged Member 1/2 export, on the CPU.

The anomaly models do not classify machine type. Automatic mode reads an explicit
fan/pump token in a filename; other recordings require the operator's selection.
"""
from __future__ import annotations

from datetime import datetime, timezone
import csv
import hashlib
import importlib
import json
import logging
import math
from pathlib import Path
import re
import sys
import threading
import time
import uuid

import numpy as np

from config import MACHINE_TYPES, NOISE_LEVELS, Settings, VERSION

LOGGER = logging.getLogger(__name__)


class PredictionError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        super().__init__(message)
        self.code, self.message, self.status = code, message, status


def identify_machine(filename: str, selected: str) -> tuple[str, dict]:
    if selected not in (*MACHINE_TYPES, "auto"):
        raise PredictionError("INVALID_MACHINE", "Select Fan, Pump, or Auto from filename.", 422)
    if selected != "auto":
        return selected, {"method": "selected", "detail": "Machine type selected by the operator; not an acoustic classification."}
    # Ignore parent paths and never use a filename's normal/abnormal label.
    name = filename.replace("\\", "/").rsplit("/", 1)[-1].lower()
    matches = set(re.findall(r"(?:^|[^a-z0-9])(fan|pump)(?=$|[^a-z0-9])", name))
    if len(matches) != 1:
        raise PredictionError("MACHINE_REQUIRED", "The filename must contain an unambiguous fan or pump token. Otherwise select the machine type.", 422)
    machine = matches.pop()
    return machine, {"method": "filename", "detail": "Machine type read from the filename, not predicted from sound. Confirm it matches the recording."}


def validate_upload(data: bytes, filename: str) -> None:
    if not filename:
        raise PredictionError("MISSING_FILENAME", "Choose a WAV recording with a filename.")
    if not data:
        raise PredictionError("EMPTY_FILE", "The uploaded recording is empty.")
    if not filename.lower().endswith(".wav"):
        raise PredictionError("UNSUPPORTED_FORMAT", "Only WAV recordings are supported. Convert your audio to WAV first.", 415)
    if len(data) < 12 or data[:4] not in (b"RIFF", b"RIFX", b"RF64") or data[8:12] != b"WAVE":
        raise PredictionError("INVALID_AUDIO", "The file does not contain a valid WAV header.")


class InferenceEngine:
    def __init__(self, export_dir: Path | None = None, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.export_dir = Path(export_dir or self.settings.export_dir).resolve()
        self.lock = threading.Lock()
        self.scorers: dict = {}
        self.model_errors: list[str] = []
        self.metrics = self._read_metrics()
        self.config = json.loads((self.export_dir / "preprocessing_config.json").read_text(encoding="utf-8"))
        module_path = self.export_dir / "smart_listener_preprocess.py"
        module_hash = hashlib.sha256(module_path.read_bytes()).hexdigest()
        expected_hash = self.config.get("module_sha256", "")
        if len(expected_hash) < 16 or not module_hash.startswith(expected_hash.lower()):
            raise ValueError("The exported preprocessing module hash does not match preprocessing_config.json.")
        parameter_blob = json.dumps(self.config["preprocessing"], sort_keys=True, separators=(",", ":"))
        parameter_hash = hashlib.sha256(parameter_blob.encode("utf-8")).hexdigest()[:12]
        if parameter_hash != self.config.get("fingerprint"):
            raise ValueError("The preprocessing parameter fingerprint does not match its configuration.")
        self.module_sha256 = module_hash
        # Import only the user-supplied, local export. All uploaded files remain data.
        export_path = str(self.export_dir)
        if export_path not in sys.path:
            sys.path.insert(0, export_path)
        self.pre_module = importlib.import_module("smart_listener_preprocess")
        if Path(self.pre_module.__file__).resolve() != module_path:
            raise ValueError("A different preprocessing module is already imported. Restart the server.")
        if self.config.get("preprocessing") != self.pre_module.FROZEN_PARAMS:
            raise ValueError("Preprocessing settings differ from the exported implementation.")
        self.pre = self.pre_module.Preprocessor.from_json(self.export_dir / "preprocessing_config.json")
        if self.pre.fingerprint != self.config.get("fingerprint"):
            raise ValueError("The preprocessing parameter fingerprint does not match its configuration.")
        self.model_module = importlib.import_module("smart_listener_model")
        if Path(self.model_module.__file__).resolve().parent != self.export_dir:
            raise ValueError("A different model module is already imported. Restart the server.")
        import torch
        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
        for machine in MACHINE_TYPES:
            for snr in NOISE_LEVELS:
                model_id = f"{machine}_{snr}"
                try:
                    scorer = self.model_module.AnomalyScorer.from_dir(
                        self.export_dir / model_id, self.export_dir / "preprocessing_config.json", device="cpu")
                    cfg = scorer.cfg
                    if cfg.get("machine_type") != machine or cfg.get("snrs") != [snr]:
                        raise ValueError("Bundle metadata does not match its machine/noise condition.")
                    if cfg.get("input_shape") != [1, *self.pre.shape[1:]]:
                        raise ValueError("Model input shape does not match preprocessing.")
                    if cfg.get("anomaly_rule") != "score > threshold":
                        raise ValueError("Unsupported anomaly decision rule.")
                    if not math.isfinite(float(cfg["threshold"])) or float(cfg["threshold"]) <= 0:
                        raise ValueError("Invalid anomaly threshold.")
                    if cfg.get("aggregation") not in ("max", "mean", "top5mean"):
                        raise ValueError("Invalid score aggregation.")
                    # Readiness includes a forward pass, not just files existing.
                    self._check_result(scorer.score_array(np.zeros(self.pre.shape, dtype=np.float32)))
                    self.scorers[model_id] = scorer
                except Exception:
                    LOGGER.exception("Could not load model %s", model_id)
                    self.model_errors.append(f"Model {model_id} could not be loaded. Check the server log and export files.")

    def _read_metrics(self) -> dict:
        path = self.export_dir.parent / "reports" / "test_results.csv"
        if not path.exists():
            return {}
        with path.open(encoding="utf-8", newline="") as source:
            result = {}
            for row in csv.DictReader(source):
                result[row["experiment"]] = {k: float(row[k]) for k in (
                    "auc", "f1", "precision", "recall", "accuracy", "n_normal", "n_abnormal")}
            return result

    @staticmethod
    def _check_result(result: dict) -> None:
        values = np.asarray([result["score"], result["threshold"], *result["window_scores"]])
        if not np.isfinite(values).all() or len(result["window_scores"]) != 19:
            raise PredictionError("INVALID_MODEL_OUTPUT", "The model returned an invalid score. Check the exported model.", 503)

    def health(self) -> dict:
        models = []
        for name, scorer in self.scorers.items():
            cfg = scorer.cfg
            models.append({"id": name, "machine_type": cfg["machine_type"], "snr": cfg["snrs"][0],
                           "aggregation": cfg["aggregation"], "threshold": cfg["threshold"],
                           "metrics": self.metrics.get(name, {}), "device": "cpu"})
        p = self.config["preprocessing"]
        return {"status": "ready" if len(models) == 6 else "degraded", "version": VERSION,
                "models": models, "max_upload_bytes": self.settings.max_upload_bytes,
                "preprocessing": {"fingerprint": self.pre.fingerprint, "verified": True,
                                  "module_sha256": self.module_sha256, **p},
                "classifier": {"available": False, "detail": "No validated acoustic classifier is included. Machine type comes from your selection or a labeled filename."},
                "errors": self.model_errors,
                "limits": {"min_duration_seconds": self.settings.min_duration_seconds,
                           "max_duration_seconds": self.settings.max_duration_seconds,
                           "analyzed_seconds": p["clip_seconds"]}}

    def predict(self, data: bytes, filename: str, machine_type: str, snr: str) -> dict:
        started = time.perf_counter()
        validate_upload(data, filename)
        if len(data) > self.settings.max_upload_bytes:
            raise PredictionError("FILE_TOO_LARGE", "The WAV recording exceeds the 25 MiB upload limit.", 413)
        machine, identification = identify_machine(filename, machine_type)
        if snr not in NOISE_LEVELS:
            raise PredictionError("INVALID_SNR", "Choose a noise condition of +6, 0, or -6 dB.", 422)
        model_id = f"{machine}_{snr}"
        scorer = self.scorers.get(model_id)
        if scorer is None:
            raise PredictionError("MODEL_UNAVAILABLE", "The selected model is unavailable. Check model readiness and the exported files.", 503)
        try:
            samples, sample_rate = self.pre_module.read_wav(data)
        except (ValueError, OSError, EOFError) as exc:
            raise PredictionError("INVALID_AUDIO", "The WAV recording could not be decoded. Use an uncompressed PCM or floating-point WAV.") from exc
        if samples.ndim != 2 or not len(samples) or samples.shape[1] < 1:
            raise PredictionError("EMPTY_AUDIO", "The recording contains no audio samples.")
        if not 4000 <= sample_rate <= 192000 or samples.shape[1] > 8:
            raise PredictionError("UNSUPPORTED_AUDIO", "Use 4–192 kHz WAV audio with at most eight channels.", 422)
        duration = len(samples) / sample_rate
        if not self.settings.min_duration_seconds <= duration <= self.settings.max_duration_seconds:
            raise PredictionError("INVALID_DURATION", f"Record between {self.settings.min_duration_seconds:g} and {self.settings.max_duration_seconds:g} seconds of audio.", 422)
        if not np.isfinite(samples).all():
            raise PredictionError("NONFINITE_AUDIO", "The recording contains NaN or infinite samples. Export a valid WAV and try again.", 422)
        mono = self.pre.to_mono_16k(samples, sample_rate)
        analyzed = mono[:self.pre.clip_len]
        rms = float(np.sqrt(np.mean(analyzed ** 2)))
        ac_rms = float(np.std(analyzed))
        if rms < 1e-5 or ac_rms < 1e-6:
            raise PredictionError("NO_USABLE_SIGNAL", "The analyzed audio is silent or nearly constant. Record audible machine sound and try again.", 422)
        warnings = []
        if duration < 10:
            warnings.append(f"Only {duration:.2f} seconds were supplied. The exported pipeline pads the remainder to 10 seconds; padded windows can affect the decision.")
        elif duration > 10:
            warnings.append("Only the first 10 seconds are analyzed. The result does not describe the rest of the recording.")
        if sample_rate != self.pre.sr:
            warnings.append(f"Audio was resampled from {sample_rate} Hz to 16000 Hz by the exported preprocessing module.")
        if samples.shape[1] > 1:
            warnings.append(f"The {samples.shape[1]} channels were averaged to mono, exactly as in the training pipeline.")
        if rms < 0.001:
            warnings.append("The recording level is low. Peak normalization is applied, but background noise can affect the decision.")
        if np.mean(np.abs(samples) >= 0.999) > 0.001:
            warnings.append("The recording may be clipped. Use a lower recording level for a more reliable result.")
        # Use the exact public training/export function, with no custom feature math.
        features = self.pre.preprocess_samples(samples, sample_rate)
        with self.lock:
            result = scorer.score_array(features)
        self._check_result(result)
        cfg = scorer.cfg
        # Keep the strict exported comparison even if the threshold is exactly equal.
        is_anomaly = bool(result["score"] > result["threshold"])
        plot_mono = self.pre.fix_length(self.pre.peak_normalize(mono))
        # Min/max envelope of actual audio; never create a decorative fake waveform.
        bins = plot_mono.reshape(800, -1)
        peaks = np.max(np.abs(bins), axis=1)
        signs = np.where(np.mean(bins, axis=1) >= 0, 1.0, -1.0)
        waveform = (peaks * signs).astype(float).tolist()
        # A continuous display spectrogram from the same normalized waveform and
        # exported power-mel transform. Display scaling is separate from inference.
        power = self.pre.power_mel(plot_mono[None, :])[0]
        logmel = np.log(power + self.pre.eps)
        display = (logmel - logmel.min()) / max(float(np.ptp(logmel)), 1e-12)
        display = display[:, ::2]
        if self.metrics.get(model_id, {}).get("auc", 1) < 0.7:
            warnings.append("This bundle has limited test discrimination (AUC below 0.70). Treat the decision as a model demonstration and review the model-library metrics.")
        return {"request_id": uuid.uuid4().hex, "timestamp": datetime.now(timezone.utc).isoformat(),
                "filename": filename.replace("\\", "/").rsplit("/", 1)[-1],
                "machine_type": machine, "machine_identification": identification,
                "condition": "abnormal" if is_anomaly else "normal", "is_anomaly": is_anomaly,
                "score": float(result["score"]), "threshold": float(result["threshold"]),
                "score_ratio": float(result["score"] / result["threshold"]),
                "aggregation": cfg["aggregation"], "model_id": model_id, "snr": snr,
                "audio": {"duration_seconds": duration, "sample_rate": sample_rate,
                          "channels": samples.shape[1], "analyzed_seconds": min(10.0, duration),
                          "padded_seconds": max(0.0, 10.0 - duration), "trimmed_seconds": max(0.0, duration - 10.0),
                          "waveform": waveform, "waveform_duration_seconds": 10},
                "window_scores": [float(s) for s in result["window_scores"]],
                "window_times": [{"start": i * 0.5, "end": i * 0.5 + 1} for i in range(19)],
                "spectrogram": {"values": display.astype(float).tolist(), "duration_seconds": 10,
                                "frequencies_hz": self.pre.mel_center_freqs().tolist()},
                "preprocessing": {"fingerprint": self.pre.fingerprint, "verified": True,
                                  "module_sha256": self.module_sha256, **self.config["preprocessing"]},
                "warnings": warnings, "elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
