"""Real exported-weight tests, including training/inference parity and edge cases.

Generated signals exercise the software; they are not labeled machine recordings
and do not measure detection accuracy. Test metrics come from the supplied reports.
"""
import io
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
from scipy.io import wavfile

from api import create_app
from config import Settings
from inference import InferenceEngine, PredictionError, identify_machine

EXPORT = Settings().export_dir


def make_wav(seconds=10.0, sr=16000, channels=1, amplitude=0.15, floating=False):
    t = np.arange(round(seconds * sr)) / sr
    rng = np.random.default_rng(42)
    signal = amplitude * (0.7 * np.sin(2 * np.pi * 173 * t) + 0.15 * np.sin(2 * np.pi * 901 * t) + 0.1 * rng.standard_normal(len(t)))
    if channels > 1:
        signal = np.column_stack([signal * (1 - i * 0.05) for i in range(channels)])
    output = io.BytesIO()
    wavfile.write(output, sr, signal.astype(np.float32) if floating else (signal * 32767).astype(np.int16))
    return output.getvalue()


@pytest.fixture(scope="session")
def engine():
    return InferenceEngine()


def test_all_six_models_ready_on_cpu(engine):
    health = engine.health()
    assert health["status"] == "ready"
    assert len(health["models"]) == 6
    assert all(model["device"] == "cpu" for model in health["models"])
    assert health["preprocessing"]["verified"] is True
    assert health["preprocessing"]["fingerprint"] == "55fac6205183"
    assert health["classifier"]["available"] is False


@pytest.mark.parametrize("machine", ["fan", "pump"])
@pytest.mark.parametrize("snr", ["6_dB", "0_dB", "-6_dB"])
def test_scores_match_unchanged_exported_scorer(engine, machine, snr):
    data = make_wav()
    actual = engine.predict(data, "synthetic_signal.wav", machine, snr)
    expected = engine.scorers[f"{machine}_{snr}"].score_wav_bytes(data)
    assert actual["score"] == pytest.approx(expected["score"], rel=1e-7)
    assert actual["window_scores"] == pytest.approx(expected["window_scores"], rel=1e-7)
    assert actual["threshold"] == expected["threshold"]
    assert actual["is_anomaly"] == (actual["score"] > actual["threshold"])
    assert actual["condition"] == ("abnormal" if expected["is_anomaly"] else "normal")
    assert actual["machine_identification"]["method"] == "selected"
    assert np.asarray(actual["spectrogram"]["values"]).shape == (64, 157)


@pytest.mark.parametrize("filename,selected,expected", [("fan_id_00.wav", "auto", "fan"), ("PUMP-test.WAV", "auto", "pump"), ("abnormal.wav", "pump", "pump"), ("normal_pump.wav", "fan", "fan")])
def test_machine_source_is_explicit(filename, selected, expected):
    machine, detail = identify_machine(filename, selected)
    assert machine == expected
    assert detail["method"] == ("filename" if selected == "auto" else "selected")


@pytest.mark.parametrize("filename", ["recording.wav", "fancy.wav", "fanpump.wav", "fan_pump.wav", "../fan/recording.wav"])
def test_auto_does_not_guess_machine(filename):
    with pytest.raises(PredictionError, match="filename") as error:
        identify_machine(filename, "auto")
    assert error.value.status == 422


@pytest.mark.parametrize("seconds,sr,channels", [(2.5, 16000, 1), (12, 16000, 1), (10, 44100, 2), (10, 8000, 1)])
def test_padding_trimming_resampling_and_stereo_use_export(engine, seconds, sr, channels):
    data = make_wav(seconds=seconds, sr=sr, channels=channels)
    actual = engine.predict(data, "recording.wav", "pump", "6_dB")
    expected = engine.scorers["pump_6_dB"].score_wav_bytes(data)
    assert actual["window_scores"] == pytest.approx(expected["window_scores"], rel=1e-7)
    assert actual["audio"]["duration_seconds"] == pytest.approx(seconds)
    assert actual["audio"]["padded_seconds"] == max(0, 10 - seconds)
    assert actual["audio"]["trimmed_seconds"] == max(0, seconds - 10)
    assert len(actual["window_scores"]) == 19
    assert len(actual["audio"]["waveform"]) == 800
    assert actual["warnings"]


@pytest.mark.parametrize("value", [0, 0.2, 1e-8])
def test_silent_near_silent_and_dc_audio_rejected(engine, value):
    output = io.BytesIO()
    wavfile.write(output, 16000, np.full(160000, value, dtype=np.float32))
    with pytest.raises(PredictionError) as error:
        engine.predict(output.getvalue(), "recording.wav", "fan", "6_dB")
    assert error.value.code == "NO_USABLE_SIGNAL"
    assert error.value.status == 422


def test_silent_analyzed_portion_is_not_rescued_by_audio_after_ten_seconds(engine):
    samples = np.r_[np.zeros(160000), np.sin(np.arange(16000))].astype(np.float32)
    output = io.BytesIO()
    wavfile.write(output, 16000, samples)
    with pytest.raises(PredictionError, match="silent"):
        engine.predict(output.getvalue(), "recording.wav", "fan", "6_dB")


def test_nonfinite_audio_rejected(engine):
    output = io.BytesIO()
    values = np.ones(16000, dtype=np.float32)
    values[100] = np.nan
    wavfile.write(output, 16000, values)
    with pytest.raises(PredictionError) as error:
        engine.predict(output.getvalue(), "recording.wav", "fan", "6_dB")
    assert error.value.code == "NONFINITE_AUDIO"


@pytest.mark.parametrize("seconds", [0.2, 61])
def test_invalid_duration_rejected(engine, seconds):
    with pytest.raises(PredictionError) as error:
        engine.predict(make_wav(seconds=seconds), "recording.wav", "fan", "6_dB")
    assert error.value.code == "INVALID_DURATION"


def copy_preprocessing(target):
    for name in ("preprocessing_config.json", "smart_listener_preprocess.py"):
        shutil.copy2(EXPORT / name, target / name)


def test_altered_preprocessing_module_fails_closed(tmp_path):
    copy_preprocessing(tmp_path)
    with (tmp_path / "smart_listener_preprocess.py").open("ab") as file:
        file.write(b"\n# changed\n")
    with pytest.raises(ValueError, match="hash"):
        InferenceEngine(export_dir=tmp_path)


def test_altered_configuration_fingerprint_fails_closed(tmp_path):
    copy_preprocessing(tmp_path)
    path = tmp_path / "preprocessing_config.json"
    config = json.loads(path.read_text())
    config["preprocessing"]["sample_rate"] = 8000
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fingerprint"):
        InferenceEngine(export_dir=tmp_path)


def test_bundle_fingerprint_mismatch_rejected_before_weights_load(engine, tmp_path):
    config = dict(engine.scorers["fan_6_dB"].cfg, preprocessing_fingerprint="mismatch")
    (tmp_path / "model_config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="fingerprint"):
        engine.model_module.AnomalyScorer.from_dir(tmp_path, EXPORT / "preprocessing_config.json", device="cpu")


def test_unavailable_model_returns_actionable_error(engine, monkeypatch):
    monkeypatch.delitem(engine.scorers, "fan_6_dB")
    assert engine.health()["status"] == "degraded"
    with pytest.raises(PredictionError) as error:
        engine.predict(make_wav(), "recording.wav", "fan", "6_dB")
    assert error.value.status == 503


def test_threshold_equality_is_normal_and_nan_output_rejected(engine, monkeypatch):
    scorer = engine.scorers["pump_6_dB"]
    threshold = scorer.cfg["threshold"]
    monkeypatch.setattr(scorer, "score_array", lambda features: {
        "score": threshold, "threshold": threshold, "window_scores": [threshold] * 19})
    result = engine.predict(make_wav(), "recording.wav", "pump", "6_dB")
    assert result["is_anomaly"] is False
    monkeypatch.setattr(scorer, "score_array", lambda features: {
        "score": float("nan"), "threshold": threshold, "window_scores": [threshold] * 19})
    with pytest.raises(PredictionError) as error:
        engine.predict(make_wav(), "recording.wav", "pump", "6_dB")
    assert error.value.code == "INVALID_MODEL_OUTPUT"


def test_real_flask_to_preprocessing_to_model_round_trip(engine):
    client = create_app(engine=engine).test_client()
    response = client.post("/api/predict", data={
        "file": (io.BytesIO(make_wav()), "pump_id_06_normal.wav"), "machine_type": "auto", "snr": "6_dB"})
    assert response.status_code == 200
    assert response.json["machine_type"] == "pump"
    assert response.json["machine_identification"]["method"] == "filename"
    assert response.json["preprocessing"]["verified"] is True
    assert response.json["request_id"] == response.headers["X-Request-ID"]
    # A normal/abnormal word in a filename must never determine the prediction.
    other = client.post("/api/predict", data={
        "file": (io.BytesIO(make_wav()), "pump_id_06_abnormal.wav"), "machine_type": "auto", "snr": "6_dB"})
    assert other.json["score"] == response.json["score"]
