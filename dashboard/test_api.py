"""HTTP boundary tests. Model behavior is tested separately against real weights."""
import io
import math
import struct
import wave

import pytest

from api import create_app
from config import Settings
from inference import PredictionError


class FakeEngine:
    def __init__(self):
        self.status, self.calls, self.error = "ready", [], None

    def health(self):
        return {"status": self.status, "models": [], "preprocessing": {}, "classifier": {}}

    def predict(self, data, filename, machine_type, snr):
        self.calls.append((data, filename, machine_type, snr))
        if self.error:
            raise self.error
        return {"machine_type": machine_type, "snr": snr, "is_anomaly": True,
                "score": 0.25, "threshold": 0.1, "window_scores": [0.25] * 19}


@pytest.fixture
def audio():
    output = io.BytesIO()
    samples = [int(10000 * math.sin(2 * math.pi * 440 * i / 16000)) for i in range(16000)]
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return output.getvalue()


@pytest.fixture
def harness(tmp_path):
    engine = FakeEngine()
    app = create_app(engine=engine, settings=Settings(export_dir=tmp_path / "unused"))
    app.config.update(TESTING=True)
    return app.test_client(), engine


def upload(client, data, filename="recording.wav", **fields):
    form = {"machine_type": "fan", "snr": "6_dB", **fields, "file": (io.BytesIO(data), filename)}
    return client.post("/api/predict", data=form, content_type="multipart/form-data")


def assert_error(response, status, code=None):
    assert response.status_code == status
    assert response.is_json
    payload = response.get_json()
    assert payload["request_id"] == response.headers["X-Request-ID"]
    assert payload["request_id"]
    assert payload["error"]["code"] and payload["error"]["message"]
    if code:
        assert payload["error"]["code"] == code
    return payload


def test_health_and_models(harness):
    client, engine = harness
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/models").is_json
    engine.status = "degraded"
    response = client.get("/api/health")
    assert response.status_code == 503
    assert response.json["status"] == "degraded"


@pytest.mark.parametrize("machine", ["fan", "pump"])
@pytest.mark.parametrize("snr", ["6_dB", "0_dB", "-6_dB"])
def test_request_forwarding_and_response_contract(harness, audio, machine, snr):
    client, engine = harness
    response = upload(client, audio, machine_type=machine, snr=snr)
    assert response.status_code == 200
    assert engine.calls == [(audio, "recording.wav", machine, snr)]
    assert response.json["score"] == 0.25
    assert response.json["window_scores"] == [0.25] * 19
    assert response.json["request_id"] == response.headers["X-Request-ID"]
    assert response.headers["Cache-Control"] == "no-store"


def test_uppercase_extension(harness, audio):
    assert upload(harness[0], audio, filename="recording.WAV").status_code == 200


def test_missing_file(harness):
    client, engine = harness
    assert_error(client.post("/api/predict"), 400, "MISSING_FILE")
    assert not engine.calls


@pytest.mark.parametrize("data,name,status", [(b"", "recording.wav", 400), (b"not WAV", "recording.wav", 400), (b"bad", "recording.mp3", 415), (b"audio", "", 400)])
def test_reject_bad_upload_before_inference(harness, data, name, status):
    client, engine = harness
    assert_error(upload(client, data, filename=name), status)
    assert not engine.calls


@pytest.mark.parametrize("fields", [{"machine_type": "compressor"}, {"snr": "10_dB"}, {"machine_type": "../../fan"}, {"snr": "../6_dB"}])
def test_invalid_selectors(harness, audio, fields):
    client, engine = harness
    assert_error(upload(client, audio, **fields), 422)
    assert not engine.calls


def test_size_limit(audio):
    engine = FakeEngine()
    app = create_app(engine=engine, settings=Settings(max_upload_bytes=1024))
    assert_error(upload(app.test_client(), audio), 413, "FILE_TOO_LARGE")
    assert not engine.calls


def test_total_request_size_limit():
    app = create_app(engine=FakeEngine(), settings=Settings(max_upload_bytes=1024))
    assert_error(upload(app.test_client(), b"a" * (2 * 1024 * 1024)), 413, "FILE_TOO_LARGE")


@pytest.mark.parametrize("code,status", [("MACHINE_REQUIRED", 422), ("NO_USABLE_SIGNAL", 422), ("MODEL_UNAVAILABLE", 503)])
def test_engine_errors_remain_structured(harness, audio, code, status):
    client, engine = harness
    engine.error = PredictionError(code, "Actionable message", status)
    assert assert_error(upload(client, audio), status, code)["error"]["message"] == "Actionable message"


def test_unexpected_error_is_sanitized(harness, audio):
    client, engine = harness
    engine.error = RuntimeError("private-path-secret")
    response = upload(client, audio)
    assert_error(response, 500, "INTERNAL_ERROR")
    assert "private-path-secret" not in response.text
    assert "Traceback" not in response.text


def test_not_found_wrong_method_and_request_ids(harness):
    client, _ = harness
    one = assert_error(client.get("/api/does-not-exist"), 404)
    two = assert_error(client.get("/api/predict"), 405)
    assert one["request_id"] != two["request_id"]


def test_ui_and_assets(harness):
    client, _ = harness
    assert client.get("/").status_code == 200
    assert "Smart Listener" in client.get("/").text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/styles.css").status_code == 200
