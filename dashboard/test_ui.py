"""Streamlit UI behavior tests with controlled HTTP replies.

The Flask dashboard is additionally exercised in a real browser; see TESTING.md.
These doubles test UI behavior, never machine-detection accuracy.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import streamlit as st
from streamlit.testing.v1 import AppTest


def reply(body, ok=True):
    return SimpleNamespace(ok=ok, json=lambda: body)


def prediction(abnormal=False):
    return {"filename": "synthetic_signal.wav", "machine_type": "fan", "condition": "abnormal" if abnormal else "normal",
            "is_anomaly": abnormal, "machine_identification": {"detail": "Selected by operator"},
            "score": 0.2 if abnormal else 0.05, "threshold": 0.1, "elapsed_ms": 75,
            "window_times": [{"start": i * 0.5, "end": i * 0.5 + 1} for i in range(19)],
            "window_scores": [0.2 if abnormal else 0.05] * 19, "aggregation": "max",
            "model_id": "fan_6_dB", "warnings": []}


@pytest.fixture
def ui(monkeypatch):
    health = {"status": "ready", "models": [dict(id=str(i)) for i in range(6)],
              "preprocessing": {"fingerprint": "55fac6205183"}}
    monkeypatch.setattr(requests, "get", lambda *args, **kwargs: reply(health))
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: SimpleNamespace(
        name="synthetic_signal.wav", getvalue=lambda: b"synthetic UI test bytes"))
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: reply(prediction()))
    return AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=30)


def analyze(app):
    app.button[0].click().run()
    assert not app.exception


def test_no_file_disables_analysis(ui, monkeypatch):
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: None)
    ui.run()
    assert not ui.exception
    assert ui.button[0].disabled


def test_offline_api_disables_analysis(ui, monkeypatch):
    def offline(*args, **kwargs):
        raise requests.ConnectionError("test offline")
    monkeypatch.setattr(requests, "get", offline)
    ui.run()
    assert not ui.exception
    assert ui.button[0].disabled
    assert any("offline" in error.value.lower() for error in ui.error)


@pytest.mark.parametrize("abnormal", [False, True])
def test_condition_and_metrics_render(ui, monkeypatch, abnormal):
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: reply(prediction(abnormal)))
    ui.run()
    analyze(ui)
    messages = [item.value for item in [*ui.success, *ui.warning]]
    assert any(("Abnormal" if abnormal else "Normal") in text for text in messages)
    assert len(ui.metric) == 3
    assert any("synthetic_signal.wav" in caption.value for caption in ui.caption)


def test_changing_machine_clears_old_result(ui):
    ui.run()
    analyze(ui)
    assert len(ui.metric) == 3
    ui.selectbox[0].select("pump").run()
    assert not ui.exception
    assert len(ui.metric) == 0
    assert any("Upload a recording" in item.value for item in ui.info)


def test_api_error_clears_old_result(ui, monkeypatch):
    ui.run()
    analyze(ui)
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: reply(
        {"error": {"code": "NO_USABLE_SIGNAL", "message": "Record audible machine sound."}}, ok=False))
    analyze(ui)
    assert any("audible" in item.value for item in ui.error)
    assert len(ui.metric) == 0


def test_timeout_is_actionable_and_clears_old_result(ui, monkeypatch):
    ui.run()
    analyze(ui)
    def timeout(*args, **kwargs):
        raise requests.Timeout("test timeout")
    monkeypatch.setattr(requests, "post", timeout)
    analyze(ui)
    assert any("Flask API" in item.value for item in ui.error)
    assert len(ui.metric) == 0
