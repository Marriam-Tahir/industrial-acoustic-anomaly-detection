"""Tests for the severity engine, incident tracker and calibration maths (pure Python, run in about a second)."""
import numpy as np
import pytest

from smart_listener import severity as sv

ZW, ZA = 2.0, 4.0  # warning and alarm limits used in these tests


def states(z, rule=None):
    return sv.run_stream(z, ZW, ZA, rule)["states"]


def test_center_scale_and_z():
    rng = np.random.default_rng(0)
    scores = np.exp(rng.normal(-3.0, 0.5, 5000))
    c, s = sv.fit_center_scale(scores)
    assert c == pytest.approx(-3.0, abs=0.05) and s == pytest.approx(0.5, abs=0.05)
    assert float(sv.to_z(np.exp(c), c, s)) == pytest.approx(0.0, abs=1e-9)
    assert float(sv.to_z(np.exp(c + 2 * s), c, s)) == pytest.approx(2.0, abs=1e-6)


def test_normal_sound_stays_normal():
    assert set(states([0.0, 0.5, -1.0, 1.5] * 10)) == {"NORMAL"}


def test_isolated_warning_level_spike_is_only_watch():
    s = states([0, 0, 3.0, 0, 0, 0, 0, 0, 0, 0])
    assert "HIGH" not in s and "WARNING" not in s and "WATCH" in s


def test_isolated_alarm_spike_never_reaches_high():
    s = states([0, 0, 9.0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    assert "HIGH" not in s


def test_sustained_alarm_level_reaches_high_after_k_windows():
    s = states([0, 0] + [9.0] * 6)
    assert s[2] != "HIGH" and s[3] != "HIGH"
    assert s[4] == "HIGH"  # third alarm-level window (k_high = 3 of the last 5)


def test_level_does_not_flicker_and_needs_clean_windows_to_drop():
    z = [9.0] * 5 + [0.0] * 4 + [9.0, 9.0]  # only 4 clean windows, then trouble again
    s = states(z)
    assert s[4] == "HIGH" and all(x == "HIGH" for x in s[4:])  # held through 4 clean windows
    s2 = states([9.0] * 5 + [0.0] * 6)
    assert s2[-1] != "HIGH"  # after 5 clean windows the level is allowed to drop


def test_incident_tracker_one_incident_for_a_long_high_period():
    trk = sv.IncidentTracker(cooldown=180)
    events = [trk.update("HIGH") for _ in range(100)]
    assert events[0] == "opened" and events.count("opened") == 1 and trk.count == 1


def test_incident_tracker_reopens_within_cooldown_but_not_after():
    trk = sv.IncidentTracker(cooldown=10)
    seq = ["HIGH"] * 3 + ["NORMAL"] * 4 + ["HIGH"] * 3 + ["NORMAL"] * 20 + ["HIGH"] * 3
    ev = [trk.update(s) for s in seq]
    assert ev.count("opened") == 2 and ev.count("reopened") == 1 and trk.count == 2


def test_engine_rejects_alarm_below_warning():
    with pytest.raises(ValueError):
        sv.SeverityEngine(3.0, 2.0)


def test_rule_overrides_apply():
    rule = {"k_high": 2, "m_high": 3}
    s = states([9.0, 9.0, 0.0], rule)
    assert s[1] == "HIGH"


def test_calibration_roundtrip(tmp_path):
    cal = {"version": 1, "machines": {"fan_id_00": {"center": -3.0, "scale": 0.4}}, "thresholds": {"z_warn": 1.6, "z_alarm": 3.0}}
    p = sv.save_calibration(cal, tmp_path / "c.json")
    back = sv.load_calibration(p)
    assert back["machines"]["fan_id_00"]["scale"] == 0.4 and "updated_utc" in back
    with pytest.raises(FileNotFoundError):
        sv.load_calibration(tmp_path / "missing.json")
