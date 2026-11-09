"""Phase 5 tests: replay source, the per-window pipeline, incidents and alert records, and the monitor manager."""
import threading
import time
from datetime import datetime, timezone

import pytest

from phase5_fixtures import analyzer, synth, synth_clip  # noqa: F401
from smart_listener import db, evidence, monitor, repo
from smart_listener.monitor import MonitorError, MonitorManager, ReplayError, ReplaySource, window_times


@pytest.fixture()
def env(tmp_path, synth, analyzer):
    path = tmp_path / "t.db"
    conn = db.connect(path)
    db.init_schema(conn)
    ids = {}
    for g in synth["groups"]:
        t = g.split("_")[0]
        ids[g] = repo.create_machine(conn, name=f"{t.title()} {g[-2:]}", machine_type=t, model_group=g,
                                     owner_name="Owner", owner_email="owner@example.com", location="Line 1")
    mgr = MonitorManager(path, analyzer, synth["cfg"], tmp_path / "evidence")
    yield {"conn": conn, "mgr": mgr, "ids": ids, "path": path, "tmp": tmp_path}
    mgr.shutdown()
    conn.close()


def machine(env, group):
    return repo.get_machine(env["conn"], env["ids"][group])


SMALL = {"scenario": "fault", "n_normal_before": 8, "n_fault": 10}


# ----------------------------------------------------------------------------- source and clock
def test_window_times_end_at_now_and_increase():
    end = datetime(2026, 11, 8, 12, 0, 0, tzinfo=timezone.utc)
    t = window_times(4, end)
    assert t[-1] == "2026-11-08T12:00:00+00:00" and t[0] == "2026-11-08T11:59:30+00:00" and t == sorted(t)
    assert datetime.fromisoformat(window_times(3)[-1]) <= datetime.now(timezone.utc)


def test_replay_plan_follows_the_scenario(synth):
    s = ReplaySource(synth["cfg"], "fan_id_06", "fault", "test", 5, 7)
    assert len(s) == 12 and [r["replay_label"] for r in s.plan] == ["normal"] * 5 + ["abnormal"] * 7
    r = ReplaySource(synth["cfg"], "fan_id_06", "fault_recovery", "test", 3, 4, 2)
    assert [x["replay_label"] for x in r.plan] == ["normal"] * 3 + ["abnormal"] * 4 + ["normal"] * 2
    n = ReplaySource(synth["cfg"], "fan_id_06", "normal", "test", n_windows=6)
    assert {x["replay_label"] for x in n.plan} == {"normal"}
    # clips are in recording order and never repeat
    ids = [x["file_id"] for x in s.plan]
    assert len(set(ids)) == len(ids) and ids[:5] == sorted(ids[:5])


def test_replay_only_uses_val_and_test_clips(synth):
    s = ReplaySource(synth["cfg"], "fan_id_06", "fault", "test", 5, 5)
    man = synth["manifest"].set_index("file_id")
    assert {man.loc[x["file_id"], "split_ad"] for x in s.plan} == {"test"}
    with pytest.raises(ReplayError, match="split"):
        ReplaySource(synth["cfg"], "fan_id_06", "fault", "train")


def test_replay_errors_are_readable(synth):
    with pytest.raises(ReplayError, match="needs"):
        ReplaySource(synth["cfg"], "fan_id_06", "fault", "test", 5, 400)
    with pytest.raises(ReplayError, match="unknown scenario"):
        ReplaySource(synth["cfg"], "fan_id_06", "nope")
    with pytest.raises(ReplayError, match="whole number"):
        ReplaySource(synth["cfg"], "fan_id_06", "fault", n_fault=-1)
    with pytest.raises(ReplayError, match="no windows"):
        ReplaySource(synth["cfg"], "fan_id_06", "fault", n_normal_before=0, n_fault=0)


def test_replay_yields_exact_length_clips(synth):
    from smart_listener import audio
    y, meta = next(iter(ReplaySource(synth["cfg"], "fan_id_06", "fault", "test", 2, 2)))
    assert len(y) == audio.expected_samples(synth["cfg"]) and meta["replay_label"] == "normal"


def test_microphone_source_is_a_documented_stub():
    with pytest.raises(NotImplementedError):
        iter(monitor.MicrophoneSource())


# ----------------------------------------------------------------------------- a full session
def test_fault_scenario_opens_one_incident_and_one_alert(env):
    m = machine(env, "fan_id_06")
    s = env["mgr"].start(m, SMALL, background=False)
    assert s["status"] == "finished" and s["simulated"] and "SIMULATED" in s["label"]
    rows = repo.list_analyses(env["conn"], m["id"], 100, 0, "simulated", s["id"])
    assert len(rows) == 18 and {r["source"] for r in rows} == {"simulated"}
    rows.reverse()
    assert {r["state"] for r in rows[:8]} == {"NORMAL"} and rows[-1]["state"] == "HIGH"
    assert all(a["window_utc"] < b["window_utc"] for a, b in zip(rows, rows[1:]))
    inc = repo.list_incidents(env["conn"], machine_id=m["id"])
    assert len(inc) == 1 and inc[0]["status"] == "closed"           # closed when the session ended
    assert inc[0]["opened_utc"] >= rows[8]["window_utc"] and inc[0]["n_windows"] >= 2 and inc[0]["peak_z"] > 6
    al = repo.list_alerts(env["conn"], machine_id=m["id"])
    assert len(al) == 1 and al[0]["status"] == "pending" and al[0]["recipient"] == "owner@example.com"
    assert al[0]["kind"] == "high_opened" and "NOT" in al[0]["body"] and f"{inc[0]['id']}" in al[0]["dedup_key"]
    ev = repo.list_severity_events(env["conn"], m["id"])
    assert [e["to_state"] for e in reversed(ev)][-1] == "HIGH" and ev[-1]["from_state"] == "NORMAL"


def test_normal_scenario_stays_quiet(env):
    m = machine(env, "pump_id_04")
    s = env["mgr"].start(m, {"scenario": "normal", "n_windows": 12}, background=False)
    rows = repo.list_analyses(env["conn"], m["id"], 100)
    assert len(rows) == 12 and {r["state"] for r in rows} == {"NORMAL"}
    assert repo.list_incidents(env["conn"]) == [] and repo.list_alerts(env["conn"]) == []
    assert s["status"] == "finished"


def test_recovery_closes_the_incident_without_a_second_alert(env):
    m = machine(env, "fan_id_06")
    env["mgr"].start(m, {"scenario": "fault_recovery", "n_normal_before": 6, "n_fault": 8, "n_normal_after": 8},
                     background=False)
    inc = repo.list_incidents(env["conn"], machine_id=m["id"])
    assert len(inc) == 1 and inc[0]["status"] == "closed"
    rows = repo.list_analyses(env["conn"], m["id"], 100)[::-1]
    assert rows[-1]["state"] == "NORMAL"                              # recovered after 5 clean windows
    assert len(repo.list_alerts(env["conn"], machine_id=m["id"])) == 1


def test_disabled_alerts_are_recorded_as_suppressed(env):
    repo.update_settings(env["conn"], {"alerts_enabled": False})
    m = machine(env, "fan_id_06")
    env["mgr"].start(m, SMALL, background=False)
    assert [a["status"] for a in repo.list_alerts(env["conn"])] == ["suppressed"]


def test_alert_for_the_same_incident_is_never_duplicated(env):
    from smart_listener import alerts
    m = machine(env, "fan_id_06")
    env["mgr"].start(m, SMALL, background=False)
    inc = repo.list_incidents(env["conn"])[0]
    assert alerts.record_high_alerts(env["conn"], m, inc["id"], inc["opened_utc"], 9.0, repo.get_settings(env["conn"])) == []
    assert len(repo.list_alerts(env["conn"])) == 1


def test_evidence_pictures_exist_for_windows_above_the_warning_limit(env):
    m = machine(env, "fan_id_06")
    env["mgr"].start(m, SMALL, background=False)
    rows = repo.list_analyses(env["conn"], m["id"], 100)
    import json
    above = [r for r in rows if r["clip_level"] != "within_normal"]
    assert above
    for r in above:
        fig = json.loads(r["evidence_json"])["figure"]
        data = (env["tmp"] / "evidence" / f"m{m['id']}" / fig).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"
    for r in rows:
        if r["clip_level"] == "within_normal":
            assert "figure" not in json.loads(r["evidence_json"])
            assert json.loads(r["evidence_json"])["simulated"] is True


def test_history_picture(env):
    m = machine(env, "fan_id_06")
    env["mgr"].start(m, SMALL, background=False)
    rows = repo.list_analyses(env["conn"], m["id"], 100)[::-1]
    png = evidence.history_figure(rows, rows[0]["z_warn"], rows[0]["z_alarm"], "t")
    assert png[:4] == b"\x89PNG" and evidence.history_figure([], 3, 6)[:4] == b"\x89PNG"


def test_clip_figure_is_a_png(synth, analyzer):
    y = synth_clip("fan", True, __import__("numpy").random.default_rng(3))
    ev = evidence.compute_evidence(analyzer, "fan_id_06", y)
    assert evidence.clip_figure(y, ev, synth["cfg"], "title")[:4] == b"\x89PNG"


# ----------------------------------------------------------------------------- starting, stopping, failing
def test_machine_cannot_be_monitored_twice_and_stop_works(env):
    m = machine(env, "fan_id_06")
    big = {"scenario": "normal", "n_windows": 40}
    s = env["mgr"].start(m, big, background=True, delay=0.2)
    assert s["status"] == "running"
    with pytest.raises(MonitorError, match="already being monitored"):
        env["mgr"].start(m, big, background=True, delay=0.2)
    time.sleep(0.7)
    assert env["mgr"].stop(s["id"]) is True
    s2 = env["mgr"].status(s["id"])
    assert s2["status"] == "stopped" and 0 < s2["progress"]["done"] < 40
    assert env["mgr"].stop(s["id"]) is False                          # already stopped
    env["mgr"].start(m, big, background=False)                        # the machine is free again


def test_stopping_during_an_incident_closes_it(env):
    m = machine(env, "fan_id_06")
    s = env["mgr"].start(m, {"scenario": "fault", "n_normal_before": 2, "n_fault": 30}, background=True, delay=0.05)
    for _ in range(100):
        if repo.list_incidents(env["conn"], "open", m["id"]):
            break
        time.sleep(0.05)
    assert repo.list_incidents(env["conn"], "open", m["id"])
    env["mgr"].stop(s["id"])
    assert repo.list_incidents(env["conn"], "open", m["id"]) == [] and len(repo.list_incidents(env["conn"])) == 1


def test_two_machines_can_run_at_once(env):
    a, b = machine(env, "fan_id_06"), machine(env, "pump_id_04")
    sa = env["mgr"].start(a, SMALL, background=True, delay=0.02)
    sb = env["mgr"].start(b, SMALL, background=True, delay=0.02)
    deadline = time.time() + 90          # generous: a busy laptop can be several times slower than usual
    while time.time() < deadline:
        if all(env["mgr"].status(s["id"])["status"] != "running" for s in (sa, sb)):
            break
        time.sleep(0.05)
    for s, mm in ((sa, a), (sb, b)):
        st = env["mgr"].status(s["id"])
        assert st["status"] == "finished", st
        assert len(repo.list_analyses(env["conn"], mm["id"], 100)) == 18
    assert len(repo.list_incidents(env["conn"])) == 2 and len(repo.list_alerts(env["conn"])) == 2


def test_limit_on_simultaneous_sessions(env, synth, analyzer):
    mgr = MonitorManager(env["path"], analyzer, synth["cfg"], max_sessions=1)
    a, b = machine(env, "fan_id_06"), machine(env, "pump_id_04")
    s = mgr.start(a, {"scenario": "normal", "n_windows": 40}, delay=0.2)
    with pytest.raises(MonitorError, match="at most 1"):
        mgr.start(b, {"scenario": "normal", "n_windows": 5}, delay=0.2)
    mgr.stop(s["id"])


def test_missing_model_and_inactive_machine_and_bad_options(env, synth):
    from smart_listener.inference import Analyzer
    a = Analyzer(synth["cfg"], env["tmp"] / "no_models", synth["calibration"])
    mgr = MonitorManager(env["path"], a, synth["cfg"])
    with pytest.raises(MonitorError, match="missing"):
        mgr.start(machine(env, "fan_id_06"), SMALL, background=False)
    repo.update_machine(env["conn"], env["ids"]["fan_id_06"], active=False)
    with pytest.raises(MonitorError, match="inactive"):
        env["mgr"].start(machine(env, "fan_id_06"), SMALL, background=False)
    with pytest.raises(MonitorError) as e:
        env["mgr"].start(machine(env, "pump_id_04"), {"scenario": "fault", "n_fault": 400}, background=False)
    assert e.value.code == 422
    assert repo.list_session_rows(env["conn"]) == []                  # nothing was created by the refused starts


def test_an_error_in_scoring_ends_the_session_cleanly(env, analyzer, monkeypatch):
    m = machine(env, "fan_id_06")
    calls = {"n": 0}
    real = analyzer.analyze_waveform

    def flaky(group, y):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("boom")
        return real(group, y)
    monkeypatch.setattr(analyzer, "analyze_waveform", flaky)
    s = env["mgr"].start(m, SMALL, background=False)
    assert s["status"] == "error" and "boom" in s["note"] and s["progress"]["done"] == 3
    assert env["mgr"].start(m, SMALL, background=False)["status"] in ("finished", "error")   # machine is free again


def test_stale_running_sessions_are_failed_on_restart(env):
    m = machine(env, "fan_id_06")
    sid = repo.create_session_row(env["conn"], m["id"], "simulated")
    repo.open_incident(env["conn"], machine_id=m["id"], opened_utc=repo.now_utc())
    assert env["mgr"].cleanup_stale() == 1
    assert repo.get_session_row(env["conn"], sid)["status"] == "error"
    assert repo.list_incidents(env["conn"], "open") == []
    assert env["mgr"].cleanup_stale() == 0


def test_fleet_shows_the_latest_simulated_state(env):
    m = machine(env, "fan_id_06")
    env["mgr"].start(m, SMALL, background=False)
    f = {r["id"]: r for r in repo.fleet(env["conn"])}
    assert f[m["id"]]["current_state"] == "HIGH" and f[m["id"]]["n_incidents"] == 1
    assert f[env["ids"]["pump_id_04"]]["current_state"] == "NO_DATA"
    # a second session whose clock overlaps the first still wins because the fleet uses insertion order
    env["mgr"].start(m, {"scenario": "normal", "n_windows": 8}, background=False)
    assert {r["id"]: r for r in repo.fleet(env["conn"])}[m["id"]]["current_state"] == "NORMAL"
