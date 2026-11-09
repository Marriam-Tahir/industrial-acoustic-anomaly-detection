"""Tests for the backend parts added with the dashboard redesign: identification (validation-gated), the
maintenance assistant, reports, operator notes, activity feed, password change and system information."""
import csv
import io
import json

import numpy as np
import pytest
import soundfile as sf

from phase5_fixtures import SR, analyzer, synth, synth_clip  # noqa: F401
from smart_listener import assistant, db, identification, repo
from smart_listener.dashboard_client import ApiClient, ApiError
from test_dashboard import PW, api_client, fast_hashing, finish, new_machine, server  # noqa: F401

RNG = np.random.default_rng(5)


def wav(kind="fan", abnormal=False, seconds=1):
    y = np.concatenate([synth_clip(kind, abnormal, RNG) for _ in range(seconds)])
    buf = io.BytesIO()
    sf.write(buf, y, SR, format="WAV")
    return buf.getvalue()


def high_incident(c, group="fan_id_06"):
    m = new_machine(c, group, group.replace("_id_", " "))
    c.update_settings(replay_seconds_per_window=0.05)
    finish(c, c.start_monitoring(m["id"], scenario="fault", n_normal_before=6, n_fault=8)["id"])
    inc = c.incidents(machine_id=m["id"])
    assert inc, "the fault replay should open an incident"
    return m, inc[0]


@pytest.fixture()
def no_validation(monkeypatch, tmp_path):
    monkeypatch.setattr(identification, "VALIDATION_FILE", tmp_path / "missing" / "validation.json")


# ----------------------------------------------------------------------------- identification
def _fake(n_per=60, groups=("fan_id_00", "fan_id_02", "pump_id_00"), gap=20.0, abnormal_share=0.3, seed=0):
    rng = np.random.default_rng(seed)
    Z, truth, split, labels = [], [], [], []
    for g in range(len(groups)):
        for k in range(n_per):
            row = rng.normal(gap, 3, len(groups))
            ab = k < n_per * abnormal_share
            row[g] = rng.normal(12 if ab else 0.5, 1)
            Z.append(row), truth.append(g), split.append("val" if k % 2 else "test"), labels.append("abnormal" if ab else "normal")
    return np.array(Z), np.array(truth), np.array(split), np.array(labels), list(groups)


def test_identification_is_enabled_only_when_held_out_accuracy_supports_it():
    Z, t, s, l, g = _fake(gap=20)
    v = identification.evaluate(Z, t, s, l, g, z_alarm=7.4)
    assert v["enabled"] and v["chosen_margin"] is not None and v["test"]["accuracy_answered"] >= identification.MIN_ACCURACY
    assert v["test"]["answered_rate_abnormal"] == 0          # abnormal clips are never matched (best z above alarm)
    Z2, t2, s2, l2, g2 = _fake(gap=0.3)                      # machines indistinguishable: must stay off
    assert not identification.evaluate(Z2, t2, s2, l2, g2, z_alarm=7.4)["enabled"]


def test_suggestion_rules(tmp_path):
    st = {"enabled": True, "margin": 5, "test": {"accuracy_answered": 0.99}}
    ok = identification.suggest([{"group": "a", "z": 0.5}, {"group": "b", "z": 30}], st, 7.4)
    assert ok["group"] == "a" and "99.0%" in ok["reason"]
    assert identification.suggest([{"group": "a", "z": 0.5}, {"group": "b", "z": 3}], st, 7.4)["group"] is None
    assert identification.suggest([{"group": "a", "z": 9}, {"group": "b", "z": 40}], st, 7.4)["group"] is None
    off = identification.status(tmp_path, ["a", "b"])
    assert not off["enabled"] and "not been validated" in off["reason"]
    assert identification.suggest([{"group": "a", "z": 0.5}, {"group": "b", "z": 30}], off, 7.4)["group"] is None
    p = tmp_path / identification.VALIDATION_FILE
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"groups": ["a", "x"], "enabled": True, "chosen_margin": 5}))
    assert "changed" in identification.status(tmp_path, ["a", "b"])["reason"]


def test_identify_endpoint_ranks_all_models_and_does_not_guess_without_validation(api_client, no_validation):
    new_machine(api_client, "fan_id_06")
    r = api_client.identify("recording.wav", wav("fan"))
    assert len(r["ranking"]) >= 2 and r["ranking"] == sorted(r["ranking"], key=lambda x: x["z"])
    assert r["suggestion"]["group"] is None and not r["status"]["enabled"]
    assert api_client.analyses(api_client.machines()[0]["id"]) == []        # identify stores nothing
    with pytest.raises(ApiError) as e:
        api_client.identify("x.wav", b"not audio")
    assert e.value.status == 422


# ----------------------------------------------------------------------------- assistant
def test_assistant_intents():
    assert assistant.intent("When will this pump fail?") == "predict"
    assert assistant.intent("What is the probability of failure?") == "predict"
    assert assistant.intent("Can you identify the exact fault?") == "fault"
    assert assistant.intent("Is it a bearing problem?") == "fault"
    assert assistant.intent("Why is the severity HIGH?") == "why_high"
    assert assistant.intent("Which machine is affected?") == "machine"
    assert assistant.intent("Was anyone notified?") == "notified"
    assert assistant.intent("What should I do next?") == "action"
    assert assistant.intent("what's the weather") == "unknown"


def test_assistant_answers_from_the_real_incident_and_says_not_determined(api_client):
    m, inc = high_incident(api_client)
    a = api_client.ask(inc["id"], "What exactly is wrong with this fan?")
    kinds = [s["kind"] for s in a["sections"]]
    assert "model" in kinds and "not_determined" in kinds and kinds.index("model") < kinds.index("not_determined")
    model_txt = " ".join(s["text"] for s in a["sections"] if s["kind"] == "model")
    assert f"{inc['peak_z']:.1f}" in model_txt and "FAN-06" in model_txt
    p = api_client.ask(inc["id"], "When will it fail?")
    assert p["sections"][0]["kind"] == "not_determined" and "not determined" in p["sections"][0]["title"]
    n = api_client.ask(inc["id"], "Was anyone notified?")
    assert "owner@example.com" in n["sections"][0]["text"]
    u = api_client.ask(inc["id"], "Tell me a joke")
    assert u["sections"][0]["kind"] == "not_determined"
    g = api_client.ask(inc["id"], "What should I do next?")
    assert any(s["kind"] == "guidance" and "Not derived from this incident" in s["text"] for s in g["sections"])
    with pytest.raises(ApiError) as e:
        api_client.ask(9999, "What happened?")
    assert e.value.status == 404


# ----------------------------------------------------------------------------- reports, notes, activity
def test_reports_are_real_files_built_from_the_database(api_client):
    m, inc = high_incident(api_client)
    rows = list(csv.reader(io.StringIO(api_client.incidents_csv().decode())))
    assert rows[0][0] == "incident" and rows[1][0] == "INC-001" and rows[1][1] == m["model_group"]
    an = list(csv.DictReader(io.StringIO(api_client.analyses_csv(machine_id=m["id"]).decode())))
    assert len(an) == 14 and {a["source"] for a in an} == {"simulated"}
    assert api_client.analyses_csv(machine_id=m["id"], date_from="2000-01-01", date_to="2000-01-02").decode().count("\n") == 1
    for pdf in (api_client.incident_pdf(inc["id"]), api_client.machine_pdf(m["id"]),
                api_client.incident_pdf(inc["id"], include_evidence=False)):
        assert pdf[:5] == b"%PDF-" and len(pdf) > 2000
    with pytest.raises(ApiError) as e:
        api_client.analyses_csv(date_from="09/11/2026")
    assert e.value.status == 422
    with pytest.raises(ApiError):
        api_client.incident_pdf(999)


def test_notes_activity_and_system_info(api_client):
    m, inc = high_incident(api_client)
    notes = api_client.add_note(inc["id"], "  Checked the guard, nothing loose.  ")
    assert [n["note"] for n in notes] == ["Checked the guard, nothing loose."] and notes[0]["author"] == "admin"
    with pytest.raises(ApiError):
        api_client.add_note(inc["id"], "   ")
    act = api_client.activity(50)
    assert any(a["kind"] == "incident_opened" and a["incident_id"] == inc["id"] and a["source"] == "simulated" for a in act)
    assert any(a["kind"] == "severity" and a["to_state"] == "HIGH" for a in act)
    assert act == sorted(act, key=lambda a: a["time_utc"], reverse=True)
    sysinfo = api_client.system()
    assert sysinfo["email_sending"] == "not_connected" and sysinfo["thresholds"]["z_alarm"] > sysinfo["thresholds"]["z_warn"]
    assert sysinfo["models"][m["model_group"]]["available"] is True


def test_password_change(server, api_client):
    with pytest.raises(ApiError) as e:
        api_client.change_password("wrong-password", "another-pass-2")
    assert e.value.status == 403
    with pytest.raises(ApiError) as e:
        api_client.change_password(PW, "short")
    assert e.value.status == 422
    other = ApiClient(http=server)
    other.login("admin", PW)
    api_client.change_password(PW, "another-pass-2")
    api_client.fleet()                                      # this session keeps working
    with pytest.raises(ApiError) as e:
        other.fleet()                                       # other sessions are signed out
    assert e.value.status == 401
    fresh = ApiClient(http=server)
    fresh.login("admin", "another-pass-2")
    assert fresh.account()["username"] == "admin"


def test_notes_table_is_added_to_an_existing_database(tmp_path):
    p = tmp_path / "old.db"
    conn = db.connect(p)
    db.init_schema(conn)
    conn.execute("DROP TABLE incident_notes")
    conn.commit()
    db.init_schema(conn)                                    # running again adds the new table, keeps the data
    assert conn.execute("SELECT COUNT(*) FROM incident_notes").fetchone()[0] == 0
    conn.close()
