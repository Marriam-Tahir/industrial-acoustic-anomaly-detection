"""Phase 5 API tests: upload analysis, simulated monitoring, evidence pictures - through the real HTTP layer
(FastAPI test client), the real models of the fake project and a real SQLite file."""
import io
import time

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from phase5_fixtures import SR, analyzer, synth, synth_clip  # noqa: F401
from smart_listener import api, db, repo, security

PW = "secret-pass-1"


@pytest.fixture(autouse=True)
def fast_hashing(monkeypatch):
    monkeypatch.setattr(security, "ITERATIONS", 1000)


@pytest.fixture()
def client(tmp_path, synth, analyzer):
    path = tmp_path / "api.db"
    app = api.create_app(db_path=path, cfg=synth["cfg"], analyzer=analyzer, evidence_dir=tmp_path / "evidence")
    conn = db.connect(path)
    repo.create_admin(conn, "admin", PW)
    conn.close()
    with TestClient(app) as c:
        c.db_path = path
        tok = c.post("/auth/login", json={"username": "admin", "password": PW}).json()["token"]
        c.headers.update({"Authorization": f"Bearer {tok}"})
        yield c


def add_machine(client, group="fan_id_06", name=None):
    t = group.split("_")[0]
    r = client.post("/machines", json={"name": name or f"{t.title()} {group[-2:]}", "machine_type": t, "model_group": group,
                                       "owner_name": "Owner", "owner_email": "owner@example.com"})
    assert r.status_code == 201, r.text
    return r.json()


def wav_bytes(y, sr=SR, channels=1):
    if channels > 1:
        y = np.stack([y] * channels, axis=1)
    b = io.BytesIO()
    sf.write(b, y, sr, format="WAV", subtype="PCM_16")
    return b.getvalue()


def upload(client, mid, data, name="clip.wav"):
    return client.post(f"/machines/{mid}/analyze-upload", files={"file": (name, data, "audio/wav")})


def wait_done(client, sid, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        s = client.get(f"/monitoring/{sid}").json()
        if s["status"] != "running":
            return s
        time.sleep(0.05)
    raise AssertionError("monitoring did not finish")


RNG = np.random.default_rng(11)


# ----------------------------------------------------------------------------- security
def test_new_endpoints_need_a_login(client):
    m = add_machine(client)
    anon = TestClient(client.app)
    for method, url in (("get", "/monitoring"), ("get", "/monitoring/1"), ("get", "/monitoring/scenarios"),
                        ("post", "/monitoring/1/stop"), ("get", "/analyses/1"), ("get", "/analyses/1/evidence.png"),
                        ("get", f"/machines/{m['id']}/history.png"), ("post", f"/machines/{m['id']}/monitoring/start"),
                        ("post", f"/machines/{m['id']}/analyze-upload")):
        assert getattr(anon, method)(url).status_code == 401, url


# ----------------------------------------------------------------------------- upload
def test_upload_normal_clip(client):
    m = add_machine(client)
    r = upload(client, m["id"], wav_bytes(synth_clip("fan", False, RNG)), "fan_id_06_00000001.wav")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["summary"]["n_windows"] == 1 and j["windows"][0]["clip_level"] == "within_normal"
    assert j["warnings"] == [] and "not kept" in j["notice"] and j["machine"]["model_group"] == "fan_id_06"
    assert j["evidence_figure"]
    png = client.get(f"/analyses/{j['windows'][0]['analysis_id']}/evidence.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"


def test_upload_abnormal_multi_window_file(client):
    m = add_machine(client)
    y = np.concatenate([synth_clip("fan", False, RNG), synth_clip("fan", True, RNG), synth_clip("fan", True, RNG)])
    j = upload(client, m["id"], wav_bytes(y), "mixed.wav").json()
    levels = [w["clip_level"] for w in j["windows"]]
    assert levels == ["within_normal", "above_alarm_limit", "above_alarm_limit"]
    assert j["summary"]["n_above_alarm_limit"] == 2 and j["worst_window"] in (1, 2)
    assert "several consecutive" in j["summary"]["file_state_note"]
    rows = client.get(f"/machines/{m['id']}/analyses?source=upload").json()
    assert len(rows) == 3 and all(r["state"] is None for r in rows)


def test_upload_never_changes_alert_state_or_creates_incidents(client):
    m = add_machine(client)
    y = np.concatenate([synth_clip("fan", True, RNG) for _ in range(4)])
    assert upload(client, m["id"], wav_bytes(y)).status_code == 200
    fleet = {r["id"]: r for r in client.get("/fleet").json()}
    assert fleet[m["id"]]["current_state"] == "NO_DATA" and fleet[m["id"]]["n_incidents"] == 0
    assert client.get("/incidents").json() == []
    # since the upload e-mail feature: a non-normal upload records ONE result e-mail (no incident, no HIGH alert)
    assert [a["kind"] for a in client.get("/alerts").json()] == ["analysis"]


def test_upload_filename_hint_for_another_machine_is_a_warning_not_a_switch(client):
    m = add_machine(client, "fan_id_06")
    j = upload(client, m["id"], wav_bytes(synth_clip("fan", False, RNG)), "pump_id_04_00000007.wav").json()
    assert any("pump id_04" in w and "fan_id_06" in w for w in j["warnings"])
    assert j["machine"]["model_group"] == "fan_id_06"


def test_upload_conversions_are_reported(client):
    m = add_machine(client)
    t = np.arange(8000 * 10) / 8000
    y8 = (0.1 * np.sin(2 * np.pi * 120 * t)).astype(np.float32)
    j = upload(client, m["id"], wav_bytes(y8, 8000, channels=2)).json()
    assert any("8000 Hz" in w for w in j["warnings"]) and any("channels" in w for w in j["warnings"])


@pytest.mark.parametrize("data,msg", [(b"not audio", "could not be read"), (None, "only")])
def test_upload_rejections(client, data, msg):
    m = add_machine(client)
    if data is None:
        data = wav_bytes(synth_clip("fan", False, RNG)[: SR * 4])
    r = upload(client, m["id"], data)
    assert r.status_code == 422 and msg in r.json()["detail"]
    assert client.get(f"/machines/{m['id']}/analyses").json() == []


def test_upload_size_limit_and_unknown_or_inactive_machine(tmp_path, synth, analyzer):
    app = api.create_app(db_path=tmp_path / "s.db", cfg=synth["cfg"], analyzer=analyzer, max_upload_bytes=50_000)
    conn = db.connect(tmp_path / "s.db")
    repo.create_admin(conn, "admin", PW)
    conn.close()
    with TestClient(app) as c:
        c.headers["Authorization"] = "Bearer " + c.post("/auth/login", json={"username": "admin", "password": PW}).json()["token"]
        m = add_machine(c)
        assert upload(c, m["id"], wav_bytes(synth_clip("fan", False, RNG))).status_code == 413
        assert upload(c, 999, b"x").status_code == 404
        c.delete(f"/machines/{m['id']}")
        assert upload(c, m["id"], b"x").status_code == 409


def test_upload_for_a_machine_whose_model_file_is_missing(tmp_path, synth):
    from smart_listener.inference import Analyzer
    a = Analyzer(synth["cfg"], tmp_path / "none", synth["calibration"])
    app = api.create_app(db_path=tmp_path / "m.db", cfg=synth["cfg"], analyzer=a)
    conn = db.connect(tmp_path / "m.db")
    repo.create_admin(conn, "admin", PW)
    conn.close()
    with TestClient(app) as c:
        c.headers["Authorization"] = "Bearer " + c.post("/auth/login", json={"username": "admin", "password": PW}).json()["token"]
        m = add_machine(c)
        r = upload(c, m["id"], wav_bytes(synth_clip("fan", False, RNG)))
        assert r.status_code == 409 and "missing" in r.json()["detail"]
        r = c.post(f"/machines/{m['id']}/monitoring/start", json={})
        assert r.status_code == 409


def test_analysis_unavailable_gives_503_not_a_crash(tmp_path, synth, monkeypatch):
    import smart_listener.inference as inf

    def broken(*a, **k):
        raise FileNotFoundError("config/severity_calibration.json not found")
    monkeypatch.setattr(inf, "Analyzer", broken)
    app = api.create_app(db_path=tmp_path / "b.db", cfg=synth["cfg"])
    conn = db.connect(tmp_path / "b.db")
    repo.create_admin(conn, "admin", PW)
    conn.close()
    with TestClient(app) as c:
        c.headers["Authorization"] = "Bearer " + c.post("/auth/login", json={"username": "admin", "password": PW}).json()["token"]
        assert c.get("/health").status_code == 200 and c.get("/fleet").status_code == 200
        m = add_machine(c)
        r = upload(c, m["id"], b"x")
        assert r.status_code == 503 and "calibration" in r.json()["detail"]


# ----------------------------------------------------------------------------- monitoring over HTTP
def test_monitoring_session_end_to_end(client):
    m = add_machine(client)
    client.put("/settings", json={"replay_seconds_per_window": 0.05})
    sc = client.get("/monitoring/scenarios").json()
    assert sc["simulated"] is True and "fault" in sc["scenarios"]
    r = client.post(f"/machines/{m['id']}/monitoring/start", json={"scenario": "fault", "n_normal_before": 8, "n_fault": 10})
    assert r.status_code == 201, r.text
    s = r.json()
    assert s["simulated"] and "SIMULATED" in s["label"] and s["source"] == "simulated"
    final = wait_done(client, s["id"])
    assert final["status"] == "finished" and final["progress"]["done"] == 18

    fleet = {r["id"]: r for r in client.get("/fleet").json()}
    assert fleet[m["id"]]["current_state"] == "HIGH" and fleet[m["id"]]["last_source"] == "simulated"
    inc = client.get("/incidents", params={"machine_id": m["id"]}).json()
    assert len(inc) == 1
    detail = client.get(f"/incidents/{inc[0]['id']}").json()
    assert len(detail["alerts"]) == 1 and detail["alerts"][0]["status"] == "pending"
    assert client.get("/alerts").json()[0]["recipient"] == "owner@example.com"
    ack = client.post(f"/incidents/{inc[0]['id']}/acknowledge", json={"note": "checked"})
    assert ack.status_code == 200 and ack.json()["acknowledged"] == 1

    rows = client.get(f"/machines/{m['id']}/analyses", params={"session_id": s["id"], "limit": 100}).json()
    assert len(rows) == 18
    flagged = [r for r in rows if r["clip_level"] != "within_normal"][0]
    png = client.get(f"/analyses/{flagged['id']}/evidence.png")
    assert png.status_code == 200 and png.content[:4] == b"\x89PNG"
    quiet = [r for r in rows if r["clip_level"] == "within_normal"][0]
    assert client.get(f"/analyses/{quiet['id']}/evidence.png").status_code == 404
    hist = client.get(f"/machines/{m['id']}/history.png")
    assert hist.status_code == 200 and hist.headers["content-type"] == "image/png"
    assert client.get(f"/machines/{m['id']}/severity-events").json()
    assert [x["status"] for x in client.get("/monitoring", params={"machine_id": m["id"]}).json()] == ["finished"]
    audit = [a["action"] for a in client.get("/audit").json()]
    assert "monitoring.start" in audit


def test_second_start_while_running_is_refused_and_stop_works(client):
    m = add_machine(client)
    client.put("/settings", json={"replay_seconds_per_window": 0.2})
    body = {"scenario": "normal", "n_windows": 40}
    s = client.post(f"/machines/{m['id']}/monitoring/start", json=body).json()
    r = client.post(f"/machines/{m['id']}/monitoring/start", json=body)
    assert r.status_code == 409 and "already" in r.json()["detail"]
    time.sleep(0.6)
    st = client.post(f"/monitoring/{s['id']}/stop")
    assert st.status_code == 200 and st.json()["status"] == "stopped"
    assert client.post(f"/monitoring/{s['id']}/stop").status_code == 409
    assert client.post("/monitoring/9999/stop").status_code == 404
    assert client.get("/monitoring/9999").status_code == 404


def test_bad_monitoring_requests(client):
    m = add_machine(client)
    assert client.post(f"/machines/{m['id']}/monitoring/start", json={"scenario": "nope"}).status_code == 422
    assert client.post(f"/machines/{m['id']}/monitoring/start", json={"n_windows": 100000}).status_code == 422
    r = client.post(f"/machines/{m['id']}/monitoring/start", json={"scenario": "fault", "n_fault": 300})
    assert r.status_code == 422 and "needs" in r.json()["detail"]
    assert client.post("/machines/999/monitoring/start", json={}).status_code == 404


def test_two_machines_monitored_together_through_the_api(client):
    a, b = add_machine(client, "fan_id_06"), add_machine(client, "pump_id_04")
    client.put("/settings", json={"replay_seconds_per_window": 0.05})
    body = {"scenario": "fault", "n_normal_before": 6, "n_fault": 8}
    ids = [client.post(f"/machines/{x['id']}/monitoring/start", json=body).json()["id"] for x in (a, b)]
    assert [wait_done(client, i)["status"] for i in ids] == ["finished", "finished"]
    assert len(client.get("/incidents").json()) == 2 and len(client.get("/alerts").json()) == 2
    # reading while another session writes must never fail with 'database is locked'
    for _ in range(3):
        assert client.get("/fleet").status_code == 200


def test_server_restart_marks_unfinished_sessions_as_errors(tmp_path, synth, analyzer):
    path = tmp_path / "r.db"
    conn = db.connect(path)
    db.init_schema(conn)
    mid = repo.create_machine(conn, name="F", machine_type="fan", model_group="fan_id_06", owner_name="o", owner_email="o@e.com")
    sid = repo.create_session_row(conn, mid, "simulated")
    conn.close()
    with TestClient(api.create_app(db_path=path, cfg=synth["cfg"], analyzer=analyzer)):
        pass
    conn = db.connect(path)
    assert repo.get_session_row(conn, sid)["status"] == "error"
