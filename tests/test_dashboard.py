"""Phase 6 tests: the API client, and the Streamlit dashboard run headless with Streamlit's AppTest."""
import io
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from phase5_fixtures import SR, analyzer, synth, synth_clip  # noqa: F401
from smart_listener import api, db, repo, security
from smart_listener.dashboard_client import ApiClient, ApiError

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
PW = "secret-pass-1"
RNG = np.random.default_rng(21)


@pytest.fixture(autouse=True)
def fast_hashing(monkeypatch):
    monkeypatch.setattr(security, "ITERATIONS", 1000)


@pytest.fixture()
def server(tmp_path, synth, analyzer):
    path = tmp_path / "d.db"
    app = api.create_app(db_path=path, cfg=synth["cfg"], analyzer=analyzer, evidence_dir=tmp_path / "ev")
    conn = db.connect(path)
    repo.create_admin(conn, "admin", PW)
    conn.close()
    with TestClient(app) as http:
        yield http


@pytest.fixture()
def api_client(server):
    c = ApiClient(http=server)
    c.login("admin", PW)
    return c


def new_machine(c, group="fan_id_06", name="Fan 06", email="owner@example.com"):
    return c.create_machine(name=name, machine_type=group.split("_")[0], model_group=group, location="Line 1",
                            owner_name="Owner", owner_email=email, notes="")


def finish(c, sid, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        s = c.session(sid)
        if s["status"] != "running":
            return s
        time.sleep(0.05)
    raise AssertionError("did not finish")


# ----------------------------------------------------------------------------- client
def test_login_errors_are_readable(server):
    c = ApiClient(http=server)
    with pytest.raises(ApiError, match="Wrong username or password") as e:
        c.login("admin", "nope-nope")
    assert e.value.status == 401
    with pytest.raises(ApiError) as e:
        c.fleet()
    assert e.value.status == 401


def test_validation_errors_are_readable(api_client):
    with pytest.raises(ApiError, match="owner_email") as e:
        new_machine(api_client, email="not-an-email")
    assert e.value.status == 422
    with pytest.raises(ApiError, match="does not belong"):
        api_client.create_machine(name="X", machine_type="pump", model_group="fan_id_06", owner_name="o", owner_email="o@e.com")


def test_unreachable_server_message():
    c = ApiClient("http://127.0.0.1:1", timeout=1)
    with pytest.raises(ApiError, match="Is it running"):
        c.health()


def test_client_full_flow(api_client):
    c = api_client
    m = new_machine(c)
    assert [x["name"] for x in c.machines()] == ["Fan 06"] and c.fleet()[0]["current_state"] == "NO_DATA"
    b = io.BytesIO()
    sf.write(b, synth_clip("fan", True, RNG), SR, format="WAV", subtype="PCM_16")
    r = c.upload(m["id"], "x.wav", b.getvalue())
    assert r["windows"][0]["clip_level"] == "above_alarm_limit" and c.evidence_png(r["windows"][0]["analysis_id"])[:4] == b"\x89PNG"
    c.update_settings(replay_seconds_per_window=0.05)
    s = c.start_monitoring(m["id"], scenario="fault", n_normal_before=6, n_fault=8)
    assert finish(c, s["id"])["status"] == "finished"
    inc = c.incidents("closed")
    assert len(inc) == 1 and c.incident(inc[0]["id"])["alerts"][0]["status"] == "pending"
    assert c.acknowledge(inc[0]["id"], "ok")["acknowledged"] == 1
    with pytest.raises(ApiError, match="already"):
        c.acknowledge(inc[0]["id"])
    assert c.history_png(m["id"])[:4] == b"\x89PNG" and c.severity_events(m["id"]) and c.audit()
    c.deactivate_machine(m["id"])
    assert c.machines(False) == []
    c.logout()
    with pytest.raises(ApiError):
        c.fleet()


# ----------------------------------------------------------------------------- the dashboard itself
PAGES = ["overview", "analyze", "machines", "incidents", "reports", "assistant", "settings"]


def run_app(server, user=True):
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["_client"] = ApiClient(http=server)
    at.run()
    if user:
        at.text_input[0].input("admin")
        at.text_input[1].input(PW)
        at.button[0].click()
        at.run()
    return at


def page(at, key):
    at.button(key=f"nav_{key}").click()
    at.run()
    assert not at.exception, (key, at.exception)
    return at


def md(at):
    return "\n".join(m.value for m in at.markdown)


def btn(at, label):
    return next(b for b in at.button if b.label == label)


def simulated_run(c, group="fan_id_06", name="Fan 06"):
    m = new_machine(c, group, name)
    c.update_settings(replay_seconds_per_window=0.05)
    finish(c, c.start_monitoring(m["id"], scenario="fault", n_normal_before=6, n_fault=8)["id"])
    return m


def test_login_screen_shows_product_and_rejects_a_wrong_password(server):
    at = run_app(server, user=False)
    t = md(at)
    assert not at.exception and "Smart Listener" in t and "Industrial Acoustic Anomaly Detection" in t
    assert "analyses industrial machinery sounds" in t and "DEMO" not in t
    at.text_input[0].input("admin")
    at.text_input[1].input("wrong-wrong")
    at.button[0].click()
    at.run()
    assert any("Wrong username or password" in e.value for e in at.error)


def test_every_screen_renders_without_errors(server, api_client):
    simulated_run(api_client)
    at = run_app(server)
    for key in PAGES:
        page(at, key)
        assert not at.error, (key, [e.value for e in at.error])
        assert at.session_state["page"] == key


def test_fleet_overview_uses_real_backend_data(server, api_client):
    simulated_run(api_client)
    new_machine(api_client, "pump_id_04", "Pump 04")
    at = run_app(server)
    t = md(at)
    for text in ("Fleet overview", "Registered machines", "FAN-06", "PUMP-04", "HIGH", "NO DATA", "Not analysed yet", "Watch, warning or high",
                 "Recent activity", "HIGH incident opened", "SIMULATED"):
        assert text in t, text
    assert t.index("FAN-06") < t.index("PUMP-04")                # HIGH machines first
    assert "DEMO ENVIRONMENT" not in t and "MIMII DATASET" not in t
    btn(at, "Details →").click()
    at.run()
    assert at.session_state["page"] == "machines" and "Machine detail" in md(at)


def test_simulated_indicator_only_while_a_replay_runs(server, api_client):
    m = new_machine(api_client)
    at = run_app(server)
    assert "SIMULATED MONITORING" not in md(at)
    api_client.update_settings(replay_seconds_per_window=5.0)
    s = api_client.start_monitoring(m["id"], scenario="normal", n_windows=3)
    at.run()
    assert "SIMULATED MONITORING" in md(at)
    api_client.stop_monitoring(s["id"])


def test_open_incident_banner_until_acknowledged(server, api_client):
    simulated_run(api_client)
    conn = db.connect(api_client.http.app.state.db_path)
    mid = api_client.machines()[0]["id"]
    iid = repo.open_incident(conn, machine_id=mid, opened_utc=repo.now_utc(), z=9.0)
    conn.close()
    at = run_app(server)
    for key in ("overview", "machines", "reports"):
        page(at, key)
        assert "not acknowledged" in md(at) and any(b.label == "Review incident" for b in at.button), key
    api_client.acknowledge(iid)
    page(at, "overview")
    assert "not acknowledged" not in md(at).split("Fleet overview")[0]


def test_machines_detail_history_and_simulated_tab(server, api_client):
    simulated_run(api_client)
    at = page(run_app(server), "machines")
    t = md(at)
    assert "Registered machines" in t and "Machine detail · FAN-06" in t and "Autoencoder" in t
    assert "SIMULATED MONITORING" in t and "not a live connection" in t
    assert any(b.label == "Start simulated monitoring" for b in at.button)
    assert len(at.get("image")) >= 1
    btn(at, "Analyze this machine").click()
    at.run()
    assert at.session_state["page"] == "analyze" and at.session_state["analyze_machine"] == api_client.machines()[0]["id"]


def test_add_machine_derives_the_type_and_reports_errors(server):
    at = page(run_app(server), "machines")          # no machines yet: the form is shown
    sel = next(s for s in at.selectbox if s.label == "Machine ID")
    sel.set_value("pump_id_04")
    at.run()
    assert "Machine type: <b>Pump</b>" in md(at) and not any(s.label == "Type" for s in at.selectbox)
    ti = {t.label: t for t in at.text_input}
    ti["Name"].input("Pump 04")
    ti["Owner name"].input("Yusra")
    ti["Owner e-mail (alerts go here)"].input("bad-email")
    btn(at, "Register machine").click()
    at.run()
    assert any("owner_email" in e.value for e in at.error)
    {t.label: t for t in at.text_input}["Owner e-mail (alerts go here)"].input("yusra@example.com")
    btn(at, "Register machine").click()
    at.run()
    assert not at.exception and "Registered Pump 04 (PUMP-04)" in md(at)
    c = ApiClient(http=server)
    c.login("admin", PW)
    assert [(m["name"], m["machine_type"], m["model_group"]) for m in c.machines()] == [("Pump 04", "pump", "pump_id_04")]


def test_analyze_screen_explains_selection_and_has_no_fake_result(server, api_client):
    new_machine(api_client, "pump_id_04", "Pump 04")
    at = page(run_app(server), "analyze")
    t = md(at)
    assert "own trained model" in t and "Analysis result" not in t and "ILLUSTRATIVE" not in t
    assert btn(at, "Run analysis").disabled                      # nothing uploaded and no machine chosen yet


def test_incident_detail_acknowledge_and_note(server, api_client):
    simulated_run(api_client)
    at = page(run_app(server), "incidents")
    t = md(at)
    for text in ("Incident history", "INC-001 · FAN-06", "Highest severity", "Start time (UTC)", "Current status", "CLOSED",
                 "has not determined a physical fault", "SIMULATED MONITORING", "PENDING"):
        assert text in t, text
    btn(at, "Acknowledge incident").click()
    at.run()
    btn(at, "Confirm acknowledgement").click()
    at.run()
    assert not at.exception and api_client.incidents()[0]["acknowledged"] == 1 and "Incident acknowledged." in md(at)
    btn(at, "Add operator note").click()
    at.run()
    at.text_area[0].input("Belt guard checked")
    btn(at, "Save note").click()
    at.run()
    assert api_client.notes(1)[0]["note"] == "Belt guard checked" and "Belt guard checked" in md(at)
    btn(at, "Download report").click()
    at.run()
    assert at.session_state["page"] == "reports"


def test_reports_generate_a_real_pdf_and_csv(server, api_client):
    simulated_run(api_client)
    at = page(run_app(server), "reports")
    assert "INC-001" in md(at) and "Not determined" in md(at)
    btn(at, "Generate report").click()
    at.run()
    assert not at.exception and at.session_state["report_file"][2][:5] == b"%PDF-"
    next(s for s in at.selectbox if s.label == "Report type").set_value("Incident list (CSV)")
    at.run()
    btn(at, "Generate report").click()
    at.run()
    assert at.session_state["report_file"][2].decode().startswith("incident,")


def test_assistant_answers_from_the_incident(server, api_client):
    simulated_run(api_client)
    at = page(run_app(server), "assistant")
    btn(at, "Can you identify the exact fault?").click()
    at.run()
    t = md(at)
    assert "not determined" in t and "Detected by model" in t and "From the model" in t
    at.text_input[0].input("When will it fail?")
    btn(at, "Send").click()
    at.run()
    assert "Failure time or probability: not determined" in md(at)


def test_settings_show_email_not_connected_and_registry(server, api_client):
    simulated_run(api_client)
    at = page(run_app(server), "settings")
    t = md(at)
    assert "NOT CONNECTED" in t and "Machine registry" in t and "FAN-06" in t and "Available" in t and "Limits z (warn / alarm)" in t


def test_names_from_the_database_cannot_inject_html(server, api_client):
    api_client.create_machine(name="<script>alert(1)</script>", machine_type="fan", model_group="fan_id_06",
                              location="<b>x</b>", owner_name="o", owner_email="o@e.com", notes="")
    at = page(run_app(server), "machines")
    t = md(at)
    assert "<script>" not in t and "&lt;script&gt;" in t and "<b>x</b>" not in t


def test_sign_out_returns_to_login(server):
    at = run_app(server)
    at.button(key="nav_signout").click()
    at.run()
    assert at.text_input and at.text_input[1].label == "Password"
