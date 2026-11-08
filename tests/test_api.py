"""Tests for the web API, using FastAPI's built-in test client (no real server needed)."""
import pytest
from fastapi.testclient import TestClient

from smart_listener import api, db, repo, security

GOOD = {"name": "Pump 04", "machine_type": "pump", "model_group": "pump_id_04", "location": "Line 2",
        "owner_name": "A. Owner", "owner_email": "owner@example.com"}


@pytest.fixture(autouse=True)
def fast_hashing(monkeypatch):
    monkeypatch.setattr(security, "ITERATIONS", 1000)


@pytest.fixture()
def client(tmp_path):
    path = tmp_path / "api.db"
    app = api.create_app(db_path=path)
    conn = db.connect(path)
    repo.create_admin(conn, "admin", "secret-pass-1")
    conn.close()
    with TestClient(app) as c:
        c.db_path = path
        yield c


def login(client, user="admin", pw="secret-pass-1"):
    return client.post("/auth/login", json={"username": user, "password": pw})


@pytest.fixture()
def auth(client):
    token = login(client).json()["token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_is_public(client):
    assert client.get("/health").json()["status"] == "ok"


def test_every_other_endpoint_needs_a_token(client):
    for path in ("/machines", "/fleet", "/incidents", "/alerts", "/settings", "/audit", "/model-profiles", "/auth/me"):
        assert client.get(path).status_code == 401, path
    assert client.post("/machines", json=GOOD).status_code == 401
    assert client.get("/machines", headers={"Authorization": "Bearer nonsense"}).status_code == 401


def test_login_success_wrong_password_and_lockout(client):
    assert login(client).status_code == 200
    for _ in range(5):
        assert login(client, pw="wrong-wrong").status_code == 401
    assert login(client, pw="wrong-wrong").status_code == 429
    assert login(client).status_code == 429                    # even the right password is refused while locked
    assert login(client, user="someone-else", pw="x").status_code == 401   # other users are not affected


def test_logout_invalidates_the_token(client, auth):
    assert client.get("/auth/me", headers=auth).json()["username"] == "admin"
    assert client.post("/auth/logout", headers=auth).status_code == 200
    assert client.get("/auth/me", headers=auth).status_code == 401


def test_register_machine_and_read_it_back(client, auth):
    r = client.post("/machines", json=GOOD, headers=auth)
    assert r.status_code == 201
    m = r.json()
    assert m["owner_email"] == "owner@example.com" and m["active"] == 1 and m["model_group"] == "pump_id_04"
    assert client.get(f"/machines/{m['id']}", headers=auth).json()["name"] == "Pump 04"
    assert [x["name"] for x in client.get("/machines", headers=auth).json()] == ["Pump 04"]
    assert client.get("/machines/999", headers=auth).status_code == 404


@pytest.mark.parametrize("change", [
    {"owner_email": "not-an-email"},
    {"owner_email": "a@b"},
    {"owner_email": "has space@example.com"},
    {"machine_type": "valve"},
    {"model_group": "pump_id_99"},
    {"model_group": "fan_id_00"},               # a fan model cannot be used for a pump
    {"name": "   "},
    {"owner_name": ""},
])
def test_machine_validation_rejects_bad_input(client, auth, change):
    r = client.post("/machines", json={**GOOD, **change}, headers=auth)
    assert r.status_code == 422, r.text


def test_duplicate_machine_name_is_a_conflict(client, auth):
    assert client.post("/machines", json=GOOD, headers=auth).status_code == 201
    assert client.post("/machines", json={**GOOD, "name": "pump 04"}, headers=auth).status_code == 409


def test_update_owner_and_deactivate_keeps_the_record(client, auth):
    mid = client.post("/machines", json=GOOD, headers=auth).json()["id"]
    r = client.put(f"/machines/{mid}", json={"owner_email": "new.owner@example.com", "owner_name": "B. Owner"}, headers=auth)
    assert r.status_code == 200 and r.json()["owner_email"] == "new.owner@example.com"
    assert client.put(f"/machines/{mid}", json={"owner_email": "bad"}, headers=auth).status_code == 422
    assert client.put(f"/machines/{mid}", json={"model_group": "fan_id_00"}, headers=auth).status_code == 422
    assert client.put("/machines/999", json={"notes": "x"}, headers=auth).status_code == 404
    assert client.delete(f"/machines/{mid}", headers=auth).json()["active"] == 0
    assert client.get("/machines", headers=auth).json() == []
    assert len(client.get("/machines?include_inactive=true", headers=auth).json()) == 1
    assert client.get(f"/machines/{mid}", headers=auth).status_code == 200      # history stays reachable


def test_fleet_incident_acknowledge_and_alert_listing(client, auth):
    mid = client.post("/machines", json=GOOD, headers=auth).json()["id"]
    fleet = client.get("/fleet", headers=auth).json()
    assert fleet[0]["current_state"] == "NO_DATA" and fleet[0]["open_incident_id"] is None

    conn = db.connect(client.db_path)
    inc = repo.open_incident(conn, machine_id=mid, opened_utc="2026-11-08T10:00:00+00:00", z=9.1, score=0.8)
    repo.add_alert(conn, machine_id=mid, incident_id=inc, kind="high_opened", recipient="owner@example.com",
                   subject="[HIGH] Pump 04", dedup_key=f"high_opened:{inc}")
    conn.close()

    assert client.get("/fleet", headers=auth).json()[0]["open_incident_id"] == inc
    assert len(client.get("/incidents?status=open", headers=auth).json()) == 1
    detail = client.get(f"/incidents/{inc}", headers=auth).json()
    assert detail["machine_name"] == "Pump 04" and len(detail["alerts"]) == 1
    assert len(client.get(f"/alerts?incident_id={inc}", headers=auth).json()) == 1

    ack = client.post(f"/incidents/{inc}/acknowledge", json={"note": "inspecting"}, headers=auth)
    assert ack.status_code == 200 and ack.json()["acknowledged"] == 1 and ack.json()["ack_by"] == "admin"
    assert client.post(f"/incidents/{inc}/acknowledge", json={}, headers=auth).status_code == 409
    assert client.post("/incidents/999/acknowledge", json={}, headers=auth).status_code == 404
    assert client.get("/incidents?status=banana", headers=auth).status_code == 422


def test_history_endpoints_for_a_machine(client, auth):
    mid = client.post("/machines", json=GOOD, headers=auth).json()["id"]
    assert client.get(f"/machines/{mid}/analyses", headers=auth).json() == []
    assert client.get(f"/machines/{mid}/severity-events", headers=auth).json() == []
    assert client.get("/machines/999/analyses", headers=auth).status_code == 404
    assert client.get(f"/machines/{mid}/analyses?limit=0", headers=auth).status_code == 422


def test_settings_get_and_validated_update(client, auth):
    assert client.get("/settings", headers=auth).json()["alerts_enabled"] is True
    assert client.put("/settings", json={"alerts_enabled": False}, headers=auth).json()["alerts_enabled"] is False
    assert client.put("/settings", json={"max_emails_per_hour": 0}, headers=auth).status_code == 422
    assert client.put("/settings", json={"smtp_password": "x"}, headers=auth).status_code == 422   # secrets never live here
    assert client.get("/settings", headers=auth).json()["alerts_enabled"] is False


def test_audit_trail_records_actions_but_never_passwords(client, auth):
    client.post("/machines", json=GOOD, headers=auth)
    login(client, pw="wrong-pass-9")
    log = client.get("/audit", headers=auth).json()
    actions = [e["action"] for e in log]
    assert "machine.create" in actions and "login.ok" in actions and "login.failed" in actions
    assert "secret-pass-1" not in str(log) and "wrong-pass-9" not in str(log)


def test_model_profiles_lists_all_eight_machines(client, auth):
    r = client.get("/model-profiles", headers=auth)
    assert r.status_code == 200 and len(r.json()) == 8
    assert {p["group"] for p in r.json()} >= {"fan_id_00", "pump_id_06"}


def test_expired_token_is_refused(client, auth):
    conn = db.connect(client.db_path)
    conn.execute("UPDATE sessions SET expires_utc = '2000-01-01T00:00:00+00:00'")
    conn.commit()
    conn.close()
    assert client.get("/machines", headers=auth).status_code == 401
