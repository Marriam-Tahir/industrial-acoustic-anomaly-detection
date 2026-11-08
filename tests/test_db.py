"""Tests for the database layer: schema, passwords, sessions, machines, incidents, alerts, settings."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from smart_listener import db, profiles, repo, security
from smart_listener.config import load_config


@pytest.fixture(autouse=True)
def fast_hashing(monkeypatch):
    monkeypatch.setattr(security, "ITERATIONS", 1000)   # real value is 600000; far too slow for tests


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "t.db")
    db.init_schema(c)
    yield c
    c.close()


def make_machine(conn, name="Pump 04", mtype="pump", group="pump_id_04"):
    return repo.create_machine(conn, name=name, machine_type=mtype, model_group=group,
                               owner_name="Owner", owner_email="owner@example.com")


def test_schema_is_idempotent_and_has_all_tables(conn):
    db.init_schema(conn)
    tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    expected = {"schema_info", "admins", "sessions", "machines", "monitoring_sessions", "analyses",
                "severity_events", "incidents", "alerts", "settings", "audit_log"}
    assert expected <= tables


def test_password_hash_and_verify():
    h = security.hash_password("correct horse")
    assert "correct horse" not in h
    assert security.verify_password("correct horse", h)
    assert not security.verify_password("wrong", h)
    assert not security.verify_password("anything", None)
    assert not security.verify_password("x", "garbage")
    assert security.hash_password("same") != security.hash_password("same")   # random salt


def test_password_strength_rules():
    with pytest.raises(ValueError):
        security.check_password_strength("short")
    with pytest.raises(ValueError):
        security.check_password_strength("        ")
    with pytest.raises(ValueError):
        security.check_password_strength("administrator", "Administrator")
    security.check_password_strength("a-fine-password", "admin")


def test_admin_creation_login_and_duplicates(conn):
    repo.create_admin(conn, "admin", "secret-pass-1")
    assert repo.authenticate(conn, "admin", "secret-pass-1")["username"] == "admin"
    assert repo.authenticate(conn, "ADMIN", "secret-pass-1") is not None      # usernames are not case sensitive
    assert repo.authenticate(conn, "admin", "wrong-pass") is None
    assert repo.authenticate(conn, "nobody", "secret-pass-1") is None
    with pytest.raises(ValueError):
        repo.create_admin(conn, "Admin", "another-pass-2")
    with pytest.raises(ValueError):
        repo.create_admin(conn, "a b", "secret-pass-1")
    assert conn.execute("SELECT password_hash FROM admins").fetchone()[0].startswith("pbkdf2_sha256$")


def test_session_token_is_stored_hashed_and_expires(conn):
    aid = repo.create_admin(conn, "admin", "secret-pass-1")
    token = repo.create_session(conn, aid, hours=1)
    stored = conn.execute("SELECT token_hash FROM sessions").fetchone()[0]
    assert stored != token and token not in stored
    assert repo.admin_for_token(conn, token)["username"] == "admin"
    assert repo.admin_for_token(conn, "not-a-token") is None
    past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(timespec="seconds")
    conn.execute("UPDATE sessions SET expires_utc = ?", (past,))
    conn.commit()
    assert repo.admin_for_token(conn, token) is None
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_machine_crud_and_unique_name(conn):
    mid = make_machine(conn)
    assert repo.get_machine(conn, mid)["owner_email"] == "owner@example.com"
    with pytest.raises(ValueError):
        make_machine(conn, name="pump 04")                      # same name, different case
    assert repo.update_machine(conn, mid, owner_email="new@example.com", location="Hall 2")
    assert repo.get_machine(conn, mid)["location"] == "Hall 2"
    assert not repo.update_machine(conn, 999, location="x")
    with pytest.raises(ValueError):
        repo.update_machine(conn, mid, colour="red")
    repo.update_machine(conn, mid, active=False)
    assert repo.list_machines(conn) == []
    assert len(repo.list_machines(conn, include_inactive=True)) == 1


def test_fleet_shows_no_data_then_latest_state_and_open_incident(conn):
    mid = make_machine(conn)
    assert repo.fleet(conn)[0]["current_state"] == "NO_DATA"
    common = dict(machine_id=mid, source="simulated", model_kind="dense", z_warn=3.5, z_alarm=7.4)
    repo.add_analysis(conn, window_utc="2026-11-08T10:00:00+00:00", score=0.1, z=0.2, clip_level="within_normal",
                      state="NORMAL", **common)
    a2 = repo.add_analysis(conn, window_utc="2026-11-08T10:00:10+00:00", score=0.9, z=9.0, clip_level="above_alarm_limit",
                           state="HIGH", **common)
    repo.add_analysis(conn, window_utc="2026-11-08T10:00:20+00:00", score=0.5, z=1.0, clip_level="within_normal",
                      state=None, **{**common, "source": "upload"})        # an upload must not change the fleet state
    inc = repo.open_incident(conn, machine_id=mid, opened_utc="2026-11-08T10:00:10+00:00", analysis_id=a2, z=9.0, score=0.9)
    row = repo.fleet(conn)[0]
    assert row["current_state"] == "HIGH" and row["open_incident_id"] == inc and row["n_incidents"] == 1
    assert [a["z"] for a in repo.list_analyses(conn, mid, limit=2)] == [1.0, 9.0]
    assert len(repo.list_analyses(conn, mid, source="upload")) == 1


def test_analysis_rejects_bad_values(conn):
    mid = make_machine(conn)
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_analysis(conn, machine_id=mid, source="simulated", window_utc="x", model_kind="dense", score=1, z=1,
                          z_warn=1, z_alarm=2, clip_level="within_normal", state="PANIC")
    with pytest.raises(sqlite3.IntegrityError):
        repo.add_analysis(conn, machine_id=999, source="simulated", window_utc="x", model_kind="dense", score=1, z=1,
                          z_warn=1, z_alarm=2, clip_level="within_normal")      # unknown machine (foreign key)


def test_incident_acknowledge_and_close(conn):
    mid = make_machine(conn)
    inc = repo.open_incident(conn, machine_id=mid, opened_utc="2026-11-08T10:00:00+00:00")
    assert repo.acknowledge_incident(conn, inc, "admin", "looking at it") == "ok"
    assert repo.acknowledge_incident(conn, inc, "admin") == "already"
    assert repo.acknowledge_incident(conn, 999, "admin") == "not_found"
    got = repo.get_incident(conn, inc)
    assert got["acknowledged"] == 1 and got["ack_by"] == "admin" and got["machine_name"] == "Pump 04"
    assert repo.close_incident(conn, inc, "2026-11-08T10:30:00+00:00")
    assert not repo.close_incident(conn, inc, "2026-11-08T10:31:00+00:00")       # already closed
    assert repo.list_incidents(conn, status="open") == []
    assert len(repo.list_incidents(conn, status="closed", machine_id=mid)) == 1


def test_alert_dedup_key_blocks_a_second_alert_for_the_same_incident(conn):
    mid = make_machine(conn)
    inc = repo.open_incident(conn, machine_id=mid, opened_utc="2026-11-08T10:00:00+00:00")
    key = f"high_opened:{inc}"
    first = repo.add_alert(conn, machine_id=mid, incident_id=inc, kind="high_opened", recipient="owner@example.com",
                           subject="[HIGH] Pump 04", dedup_key=key)
    second = repo.add_alert(conn, machine_id=mid, incident_id=inc, kind="high_opened", recipient="owner@example.com",
                            subject="[HIGH] Pump 04", dedup_key=key)
    assert first is not None and second is None
    assert len(repo.list_alerts(conn, incident_id=inc)) == 1


def test_settings_defaults_validation_and_update(conn):
    s = repo.get_settings(conn)
    assert s == {"alerts_enabled": True, "max_emails_per_hour": 10, "reminder_minutes": 0, "replay_seconds_per_window": 1.0}
    new = repo.update_settings(conn, {"alerts_enabled": False, "max_emails_per_hour": 5})
    assert new["alerts_enabled"] is False and new["max_emails_per_hour"] == 5
    for bad in ({"colour": 1}, {"max_emails_per_hour": 0}, {"max_emails_per_hour": "5"}, {"alerts_enabled": "yes"},
                {"replay_seconds_per_window": 99}, {"max_emails_per_hour": True}):
        with pytest.raises(ValueError):
            repo.update_settings(conn, bad)
    assert repo.get_settings(conn)["max_emails_per_hour"] == 5            # failed updates changed nothing
    with pytest.raises(ValueError):
        repo.update_settings(conn, {"alerts_enabled": True, "max_emails_per_hour": 999})
    assert repo.get_settings(conn)["alerts_enabled"] is False             # all-or-nothing


def test_audit_trail(conn):
    repo.audit(conn, "admin", "machine.create", "id=1")
    assert repo.list_audit(conn)[0]["action"] == "machine.create"


def test_model_profiles_without_report_files_and_with_loudness_caution(tmp_path):
    cfg = load_config()
    empty = profiles.model_profiles(cfg, tmp_path)
    assert len(empty) == 8 and all(p["test_auc"] is None and p["caution"] is None for p in empty)

    (tmp_path / "config").mkdir()
    (tmp_path / "reports" / "phase2").mkdir(parents=True)
    (tmp_path / "reports" / "phase3").mkdir(parents=True)
    (tmp_path / "config" / "severity_calibration.json").write_text(json.dumps(
        {"machines": {"pump_id_02": {"model": "dense", "n_val_normal": 50}, "fan_id_00": {"model": "dense", "n_val_normal": 183}}}))
    import pandas as pd
    pd.DataFrame([{"model": "dense", "group": "pump_id_02", "split": "test", "auc": 0.826, "auc_ci_low": 0.737, "auc_ci_high": 0.902},
                  {"model": "dense", "group": "fan_id_00", "split": "test", "auc": 0.872, "auc_ci_low": 0.835, "auc_ci_high": 0.906}]
                 ).to_csv(tmp_path / "reports" / "phase2" / "metrics_by_group.csv", index=False)
    pd.DataFrame([{"group": "pump_id_02", "split": "test", "auc_level_db_free": 0.859, "auc_rms_free": 0.967},
                  {"group": "fan_id_00", "split": "test", "auc_level_db_free": 0.521, "auc_rms_free": 0.655}]
                 ).to_csv(tmp_path / "reports" / "phase3" / "sanity_level_baseline.csv", index=False)
    by = {p["group"]: p for p in profiles.model_profiles(cfg, tmp_path)}
    assert by["pump_id_02"]["test_auc"] == 0.826 and by["pump_id_02"]["test_auc_low"] == 0.737
    assert by["pump_id_02"]["caution"] and "loudness" in by["pump_id_02"]["caution"]
    assert by["fan_id_00"]["caution"] is None and by["fan_id_00"]["loudness_auc"] == 0.655
