"""Upload result e-mails (PDF + CSV + evidence attached), Pakistan-time display, and the database upgrade they need.

No real e-mail is ever sent: the API runs without a mail server (test mode), so delivered messages are saved as .eml
files and read back here."""
import csv
import email
import io
import json
import sqlite3
from email import policy

import numpy as np
import pytest

from phase5_fixtures import analyzer, synth, synth_clip  # noqa: F401
from smart_listener import alerts, db, notifier, reports, repo
from test_phase5_api import add_machine, client, fast_hashing, upload, wav_bytes  # noqa: F401

RNG = np.random.default_rng(23)


def clips(kind_list):
    return wav_bytes(np.concatenate([synth_clip("fan", k == "a", RNG) for k in kind_list]))


def outbox_messages(client):
    out = client.db_path.parent / "outbox"
    return [email.message_from_bytes(p.read_bytes(), policy=policy.default) for p in sorted(out.glob("*.eml"))]


# ----------------------------------------------------------------------------- when an upload e-mails
def summary(n, state, levels):
    return ({"n_windows": n, "file_state": state, "n_above_warning_limit": 0, "n_above_alarm_limit": 0},
            [{"z": float(i), "score": 0.1, "clip_level": lv, "z_warn": 3.57, "z_alarm": 7.42, "offset_s": i * 10.0}
             for i, lv in enumerate(levels)])


@pytest.mark.parametrize("state, notify", [("NORMAL", False), ("WATCH", True), ("WARNING", True), ("HIGH", True)])
def test_long_file_uses_its_severity(state, notify):
    s, w = summary(6, state, ["within_normal"] * 6)
    o = alerts.analysis_outcome(s, w, 6)
    assert o["assessed"] and o["notify"] is notify and o["severity"] == state and o["headline"] == state


@pytest.mark.parametrize("level, notify", [("within_normal", False), ("above_warning_limit", False),
                                           ("above_alarm_limit", True)])
def test_short_file_e_mails_only_above_the_alarm_limit(level, notify):
    s, w = summary(1, "WARNING", [level])
    o = alerts.analysis_outcome(s, w, 6)
    assert not o["assessed"] and o["severity"] is None and o["notify"] is notify


# ----------------------------------------------------------------------------- through the API
def test_normal_upload_sends_no_e_mail(client):
    m = add_machine(client)
    j = upload(client, m["id"], clips("n"), "fan_id_06_normal.wav").json()
    assert j["email"]["needed"] is False and j["email"]["alert_id"] is None
    assert client.get("/alerts").json() == []


def test_long_normal_file_sends_no_e_mail(client):
    m = add_machine(client)
    j = upload(client, m["id"], clips("nnnnnn")).json()
    assert j["summary"]["file_state"] == "NORMAL" and not j["summary"]["file_state_note"]
    assert j["email"]["needed"] is False and client.get("/alerts").json() == []


def test_short_abnormal_upload_e_mails_the_owner_with_pdf_csv_and_evidence(client):
    m = add_machine(client)
    j = upload(client, m["id"], clips("a"), "fan_id_06_abnormal_1.wav").json()
    e = j["email"]
    assert e["needed"] and e["status"] == "pending" and e["recipient"] == "owner@example.com"
    a = client.get("/alerts").json()[0]
    assert a["kind"] == "analysis" and a["incident_id"] is None
    assert a["subject"] == "[Smart Listener] Upload result ABOVE ALARM LIMIT: FAN-06 (fan_id_06_abnormal_1.wav)"
    for text in ("Machine ID      : FAN-06 (fan_id_06)", "Machine type    : Fan", "Analysed (PKT)", "not assessed",
                 "Anomaly level   : z =", "Anomaly score   :", "does NOT give a probability of failure"):
        assert text in a["body"], text
    files = json.loads(a["attachments_json"])
    assert [f.rsplit(".", 1)[1] for f in files] == ["pdf", "csv", "png"]
    root = client.db_path.parent
    assert (root / files[0]).read_bytes()[:4] == b"%PDF" and (root / files[2]).read_bytes()[:4] == b"\x89PNG"
    rows = list(csv.DictReader(io.StringIO((root / files[1]).read_text(encoding="utf-8"))))
    w = j["windows"][0]
    assert len(rows) == 1 and rows[0]["machine_id"] == "FAN-06" and rows[0]["window_result"] == "ABOVE ALARM LIMIT"
    assert float(rows[0]["z"]) == pytest.approx(w["z"], abs=1e-3) and rows[0]["window_pkt"].endswith("+05:00")
    assert rows[0]["file_severity"] == "not assessed"

    # deliver (test mode: saved, never claimed as sent) and check the attachments inside the e-mail itself
    assert client.post("/mail/process").json()["saved_test_mode"] == 1
    msg = outbox_messages(client)[0]
    names = [p.get_filename() for p in msg.iter_attachments()]
    assert names == [f.rsplit("/", 1)[1] for f in files]
    types = {p.get_filename().rsplit(".", 1)[1]: p.get_content_type() for p in msg.iter_attachments()}
    assert types == {"pdf": "application/pdf", "csv": "text/csv", "png": "image/png"}   # same on Windows and Linux
    att_csv = next(p for p in msg.iter_attachments() if p.get_filename().endswith(".csv")).get_content()
    assert "ABOVE ALARM LIMIT" in att_csv
    assert client.get(f"/alerts?status=dry_run").json()[0]["id"] == a["id"]


def test_long_abnormal_file_e_mails_its_severity(client):
    m = add_machine(client)
    j = upload(client, m["id"], clips("aaaaaa"), "line2.wav").json()
    sev = j["summary"]["file_state"]
    assert sev in ("WATCH", "WARNING", "HIGH") and j["email"]["needed"]
    a = client.get("/alerts").json()[0]
    assert a["subject"].startswith(f"[Smart Listener] Upload result {sev}: FAN-06") and f"Severity        : {sev}" in a["body"]
    pdf = (client.db_path.parent / json.loads(a["attachments_json"])[0]).read_bytes()
    assert pdf[:4] == b"%PDF" and len(pdf) > 20000           # the evidence picture is embedded


def test_each_upload_e_mails_once_and_no_duplicates(client):
    m = add_machine(client)
    j1 = upload(client, m["id"], clips("a")).json()
    upload(client, m["id"], clips("a"))
    assert len(client.get("/alerts").json()) == 2           # two analyses -> two e-mails
    conn = db.connect(client.db_path)
    try:
        machine = repo.get_machine(conn, m["id"])
        again = alerts.record_analysis_alert(conn, machine, j1, repo.get_settings(conn), client.db_path.parent, 6,
                                             "clip.wav", evidence_dir=client.db_path.parent / "evidence")
    finally:
        conn.close()
    assert again["status"] == "duplicate" and again["alert_id"] is None
    assert len(client.get("/alerts").json()) == 2
    client.post("/mail/process")
    client.post("/mail/process")
    assert len(outbox_messages(client)) == 2


def test_alerts_switched_off_records_but_does_not_send(client):
    m = add_machine(client)
    client.put("/settings", json={"alerts_enabled": False})
    j = upload(client, m["id"], clips("a")).json()
    assert j["email"]["status"] == "suppressed"
    assert client.post("/mail/process").json()["saved_test_mode"] == 0


def test_missing_attachment_is_retried_with_a_clear_reason(client):
    m = add_machine(client)
    upload(client, m["id"], clips("a"))
    a = client.get("/alerts").json()[0]
    (client.db_path.parent / json.loads(a["attachments_json"])[0]).unlink()
    assert client.post("/mail/process").json()["retry_later"] == 1
    a = client.get(f"/alerts?status=pending").json()[0]
    assert "attachment missing" in a["last_error"] and "mail server" not in a["last_error"]


def test_upload_e_mail_is_sent_by_smtp_with_attachments(client, tmp_path):
    m = add_machine(client)
    upload(client, m["id"], clips("a"))
    sent = []

    class Fake:
        def send(self, msg):
            sent.append(msg)
    conn = db.connect(client.db_path)
    try:
        cfg = notifier.MailConfig(host="smtp.test", user="me@test", password="x", sender="me@test", security="none")
        res = notifier.process_once(conn, cfg, Fake(), tmp_path / "outbox", attach_root=client.db_path.parent)
        assert res["sent"] == 1 and repo.list_alerts(conn)[0]["status"] == "sent"
    finally:
        conn.close()
    assert len(list(sent[0].iter_attachments())) == 3 and sent[0]["To"] == "owner@example.com"


# ----------------------------------------------------------------------------- reports and time
def test_pdf_and_csv_builders_use_the_saved_windows(client):
    m = add_machine(client)
    j = upload(client, m["id"], clips("naa")).json()
    conn = db.connect(client.db_path)
    try:
        machine = repo.get_machine(conn, m["id"])
        rows = reports.upload_rows(conn, m["id"], j["session_id"])
        o = alerts.analysis_outcome(j["summary"], j["windows"], 6)
        text = reports.analysis_csv(machine, rows, o)
        pdf = reports.analysis_pdf(conn, machine, j["session_id"], o, "x.wav", ["resampled"], "weak model")
    finally:
        conn.close()
    lines = list(csv.DictReader(io.StringIO(text)))
    assert [r["start_s"] for r in lines] == ["0", "10", "20"]
    assert [r["window_result"] for r in lines] == ["WITHIN NORMAL RANGE", "ABOVE ALARM LIMIT", "ABOVE ALARM LIMIT"]
    assert pdf[:4] == b"%PDF"


def test_report_csv_exports_keep_utc_and_add_pakistan_time(client):
    m = add_machine(client)
    upload(client, m["id"], clips("n"))
    header = client.get("/reports/analyses.csv").text.splitlines()[0].split(",")
    assert "window_utc" in header and header[-1] == "window_pkt"
    assert client.get("/reports/incidents.csv").text.splitlines()[0].endswith("opened_pkt,closed_pkt,ack_pkt")


def test_report_dates_are_pakistan_calendar_days(tmp_path):
    conn = db.connect(tmp_path / "t.db")
    db.init_schema(conn)
    mid = repo.create_machine(conn, name="P", machine_type="pump", model_group="pump_id_04", owner_name="O",
                              owner_email="o@x.org")
    for ts in ("2026-11-09T18:59:00+00:00", "2026-11-09T19:00:00+00:00", "2026-11-10T18:59:59+00:00",
               "2026-11-10T19:00:00+00:00"):
        repo.add_analysis(conn, machine_id=mid, source="upload", window_utc=ts, model_kind="dense", score=0.1, z=0.0,
                          z_warn=3.57, z_alarm=7.42, clip_level="within_normal")
    got = [r["window_utc"] for r in repo.list_analyses_between(conn, mid, "2026-11-10", "2026-11-10")]
    # 10 Nov in Pakistan = 9 Nov 19:00 UTC up to (not including) 10 Nov 19:00 UTC
    assert got == ["2026-11-09T19:00:00+00:00", "2026-11-10T18:59:59+00:00"]
    conn.close()


# ----------------------------------------------------------------------------- database upgrade
def test_older_database_is_upgraded_and_keeps_every_alert(tmp_path):
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.executescript(db.SCHEMA.replace(", 'analysis'))", "))"))
    c.execute("INSERT INTO schema_info VALUES ('schema_version', '1')")
    c.execute("INSERT INTO machines(name, machine_type, model_group, owner_name, owner_email, created_utc, updated_utc) "
              "VALUES ('P', 'pump', 'pump_id_04', 'O', 'o@x.org', 't', 't')")
    c.execute("INSERT INTO alerts(machine_id, kind, recipient, subject, dedup_key, created_utc, status) "
              "VALUES (1, 'high_opened', 'o@x.org', 'old', 'k-old', 't', 'sent')")
    c.commit()
    with pytest.raises(sqlite3.IntegrityError):                # the old rule really refuses the new kind
        c.execute("INSERT INTO alerts(machine_id, kind, recipient, subject, dedup_key, created_utc) "
                  "VALUES (1, 'analysis', 'o', 's', 'k', 't')")
    c.close()
    conn = db.connect(path)
    db.init_schema(conn)
    db.init_schema(conn)                                        # running it again changes nothing
    old = repo.list_alerts(conn)
    assert len(old) == 1 and old[0]["subject"] == "old" and old[0]["status"] == "sent" and old[0]["id"] == 1
    assert repo.add_alert(conn, machine_id=1, kind="analysis", recipient="o@x.org", subject="new", dedup_key="k-new",
                          attachments=["attachments/upload_1/a.pdf"])
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
    conn.close()
