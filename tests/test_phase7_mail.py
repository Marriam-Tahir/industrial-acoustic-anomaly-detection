"""Phase 7: e-mail delivery (test mode, SMTP, retries, hourly limit, reminders) and its API."""
import socketserver
import threading
from datetime import datetime, timedelta, timezone

import pytest

from phase5_fixtures import analyzer, synth  # noqa: F401
from smart_listener import alerts, db, notifier, repo
from smart_listener.dashboard_client import ApiError
from test_dashboard import api_client, fast_hashing, finish, new_machine, server  # noqa: F401

NOW = datetime(2026, 11, 9, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def conn(tmp_path):
    c = db.connect(tmp_path / "m.db")
    db.init_schema(c)
    yield c
    c.close()


def machine(conn, email="owner@example.com", name="Pump 04"):
    mid = repo.create_machine(conn, name=name, machine_type="pump", model_group="pump_id_04", location="Line 2",
                              owner_name="Owner", owner_email=email, notes="")
    return repo.get_machine(conn, mid)


def incident(conn, m, opened=NOW, source="simulated"):
    iid = repo.open_incident(conn, machine_id=m["id"], opened_utc=opened.isoformat(), z=12.5)
    alerts.record_high_alerts(conn, m, iid, opened.isoformat(), 12.5, repo.get_settings(conn),
                              "1730-1904 Hz", source=source, z_alarm=7.42)
    return iid


class FakeSender:
    def __init__(self, fail=0, error=None):
        self.fail, self.error, self.sent = fail, error or ConnectionError("refused"), []

    def send(self, msg):
        if self.fail:
            self.fail -= 1
            raise self.error
        self.sent.append(msg)


SMTP_CFG = notifier.MailConfig(host="smtp.test", user="me@test", password="x", sender="me@test", security="none")


def test_alert_text_uses_codes_states_limits_and_simulated(conn):
    m = machine(conn)
    incident(conn, m)
    a = repo.list_alerts(conn)[0]
    assert a["subject"] == "[Smart Listener] HIGH: PUMP-04 INC-001"
    assert "SIMULATED MONITORING" in a["body"] and "z = 12.5 (alarm limit z > 7.42)" in a["body"]
    assert "does NOT give a probability of failure" in a["body"] and "1730-1904 Hz" in a["body"]


def test_test_mode_saves_the_message_and_never_claims_it_was_sent(conn, tmp_path):
    m = machine(conn)
    incident(conn, m, source="upload")
    res = notifier.process_once(conn, notifier.MailConfig(), None, tmp_path / "outbox", NOW)
    assert res["saved_test_mode"] == 1 and res["sent"] == 0
    a = repo.list_alerts(conn)[0]
    assert a["status"] == "dry_run" and a["sent_utc"] and a["attempts"] == 1
    eml = (tmp_path / "outbox" / f"alert_{a['id']}.eml").read_text()
    assert "To: owner@example.com" in eml and "PUMP-04" in eml and "SIMULATED" not in eml


def test_smtp_mode_sends_once_and_dedup_holds(conn, tmp_path):
    m = machine(conn)
    iid = incident(conn, m)
    alerts.record_high_alerts(conn, m, iid, NOW.isoformat(), 13.0, repo.get_settings(conn))   # same incident again
    s = FakeSender()
    assert notifier.process_once(conn, SMTP_CFG, s, tmp_path, NOW)["sent"] == 1
    assert notifier.process_once(conn, SMTP_CFG, s, tmp_path, NOW)["sent"] == 0
    assert len(s.sent) == 1 and s.sent[0]["To"] == "owner@example.com" and s.sent[0]["From"] == "me@test"
    assert repo.list_alerts(conn)[0]["status"] == "sent"


def test_failures_are_retried_with_backoff_then_marked_failed_and_can_be_requeued(conn, tmp_path):
    m = machine(conn)
    incident(conn, m)
    s = FakeSender(fail=10)
    t = NOW
    r = notifier.process_once(conn, SMTP_CFG, s, tmp_path, t)
    a = repo.list_alerts(conn)[0]
    assert r["retry_later"] == 1 and a["status"] == "pending" and a["attempts"] == 1 and "Could not connect" in a["last_error"]
    assert notifier.process_once(conn, SMTP_CFG, s, tmp_path, t + timedelta(seconds=30))["retry_later"] == 0   # waits 1 min
    notifier.process_once(conn, SMTP_CFG, s, tmp_path, t + timedelta(minutes=1, seconds=1))
    notifier.process_once(conn, SMTP_CFG, s, tmp_path, t + timedelta(minutes=6))
    a = repo.list_alerts(conn)[0]
    assert a["status"] == "failed" and a["attempts"] == notifier.MAX_ATTEMPTS
    assert repo.requeue_alert(conn, a["id"]) and not repo.requeue_alert(conn, a["id"])
    ok = FakeSender()
    assert notifier.process_once(conn, SMTP_CFG, ok, tmp_path, t + timedelta(minutes=7))["sent"] == 1


def test_wrong_gmail_password_gets_a_clear_message():
    import smtplib
    assert "app password" in notifier.friendly_error(smtplib.SMTPAuthenticationError(535, b"bad"))


def test_hourly_limit(conn, tmp_path):
    repo.update_settings(conn, {"max_emails_per_hour": 2})
    for i in range(5):
        m = machine(conn, f"o{i}@example.com", f"Pump {i}")
        incident(conn, m)
    s = FakeSender()
    r = notifier.process_once(conn, SMTP_CFG, s, tmp_path, NOW)
    assert r["sent"] == 2 and r["waiting_for_hourly_limit"] == 3
    assert notifier.process_once(conn, SMTP_CFG, s, tmp_path, NOW + timedelta(minutes=30))["sent"] == 0
    assert notifier.process_once(conn, SMTP_CFG, s, tmp_path, NOW + timedelta(minutes=61))["sent"] == 2


def test_reminders_for_open_unacknowledged_incidents(conn, tmp_path):
    repo.update_settings(conn, {"reminder_minutes": 10})
    m = machine(conn)
    iid = incident(conn, m, opened=NOW)
    conn.execute("UPDATE alerts SET created_utc = ?", (NOW.isoformat(),))
    conn.commit()
    assert alerts.record_reminders(conn, repo.get_settings(conn), NOW + timedelta(minutes=5)) == []
    first = alerts.record_reminders(conn, repo.get_settings(conn), NOW + timedelta(minutes=11))
    assert len(first) == 1 and repo.get_alert(conn, first[0])["kind"] == "reminder"
    assert "INC-001" in repo.get_alert(conn, first[0])["subject"]
    assert alerts.record_reminders(conn, repo.get_settings(conn), NOW + timedelta(minutes=12)) == []  # just reminded
    repo.acknowledge_incident(conn, iid, "admin", "")
    assert alerts.record_reminders(conn, repo.get_settings(conn), NOW + timedelta(hours=5)) == []


def test_config_file_and_environment(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "smtp.env").write_text(
        "# comment\nSL_SMTP_HOST=smtp.gmail.com\nSL_SMTP_USER=me@gmail.com\nSL_SMTP_PASSWORD=wxyz qrst uvwx yzab\n")
    c = notifier.load_mail_config(tmp_path, environ={})
    assert c.mode == "smtp" and c.port == 587 and c.security == "starttls" and c.password == "wxyzqrstuvwxyzab"
    assert c.sender == "me@gmail.com" and "password" not in str(c.public()).replace("password_set", "")
    assert notifier.load_mail_config(tmp_path, environ={"SL_SMTP_HOST": "other", "SL_SMTP_SECURITY": "ssl"}).port == 465
    assert notifier.load_mail_config(tmp_path / "none", environ={}).mode == "test"
    with pytest.raises(ValueError):
        notifier.load_mail_config(tmp_path, environ={"SL_SMTP_SECURITY": "tls13"})
    assert notifier.load_mail_config(tmp_path).mode == "test"            # never real e-mail while tests run


class _SmtpHandler(socketserver.StreamRequestHandler):
    """Just enough SMTP to accept one message (no TLS, no login)."""
    def handle(self):
        w = lambda s: self.wfile.write((s + "\r\n").encode())
        w("220 test")
        data = []
        while True:
            line = self.rfile.readline().decode(errors="replace").rstrip("\r\n")
            cmd = line.upper()
            if cmd.startswith(("EHLO", "HELO")):
                w("250 ok")
            elif cmd.startswith(("MAIL", "RCPT", "RSET", "NOOP")):
                w("250 ok")
            elif cmd == "DATA":
                w("354 go")
                while True:
                    l = self.rfile.readline().decode(errors="replace")
                    if l.rstrip("\r\n") == ".":
                        break
                    data.append(l)
                self.server.received.append("".join(data))
                w("250 queued")
            elif cmd == "QUIT" or not line:
                w("221 bye")
                return
            else:
                w("502 no")


def test_real_smtp_conversation_with_a_local_server(conn, tmp_path):
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _SmtpHandler)
    srv.received = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cfg = notifier.MailConfig(host="127.0.0.1", port=srv.server_address[1], security="none", sender="sl@test")
        m = machine(conn)
        incident(conn, m)
        r = notifier.process_once(conn, cfg, notifier.SmtpSender(cfg, timeout=5), tmp_path, NOW)
        assert r["sent"] == 1 and "Subject: [Smart Listener] HIGH: PUMP-04 INC-001" in srv.received[0]
    finally:
        srv.shutdown()
        srv.server_close()


def test_new_column_is_added_to_an_older_database(tmp_path):
    c = db.connect(tmp_path / "old.db")
    db.init_schema(c)
    c.executescript("CREATE TABLE a2 AS SELECT * FROM alerts; DROP TABLE alerts;")  # simulate the old table
    c.execute("""CREATE TABLE alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, incident_id INTEGER, machine_id INTEGER NOT NULL,
                 kind TEXT NOT NULL, recipient TEXT NOT NULL, subject TEXT NOT NULL, body TEXT NOT NULL DEFAULT '',
                 status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                 last_error TEXT NOT NULL DEFAULT '', dedup_key TEXT NOT NULL UNIQUE, created_utc TEXT NOT NULL,
                 sent_utc TEXT)""")
    c.commit()
    db.init_schema(c)
    assert "next_attempt_utc" in {r[1] for r in c.execute("PRAGMA table_info(alerts)")}
    c.close()


# ----------------------------------------------------------------------------- API
def test_mail_api(api_client):
    st = api_client.mail_status()
    assert st["mode"] == "test" and "password" not in st and st["password_set"] is False
    with pytest.raises(ApiError) as e:
        api_client.mail_test("someone@example.com")
    assert e.value.status == 409                                     # needs a registered machine first
    new_machine(api_client)
    t = api_client.mail_test("someone@example.com")
    assert t["status"] == "dry_run" and t["recipient"] == "someone@example.com"
    with pytest.raises(ApiError):
        api_client.mail_test("not-an-address")
    rows = api_client.alerts()
    assert rows[0]["kind"] == "test" and api_client.alerts(status="dry_run")[0]["id"] == rows[0]["id"]
    assert api_client.alerts(status="failed") == []
    with pytest.raises(ApiError) as e:
        api_client.retry_alert(rows[0]["id"])
    assert e.value.status == 409
    assert api_client.mail_process()["mode"] == "test"
    assert api_client.system()["email_sending"] == "test"


def test_simulated_high_incident_reaches_the_outbox_and_the_assistant_says_so(api_client):
    m = new_machine(api_client)
    api_client.update_settings(replay_seconds_per_window=0.05)
    finish(api_client, api_client.start_monitoring(m["id"], scenario="fault", n_normal_before=6, n_fault=8)["id"])
    api_client.mail_process()
    inc = api_client.incidents()[0]
    al = api_client.incident(inc["id"])["alerts"]
    assert [a["status"] for a in al] == ["dry_run"]
    txt = api_client.ask(inc["id"], "Was anyone notified?")["sections"][0]["text"]
    assert "test mode" in txt and "NOT sent" in txt
