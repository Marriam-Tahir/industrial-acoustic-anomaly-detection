"""Phase 7: deliver alert e-mails.

Reads 'pending' rows from the `alerts` table and delivers them, oldest first:
  - SMTP mode : sent through the configured mail server (for example Gmail with an app password).
                Status becomes 'sent' only after the server accepted the message.
  - test mode : used when no SMTP server is configured. The message is saved as an .eml file in data/app/outbox and the
                status becomes 'dry_run' (shown as TEST MODE - NOT SENT). Nothing is ever reported as sent in test mode.
A failed delivery is retried up to MAX_ATTEMPTS times with growing waits (1, 4, 16 minutes); after that the row is
'failed' and an administrator can put it back in the queue. At most `max_emails_per_hour` messages are delivered in any
60 minutes (all machines together); the rest wait in the queue. Reminders for unacknowledged incidents are added here.

Configuration (environment variables, or a file config/smtp.env with KEY=VALUE lines; the environment wins):
  SL_SMTP_HOST      e.g. smtp.gmail.com          (empty = test mode)
  SL_SMTP_PORT      587 for STARTTLS, 465 for SSL
  SL_SMTP_SECURITY  starttls | ssl | none        (default starttls)
  SL_SMTP_USER      the account name, e.g. you@gmail.com
  SL_SMTP_PASSWORD  for Gmail: a 16-character app password, NOT your normal password
  SL_SMTP_FROM      sender address (default: SL_SMTP_USER)
The password is never returned by the API, logged, or stored in the database.
"""
import smtplib
import ssl
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from email.utils import formatdate, make_msgid
from pathlib import Path
from typing import Optional

from smart_listener import alerts, db, repo

MAX_ATTEMPTS = 3
BACKOFF_MINUTES = [1, 4, 16]
BATCH = 20
ENV_FILE = Path("config") / "smtp.env"
KEYS = ["SL_SMTP_HOST", "SL_SMTP_PORT", "SL_SMTP_SECURITY", "SL_SMTP_USER", "SL_SMTP_PASSWORD", "SL_SMTP_FROM"]


@dataclass
class MailConfig:
    host: str = ""
    port: int = 587
    security: str = "starttls"
    user: str = ""
    password: str = ""
    sender: str = ""
    source: str = "not configured"

    @property
    def mode(self) -> str:
        return "smtp" if self.host else "test"

    def public(self) -> dict:
        """Safe to show: never includes the password."""
        return {"mode": self.mode, "host": self.host, "port": self.port, "security": self.security,
                "user": self.user, "sender": self.sender or self.user, "password_set": bool(self.password),
                "source": self.source}


def read_env_file(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def load_mail_config(root: Path, environ: Optional[dict] = None) -> MailConfig:
    import os
    if environ is None and "PYTEST_CURRENT_TEST" in os.environ:
        return MailConfig(source="disabled while tests run")   # the test suite must never send real e-mail
    env = os.environ if environ is None else environ
    path = Path(root) / ENV_FILE
    file_vals = read_env_file(path)
    vals = {k: env.get(k) or file_vals.get(k, "") for k in KEYS}
    src = "environment" if any(env.get(k) for k in KEYS) else (str(ENV_FILE) if file_vals else "not configured")
    security = (vals["SL_SMTP_SECURITY"] or "starttls").lower()
    if security not in ("starttls", "ssl", "none"):
        raise ValueError("SL_SMTP_SECURITY must be starttls, ssl or none")
    port = int(vals["SL_SMTP_PORT"] or (465 if security == "ssl" else 587))
    password = vals["SL_SMTP_PASSWORD"].replace(" ", "")   # Gmail shows app passwords in groups of four
    return MailConfig(vals["SL_SMTP_HOST"], port, security, vals["SL_SMTP_USER"], password,
                      vals["SL_SMTP_FROM"] or vals["SL_SMTP_USER"], src)


def build_message(alert: dict, sender: str) -> EmailMessage:
    msg = EmailMessage()
    msg["From"] = sender or "smart-listener@localhost"
    msg["To"] = alert["recipient"]
    msg["Subject"] = alert["subject"]
    msg["Date"] = formatdate(localtime=False)
    msg["Message-ID"] = make_msgid(domain="smart-listener.local")
    msg["X-Smart-Listener-Alert"] = str(alert["id"])
    msg.set_content(alert["body"] or alert["subject"])
    return msg


class SmtpSender:
    def __init__(self, cfg: MailConfig, timeout: float = 20.0):
        self.cfg, self.timeout = cfg, timeout

    def send(self, msg: EmailMessage) -> None:
        c = self.cfg
        if c.security == "ssl":
            server = smtplib.SMTP_SSL(c.host, c.port, timeout=self.timeout, context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(c.host, c.port, timeout=self.timeout)
        with server:
            if c.security == "starttls":
                server.starttls(context=ssl.create_default_context())
            if c.user:
                server.login(c.user, c.password)
            server.send_message(msg)


class OutboxSender:
    """Test mode: writes the message to a file instead of sending it."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)

    def send(self, msg: EmailMessage) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / f"alert_{msg['X-Smart-Listener-Alert']}.eml").write_bytes(bytes(msg))


def _iso(d: datetime) -> str:
    return d.astimezone(timezone.utc).isoformat(timespec="seconds")


def friendly_error(e: Exception) -> str:
    if isinstance(e, smtplib.SMTPAuthenticationError):
        return ("The mail server refused the user name or password. For Gmail use an app password "
                "(Google Account > Security > App passwords), not your normal password.")
    if isinstance(e, smtplib.SMTPRecipientsRefused):
        return "The mail server refused the recipient address."
    if isinstance(e, (TimeoutError, ConnectionError, OSError)) and not isinstance(e, smtplib.SMTPException):
        return f"Could not connect to the mail server ({type(e).__name__}: {e})."
    return f"{type(e).__name__}: {e}"


def process_once(conn, cfg: MailConfig, sender, outbox: Path, now: Optional[datetime] = None) -> dict:
    """One pass: add due reminders, then deliver due alerts within the hourly limit. Returns what happened."""
    now = now or datetime.now(timezone.utc)
    settings = repo.get_settings(conn)
    reminders = alerts.record_reminders(conn, settings, now)
    room = settings["max_emails_per_hour"] - repo.count_delivered_since(conn, _iso(now - timedelta(hours=1)))
    out = {"reminders_added": len(reminders), "sent": 0, "saved_test_mode": 0, "retry_later": 0, "failed": 0,
           "waiting_for_hourly_limit": 0, "mode": cfg.mode}
    due = repo.due_alerts(conn, _iso(now), BATCH)
    if room <= 0:
        out["waiting_for_hourly_limit"] = len(due)
        return out
    if len(due) > room:
        out["waiting_for_hourly_limit"] = len(due) - room
        due = due[:room]
    for a in due:
        attempts = a["attempts"] + 1
        try:
            if cfg.mode == "smtp":
                sender.send(build_message(a, cfg.sender))
                status, key = "sent", "sent"
            else:
                OutboxSender(outbox).send(build_message(a, cfg.sender))
                status, key = "dry_run", "saved_test_mode"
            repo.update_alert_delivery(conn, a["id"], status=status, attempts=attempts, sent_utc=_iso(now))
            out[key] += 1
        except Exception as e:  # any delivery problem is recorded on the row; the worker keeps going
            err = friendly_error(e)
            if attempts >= MAX_ATTEMPTS:
                repo.update_alert_delivery(conn, a["id"], status="failed", attempts=attempts, last_error=err)
                out["failed"] += 1
            else:
                nxt = now + timedelta(minutes=BACKOFF_MINUTES[attempts - 1])
                repo.update_alert_delivery(conn, a["id"], status="pending", attempts=attempts, last_error=err,
                                           next_attempt_utc=_iso(nxt))
                out["retry_later"] += 1
    return out


class MailWorker:
    """Runs process_once every `interval` seconds in a background thread with its own database connection."""

    def __init__(self, db_path, cfg: MailConfig, outbox: Path, interval: float = 30.0, sender=None):
        self.db_path, self.cfg, self.outbox, self.interval = db_path, cfg, Path(outbox), interval
        self.sender = sender or (SmtpSender(cfg) if cfg.mode == "smtp" else None)
        self.last = {"time_utc": None, "result": None, "error": ""}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread = None

    def run_now(self) -> dict:
        with self._lock:                       # the timer and a "send now" click never deliver the same row twice
            conn = db.connect(self.db_path)
            try:
                res = process_once(conn, self.cfg, self.sender, self.outbox)
                self.last = {"time_utc": _iso(datetime.now(timezone.utc)), "result": res, "error": ""}
                return res
            except Exception as e:
                self.last = {"time_utc": _iso(datetime.now(timezone.utc)), "result": None, "error": friendly_error(e)}
                raise
            finally:
                conn.close()

    def _loop(self):
        while not self._stop.wait(self.interval):
            try:
                self.run_now()
            except Exception:
                pass                           # recorded in self.last; try again next time

    def start(self):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="mail-worker", daemon=True)
            self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
