"""SQLite storage: file location, connection and table definitions.

One database file holds everything the application remembers: admins and login sessions, registered machines and
their owners, every analysed window, severity changes, incidents, e-mail alerts, settings and an audit trail.

Times are stored as UTC text such as 2026-11-08T07:30:00+00:00 (they sort correctly as text).
`window_utc` is the time the analysed sound belongs to (a simulated clock when monitoring is simulated);
`created_utc` is the real time the row was written.
"""
import os
import sqlite3
from pathlib import Path

from smart_listener.config import project_root

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_info (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS admins (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT NOT NULL UNIQUE COLLATE NOCASE,
  password_hash TEXT NOT NULL,
  active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
  created_utc TEXT NOT NULL,
  last_login_utc TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
  token_hash TEXT PRIMARY KEY,
  admin_id INTEGER NOT NULL REFERENCES admins(id) ON DELETE CASCADE,
  created_utc TEXT NOT NULL,
  expires_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS machines (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE COLLATE NOCASE,
  machine_type TEXT NOT NULL CHECK (machine_type IN ('fan', 'pump')),
  model_group TEXT NOT NULL,
  location TEXT NOT NULL DEFAULT '',
  owner_name TEXT NOT NULL,
  owner_email TEXT NOT NULL,
  notes TEXT NOT NULL DEFAULT '',
  active INTEGER NOT NULL DEFAULT 1 CHECK (active IN (0, 1)),
  created_utc TEXT NOT NULL,
  updated_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS monitoring_sessions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  machine_id INTEGER NOT NULL REFERENCES machines(id),
  source TEXT NOT NULL CHECK (source IN ('simulated', 'upload')),
  status TEXT NOT NULL DEFAULT 'running' CHECK (status IN ('running', 'stopped', 'finished', 'error')),
  started_utc TEXT NOT NULL,
  ended_utc TEXT,
  note TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS analyses (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  machine_id INTEGER NOT NULL REFERENCES machines(id),
  session_id INTEGER REFERENCES monitoring_sessions(id),
  source TEXT NOT NULL CHECK (source IN ('simulated', 'upload')),
  source_ref TEXT NOT NULL DEFAULT '',
  window_utc TEXT NOT NULL,
  created_utc TEXT NOT NULL,
  model_kind TEXT NOT NULL,
  score REAL NOT NULL,
  z REAL NOT NULL,
  z_warn REAL NOT NULL,
  z_alarm REAL NOT NULL,
  clip_level TEXT NOT NULL CHECK (clip_level IN ('within_normal', 'above_warning_limit', 'above_alarm_limit')),
  state TEXT CHECK (state IN ('NORMAL', 'WATCH', 'WARNING', 'HIGH')),
  reason TEXT NOT NULL DEFAULT '',
  evidence_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_analyses_machine_time ON analyses(machine_id, window_utc);

CREATE TABLE IF NOT EXISTS severity_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  machine_id INTEGER NOT NULL REFERENCES machines(id),
  analysis_id INTEGER REFERENCES analyses(id),
  from_state TEXT NOT NULL,
  to_state TEXT NOT NULL,
  reason TEXT NOT NULL DEFAULT '',
  event_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_severity_machine ON severity_events(machine_id, event_utc);

CREATE TABLE IF NOT EXISTS incidents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  machine_id INTEGER NOT NULL REFERENCES machines(id),
  status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
  opened_utc TEXT NOT NULL,
  closed_utc TEXT,
  opened_analysis_id INTEGER REFERENCES analyses(id),
  peak_z REAL NOT NULL DEFAULT 0,
  peak_score REAL NOT NULL DEFAULT 0,
  n_windows INTEGER NOT NULL DEFAULT 0,
  n_reopened INTEGER NOT NULL DEFAULT 0,
  acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (acknowledged IN (0, 1)),
  ack_by TEXT,
  ack_utc TEXT,
  ack_note TEXT NOT NULL DEFAULT '',
  summary_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS ix_incidents_machine ON incidents(machine_id, status);

-- operator notes on an incident (added with the dashboard redesign; additive, so schema version 1 still applies)
CREATE TABLE IF NOT EXISTS incident_notes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  incident_id INTEGER NOT NULL REFERENCES incidents(id),
  author TEXT NOT NULL,
  note TEXT NOT NULL,
  created_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  incident_id INTEGER REFERENCES incidents(id),
  machine_id INTEGER NOT NULL REFERENCES machines(id),
  kind TEXT NOT NULL CHECK (kind IN ('high_opened', 'reminder', 'closed', 'test')),
  recipient TEXT NOT NULL,
  subject TEXT NOT NULL,
  body TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'sent', 'failed', 'suppressed', 'dry_run')),
  attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT NOT NULL DEFAULT '',
  dedup_key TEXT NOT NULL UNIQUE,
  created_utc TEXT NOT NULL,
  sent_utc TEXT
);

CREATE TABLE IF NOT EXISTS settings (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_utc TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  actor TEXT NOT NULL,
  action TEXT NOT NULL,
  detail TEXT NOT NULL DEFAULT '',
  created_utc TEXT NOT NULL
);
"""


def default_db_path() -> Path:
    """data/app/smartlistener.db inside the project, or the file named by the SL_DB_PATH environment variable."""
    env = os.environ.get("SL_DB_PATH")
    return Path(env) if env else project_root() / "data" / "app" / "smartlistener.db"


def connect(path=None) -> sqlite3.Connection:
    """Open the database (creating the folder if needed). Rows behave like dictionaries.

    check_same_thread=False because the web server may use the connection from a worker thread; each request
    gets its own connection, so a connection is never used by two requests at the same time."""
    p = Path(path) if path else default_db_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), timeout=15, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 15000")
    try:
        conn.execute("PRAGMA journal_mode = WAL")
    except sqlite3.DatabaseError:
        pass  # some drives do not support WAL; the default mode still works
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    """Create every table that does not exist yet. Safe to run again and again; never deletes data."""
    conn.executescript(SCHEMA)
    with conn:
        conn.execute("INSERT OR IGNORE INTO schema_info(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    row = conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
    if int(row["value"]) != SCHEMA_VERSION:
        raise RuntimeError(f"database schema version is {row['value']}, this program needs {SCHEMA_VERSION}. "
                           "Ask for a migration instead of deleting the database.")
