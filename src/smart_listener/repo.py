"""Reading and writing the database: admins, sessions, machines, analyses, incidents, alerts, settings, audit.

Every function takes an open connection first. Every write is committed before the function returns.
Functions return plain dictionaries (or lists of them) so the API and the dashboard can use them directly.
"""
import json
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from smart_listener import security


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _one(row) -> Optional[dict]:
    return dict(row) if row is not None else None


def _all(rows) -> list:
    return [dict(r) for r in rows]


# ----------------------------------------------------------------------------- admins and sessions
def create_admin(conn, username: str, password: str) -> int:
    username = username.strip()
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,40}", username):
        raise ValueError("the username must be 3 to 40 characters: letters, digits, dot, dash or underscore")
    security.check_password_strength(password, username)
    try:
        with conn:
            cur = conn.execute("INSERT INTO admins(username, password_hash, created_utc) VALUES (?, ?, ?)",
                               (username, security.hash_password(password), now_utc()))
    except sqlite3.IntegrityError:
        raise ValueError("an admin with that username already exists") from None
    return cur.lastrowid


def count_admins(conn) -> int:
    return conn.execute("SELECT COUNT(*) AS n FROM admins").fetchone()["n"]


def authenticate(conn, username: str, password: str) -> Optional[dict]:
    """Return {id, username} if the login is right, otherwise None."""
    row = conn.execute("SELECT * FROM admins WHERE username = ? AND active = 1", (username.strip(),)).fetchone()
    ok = security.verify_password(password, row["password_hash"] if row else None)
    if row is None or not ok:
        return None
    with conn:
        conn.execute("UPDATE admins SET last_login_utc = ? WHERE id = ?", (now_utc(), row["id"]))
    return {"id": row["id"], "username": row["username"]}


def get_admin(conn, admin_id: int) -> Optional[dict]:
    return _one(conn.execute("SELECT id, username, active, created_utc, last_login_utc FROM admins WHERE id = ?",
                             (admin_id,)).fetchone())


def change_password(conn, admin_id: int, current: str, new: str, keep_token: Optional[str] = None) -> str:
    """Returns 'ok' or 'wrong_current'. Raises ValueError (plain English) if the new password is too weak.
    All other sessions of this admin are ended, so a leaked old session cannot be used any more."""
    row = conn.execute("SELECT * FROM admins WHERE id = ?", (admin_id,)).fetchone()
    if row is None or not security.verify_password(current, row["password_hash"]):
        return "wrong_current"
    security.check_password_strength(new, row["username"])
    with conn:
        conn.execute("UPDATE admins SET password_hash = ? WHERE id = ?", (security.hash_password(new), admin_id))
        conn.execute("DELETE FROM sessions WHERE admin_id = ? AND token_hash != ?",
                     (admin_id, security.hash_token(keep_token) if keep_token else ""))
    return "ok"


def create_session(conn, admin_id: int, hours: float = 12) -> str:
    token = security.new_token()
    start = datetime.now(timezone.utc)
    with conn:
        conn.execute("INSERT INTO sessions(token_hash, admin_id, created_utc, expires_utc) VALUES (?, ?, ?, ?)",
                     (security.hash_token(token), admin_id, start.isoformat(timespec="seconds"),
                      (start + timedelta(hours=hours)).isoformat(timespec="seconds")))
        conn.execute("DELETE FROM sessions WHERE expires_utc <= ?", (now_utc(),))
    return token


def admin_for_token(conn, token: str) -> Optional[dict]:
    row = conn.execute("""SELECT a.id, a.username, s.expires_utc FROM sessions s
                          JOIN admins a ON a.id = s.admin_id
                          WHERE s.token_hash = ? AND a.active = 1""", (security.hash_token(token),)).fetchone()
    if row is None:
        return None
    if row["expires_utc"] <= now_utc():
        delete_session(conn, token)
        return None
    return {"id": row["id"], "username": row["username"]}


def delete_session(conn, token: str) -> None:
    with conn:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (security.hash_token(token),))


# ----------------------------------------------------------------------------- machines
MACHINE_FIELDS = ("name", "machine_type", "model_group", "location", "owner_name", "owner_email", "notes", "active")


def create_machine(conn, *, name: str, machine_type: str, model_group: str, owner_name: str, owner_email: str,
                   location: str = "", notes: str = "") -> int:
    now = now_utc()
    try:
        with conn:
            cur = conn.execute(
                """INSERT INTO machines(name, machine_type, model_group, location, owner_name, owner_email, notes,
                                        created_utc, updated_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (name, machine_type, model_group, location, owner_name, owner_email, notes, now, now))
    except sqlite3.IntegrityError:
        raise ValueError("a machine with that name already exists") from None
    return cur.lastrowid


def get_machine(conn, machine_id: int) -> Optional[dict]:
    return _one(conn.execute("SELECT * FROM machines WHERE id = ?", (machine_id,)).fetchone())


def list_machines(conn, include_inactive: bool = False) -> list:
    sql = "SELECT * FROM machines" + ("" if include_inactive else " WHERE active = 1") + " ORDER BY name"
    return _all(conn.execute(sql).fetchall())


def update_machine(conn, machine_id: int, **changes) -> bool:
    """Change some fields of a machine. Returns False if the machine does not exist."""
    unknown = set(changes) - set(MACHINE_FIELDS)
    if unknown:
        raise ValueError(f"unknown machine fields: {sorted(unknown)}")
    if not changes:
        return get_machine(conn, machine_id) is not None
    if "active" in changes:
        changes["active"] = 1 if changes["active"] else 0
    sets = ", ".join(f"{k} = ?" for k in changes)
    try:
        with conn:
            cur = conn.execute(f"UPDATE machines SET {sets}, updated_utc = ? WHERE id = ?",
                               (*changes.values(), now_utc(), machine_id))
    except sqlite3.IntegrityError:
        raise ValueError("a machine with that name already exists") from None
    return cur.rowcount > 0


def fleet(conn, include_inactive: bool = False) -> list:
    """Every machine with its current severity state, last analysed window and open incident (if any)."""
    sql = """SELECT m.*, a.state AS current_state, a.window_utc AS last_window_utc, a.z AS last_z,
                    a.score AS last_score, a.source AS last_source,
                    (SELECT id FROM incidents WHERE machine_id = m.id AND status = 'open' ORDER BY id DESC LIMIT 1)
                        AS open_incident_id,
                    (SELECT COUNT(*) FROM incidents WHERE machine_id = m.id) AS n_incidents
             FROM machines m
             LEFT JOIN analyses a ON a.id = (SELECT id FROM analyses WHERE machine_id = m.id AND state IS NOT NULL
                                             ORDER BY id DESC LIMIT 1)"""
    if not include_inactive:
        sql += " WHERE m.active = 1"
    out = _all(conn.execute(sql + " ORDER BY m.name").fetchall())
    for r in out:
        r["current_state"] = r["current_state"] or "NO_DATA"
    return out


# ----------------------------------------------------------------------------- analyses
def add_analysis(conn, *, machine_id: int, source: str, window_utc: str, model_kind: str, score: float, z: float,
                 z_warn: float, z_alarm: float, clip_level: str, state: Optional[str] = None, reason: str = "",
                 session_id: Optional[int] = None, source_ref: str = "", evidence: Optional[dict] = None) -> int:
    with conn:
        cur = conn.execute(
            """INSERT INTO analyses(machine_id, session_id, source, source_ref, window_utc, created_utc, model_kind,
                                    score, z, z_warn, z_alarm, clip_level, state, reason, evidence_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (machine_id, session_id, source, source_ref, window_utc, now_utc(), model_kind, score, z, z_warn, z_alarm,
             clip_level, state, reason, json.dumps(evidence or {})))
    return cur.lastrowid


def list_analyses(conn, machine_id: int, limit: int = 100, offset: int = 0, source: Optional[str] = None,
                  session_id: Optional[int] = None) -> list:
    sql, args = "SELECT * FROM analyses WHERE machine_id = ?", [machine_id]
    if source:
        sql, args = sql + " AND source = ?", args + [source]
    if session_id:
        sql, args = sql + " AND session_id = ?", args + [session_id]
    sql += " ORDER BY window_utc DESC, id DESC LIMIT ? OFFSET ?"
    return _all(conn.execute(sql, (*args, limit, offset)).fetchall())


def get_analysis(conn, analysis_id: int) -> Optional[dict]:
    return _one(conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone())


def set_analysis_evidence(conn, analysis_id: int, evidence: dict) -> None:
    with conn:
        conn.execute("UPDATE analyses SET evidence_json = ? WHERE id = ?", (json.dumps(evidence), analysis_id))


def add_severity_event(conn, *, machine_id: int, from_state: str, to_state: str, event_utc: str,
                       analysis_id: Optional[int] = None, reason: str = "") -> int:
    with conn:
        cur = conn.execute("""INSERT INTO severity_events(machine_id, analysis_id, from_state, to_state, reason, event_utc)
                              VALUES (?, ?, ?, ?, ?, ?)""", (machine_id, analysis_id, from_state, to_state, reason, event_utc))
    return cur.lastrowid


def list_severity_events(conn, machine_id: int, limit: int = 100) -> list:
    return _all(conn.execute("SELECT * FROM severity_events WHERE machine_id = ? ORDER BY event_utc DESC, id DESC LIMIT ?",
                             (machine_id, limit)).fetchall())


def recent_activity(conn, limit: int = 20) -> list:
    """Latest severity changes and incident openings across the whole fleet, newest first."""
    sql = """SELECT * FROM (
               SELECT e.event_utc AS time_utc, 'severity' AS kind, e.machine_id, m.name AS machine_name,
                      m.model_group, e.from_state, e.to_state, NULL AS incident_id, a.source AS source
               FROM severity_events e JOIN machines m ON m.id = e.machine_id
               LEFT JOIN analyses a ON a.id = e.analysis_id
               UNION ALL
               SELECT i.opened_utc, 'incident_opened', i.machine_id, m.name, m.model_group, NULL, 'HIGH', i.id,
                      a.source
               FROM incidents i JOIN machines m ON m.id = i.machine_id
               LEFT JOIN analyses a ON a.id = i.opened_analysis_id)
             ORDER BY time_utc DESC LIMIT ?"""
    return _all(conn.execute(sql, (limit,)).fetchall())


# ----------------------------------------------------------------------------- incidents
_INCIDENT_SELECT = """SELECT i.*, m.name AS machine_name, m.machine_type, m.model_group, m.owner_name, m.owner_email
                      FROM incidents i JOIN machines m ON m.id = i.machine_id"""


def open_incident(conn, *, machine_id: int, opened_utc: str, analysis_id: Optional[int] = None, z: float = 0.0,
                  score: float = 0.0, summary: Optional[dict] = None) -> int:
    with conn:
        cur = conn.execute("""INSERT INTO incidents(machine_id, opened_utc, opened_analysis_id, peak_z, peak_score,
                                                    n_windows, summary_json) VALUES (?, ?, ?, ?, ?, 1, ?)""",
                           (machine_id, opened_utc, analysis_id, z, score, json.dumps(summary or {})))
    return cur.lastrowid


def reopen_incident(conn, incident_id: int) -> bool:
    """HIGH came back within the cool-down time: the same incident continues (no new incident, no new alert)."""
    with conn:
        cur = conn.execute("""UPDATE incidents SET status = 'open', closed_utc = NULL, n_reopened = n_reopened + 1
                              WHERE id = ? AND status = 'closed'""", (incident_id,))
    return cur.rowcount > 0


def update_incident_progress(conn, incident_id: int, z: float, score: float) -> None:
    """One more HIGH window inside the incident: count it and keep the peak."""
    with conn:
        conn.execute("""UPDATE incidents SET n_windows = n_windows + 1,
                              peak_score = CASE WHEN ? > peak_z THEN ? ELSE peak_score END,
                              peak_z = MAX(peak_z, ?) WHERE id = ?""", (z, score, z, incident_id))


def latest_incident(conn, machine_id: int) -> Optional[dict]:
    return _one(conn.execute("SELECT * FROM incidents WHERE machine_id = ? ORDER BY id DESC LIMIT 1", (machine_id,)).fetchone())


def get_incident(conn, incident_id: int) -> Optional[dict]:
    return _one(conn.execute(_INCIDENT_SELECT + " WHERE i.id = ?", (incident_id,)).fetchone())


def list_incidents(conn, status: Optional[str] = None, machine_id: Optional[int] = None, limit: int = 100) -> list:
    where, args = [], []
    if status:
        where.append("i.status = ?")
        args.append(status)
    if machine_id:
        where.append("i.machine_id = ?")
        args.append(machine_id)
    sql = _INCIDENT_SELECT + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY i.opened_utc DESC, i.id DESC LIMIT ?"
    return _all(conn.execute(sql, (*args, limit)).fetchall())


def acknowledge_incident(conn, incident_id: int, by: str, note: str = "") -> str:
    """Returns 'ok', 'not_found' or 'already'. An incident can be acknowledged while open or after it closed."""
    row = conn.execute("SELECT acknowledged FROM incidents WHERE id = ?", (incident_id,)).fetchone()
    if row is None:
        return "not_found"
    if row["acknowledged"]:
        return "already"
    with conn:
        conn.execute("UPDATE incidents SET acknowledged = 1, ack_by = ?, ack_utc = ?, ack_note = ? WHERE id = ?",
                     (by, now_utc(), note, incident_id))
    return "ok"


def add_incident_note(conn, incident_id: int, author: str, note: str) -> Optional[int]:
    if conn.execute("SELECT 1 FROM incidents WHERE id = ?", (incident_id,)).fetchone() is None:
        return None
    with conn:
        cur = conn.execute("INSERT INTO incident_notes(incident_id, author, note, created_utc) VALUES (?, ?, ?, ?)",
                           (incident_id, author, note, now_utc()))
    return cur.lastrowid


def list_incident_notes(conn, incident_id: int) -> list:
    return _all(conn.execute("SELECT * FROM incident_notes WHERE incident_id = ? ORDER BY id", (incident_id,)).fetchall())


def list_analyses_between(conn, machine_id: Optional[int], date_from: Optional[str], date_to: Optional[str],
                          source: Optional[str] = None, limit: int = 100000) -> list:
    """Analysed windows (oldest first) for reports. Dates are 'YYYY-MM-DD' (UTC), both ends included."""
    where, args = [], []
    if machine_id:
        where.append("a.machine_id = ?"); args.append(machine_id)
    if date_from:
        where.append("a.window_utc >= ?"); args.append(date_from)
    if date_to:
        where.append("a.window_utc < date(?, '+1 day')"); args.append(date_to)
    if source:
        where.append("a.source = ?"); args.append(source)
    sql = ("SELECT a.*, m.name AS machine_name, m.model_group FROM analyses a JOIN machines m ON m.id = a.machine_id"
           + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY a.window_utc, a.id LIMIT ?")
    return _all(conn.execute(sql, (*args, limit)).fetchall())


def close_incident(conn, incident_id: int, closed_utc: str) -> bool:
    with conn:
        cur = conn.execute("UPDATE incidents SET status = 'closed', closed_utc = ? WHERE id = ? AND status = 'open'",
                           (closed_utc, incident_id))
    return cur.rowcount > 0


# ----------------------------------------------------------------------------- monitoring sessions
def create_session_row(conn, machine_id: int, source: str, note: str = "") -> int:
    with conn:
        cur = conn.execute("INSERT INTO monitoring_sessions(machine_id, source, status, started_utc, note) "
                           "VALUES (?, ?, 'running', ?, ?)", (machine_id, source, now_utc(), note[:500]))
    return cur.lastrowid


def finish_session_row(conn, session_id: int, status: str, note: str = "") -> None:
    with conn:
        conn.execute("UPDATE monitoring_sessions SET status = ?, ended_utc = ?, "
                     "note = CASE WHEN ? = '' THEN note ELSE ? END WHERE id = ? AND status = 'running'",
                     (status, now_utc(), note[:500], note[:500], session_id))


def get_session_row(conn, session_id: int) -> Optional[dict]:
    return _one(conn.execute("SELECT * FROM monitoring_sessions WHERE id = ?", (session_id,)).fetchone())


def list_session_rows(conn, machine_id: Optional[int] = None, status: Optional[str] = None, limit: int = 50) -> list:
    where, args = [], []
    if machine_id:
        where.append("machine_id = ?")
        args.append(machine_id)
    if status:
        where.append("status = ?")
        args.append(status)
    sql = "SELECT * FROM monitoring_sessions" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT ?"
    return _all(conn.execute(sql, (*args, limit)).fetchall())


def fail_stale_sessions(conn) -> int:
    """Sessions still marked 'running' when the server starts belong to a previous server run (their threads are gone).
    Mark them as errors and close their open incidents. Returns how many were fixed."""
    rows = _all(conn.execute("SELECT id, machine_id FROM monitoring_sessions WHERE status = 'running'").fetchall())
    for r in rows:
        finish_session_row(conn, r["id"], "error", "the server was restarted while this session was running")
    if rows:
        ids = [r["machine_id"] for r in rows]
        marks = ",".join("?" * len(ids))
        with conn:
            conn.execute(f"UPDATE incidents SET status = 'closed', closed_utc = ? WHERE status = 'open' "
                         f"AND machine_id IN ({marks})", (now_utc(), *ids))
    return len(rows)


# ----------------------------------------------------------------------------- alerts (e-mail records)
def add_alert(conn, *, machine_id: int, kind: str, recipient: str, subject: str, dedup_key: str,
              incident_id: Optional[int] = None, body: str = "", status: str = "pending") -> Optional[int]:
    """Record an alert. Returns None if an alert with the same dedup_key already exists (this is what stops
    the same incident from being e-mailed twice)."""
    try:
        with conn:
            cur = conn.execute("""INSERT INTO alerts(incident_id, machine_id, kind, recipient, subject, body, status,
                                                     dedup_key, created_utc) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                               (incident_id, machine_id, kind, recipient, subject, body, status, dedup_key, now_utc()))
    except sqlite3.IntegrityError:
        return None
    return cur.lastrowid


def list_alerts(conn, incident_id: Optional[int] = None, machine_id: Optional[int] = None, limit: int = 100) -> list:
    where, args = [], []
    if incident_id:
        where.append("incident_id = ?")
        args.append(incident_id)
    if machine_id:
        where.append("machine_id = ?")
        args.append(machine_id)
    sql = "SELECT * FROM alerts" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY id DESC LIMIT ?"
    return _all(conn.execute(sql, (*args, limit)).fetchall())


# ----------------------------------------------------------------------------- settings
# name: (type, default, minimum, maximum)
SETTING_SPECS = {
    "alerts_enabled": ("bool", True, None, None),            # master switch for e-mail alerts
    "max_emails_per_hour": ("int", 10, 1, 100),              # global e-mail limit, all machines together
    "reminder_minutes": ("int", 0, 0, 1440),                 # 0 = no reminder e-mails for an unacknowledged incident
    "replay_seconds_per_window": ("float", 1.0, 0.05, 10.0),  # simulated monitoring: real seconds per 10-second clip
}


def _coerce_setting(key: str, value: Any):
    if key not in SETTING_SPECS:
        raise ValueError(f"unknown setting '{key}'. Allowed: {sorted(SETTING_SPECS)}")
    kind, _, lo, hi = SETTING_SPECS[key]
    if kind == "bool":
        if not isinstance(value, bool):
            raise ValueError(f"{key} must be true or false")
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    value = int(value) if kind == "int" else float(value)
    if (kind == "int" and value != value) or not lo <= value <= hi:
        raise ValueError(f"{key} must be between {lo} and {hi}")
    return value


def get_settings(conn) -> dict:
    out = {k: spec[1] for k, spec in SETTING_SPECS.items()}
    for row in conn.execute("SELECT key, value FROM settings").fetchall():
        if row["key"] in out:
            out[row["key"]] = json.loads(row["value"])
    return out


def update_settings(conn, changes: dict) -> dict:
    clean = {k: _coerce_setting(k, v) for k, v in changes.items()}   # validate everything before saving anything
    with conn:
        for k, v in clean.items():
            conn.execute("""INSERT INTO settings(key, value, updated_utc) VALUES (?, ?, ?)
                            ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_utc = excluded.updated_utc""",
                         (k, json.dumps(v), now_utc()))
    return get_settings(conn)


# ----------------------------------------------------------------------------- audit trail
def audit(conn, actor: str, action: str, detail: str = "") -> None:
    with conn:
        conn.execute("INSERT INTO audit_log(actor, action, detail, created_utc) VALUES (?, ?, ?, ?)",
                     (actor, action, detail[:500], now_utc()))


def list_audit(conn, limit: int = 100) -> list:
    return _all(conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall())
