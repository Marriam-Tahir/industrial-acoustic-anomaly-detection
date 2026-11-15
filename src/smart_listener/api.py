"""The backend web API (FastAPI). Everything except /health and /auth/login needs a login token.

Start it with:   python scripts/run_api.py        then open  http://127.0.0.1:8000/docs
Phase 4 provides: login, the machine registry (owners and e-mail addresses), fleet overview, incident
acknowledgement, history and alert listings, settings, model profiles and an audit trail.
Phase 5 adds audio upload and analysis, simulated continuous monitoring and evidence pictures.
Phases 6 to 7 add the dashboard, e-mail sending, the assistant and reports on top of this.
"""
import json
import os
import re
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import Body, Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, field_validator

from smart_listener import db, profiles, repo
from smart_listener.config import load_config, project_root

EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
MAX_LOGIN_FAILURES = 5
LOGIN_WINDOW_SECONDS = 300

bearer = HTTPBearer(auto_error=False)


# ----------------------------------------------------------------------------- request bodies
class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=60)
    password: str = Field(min_length=1, max_length=200)


def _clean_email(v: str) -> str:
    import re
    v = v.strip()
    if len(v) > 254 or not re.match(EMAIL_RE, v):
        raise ValueError("owner_email must look like name@example.com")
    return v


class MachineIn(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    machine_type: Literal["fan", "pump"]
    model_group: str = Field(min_length=1, max_length=40)
    location: str = Field("", max_length=120)
    owner_name: str = Field(min_length=1, max_length=80)
    owner_email: str
    notes: str = Field("", max_length=500)

    @field_validator("name", "owner_name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("must not be empty")
        return v

    @field_validator("owner_email")
    @classmethod
    def _email(cls, v: str) -> str:
        return _clean_email(v)


class MachineUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=60)
    machine_type: Optional[Literal["fan", "pump"]] = None
    model_group: Optional[str] = Field(None, min_length=1, max_length=40)
    location: Optional[str] = Field(None, max_length=120)
    owner_name: Optional[str] = Field(None, min_length=1, max_length=80)
    owner_email: Optional[str] = None
    notes: Optional[str] = Field(None, max_length=500)
    active: Optional[bool] = None

    @field_validator("owner_email")
    @classmethod
    def _email(cls, v):
        return None if v is None else _clean_email(v)


class AckIn(BaseModel):
    note: str = Field("", max_length=500)


class PasswordIn(BaseModel):
    current_password: str = Field(..., max_length=200)
    new_password: str = Field(..., max_length=200)


class TestMailIn(BaseModel):
    recipient: str

    @field_validator("recipient")
    @classmethod
    def _ok(cls, v):
        v = v.strip()
        if len(v) > 254 or not re.match(EMAIL_RE, v):
            raise ValueError("recipient must look like name@example.com")
        return v


class NoteIn(BaseModel):
    note: str = Field(..., min_length=1, max_length=1000)


class AskIn(BaseModel):
    incident_id: int
    question: str = Field(..., min_length=1, max_length=500)


class MonitorStartIn(BaseModel):
    """Options of a SIMULATED monitoring session (replayed dataset recordings)."""
    scenario: Literal["normal", "fault", "fault_recovery"] = "fault"
    split: Literal["val", "test"] = "test"
    n_normal_before: int = Field(12, ge=0, le=500)
    n_fault: int = Field(24, ge=0, le=500)
    n_normal_after: int = Field(12, ge=0, le=500)
    n_windows: int = Field(30, ge=1, le=500)


MAX_UPLOAD_BYTES = 60 * 1024 * 1024      # 10 minutes of 16 kHz 16-bit mono is about 19 MB; 8-channel files are larger


# ----------------------------------------------------------------------------- dependencies
def get_conn(request: Request):
    conn = db.connect(request.app.state.db_path)
    try:
        yield conn
    finally:
        conn.close()


def current_admin(request: Request, creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer),
                  conn=Depends(get_conn)) -> dict:
    admin = repo.admin_for_token(conn, creds.credentials) if creds else None
    if admin is None:
        raise HTTPException(status_code=401, detail="Please log in (missing, wrong or expired token).",
                            headers={"WWW-Authenticate": "Bearer"})
    return admin


def _check_group(groups: set, machine_type: str, model_group: str) -> None:
    if model_group not in groups:
        raise HTTPException(422, f"model_group must be one of {sorted(groups)}")
    if not model_group.startswith(machine_type + "_"):
        raise HTTPException(422, f"model_group '{model_group}' does not belong to machine type '{machine_type}'")


# ----------------------------------------------------------------------------- analysis helpers
def _analyzer(app):
    """The shared Analyzer, created on first use. None (and a stored reason) if the calibration is missing."""
    if app.state.analyzer is None and app.state.analyzer_error is None:
        try:
            from smart_listener.inference import Analyzer
            app.state.analyzer = Analyzer(app.state.cfg)
        except Exception as e:  # missing calibration file, torch not installed, ...
            app.state.analyzer_error = f"{type(e).__name__}: {e}"
    return app.state.analyzer


def _manager(app):
    if app.state.monitor is None:
        from smart_listener.monitor import MonitorManager
        app.state.monitor = MonitorManager(app.state.db_path, _analyzer(app), app.state.cfg, app.state.evidence_dir,
                                           source_factory=app.state.source_factory)
    return app.state.monitor


def _cleanup_only(app) -> int:
    conn = db.connect(app.state.db_path)
    try:
        return repo.fail_stale_sessions(conn)
    finally:
        conn.close()


def need_analyzer(request: Request):
    a = _analyzer(request.app)
    if a is None:
        raise HTTPException(503, f"Analysis is not available: {request.app.state.analyzer_error}")
    return a


def _save_upload(file, limit: int) -> str:
    """Copy the uploaded file to a temporary path (the caller deletes it). Refuses files above `limit` bytes."""
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    try:
        size = 0
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, f"The file is larger than {limit // (1024 * 1024)} MB.")
                out.write(chunk)
    except BaseException:
        os.remove(tmp)
        raise
    return tmp


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", Path(name or "upload.wav").name)[:100] or "upload.wav"


# ----------------------------------------------------------------------------- the application
def create_app(db_path=None, cfg: Optional[dict] = None, session_hours: float = 12, analyzer=None,
               evidence_dir=None, source_factory=None, max_upload_bytes: int = MAX_UPLOAD_BYTES,
               mail_config=None, mail_sender=None, mail_interval: float = 30.0, start_mail_worker: bool = False) -> FastAPI:
    """analyzer / source_factory can be passed in (tests do this); otherwise the real models and the dataset replay
    are used. The analyzer is created on first use, so the API starts even when no model file exists yet."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.monitor_stale_fixed = _cleanup_only(app)   # sessions left 'running' by a previous server run
        if start_mail_worker:
            app.state.mail.start()
        yield
        app.state.mail.stop()
        if getattr(app.state, "monitor", None) is not None:
            app.state.monitor.shutdown()

    app = FastAPI(title="Smart Listener API", version="0.5.0", lifespan=lifespan,
                  description="Industrial acoustic anomaly monitoring: machine registry, audio upload, simulated "
                              "monitoring, incidents and alerts. Anomaly scores are not failure probabilities and do "
                              "not identify a specific fault.")
    app.state.db_path = db_path or db.default_db_path()
    app.state.cfg = cfg or load_config()
    app.state.analyzer = analyzer
    app.state.analyzer_error = None
    app.state.monitor = None
    app.state.source_factory = source_factory
    app.state.evidence_dir = Path(evidence_dir) if evidence_dir else Path(app.state.db_path).parent / "evidence"
    app.state.max_upload_bytes = max_upload_bytes
    app.state.groups = set(profiles.model_groups(app.state.cfg))
    app.state.session_hours = session_hours
    app.state.failures = {}
    from smart_listener import notifier
    app.state.mail_config_error = ""
    if mail_config is None:
        try:
            mail_config = notifier.load_mail_config(project_root())
        except ValueError as e:     # a broken mail setting must not stop the API; fall back to test mode and say why
            app.state.mail_config_error = str(e)
            mail_config = notifier.MailConfig()
    app.state.mail = notifier.MailWorker(app.state.db_path, mail_config, Path(app.state.db_path).parent / "outbox",
                                         mail_interval, mail_sender)
    app.state.mail_worker_started = start_mail_worker
    conn = db.connect(app.state.db_path)
    try:
        db.init_schema(conn)
    finally:
        conn.close()

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        """Basic hardening for every response: no MIME sniffing, no framing, no caching of private data."""
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if not request.url.path.startswith(("/docs", "/redoc", "/openapi.json")):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    # ---- health and login
    @app.get("/health")
    def health():
        return {"status": "ok", "schema_version": db.SCHEMA_VERSION}

    @app.post("/auth/login")
    def login(body: LoginIn, request: Request, conn=Depends(get_conn)):
        key = f"{body.username.strip().lower()}|{request.client.host if request.client else '-'}"
        now = time.time()
        recent = [t for t in request.app.state.failures.get(key, []) if now - t < LOGIN_WINDOW_SECONDS]
        if len(recent) >= MAX_LOGIN_FAILURES:
            raise HTTPException(429, "Too many failed logins. Wait a few minutes and try again.",
                                headers={"Retry-After": str(LOGIN_WINDOW_SECONDS)})
        admin = repo.authenticate(conn, body.username, body.password)
        if admin is None:
            recent.append(now)
            request.app.state.failures[key] = recent
            repo.audit(conn, body.username.strip()[:40], "login.failed")
            raise HTTPException(401, "Wrong username or password.")
        request.app.state.failures.pop(key, None)
        token = repo.create_session(conn, admin["id"], request.app.state.session_hours)
        repo.audit(conn, admin["username"], "login.ok")
        return {"token": token, "token_type": "bearer", "username": admin["username"],
                "expires_in_hours": request.app.state.session_hours}

    @app.post("/auth/logout")
    def logout(creds: HTTPAuthorizationCredentials = Depends(bearer), admin=Depends(current_admin), conn=Depends(get_conn)):
        repo.delete_session(conn, creds.credentials)
        repo.audit(conn, admin["username"], "logout")
        return {"status": "logged out"}

    @app.get("/auth/me")
    def me(admin=Depends(current_admin)):
        return admin

    # ---- model profiles
    @app.get("/model-profiles")
    def model_profiles(request: Request, admin=Depends(current_admin)):
        return profiles.model_profiles(request.app.state.cfg, project_root())

    # ---- machine registry
    @app.get("/machines")
    def machines(include_inactive: bool = False, admin=Depends(current_admin), conn=Depends(get_conn)):
        return repo.list_machines(conn, include_inactive)

    @app.post("/machines", status_code=201)
    def create_machine(body: MachineIn, request: Request, admin=Depends(current_admin), conn=Depends(get_conn)):
        _check_group(request.app.state.groups, body.machine_type, body.model_group)
        try:
            mid = repo.create_machine(conn, **body.model_dump())
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        repo.audit(conn, admin["username"], "machine.create", f"id={mid} name={body.name} group={body.model_group}")
        return repo.get_machine(conn, mid)

    @app.get("/machines/{machine_id}")
    def machine(machine_id: int, admin=Depends(current_admin), conn=Depends(get_conn)):
        m = repo.get_machine(conn, machine_id)
        if m is None:
            raise HTTPException(404, "machine not found")
        return m

    @app.put("/machines/{machine_id}")
    def update_machine(machine_id: int, body: MachineUpdate, request: Request, admin=Depends(current_admin),
                       conn=Depends(get_conn)):
        current = repo.get_machine(conn, machine_id)
        if current is None:
            raise HTTPException(404, "machine not found")
        changes = body.model_dump(exclude_none=True)
        merged = {**current, **changes}
        _check_group(request.app.state.groups, merged["machine_type"], merged["model_group"])
        try:
            repo.update_machine(conn, machine_id, **changes)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        repo.audit(conn, admin["username"], "machine.update", f"id={machine_id} fields={sorted(changes)}")
        return repo.get_machine(conn, machine_id)

    @app.delete("/machines/{machine_id}")
    def deactivate_machine(machine_id: int, admin=Depends(current_admin), conn=Depends(get_conn)):
        """Machines are never erased (their history must stay); this marks the machine inactive."""
        if not repo.update_machine(conn, machine_id, active=False):
            raise HTTPException(404, "machine not found")
        repo.audit(conn, admin["username"], "machine.deactivate", f"id={machine_id}")
        return repo.get_machine(conn, machine_id)

    @app.get("/machines/{machine_id}/analyses")
    def machine_analyses(machine_id: int, limit: int = Query(100, ge=1, le=1000), offset: int = Query(0, ge=0),
                         source: Optional[Literal["simulated", "upload"]] = None, session_id: Optional[int] = None,
                         admin=Depends(current_admin), conn=Depends(get_conn)):
        if repo.get_machine(conn, machine_id) is None:
            raise HTTPException(404, "machine not found")
        return repo.list_analyses(conn, machine_id, limit, offset, source, session_id)

    @app.get("/machines/{machine_id}/severity-events")
    def machine_severity_events(machine_id: int, limit: int = Query(100, ge=1, le=1000), admin=Depends(current_admin),
                                conn=Depends(get_conn)):
        if repo.get_machine(conn, machine_id) is None:
            raise HTTPException(404, "machine not found")
        return repo.list_severity_events(conn, machine_id, limit)

    # ---- fleet, incidents, alerts
    @app.get("/fleet")
    def fleet(include_inactive: bool = False, admin=Depends(current_admin), conn=Depends(get_conn)):
        return repo.fleet(conn, include_inactive)

    @app.get("/incidents")
    def incidents(status: Optional[Literal["open", "closed"]] = None, machine_id: Optional[int] = None,
                  limit: int = Query(100, ge=1, le=1000), admin=Depends(current_admin), conn=Depends(get_conn)):
        return repo.list_incidents(conn, status, machine_id, limit)

    @app.get("/incidents/{incident_id}")
    def incident(incident_id: int, admin=Depends(current_admin), conn=Depends(get_conn)):
        inc = repo.get_incident(conn, incident_id)
        if inc is None:
            raise HTTPException(404, "incident not found")
        inc["alerts"] = repo.list_alerts(conn, incident_id=incident_id)
        return inc

    @app.post("/incidents/{incident_id}/acknowledge")
    def acknowledge(incident_id: int, body: AckIn, admin=Depends(current_admin), conn=Depends(get_conn)):
        result = repo.acknowledge_incident(conn, incident_id, admin["username"], body.note.strip())
        if result == "not_found":
            raise HTTPException(404, "incident not found")
        if result == "already":
            raise HTTPException(409, "this incident was already acknowledged")
        repo.audit(conn, admin["username"], "incident.acknowledge", f"id={incident_id}")
        return repo.get_incident(conn, incident_id)

    @app.get("/alerts")
    def alerts(incident_id: Optional[int] = None, machine_id: Optional[int] = None,
               status: Optional[Literal["pending", "sent", "failed", "suppressed", "dry_run"]] = None,
               limit: int = Query(100, ge=1, le=1000), admin=Depends(current_admin), conn=Depends(get_conn)):
        return repo.list_alerts(conn, incident_id, machine_id, limit, status)

    # ---- Phase 5: upload and analysis
    def _active_machine(conn, machine_id: int) -> dict:
        m = repo.get_machine(conn, machine_id)
        if m is None:
            raise HTTPException(404, "machine not found")
        if not m["active"]:
            raise HTTPException(409, "this machine is inactive; activate it first")
        return m

    def _upload_email(request: Request, conn, machine: dict, result: dict, file_name: str, warnings, caution: str) -> dict:
        """E-mail the owner when an uploaded file's result is not normal (see alerts.analysis_outcome). A problem here
        is reported in the answer but never stops the analysis result from being returned."""
        from smart_listener import alerts, severity
        try:
            out = alerts.record_analysis_alert(conn, machine, result, repo.get_settings(conn),
                                               Path(request.app.state.db_path).parent,
                                               {**severity.DEFAULT_RULE, **(request.app.state.analyzer.rule or {})}["n_window"], file_name, warnings, caution,
                                               request.app.state.evidence_dir)
        except Exception as e:   # the analysis itself is already saved
            return {"needed": None, "reason": f"could not prepare the e-mail: {type(e).__name__}: {e}",
                    "alert_id": None, "status": "error"}
        if out["alert_id"] and out["status"] == "pending" and request.app.state.mail_worker_started:
            def deliver():
                try:
                    request.app.state.mail.run_now()      # same lock as the timer: never sent twice
                except Exception:
                    pass                                   # recorded on the worker; the timer tries again
            threading.Thread(target=deliver, name="mail-upload", daemon=True).start()
        return out

    @app.post("/machines/{machine_id}/analyze-upload")
    def analyze_upload(machine_id: int, request: Request, file: UploadFile = File(...), admin=Depends(current_admin),
                       conn=Depends(get_conn)):
        """Analyse an uploaded WAV file with the model of the SELECTED machine. The machine is chosen by you; it is
        not recognised from the sound. A file name such as fan_id_06_... is only compared with your choice."""
        from smart_listener import inference, monitor
        machine = _active_machine(conn, machine_id)
        analyzer = need_analyzer(request)
        ok, why = analyzer.available(machine["model_group"])
        if not ok:
            raise HTTPException(409, f"Cannot analyse for {machine['name']}: {why}")
        tmp = _save_upload(file, request.app.state.max_upload_bytes)
        try:
            try:
                prepared = inference.prepare_upload(tmp, request.app.state.cfg)
            except inference.AudioRejected as e:
                raise HTTPException(422, str(e)) from None
            warning = inference.check_hint(inference.filename_hint(file.filename or ""), machine["model_group"])
            if warning:
                prepared.warnings.append(warning)
            result = monitor.analyze_upload(conn, analyzer, machine, prepared, _safe_name(file.filename),
                                            request.app.state.evidence_dir)
        finally:
            try:
                os.remove(tmp)                       # the uploaded audio itself is never kept
            except OSError:
                pass
        profile = next((p for p in profiles.model_profiles(request.app.state.cfg, project_root())
                        if p["group"] == machine["model_group"]), {})
        repo.audit(conn, admin["username"], "upload.analyze",
                   f"machine={machine_id} file={_safe_name(file.filename)} windows={result['summary']['n_windows']}")
        email = _upload_email(request, conn, machine, result, _safe_name(file.filename), prepared.warnings,
                              profile.get("caution") or "")
        return {"machine": {k: machine[k] for k in ("id", "name", "machine_type", "model_group")},
                "file": {"name": _safe_name(file.filename), **prepared.info}, "warnings": prepared.warnings,
                "model_caution": profile.get("caution"), **result, "email": email,
                "notice": "Anomaly levels show how unlike this machine's normal sound a recording is. They are not "
                          "failure probabilities and do not identify a fault. The audio file was not kept."}

    # ---- Phase 5: simulated monitoring
    @app.get("/monitoring/scenarios")
    def monitoring_scenarios(admin=Depends(current_admin)):
        from smart_listener.monitor import SCENARIOS
        return {"simulated": True, "scenarios": SCENARIOS,
                "note": "Simulated monitoring replays recorded dataset clips one after another. It is a demonstration, "
                        "not a connection to a real machine."}

    @app.post("/machines/{machine_id}/monitoring/start", status_code=201)
    def start_monitoring(machine_id: int, body: MonitorStartIn, request: Request, admin=Depends(current_admin),
                         conn=Depends(get_conn)):
        from smart_listener.monitor import MonitorError
        machine = _active_machine(conn, machine_id)
        need_analyzer(request)
        try:
            session = _manager(request.app).start(machine, body.model_dump())
        except MonitorError as e:
            raise HTTPException(e.code, str(e)) from None
        repo.audit(conn, admin["username"], "monitoring.start", f"machine={machine_id} session={session['id']} {body.scenario}")
        return session

    @app.get("/monitoring")
    def monitoring_sessions(machine_id: Optional[int] = None, status: Optional[str] = None,
                            limit: int = Query(50, ge=1, le=500), admin=Depends(current_admin), conn=Depends(get_conn)):
        return repo.list_session_rows(conn, machine_id, status, limit)

    @app.get("/monitoring/{session_id}")
    def monitoring_status(session_id: int, request: Request, admin=Depends(current_admin)):
        if request.app.state.monitor is None:
            conn = db.connect(request.app.state.db_path)
            try:
                row = repo.get_session_row(conn, session_id)
            finally:
                conn.close()
            if row is None:
                raise HTTPException(404, "monitoring session not found")
            return {**row, "progress": None, "simulated": row["source"] == "simulated"}
        st = request.app.state.monitor.status(session_id)
        if st is None:
            raise HTTPException(404, "monitoring session not found")
        return st

    @app.post("/monitoring/{session_id}/stop")
    def monitoring_stop(session_id: int, request: Request, admin=Depends(current_admin), conn=Depends(get_conn)):
        row = repo.get_session_row(conn, session_id)
        if row is None:
            raise HTTPException(404, "monitoring session not found")
        mgr = request.app.state.monitor
        if mgr is None or not mgr.stop(session_id):
            raise HTTPException(409, "this session is not running")
        repo.audit(conn, admin["username"], "monitoring.stop", f"session={session_id}")
        return mgr.status(session_id)

    # ---- Phase 5: evidence pictures
    @app.get("/analyses/{analysis_id}")
    def analysis(analysis_id: int, admin=Depends(current_admin), conn=Depends(get_conn)):
        a = repo.get_analysis(conn, analysis_id)
        if a is None:
            raise HTTPException(404, "analysis not found")
        return a

    @app.get("/analyses/{analysis_id}/evidence.png")
    def analysis_evidence(analysis_id: int, request: Request, admin=Depends(current_admin), conn=Depends(get_conn)):
        import json as _json
        a = repo.get_analysis(conn, analysis_id)
        if a is None:
            raise HTTPException(404, "analysis not found")
        fig = _json.loads(a["evidence_json"] or "{}").get("figure")
        path = request.app.state.evidence_dir / f"m{a['machine_id']}" / fig if fig and re.fullmatch(r"a\d+\.png", fig) else None
        if path is None or not path.exists():
            raise HTTPException(404, "no evidence picture was saved for this window (pictures are made for windows "
                                     "above the warning limit and for the highest window of an upload)")
        return Response(path.read_bytes(), media_type="image/png")

    @app.get("/machines/{machine_id}/history.png")
    def machine_history_png(machine_id: int, request: Request, limit: int = Query(200, ge=2, le=2000),
                            session_id: Optional[int] = None, admin=Depends(current_admin), conn=Depends(get_conn)):
        from smart_listener import evidence
        machine = repo.get_machine(conn, machine_id)
        if machine is None:
            raise HTTPException(404, "machine not found")
        rows = [r for r in repo.list_analyses(conn, machine_id, limit, 0, "simulated", session_id) if r["state"]]
        rows.reverse()
        zw = rows[0]["z_warn"] if rows else 0.0
        za = rows[0]["z_alarm"] if rows else 0.0
        png = evidence.history_figure(rows, zw, za, f"{machine['name']} - anomaly level over time (simulated monitoring)")
        return Response(png, media_type="image/png")

    # ---- settings and audit
    @app.get("/settings")
    def get_settings(admin=Depends(current_admin), conn=Depends(get_conn)):
        return repo.get_settings(conn)

    @app.put("/settings")
    def put_settings(changes: dict[str, Any] = Body(...), admin=Depends(current_admin), conn=Depends(get_conn)):
        try:
            new = repo.update_settings(conn, changes)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        repo.audit(conn, admin["username"], "settings.update", ", ".join(f"{k}={v}" for k, v in changes.items()))
        return new

    @app.get("/audit")
    def audit_log(limit: int = Query(100, ge=1, le=1000), admin=Depends(current_admin), conn=Depends(get_conn)):
        return repo.list_audit(conn, limit)

    # ---- redesign additions: account, activity, notes, identification, assistant, reports, system info
    @app.post("/auth/password")
    def change_password(body: PasswordIn, creds: HTTPAuthorizationCredentials = Depends(bearer),
                        admin=Depends(current_admin), conn=Depends(get_conn)):
        try:
            r = repo.change_password(conn, admin["id"], body.current_password, body.new_password, creds.credentials)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        if r == "wrong_current":
            raise HTTPException(403, "The current password is wrong.")
        repo.audit(conn, admin["username"], "password.change")
        return {"status": "password changed; other sessions were signed out"}

    @app.get("/account")
    def account(admin=Depends(current_admin), conn=Depends(get_conn)):
        return repo.get_admin(conn, admin["id"])

    @app.get("/activity")
    def activity(limit: int = Query(20, ge=1, le=200), admin=Depends(current_admin), conn=Depends(get_conn)):
        return repo.recent_activity(conn, limit)

    @app.get("/incidents/{incident_id}/notes")
    def incident_notes(incident_id: int, admin=Depends(current_admin), conn=Depends(get_conn)):
        if repo.get_incident(conn, incident_id) is None:
            raise HTTPException(404, "incident not found")
        return repo.list_incident_notes(conn, incident_id)

    @app.post("/incidents/{incident_id}/notes", status_code=201)
    def add_incident_note(incident_id: int, body: NoteIn, admin=Depends(current_admin), conn=Depends(get_conn)):
        note = body.note.strip()
        if not note:
            raise HTTPException(422, "the note is empty")
        nid = repo.add_incident_note(conn, incident_id, admin["username"], note)
        if nid is None:
            raise HTTPException(404, "incident not found")
        repo.audit(conn, admin["username"], "incident.note", f"id={incident_id}")
        return repo.list_incident_notes(conn, incident_id)

    @app.get("/system")
    def system_info(request: Request, admin=Depends(current_admin)):
        a = _analyzer(request.app)
        mode = request.app.state.mail.cfg.mode
        out = {"email_sending": mode,
               "email_note": ("Alert e-mails are sent through the configured mail server." if mode == "smtp" else
                              "No mail server is configured, so alert e-mails are saved as files in data/app/outbox "
                              "and are NOT sent (test mode)."),
               "analysis_available": a is not None, "analysis_error": request.app.state.analyzer_error,
               "sample_rate": request.app.state.cfg["audio"]["sample_rate"]}
        if a is not None:
            out.update(thresholds=a.thresholds, rule=a.rule,
                       models={g: {"available": a.available(g)[0], "reason": a.available(g)[1]}
                               for g in sorted(request.app.state.groups)})
        return out

    @app.get("/identification/status")
    def identification_status(request: Request, admin=Depends(current_admin)):
        from smart_listener import identification
        return identification.status(project_root(), sorted(request.app.state.groups))

    @app.post("/identify")
    def identify(request: Request, file: UploadFile = File(...), admin=Depends(current_admin), conn=Depends(get_conn)):
        """Score an uploaded file with EVERY machine model and suggest the machine only if validation supports it.
        Nothing is stored. The caller still chooses the machine and then calls analyze-upload."""
        from smart_listener import identification, inference
        analyzer = need_analyzer(request)
        groups = [g for g in sorted(request.app.state.groups) if analyzer.available(g)[0]]
        tmp = _save_upload(file, request.app.state.max_upload_bytes)
        try:
            try:
                prepared = inference.prepare_upload(tmp, request.app.state.cfg)
            except inference.AudioRejected as e:
                raise HTTPException(422, str(e)) from None
            ranking = identification.rank(analyzer, prepared.windows, groups) if groups else []
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
        st = identification.status(project_root(), sorted(request.app.state.groups))
        sug = identification.suggest(ranking, st, analyzer.thresholds["z_alarm"])
        registered = {m["model_group"]: m["id"] for m in repo.list_machines(conn)}
        if sug["group"] and sug["group"] not in registered:
            sug = {"group": None, "reason": f"The closest model is {sug['group']}, but no active machine is registered "
                                            f"with that ID."}
        hint = inference.filename_hint(file.filename or "")
        return {"file": {"name": _safe_name(file.filename), **prepared.info}, "warnings": prepared.warnings,
                "ranking": ranking, "status": st, "suggestion": {**sug, "machine_id": registered.get(sug["group"])},
                "filename_hint": hint}

    @app.get("/assistant/questions")
    def assistant_questions(admin=Depends(current_admin)):
        from smart_listener import assistant
        return {"questions": assistant.SUGGESTED_QUESTIONS}

    @app.post("/assistant/ask")
    def assistant_ask(body: AskIn, request: Request, admin=Depends(current_admin), conn=Depends(get_conn)):
        from smart_listener import assistant, severity
        a = _analyzer(request.app)
        thr = a.thresholds if a else None
        rule = a.rule if a else dict(severity.DEFAULT_RULE)
        if thr is None:   # no models loaded: use the limits stored with the incident's own analysis
            inc = repo.get_incident(conn, body.incident_id)
            op = repo.get_analysis(conn, inc["opened_analysis_id"]) if inc and inc["opened_analysis_id"] else None
            if op is None:
                raise HTTPException(503, "The alert limits are not available.")
            thr = {"z_warn": op["z_warn"], "z_alarm": op["z_alarm"]}
        try:
            return assistant.answer(conn, body.incident_id, body.question.strip(), thr, rule)
        except LookupError:
            raise HTTPException(404, "incident not found") from None

    def _date(v: Optional[str]) -> Optional[str]:
        if v and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise HTTPException(422, "dates must look like 2026-11-09")
        return v or None

    def _download(data, media, name):
        return Response(data, media_type=media, headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @app.get("/reports/incidents.csv")
    def report_incidents_csv(status: Optional[Literal["open", "closed"]] = None, machine_id: Optional[int] = None,
                             admin=Depends(current_admin), conn=Depends(get_conn)):
        from smart_listener import reports
        repo.audit(conn, admin["username"], "report.incidents_csv")
        return _download(reports.incidents_csv(repo.list_incidents(conn, status, machine_id, 100000)), "text/csv",
                         "incidents.csv")

    @app.get("/reports/analyses.csv")
    def report_analyses_csv(machine_id: Optional[int] = None, date_from: Optional[str] = None,
                            date_to: Optional[str] = None, admin=Depends(current_admin), conn=Depends(get_conn)):
        from smart_listener import reports
        rows = repo.list_analyses_between(conn, machine_id, _date(date_from), _date(date_to))
        repo.audit(conn, admin["username"], "report.analyses_csv", f"machine={machine_id} rows={len(rows)}")
        return _download(reports.analyses_csv(rows), "text/csv", "analyses.csv")

    @app.get("/reports/incidents/{incident_id}.pdf")
    def report_incident_pdf(incident_id: int, request: Request, include_evidence: bool = True,
                            admin=Depends(current_admin), conn=Depends(get_conn)):
        from smart_listener import reports
        try:
            pdf = reports.incident_pdf(conn, incident_id, request.app.state.evidence_dir, include_evidence)
        except LookupError:
            raise HTTPException(404, "incident not found") from None
        repo.audit(conn, admin["username"], "report.incident_pdf", f"id={incident_id}")
        return _download(pdf, "application/pdf", f"incident_{incident_id:03d}.pdf")

    @app.get("/reports/machines/{machine_id}.pdf")
    def report_machine_pdf(machine_id: int, request: Request, date_from: Optional[str] = None,
                           date_to: Optional[str] = None, include_evidence: bool = True,
                           admin=Depends(current_admin), conn=Depends(get_conn)):
        from smart_listener import reports
        try:
            pdf = reports.machine_pdf(conn, machine_id, _date(date_from), _date(date_to), request.app.state.evidence_dir,
                                      include_evidence)
        except LookupError:
            raise HTTPException(404, "machine not found") from None
        repo.audit(conn, admin["username"], "report.machine_pdf", f"id={machine_id}")
        return _download(pdf, "application/pdf", f"machine_{machine_id}.pdf")

    # ---- Phase 7: e-mail delivery
    from smart_listener import notifier

    @app.get("/mail/status")
    def mail_status(request: Request, admin=Depends(current_admin), conn=Depends(get_conn)):
        w = request.app.state.mail
        return {**w.cfg.public(), "config_error": request.app.state.mail_config_error,
                "worker_running": request.app.state.mail_worker_started, "interval_s": w.interval,
                "last_run": w.last, "counts": repo.alert_counts(conn), "outbox": str(w.outbox),
                "max_attempts": notifier.MAX_ATTEMPTS}

    @app.post("/mail/process")
    def mail_process(request: Request, admin=Depends(current_admin), conn=Depends(get_conn)):
        res = request.app.state.mail.run_now()
        repo.audit(conn, admin["username"], "mail.process", json.dumps(res))
        return res

    @app.post("/mail/test", status_code=201)
    def mail_test(body: TestMailIn, request: Request, admin=Depends(current_admin), conn=Depends(get_conn)):
        from smart_listener import alerts
        try:
            aid = alerts.record_test(conn, body.recipient, admin["username"])
        except LookupError as e:
            raise HTTPException(409, str(e)) from None
        repo.audit(conn, admin["username"], "mail.test", body.recipient)
        request.app.state.mail.run_now()
        a = repo.get_alert(conn, aid)
        return {k: a[k] for k in ("id", "recipient", "status", "attempts", "last_error", "sent_utc")}

    @app.post("/alerts/{alert_id}/retry")
    def alert_retry(alert_id: int, admin=Depends(current_admin), conn=Depends(get_conn)):
        if repo.get_alert(conn, alert_id) is None:
            raise HTTPException(404, "alert not found")
        if not repo.requeue_alert(conn, alert_id):
            raise HTTPException(409, "Only a FAILED e-mail can be put back in the queue.")
        repo.audit(conn, admin["username"], "alert.retry", f"id={alert_id}")
        return repo.get_alert(conn, alert_id)

    return app
