"""The backend web API (FastAPI). Everything except /health and /auth/login needs a login token.

Start it with:   python scripts/run_api.py        then open  http://127.0.0.1:8000/docs
Phase 4 provides: login, the machine registry (owners and e-mail addresses), fleet overview, incident
acknowledgement, history and alert listings, settings, model profiles and an audit trail.
Phases 5 to 7 add audio analysis, monitoring, e-mail sending, the assistant and reports on top of this.
"""
import time
from typing import Any, Literal, Optional

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
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


# ----------------------------------------------------------------------------- the application
def create_app(db_path=None, cfg: Optional[dict] = None, session_hours: float = 12) -> FastAPI:
    app = FastAPI(title="Smart Listener API", version="0.4.0",
                  description="Industrial acoustic anomaly monitoring: machine registry, incidents and alerts. "
                              "Anomaly scores are not failure probabilities and do not identify a specific fault.")
    app.state.db_path = db_path or db.default_db_path()
    app.state.cfg = cfg or load_config()
    app.state.groups = set(profiles.model_groups(app.state.cfg))
    app.state.session_hours = session_hours
    app.state.failures = {}
    conn = db.connect(app.state.db_path)
    try:
        db.init_schema(conn)
    finally:
        conn.close()

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
                         source: Optional[Literal["simulated", "upload"]] = None, admin=Depends(current_admin),
                         conn=Depends(get_conn)):
        if repo.get_machine(conn, machine_id) is None:
            raise HTTPException(404, "machine not found")
        return repo.list_analyses(conn, machine_id, limit, offset, source)

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
               limit: int = Query(100, ge=1, le=1000), admin=Depends(current_admin), conn=Depends(get_conn)):
        return repo.list_alerts(conn, incident_id, machine_id, limit)

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

    return app
