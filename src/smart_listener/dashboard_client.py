"""A small client for the backend API, used by the Streamlit dashboard (and usable from any script).

The dashboard never opens the database itself: everything goes through the API, so there is one set of rules
(login, validation, audit trail) no matter who asks. Pass `http=` (any httpx.Client, for example FastAPI's test
client) to talk to an in-process app in tests.
"""
from typing import Optional

import httpx


class ApiError(RuntimeError):
    """The API refused or failed. `status` is the HTTP code (0 = could not connect); the message is readable."""

    def __init__(self, message: str, status: int = 0):
        super().__init__(message)
        self.status = status


def _detail(r: httpx.Response) -> str:
    try:
        d = r.json().get("detail", r.text)
    except ValueError:
        return r.text[:200] or f"HTTP {r.status_code}"
    if isinstance(d, list):   # validation errors from FastAPI
        return "; ".join(f"{'.'.join(str(x) for x in e.get('loc', [])[1:])}: {e.get('msg')}" for e in d)
    return str(d)


class ApiClient:
    def __init__(self, base_url: str = "http://127.0.0.1:8000", token: Optional[str] = None, http: Optional[httpx.Client] = None,
                 timeout: float = 120.0):
        self.http = http or httpx.Client(base_url=base_url, timeout=timeout)
        self.token = token
        self.username: Optional[str] = None

    # ---- plumbing
    def _call(self, method: str, url: str, raw: bool = False, **kw):
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        try:
            r = self.http.request(method, url, headers=headers, **kw)
        except httpx.HTTPError as e:
            raise ApiError("Cannot reach the Smart Listener API. Is it running (python scripts/run_api.py)?") from e
        if r.status_code >= 400:
            raise ApiError(_detail(r), r.status_code)
        return r.content if raw else r.json()

    # ---- login
    def login(self, username: str, password: str) -> None:
        j = self._call("POST", "/auth/login", json={"username": username, "password": password})
        self.token, self.username = j["token"], j["username"]

    def logout(self) -> None:
        try:
            self._call("POST", "/auth/logout")
        finally:
            self.token = self.username = None

    def health(self) -> dict:
        return self._call("GET", "/health")

    # ---- reading
    def fleet(self, include_inactive=False): return self._call("GET", "/fleet", params={"include_inactive": include_inactive})
    def machines(self, include_inactive=True): return self._call("GET", "/machines", params={"include_inactive": include_inactive})
    def profiles(self): return self._call("GET", "/model-profiles")
    def analyses(self, machine_id, limit=200, source=None, session_id=None):
        return self._call("GET", f"/machines/{machine_id}/analyses",
                          params={k: v for k, v in {"limit": limit, "source": source, "session_id": session_id}.items() if v})
    def severity_events(self, machine_id, limit=50): return self._call("GET", f"/machines/{machine_id}/severity-events", params={"limit": limit})
    def incidents(self, status=None, machine_id=None, limit=100):
        return self._call("GET", "/incidents", params={k: v for k, v in {"status": status, "machine_id": machine_id, "limit": limit}.items() if v})
    def incident(self, incident_id): return self._call("GET", f"/incidents/{incident_id}")
    def alerts(self, machine_id=None, limit=100):
        return self._call("GET", "/alerts", params={k: v for k, v in {"machine_id": machine_id, "limit": limit}.items() if v})
    def settings(self): return self._call("GET", "/settings")
    def audit(self, limit=100): return self._call("GET", "/audit", params={"limit": limit})
    def scenarios(self): return self._call("GET", "/monitoring/scenarios")
    def sessions(self, machine_id=None, limit=20):
        return self._call("GET", "/monitoring", params={k: v for k, v in {"machine_id": machine_id, "limit": limit}.items() if v})
    def session(self, session_id): return self._call("GET", f"/monitoring/{session_id}")

    # ---- pictures (PNG bytes)
    def evidence_png(self, analysis_id): return self._call("GET", f"/analyses/{analysis_id}/evidence.png", raw=True)
    def history_png(self, machine_id, session_id=None):
        return self._call("GET", f"/machines/{machine_id}/history.png", raw=True, params={"session_id": session_id} if session_id else None)

    # ---- changing things
    def create_machine(self, **fields): return self._call("POST", "/machines", json=fields)
    def update_machine(self, machine_id, **fields): return self._call("PUT", f"/machines/{machine_id}", json=fields)
    def deactivate_machine(self, machine_id): return self._call("DELETE", f"/machines/{machine_id}")
    def acknowledge(self, incident_id, note=""): return self._call("POST", f"/incidents/{incident_id}/acknowledge", json={"note": note})
    def update_settings(self, **changes): return self._call("PUT", "/settings", json=changes)
    def upload(self, machine_id, filename, data: bytes):
        return self._call("POST", f"/machines/{machine_id}/analyze-upload", files={"file": (filename, data, "audio/wav")})
    def start_monitoring(self, machine_id, **options): return self._call("POST", f"/machines/{machine_id}/monitoring/start", json=options)
    def stop_monitoring(self, session_id): return self._call("POST", f"/monitoring/{session_id}/stop")

    # ---- redesign additions
    def account(self): return self._call("GET", "/account")
    def change_password(self, current, new):
        return self._call("POST", "/auth/password", json={"current_password": current, "new_password": new})
    def activity(self, limit=20): return self._call("GET", "/activity", params={"limit": limit})
    def system(self): return self._call("GET", "/system")
    def notes(self, incident_id): return self._call("GET", f"/incidents/{incident_id}/notes")
    def add_note(self, incident_id, note): return self._call("POST", f"/incidents/{incident_id}/notes", json={"note": note})
    def identification_status(self): return self._call("GET", "/identification/status")
    def identify(self, filename, data: bytes):
        return self._call("POST", "/identify", files={"file": (filename, data, "audio/wav")})
    def assistant_questions(self): return self._call("GET", "/assistant/questions")["questions"]
    def ask(self, incident_id, question):
        return self._call("POST", "/assistant/ask", json={"incident_id": incident_id, "question": question})

    def _params(self, **kw): return {k: v for k, v in kw.items() if v not in (None, "")}
    def incidents_csv(self, status=None, machine_id=None):
        return self._call("GET", "/reports/incidents.csv", raw=True, params=self._params(status=status, machine_id=machine_id))
    def analyses_csv(self, machine_id=None, date_from=None, date_to=None):
        return self._call("GET", "/reports/analyses.csv", raw=True,
                          params=self._params(machine_id=machine_id, date_from=date_from, date_to=date_to))
    def incident_pdf(self, incident_id, include_evidence=True):
        return self._call("GET", f"/reports/incidents/{incident_id}.pdf", raw=True, params={"include_evidence": include_evidence})
    def machine_pdf(self, machine_id, date_from=None, date_to=None, include_evidence=True):
        return self._call("GET", f"/reports/machines/{machine_id}.pdf", raw=True,
                          params=self._params(date_from=date_from, date_to=date_to, include_evidence=include_evidence))
    def analysis(self, analysis_id): return self._call("GET", f"/analyses/{analysis_id}")
