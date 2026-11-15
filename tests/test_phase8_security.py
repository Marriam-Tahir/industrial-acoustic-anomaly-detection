"""Phase 8: basic security checks across the whole API."""
import re

import pytest
from fastapi.routing import APIRoute

from phase5_fixtures import analyzer, synth  # noqa: F401
from smart_listener import db, notifier
from test_dashboard import PW, api_client, fast_hashing, server  # noqa: F401

PUBLIC = {("GET", "/health"), ("POST", "/auth/login")}


def _routes(app):
    for r in app.routes:
        if isinstance(r, APIRoute):
            for m in r.methods:
                yield m, r.path


def test_every_route_except_health_and_login_requires_a_valid_session(server):
    app = server.app
    checked = 0
    for method, path in _routes(app):
        if (method, path) in PUBLIC:
            continue
        url = re.sub(r"\{[^}]+\}", "1", path)
        for headers in ({}, {"Authorization": "Bearer not-a-real-token"}):
            r = server.request(method, url, headers=headers, json={} if method in ("POST", "PUT") else None)
            assert r.status_code in (401, 403), (method, path, headers, r.status_code)
        checked += 1
    assert checked >= 40                       # the API has grown; make sure the walk really covered it


def test_public_routes_reveal_nothing_private(server):
    h = server.get("/health").json()
    assert set(h) == {"status", "schema_version"}


def test_security_headers_on_responses(server, api_client):
    for r in (server.get("/health"), server.get("/fleet", headers={"Authorization": f"Bearer {api_client.token}"})):
        assert r.headers["X-Content-Type-Options"] == "nosniff" and r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["Cache-Control"] == "no-store"


def test_passwords_and_session_tokens_are_not_stored_in_plain_text(server, api_client):
    conn = db.connect(server.app.state.db_path)
    try:
        dump = "\n".join(conn.iterdump())
    finally:
        conn.close()
    assert PW not in dump and api_client.token not in dump


def test_login_lockout_after_repeated_failures(server):
    for _ in range(5):
        assert server.post("/auth/login", json={"username": "admin", "password": "wrong-pass-1"}).status_code == 401
    r = server.post("/auth/login", json={"username": "admin", "password": PW})
    assert r.status_code == 429 and "Retry-After" in r.headers          # even the right password waits


def test_unedited_example_mail_file_is_refused(tmp_path):
    (tmp_path / "config").mkdir()
    example = (notifier.Path(__file__).resolve().parents[1] / "config" / "smtp.env.example").read_text()
    (tmp_path / "config" / "smtp.env").write_text(example)
    with pytest.raises(ValueError, match="example address or password"):
        notifier.load_mail_config(tmp_path, environ={})
    (tmp_path / "config" / "smtp.env").write_text(example.replace("your.address@gmail.com", "yusra@gmail.com")
                                                 .replace("abcd efgh ijkl mnop", "qwer tyui opas dfgh"))
    assert notifier.load_mail_config(tmp_path, environ={}).mode == "smtp"


def test_sql_like_input_is_treated_as_text(api_client):
    m = api_client.create_machine(name="x'); DROP TABLE machines;--", machine_type="fan", model_group="fan_id_06",
                                  location="", owner_name="o", owner_email="o@e.com", notes="")
    assert api_client.machines()[0]["name"] == "x'); DROP TABLE machines;--" and len(api_client.machines()) == 1
