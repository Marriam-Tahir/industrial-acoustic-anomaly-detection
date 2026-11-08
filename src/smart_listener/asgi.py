"""Entry point for the web server: `uvicorn smart_listener.asgi:app`. Kept separate from api.py so that importing
the API code (for example in tests) never creates a database file by accident."""
import os

from smart_listener.api import create_app

app = create_app(session_hours=float(os.environ.get("SL_SESSION_HOURS", "12")))
