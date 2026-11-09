"""Start the local Flask dashboard: python main.py. Exposes app for WSGI tooling."""
import logging

from api import create_app
from config import Settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()

if __name__ == "__main__":
    settings = Settings()
    print(f"Smart Listener dashboard: http://{settings.host}:{settings.port}", flush=True)
    app.run(host=settings.host, port=settings.port, debug=False, threaded=True, use_reloader=False)
