"""Start the backend API on your own computer only (127.0.0.1), so nobody else on the network can reach it.

    python scripts/run_api.py              # then open http://127.0.0.1:8000/docs in your browser
    python scripts/run_api.py --port 8001

Press Ctrl+C in the terminal to stop it.
"""
import argparse
import sys

import uvicorn

from smart_listener import db


def main() -> int:
    ap = argparse.ArgumentParser(description="Run the Smart Listener API")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1", help="leave as 127.0.0.1 unless you really need other computers to connect")
    args = ap.parse_args()
    if not db.default_db_path().exists():
        print("No database yet. Run first:  python scripts/init_db.py --create-admin admin")
        return 1
    print(f"Database: {db.default_db_path()}\nAPI docs: http://{args.host}:{args.port}/docs   (Ctrl+C to stop)")
    uvicorn.run("smart_listener.asgi:app", host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
