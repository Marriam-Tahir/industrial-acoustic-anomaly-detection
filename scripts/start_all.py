"""Start the API and the dashboard together (for demos). Press Ctrl+C once to stop both.

    python scripts/start_all.py
    python scripts/start_all.py --api-port 8000 --dashboard-port 8501

The dashboard opens at http://localhost:8501. Logs from both programs appear in this window.
"""
import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def wait_for(url: str, seconds: float, proc=None) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if proc is not None and proc.poll() is not None:
            return False                      # the program already stopped: no point waiting

        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Start the Smart Listener API and dashboard")
    ap.add_argument("--api-port", type=int, default=8000)
    ap.add_argument("--dashboard-port", type=int, default=8501)
    args = ap.parse_args()
    # the project's code is in src/; adding it here means both programs find it even if `pip install -e .` was not run
    path = os.pathsep.join(p for p in (str(ROOT / "src"), os.environ.get("PYTHONPATH", "")) if p)
    env = dict(os.environ, SL_API_URL=f"http://127.0.0.1:{args.api_port}", PYTHONPATH=path)
    api = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "run_api.py"), "--port", str(args.api_port)],
                           cwd=ROOT, env=env)
    if not wait_for(f"http://127.0.0.1:{args.api_port}/health", 90, api):
        print("\nThe API did not start. Read the error above. Common causes: a package is missing "
              "(python -m pip install -r requirements-app.txt), or the port is in use (--api-port 8001).")
        api.terminate()
        return 1
    dash = subprocess.Popen([sys.executable, "-m", "streamlit", "run", str(ROOT / "dashboard" / "app.py"),
                             "--server.port", str(args.dashboard_port)], cwd=ROOT, env=env)
    print(f"\nSmart Listener is running.\n  Dashboard: http://localhost:{args.dashboard_port}\n"
          f"  API docs : http://127.0.0.1:{args.api_port}/docs\nPress Ctrl+C to stop both.\n", flush=True)
    try:
        while api.poll() is None and dash.poll() is None:
            time.sleep(1)
        print("One of the programs stopped; stopping the other.")
    except KeyboardInterrupt:
        print("\nStopping ...")
    finally:
        for p in (dash, api):
            if p.poll() is None:
                p.terminate()
        for p in (dash, api):
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
