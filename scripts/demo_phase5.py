"""Walk through Phase 5 against the RUNNING API (start it first with: python scripts/run_api.py).

    python scripts/demo_phase5.py --machine "Pump 04" --wav path/to/some_clip.wav

It logs in, uploads the file (if given), runs a short SIMULATED monitoring session on the machine, waits for it,
prints what happened (states, incident, alert record) and saves the pictures to reports/phase5/.
The admin password is read from the SL_ADMIN_PASSWORD variable or typed (not shown).
"""
import argparse
import getpass
import os
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--user", default="admin")
    ap.add_argument("--machine", required=True, help="registered machine name, e.g. 'Pump 04'")
    ap.add_argument("--wav", help="optional WAV file to upload first")
    ap.add_argument("--scenario", default="fault", choices=["normal", "fault", "fault_recovery"])
    ap.add_argument("--normal-before", type=int, default=10)
    ap.add_argument("--fault", type=int, default=20)
    ap.add_argument("--out", default="reports/phase5")
    args = ap.parse_args()

    pw = os.environ.get("SL_ADMIN_PASSWORD") or getpass.getpass("Admin password: ")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with httpx.Client(base_url=args.url, timeout=120) as c:
        r = c.post("/auth/login", json={"username": args.user, "password": pw})
        if r.status_code != 200:
            print("Login failed:", r.json().get("detail"))
            return 1
        c.headers["Authorization"] = "Bearer " + r.json()["token"]
        machines = {m["name"].lower(): m for m in c.get("/machines").json()}
        m = machines.get(args.machine.lower())
        if m is None:
            print(f"No machine named '{args.machine}'. Registered: {sorted(machines)}")
            return 1
        print(f"Machine: {m['name']} ({m['model_group']})")

        if args.wav:
            with open(args.wav, "rb") as fh:
                r = c.post(f"/machines/{m['id']}/analyze-upload", files={"file": (Path(args.wav).name, fh, "audio/wav")})
            if r.status_code != 200:
                print("Upload refused:", r.json().get("detail"))
            else:
                j = r.json()
                print(f"\nUPLOAD {j['file']['name']}: {j['summary']['n_windows']} window(s), highest z {j['summary']['max_z']:.2f}")
                for w in j["warnings"]:
                    print("  note:", w)
                if j.get("model_caution"):
                    print("  caution:", j["model_caution"])
                for w in j["windows"]:
                    print(f"  window at {w['offset_s']:6.1f} s: z {w['z']:6.2f}  {w['clip_level']}")
                png = c.get(f"/analyses/{j['windows'][j['worst_window']]['analysis_id']}/evidence.png")
                (out / "upload_evidence.png").write_bytes(png.content)
                print(f"  picture saved: {out / 'upload_evidence.png'}")

        c.put("/settings", json={"replay_seconds_per_window": 0.05})      # fast demo
        r = c.post(f"/machines/{m['id']}/monitoring/start",
                   json={"scenario": args.scenario, "n_normal_before": args.normal_before, "n_fault": args.fault})
        if r.status_code != 201:
            print("Monitoring did not start:", r.json().get("detail"))
            return 1
        sid = r.json()["id"]
        print(f"\nSIMULATED monitoring session {sid} started (replayed dataset recordings, not a live machine)")
        while True:
            s = c.get(f"/monitoring/{sid}").json()
            p = s["progress"] or {}
            print(f"  {p.get('done', 0)}/{p.get('total', '?')} windows, latest state: {(p.get('latest') or {}).get('state', '-')}   ", end="\r")
            if s["status"] != "running":
                break
            time.sleep(0.5)
        print(f"\nSession finished with status: {s['status']} {s['note'] if s['status'] == 'error' else ''}")
        for e in reversed(c.get(f"/machines/{m['id']}/severity-events").json()):
            print(f"  {e['event_utc']}  {e['from_state']} -> {e['to_state']}   ({e['reason']})")
        for inc in c.get("/incidents", params={"machine_id": m["id"], "limit": 3}).json():
            print(f"  incident {inc['id']}: {inc['status']}, peak z {inc['peak_z']:.1f}, acknowledged: {bool(inc['acknowledged'])}")
        for a in c.get("/alerts", params={"machine_id": m["id"], "limit": 3}).json():
            print(f"  alert record {a['id']}: to {a['recipient']}, status {a['status']} (e-mail sending is added in Phase 7)")
        (out / "history.png").write_bytes(c.get(f"/machines/{m['id']}/history.png").content)
        print(f"  history picture saved: {out / 'history.png'}")
        rows = c.get(f"/machines/{m['id']}/analyses", params={"session_id": sid, "limit": 200}).json()
        hot = [x for x in rows if x["clip_level"] != "within_normal"]
        if hot:
            top = max(hot, key=lambda x: x["z"])
            png = c.get(f"/analyses/{top['id']}/evidence.png")
            if png.status_code == 200:
                (out / "monitoring_evidence.png").write_bytes(png.content)
                print(f"  evidence picture of the highest window saved: {out / 'monitoring_evidence.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
