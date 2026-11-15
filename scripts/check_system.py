"""Pre-demo system check: is everything in place to run Smart Listener?

    python scripts/check_system.py

Prints one line per check (OK / WARN / FAIL) and exits with code 1 if anything FAILs. It only reads files; it changes
nothing. Run it before a demo or after moving the project to another computer.
"""
import importlib
import shutil
import socket
import sys
from pathlib import Path

RESULTS = []


def report(level: str, what: str, detail: str = ""):
    RESULTS.append(level)
    print(f"  [{level:<4}] {what}" + (f" - {detail}" if detail else ""))


def port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def main() -> int:
    print("Smart Listener - system check\n")
    print("Software")
    v = sys.version_info
    report("OK" if v >= (3, 10) else "FAIL", f"Python {v.major}.{v.minor}.{v.micro}")
    for mod in ("numpy", "pandas", "soundfile", "librosa", "torch", "matplotlib", "fastapi", "uvicorn", "httpx",
                "streamlit", "multipart"):
        try:
            m = importlib.import_module(mod)
            report("OK", mod, getattr(m, "__version__", ""))
        except Exception as e:
            report("FAIL", mod, f"not importable ({type(e).__name__}). pip install -r requirements-app.txt")

    from smart_listener import db, identification, notifier, profiles, severity
    from smart_listener.config import config_fingerprint, load_config, project_root, resolve_path
    from smart_listener.inference import Analyzer
    root = project_root()
    cfg = load_config()

    print("\nModels and limits")
    try:
        cal = severity.load_calibration()
        t = cal["thresholds"]
        report("OK", "severity calibration", f"warning z > {t['z_warn']:.2f}, alarm z > {t['z_alarm']:.2f}")
        fp = cal.get("config_fingerprint")
        if fp and fp != config_fingerprint(cfg):
            report("FAIL", "audio settings changed since calibration", "rebuild features and recalibrate")
    except FileNotFoundError as e:
        report("FAIL", "severity calibration", str(e))
        cal = None
    groups = profiles.model_groups(cfg)
    if cal:
        a = Analyzer(cfg, calibration=cal)
        ok = [g for g in groups if a.available(g)[0]]
        for g in groups:
            okg, why = a.available(g)
            report("OK" if okg else "FAIL", f"model {g}", "" if okg else why)
    else:
        ok = []

    print("\nRecordings for simulated monitoring (validation/test clips)")
    try:
        import pandas as pd
        man = pd.read_csv(resolve_path(cfg, "manifest_dir") / "manifest.csv", keep_default_na=False)
        mono = resolve_path(cfg, "mono_dir")
        for g in ok:
            m = man[(man["group"] == g) & (man["split_ad"] == "test") & (man["exclude_reason"] == "")]
            sample = m.head(3)["path"]
            missing = [p for p in sample if not (mono / p).exists()]
            n_n, n_a = (m["label_name"] == "normal").sum(), (m["label_name"] == "abnormal").sum()
            report("OK" if not missing and len(m) else "FAIL", f"{g} test clips", f"{n_n} normal, {n_a} abnormal"
                   + (f"; missing files, e.g. {missing[0]}" if missing else ""))
    except FileNotFoundError:
        report("FAIL", "manifest", "data/manifests/manifest.csv not found (Phase 1)")

    print("\nDatabase")
    p = db.default_db_path()
    if not p.exists():
        report("WARN", "database", f"{p} does not exist yet; run scripts/init_db.py --create-admin <name>")
    else:
        conn = db.connect(p)
        try:
            ver = conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
            report("OK" if ver and int(ver[0]) == db.SCHEMA_VERSION else "FAIL", "schema version", str(ver[0] if ver else "?"))
            n_admin = conn.execute("SELECT COUNT(*) FROM admins WHERE active = 1").fetchone()[0]
            report("OK" if n_admin else "FAIL", "administrator accounts", str(n_admin))
            rows = conn.execute("SELECT model_group, name FROM machines WHERE active = 1").fetchall()
            report("OK" if rows else "WARN", "registered machines", str(len(rows)) if rows else "none; register them in the dashboard")
            for g, name in rows:
                if g not in ok:
                    report("WARN", f"machine '{name}' ({g})", "its model is missing, so it cannot be analysed")
            n_open = conn.execute("SELECT COUNT(*) FROM incidents WHERE status = 'open'").fetchone()[0]
            report("OK", "open incidents", str(n_open))
        except Exception as e:
            report("FAIL", "database", f"{type(e).__name__}: {e}")
        finally:
            conn.close()

    print("\nE-mail")
    try:
        mc = notifier.load_mail_config(root)
        if mc.mode == "smtp":
            report("OK", "mode SMTP", f"{mc.host}:{mc.port} as {mc.sender}; use Settings > Send test to confirm delivery")
        else:
            report("WARN", "mode TEST", "no config/smtp.env: e-mails are saved in data/app/outbox and NOT sent")
    except ValueError as e:
        report("FAIL", "config/smtp.env", str(e))
    gi = root / ".gitignore"
    if (root / "config" / "smtp.env").exists():
        listed = gi.exists() and "config/smtp.env" in gi.read_text(encoding="utf-8", errors="replace")
        report("OK" if listed else "FAIL", "config/smtp.env kept out of Git", "" if listed else "add config/smtp.env to .gitignore")

    print("\nMachine identification")
    st = identification.status(root, sorted(groups))
    report("OK", "suggestions " + ("ON" if st["enabled"] else "OFF (manual selection)"), st.get("reason", ""))

    print("\nThis computer")
    free_gb = shutil.disk_usage(root).free / 1e9
    report("OK" if free_gb > 2 else "WARN", "free disk space", f"{free_gb:.1f} GB")
    for port, what in ((8000, "API"), (8501, "dashboard")):
        report("OK" if port_free(port) else "WARN", f"port {port} ({what})",
               "free" if port_free(port) else "in use - is it already running?")

    n_fail, n_warn = RESULTS.count("FAIL"), RESULTS.count("WARN")
    print(f"\n{len(RESULTS)} checks: {n_fail} FAIL, {n_warn} WARN")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
