"""Alert RECORDS: what should be e-mailed, and to whom.

This module only writes rows to the `alerts` table ('pending', or 'suppressed' when alerts are switched off).
Sending is done by smart_listener.notifier, which reads the pending rows, sends them by SMTP (or saves them in test
mode), retries failures and keeps to the hourly limit. Keeping the two steps apart means monitoring never waits for,
or fails because of, a mail server.

Kinds of alert:
  high_opened - a new HIGH incident (one per incident and recipient: the dedup key contains the incident number)
  reminder    - the incident is still open and not acknowledged after `reminder_minutes` (repeats at that interval)
  test        - a test message sent from Settings
  analysis    - the result of an uploaded recording that is not normal (one per upload and recipient), with a PDF
                report, a CSV of every window and the evidence picture attached

When an upload e-mails (decided by `analysis_outcome`):
  file long enough for a severity (at least n_window windows, 60 s): severity WATCH, WARNING or HIGH -> e-mail;
                                                                      NORMAL -> no e-mail
  shorter file (severity "not assessed"): any window above the ALARM limit -> e-mail; otherwise no e-mail
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from smart_listener import repo
from smart_listener.dashboard_ui import CLIP, fmt_time, inc_code, machine_code, machine_type_of

LIMITS_TEXT = ("What this means: the machine sounds unlike its learned normal sound. It does NOT give a probability of "
               "failure, does NOT estimate when the machine will fail, and does NOT say which part is faulty.")
SIMULATED_TEXT = ("NOTE: SIMULATED MONITORING. This incident comes from replayed MIMII dataset recordings, not from a "
                  "live microphone on a real machine.")
FOOTER = "This message was sent automatically by Smart Listener - Industrial Acoustic Anomaly Detection.\n"


def _where(machine: dict) -> str:
    return f"{machine['name']} ({machine_code(machine['model_group'])}, {machine['machine_type']}" + \
           (f", {machine['location']})" if machine.get("location") else ")")


def high_incident_text(machine: dict, incident_id: int, window_utc: str, z: float, sentence: str = "",
                       source: str = "", z_alarm: float = None) -> tuple:
    subject = f"[Smart Listener] HIGH: {machine_code(machine['model_group'])} {inc_code(incident_id)}"
    body = (
        f"Hello {machine['owner_name']},\n\n"
        + (SIMULATED_TEXT + "\n\n" if source == "simulated" else "") +
        f"The sound of {_where(machine)} has stayed far from its normal sound for several consecutive 10-second "
        f"recordings. Operational severity is now HIGH.\n\n"
        f"  Incident        : {inc_code(incident_id)}\n"
        f"  Started (PKT)   : {fmt_time(window_utc) or window_utc}\n"
        f"  Anomaly level   : z = {z:.1f}" + (f" (alarm limit z > {z_alarm:.2f})" if z_alarm is not None else "") + "\n"
        + (f"  Where it differs: {sentence}\n" if sentence else "") +
        f"\n{LIMITS_TEXT}\n\nPlease have the machine inspected, then acknowledge {inc_code(incident_id)} in Smart "
        f"Listener.\n\n" + FOOTER)
    return subject, body


def record_high_alerts(conn, machine: dict, incident_id: int, window_utc: str, z: float, settings: dict,
                       sentence: str = "", source: str = "", z_alarm: float = None) -> list:
    """Create the alert rows for a new HIGH incident. Returns the ids of the rows that were created (empty if the
    same incident already has its alert)."""
    subject, body = high_incident_text(machine, incident_id, window_utc, z, sentence, source, z_alarm)
    status = "pending" if settings.get("alerts_enabled", True) else "suppressed"
    recipient = machine["owner_email"].strip()
    key = f"incident-{incident_id}-high_opened-{recipient.lower()}"
    aid = repo.add_alert(conn, machine_id=machine["id"], kind="high_opened", recipient=recipient, subject=subject,
                         dedup_key=key, incident_id=incident_id, body=body, status=status)
    return [aid] if aid else []


def _parse(ts):
    d = datetime.fromisoformat(ts)
    return d if d.tzinfo else d.replace(tzinfo=timezone.utc)


def record_reminders(conn, settings: dict, now: datetime) -> list:
    """For each open, unacknowledged incident, add a reminder when `reminder_minutes` have passed since the last
    message about it (the first alert or the previous reminder). Returns the ids created."""
    minutes = int(settings.get("reminder_minutes", 0))
    if minutes <= 0 or not settings.get("alerts_enabled", True):
        return []
    made = []
    for inc in repo.list_incidents(conn, "open", None, 1000):
        if inc["acknowledged"]:
            continue
        sent = repo.list_alerts(conn, incident_id=inc["id"], limit=1000)
        if not sent:
            continue                                  # alerts were off when it opened: do not start reminding now
        last = max(_parse(a["created_utc"]) for a in sent)
        if now - last < timedelta(minutes=minutes):
            continue
        n = sum(1 for a in sent if a["kind"] == "reminder") + 1
        recipient = inc["owner_email"].strip()
        machine = repo.get_machine(conn, inc["machine_id"])
        open_for = int((now - _parse(inc["opened_utc"])).total_seconds() // 60)
        subject = f"[Smart Listener] Reminder: {machine_code(inc['model_group'])} {inc_code(inc['id'])} still open"
        body = (f"Hello {inc['owner_name']},\n\n{inc_code(inc['id'])} on {_where(machine)} has been open for "
                f"{open_for} minutes and has not been acknowledged. Highest anomaly level so far: z = {inc['peak_z']:.1f}."
                f"\n\n{LIMITS_TEXT}\n\nPlease review it in Smart Listener and acknowledge it once the machine has been "
                f"checked.\n\n" + FOOTER)
        aid = repo.add_alert(conn, machine_id=inc["machine_id"], kind="reminder", recipient=recipient, subject=subject,
                             dedup_key=f"incident-{inc['id']}-reminder-{n}-{recipient.lower()}", incident_id=inc["id"],
                             body=body)
        if aid:
            made.append(aid)
    return made


def record_test(conn, recipient: str, requested_by: str) -> int:
    """A test message (from Settings). Needs a registered machine only because every alert row belongs to one."""
    m = next(iter(repo.list_machines(conn, include_inactive=True)), None)
    if m is None:
        raise LookupError("Register at least one machine before sending a test e-mail.")
    now = repo.now_utc()
    body = (f"This is a test message from Smart Listener, requested by {requested_by} at {fmt_time(now)} (PKT, Asia/Karachi).\n\n"
            f"If you received it, e-mail alerts for new HIGH incidents can reach this address.\n\n" + FOOTER)
    return repo.add_alert(conn, machine_id=m["id"], kind="test", recipient=recipient.strip(),
                          subject="[Smart Listener] Test e-mail", body=body, dedup_key=f"test-{now}-{recipient.lower()}")


# ----------------------------------------------------------------------------- uploaded recordings
def analysis_outcome(summary: dict, windows: list, n_needed: int) -> dict:
    """Decide whether an uploaded file's result is e-mailed. `summary` and `windows` come from
    monitor.analyze_upload; `n_needed` is the severity rule's window count (n_window)."""
    n = summary["n_windows"]
    assessed = n >= n_needed
    worst = max(windows, key=lambda w: w["z"])
    severity = summary["file_state"] if assessed else None
    if assessed:
        notify = severity in ("WATCH", "WARNING", "HIGH")
        why = f"severity {severity}" + ("" if notify else " (normal: no e-mail)")
        headline = severity
    else:
        notify = worst["clip_level"] == "above_alarm_limit"
        headline = CLIP.get(worst["clip_level"], (worst["clip_level"],))[0]
        why = (f"short file ({n} window(s), severity not assessed); highest window {headline}"
               + ("" if notify else " (not above the alarm limit: no e-mail)"))
    return {"notify": notify, "assessed": assessed, "severity": severity, "headline": headline, "reason": why,
            "n_windows": n, "n_needed": n_needed, "worst_z": worst["z"], "worst_score": worst["score"],
            "worst_level": worst["clip_level"], "worst_offset_s": worst.get("offset_s", 0.0),
            "z_warn": worst["z_warn"], "z_alarm": worst["z_alarm"]}


def analysis_text(machine: dict, outcome: dict, file_name: str, analysed_utc: str, summary: dict, warnings=(),
                  caution: str = "", sentence: str = "", attachments=()) -> tuple:
    code = machine_code(machine["model_group"])
    subject = f"[Smart Listener] Upload result {outcome['headline']}: {code} ({file_name})"
    sev = outcome["severity"] if outcome["assessed"] else \
        f"not assessed (file has {outcome['n_windows']} window(s); severity needs {outcome['n_needed']}, about " \
        f"{outcome['n_needed'] * 10} s)"
    notes = [w for w in warnings if w] + ([f"Model reliability: {caution}"] if caution else [])
    body = (
        f"Hello {machine['owner_name']},\n\n"
        f"An uploaded recording for {_where(machine)} was analysed and the result is not normal.\n\n"
        f"  Machine ID      : {code} ({machine['model_group']})\n"
        f"  Machine type    : {machine_type_of(machine['model_group'])}\n"
        f"  File            : {file_name}\n"
        f"  Analysed (PKT)  : {fmt_time(analysed_utc)}\n"
        f"  Severity        : {sev}\n"
        f"  Highest window  : {CLIP.get(outcome['worst_level'], (outcome['worst_level'],))[0]} at "
        f"{outcome['worst_offset_s']:.0f}-{outcome['worst_offset_s'] + 10:.0f} s\n"
        f"  Anomaly level   : z = {outcome['worst_z']:.2f} (warning limit z > {outcome['z_warn']:.2f}, "
        f"alarm limit z > {outcome['z_alarm']:.2f})\n"
        f"  Anomaly score   : {outcome['worst_score']:.6f} (reconstruction error, MSE)\n"
        f"  Windows         : {summary['n_windows']} analysed, {summary['n_above_warning_limit']} above warning, "
        f"{summary['n_above_alarm_limit']} above alarm\n"
        + (f"  Where it differs: {sentence}\n" if sentence else "")
        + ("\nWarnings:\n" + "".join(f"  - {w}\n" for w in notes) if notes else "")
        + ("\nAttached: " + ", ".join(Path(a).name for a in attachments) + "\n" if attachments else "")
        + f"\nThis was a manual check of an uploaded file. It does not change the machine's monitoring state and "
          f"does not open an incident.\n\n{LIMITS_TEXT}\n\n" + FOOTER)
    return subject, body


def record_analysis_alert(conn, machine: dict, result: dict, settings: dict, attach_root, n_needed: int,
                          file_name: str, warnings=(), caution: str = "", evidence_dir=None) -> dict:
    """After an upload: if the result is not normal, write the PDF and CSV (and copy the evidence picture) to
    <attach_root>/attachments/upload_<session>/ and record ONE 'analysis' alert for the machine owner.
    Returns {"needed", "reason", "alert_id", "status"}; never raises for a normal result."""
    from smart_listener import reports
    outcome = analysis_outcome(result["summary"], result["windows"], n_needed)
    out = {"needed": outcome["notify"], "reason": outcome["reason"], "alert_id": None, "status": None,
           "recipient": machine["owner_email"].strip()}
    if not outcome["notify"]:
        return out
    sid = result["session_id"]
    rel = Path("attachments") / f"upload_{sid}"
    folder = Path(attach_root) / rel
    folder.mkdir(parents=True, exist_ok=True)
    rows = reports.upload_rows(conn, machine["id"], sid)
    code = machine_code(machine["model_group"]).lower()
    files = {f"{code}_upload_{sid}_report.pdf": reports.analysis_pdf(conn, machine, sid, outcome, file_name, warnings,
                                                                      caution, evidence_dir),
             f"{code}_upload_{sid}_windows.csv": reports.analysis_csv(machine, rows, outcome).encode("utf-8")}
    worst = max(rows, key=lambda a: a["z"])
    png = reports._evidence_png(evidence_dir, worst)
    if png:
        files[f"{code}_upload_{sid}_evidence.png"] = png
    for name, data in files.items():
        (folder / name).write_bytes(data)
    attachments = [str(rel / n).replace("\\", "/") for n in files]
    sess = repo.get_session_row(conn, sid) or {}
    try:
        sentence = json.loads(worst.get("evidence_json") or "{}").get("sentence", "")
    except ValueError:
        sentence = ""
    subject, body = analysis_text(machine, outcome, file_name, sess.get("started_utc") or worst["created_utc"],
                                  result["summary"], warnings, caution, sentence, attachments)
    status = "pending" if settings.get("alerts_enabled", True) else "suppressed"
    aid = repo.add_alert(conn, machine_id=machine["id"], kind="analysis", recipient=out["recipient"], subject=subject,
                         dedup_key=f"upload-{sid}-analysis-{out['recipient'].lower()}", body=body, status=status,
                         attachments=attachments)
    out.update(alert_id=aid, status=status if aid else "duplicate")
    return out
