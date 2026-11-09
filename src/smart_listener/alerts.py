"""Alert RECORDS: what should be e-mailed, and to whom.

This module only writes rows to the `alerts` table ('pending', or 'suppressed' when alerts are switched off).
Sending is done by smart_listener.notifier, which reads the pending rows, sends them by SMTP (or saves them in test
mode), retries failures and keeps to the hourly limit. Keeping the two steps apart means monitoring never waits for,
or fails because of, a mail server.

Kinds of alert:
  high_opened - a new HIGH incident (one per incident and recipient: the dedup key contains the incident number)
  reminder    - the incident is still open and not acknowledged after `reminder_minutes` (repeats at that interval)
  test        - a test message sent from Settings
"""
from datetime import datetime, timedelta, timezone

from smart_listener import repo
from smart_listener.dashboard_ui import fmt_time, inc_code, machine_code

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
        f"  Started (UTC)   : {fmt_time(window_utc) or window_utc}\n"
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
    body = (f"This is a test message from Smart Listener, requested by {requested_by} at {fmt_time(now)} (UTC).\n\n"
            f"If you received it, e-mail alerts for new HIGH incidents can reach this address.\n\n" + FOOTER)
    return repo.add_alert(conn, machine_id=m["id"], kind="test", recipient=recipient.strip(),
                          subject="[Smart Listener] Test e-mail", body=body, dedup_key=f"test-{now}-{recipient.lower()}")
