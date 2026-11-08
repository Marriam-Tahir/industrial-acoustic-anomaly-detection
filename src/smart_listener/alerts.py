"""Phase 5: alert RECORDS for new HIGH incidents.

This module only writes rows to the `alerts` table with status 'pending' (or 'suppressed' when alerts are switched off).
Nothing is sent from here. Phase 7 adds the e-mail sender, which reads the pending rows, sends them with SMTP, retries
failures and updates their status. Keeping the two steps apart means monitoring never waits for, or fails because of, a
mail server.

One alert per incident and recipient: the dedup key contains the incident number, so recording the same incident twice
creates no second row.
"""
from smart_listener import repo


def high_incident_text(machine: dict, incident_id: int, window_utc: str, z: float, sentence: str = "") -> tuple:
    subject = f"[Smart Listener] HIGH alert: {machine['name']} ({machine['model_group']})"
    body = (
        f"Hello {machine['owner_name']},\n\n"
        f"The sound of '{machine['name']}' ({machine['machine_type']}, {machine['location'] or 'location not set'}) has "
        f"stayed far above its normal range for several consecutive 10-second recordings. The level is now HIGH.\n\n"
        f"  Incident number : {incident_id}\n"
        f"  Time            : {window_utc}\n"
        f"  Anomaly level   : z = {z:.1f} (how far the sound is above this machine's normal range)\n"
        + (f"  Where it differs: {sentence}\n" if sentence else "") +
        "\nWhat this means: the sound is unlike what this machine normally sounds like. What it does NOT mean: it is not "
        "a probability of failure, it is not an estimate of remaining life, and it does not say which part is faulty. "
        "Please have someone listen to or inspect the machine, then acknowledge the incident in the Smart Listener "
        "dashboard.\n\n"
        "This message was created automatically by Smart Listener.\n")
    return subject, body


def record_high_alerts(conn, machine: dict, incident_id: int, window_utc: str, z: float, settings: dict,
                       sentence: str = "") -> list:
    """Create the alert rows for a new HIGH incident. Returns the ids of the rows that were created (empty if the
    same incident already has its alert)."""
    subject, body = high_incident_text(machine, incident_id, window_utc, z, sentence)
    status = "pending" if settings.get("alerts_enabled", True) else "suppressed"
    recipient = machine["owner_email"].strip()
    key = f"incident-{incident_id}-high_opened-{recipient.lower()}"
    aid = repo.add_alert(conn, machine_id=machine["id"], kind="high_opened", recipient=recipient, subject=subject,
                         dedup_key=key, incident_id=incident_id, body=body, status=status)
    return [aid] if aid else []
