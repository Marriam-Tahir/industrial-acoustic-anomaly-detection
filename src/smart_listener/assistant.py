"""Maintenance assistant: answers questions about ONE incident from the recorded data only. No language model.

Every answer is a list of sections, and each section says where its content comes from:
  record        - facts stored by the system (machine, times, status, acknowledgement, notifications)
  model         - what the anomaly model measured (anomaly levels, limits, severity rule, evidence)
  not_determined- the question cannot be answered from the model or the records; the answer says so
  guidance      - general maintenance practice, NOT derived from this incident's data
  action        - a recommended next step for the operator (general practice)
The model detects that a machine sounds unlike its learned normal sound. It does not identify the physical fault,
predict when a machine will fail, or give a probability of failure, and the assistant never claims otherwise.
"""
import json
import re

from smart_listener.dashboard_ui import ALERT_KIND, alert_text, fmt_time, inc_code, machine_code

NOT_DETERMINED = "not determined"

SUGGESTED_QUESTIONS = [
    "What did the model detect?",
    "Why is the severity HIGH?",
    "Which machine is affected?",
    "Can you identify the exact fault?",
    "When did it start and is it still open?",
    "Was anyone notified?",
    "What should I do next?",
]

_INTENTS = [   # first match wins; order matters (e.g. "when will it fail" is a prediction, not a time question)
    ("predict", r"\b(fail(ure|s|ing)?|break ?down|remaining|rul\b|how long|life ?time|probabilit|chance|likel|risk|predict)"),
    ("fault", r"(fault|cause|caus|diagnos|bearing|impeller|motor|leak|imbalance|misalign|which part|what part|"
              r"broken|damag|exact|root)"),
    ("why_high", r"(why|how come|reason).*(high|severity|level|incident|alarm)|(severity|high).*(why|mean)"),
    ("notified", r"(notif|e-?mail|alert|told|inform|sent)"),
    ("ack", r"(acknowledg|who (has )?(seen|reviewed)|note)"),
    ("evidence", r"(evidence|frequenc|band|spectrogram|heatmap|hz\b|sound like|score|z[- ]?value|anomaly level)"),
    ("machine", r"(which|what) machine|affected|machine (is|was)|where is"),
    ("when", r"(when|start|began|time|how long ago|ongoing|still|open|closed|status|over\b|resolved)"),
    ("source", r"(simulat|real|live|demo|replay|dataset|mimii|microphone)"),
    ("action", r"(what (should|do|can) (i|we)|next step|recommend|action|inspect|what now|to do|fix|repair)"),
    ("detect", r"(detect|found|wrong|happen|problem|issue|matter|what is it|summary|explain)"),
]


def intent(question: str) -> str:
    q = question.lower().strip()
    for name, pat in _INTENTS:
        if re.search(pat, q):
            return name
    return "unknown"


def _sec(kind, title, text):
    return {"kind": kind, "title": title, "text": text}


def _context(inc: dict, events: list, alerts: list, notes: list, opening: dict, thresholds: dict, rule: dict) -> dict:
    ev = {}
    if opening and opening.get("evidence_json"):
        try:
            ev = json.loads(opening["evidence_json"]) or {}
        except ValueError:
            ev = {}
    return {"inc": inc, "events": events, "alerts": alerts, "notes": notes, "opening": opening or {}, "ev": ev,
            "zw": thresholds["z_warn"], "za": thresholds["z_alarm"], "rule": rule}


def _machine_line(c):
    i = c["inc"]
    return f"{machine_code(i['model_group'])} ({i['machine_name']}, {i['machine_type']})"


def _detected(c):
    i, za = c["inc"], c["za"]
    return _sec("model", "Detected by model",
                f"The sound of {_machine_line(c)} moved far from that machine's learned normal sound. The highest anomaly "
                f"level in this incident was z = {i['peak_z']:.1f}, against an alarm limit of {za:.1f}; "
                f"{i['n_windows']} ten-second window(s) were at HIGH severity. Operational severity: HIGH.")


def _evidence(c):
    ev, op = c["ev"], c["opening"]
    if not ev.get("summary"):
        return _sec("model", "Evidence", "No frequency evidence was saved for the window that opened this incident.")
    s = ev["summary"]
    bands = ", ".join(f"{b['hz_low']}-{b['hz_high']} Hz ({b['share_of_error'] * 100:.0f}% of the error)"
                      for b in s.get("top_bands", [])[:3]) or NOT_DETERMINED
    when = (f"strongest around {s['peak_time_s']:.1f} s into the window"
            if s.get("peak_time_s") is not None and s.get("peak_time_error_ratio", 0) >= 2 else "spread through the window")
    score = f"Score of the opening window: {op['score']:.4f} (z = {op['z']:.1f}). " if op.get("score") is not None else ""
    return _sec("model", "Evidence", f"{score}The model's reconstruction error was largest in: {bands}"
                                     f"{'; ' + when if when else ''}. This shows where the sound differs from normal; "
                                     f"it does not say what causes the difference.")


def _not_fault():
    return _sec("not_determined", "Physical fault: " + NOT_DETERMINED,
                "The model only measures how unlike its normal sound a recording is. It was not trained to name "
                "faults, so the specific physical fault is not determined by this system.")


def _guidance():
    return _sec("guidance", "General guidance",
                "Not derived from this incident's data: follow the site's normal inspection procedure for this type of "
                "machine. Listening at the machine, checking mounting and guards, and comparing with a known-good "
                "machine of the same type are common first checks.")


def _action(c):
    i = c["inc"]
    steps = "review the evidence, arrange an appropriate physical inspection"
    if i["status"] == "open":
        steps += ", and acknowledge the incident when it has been reviewed"
    elif not i["acknowledged"]:
        steps += ", and acknowledge the incident to record that it was reviewed"
    return _sec("action", "Recommended operator action", steps[0].upper() + steps[1:] + ".")


def _source(c):
    src = c["opening"].get("source")
    if src == "simulated":
        return _sec("record", "Data source", "SIMULATED MONITORING: this incident came from replayed MIMII dataset "
                                             "recordings, not from a live microphone on a real machine.")
    if src:
        return _sec("record", "Data source", f"Recorded from source '{src}'.")
    return _sec("record", "Data source", "The data source of the opening window is " + NOT_DETERMINED + ".")


def answer_from_context(c: dict, question: str) -> dict:
    kind, i, rule = intent(question), c["inc"], c["rule"]
    code = inc_code(i["id"])
    if kind == "predict":
        secs = [_sec("not_determined", "Failure time or probability: " + NOT_DETERMINED,
                     "This system does not predict when a machine will fail and does not calculate a probability of "
                     "failure. It only reports how unusual the machine sounds compared with its normal sound."),
                _detected(c), _action(c)]
    elif kind == "fault":
        secs = [_detected(c), _evidence(c), _not_fault(), _guidance(), _action(c)]
    elif kind == "why_high":
        secs = [_sec("model", "Why the severity is HIGH",
                     f"HIGH is set when at least {rule['k_high']} of the last {rule['m_high']} ten-second windows are "
                     f"above the alarm limit (z > {c['za']:.1f}). This incident reached a peak of z = {i['peak_z']:.1f} "
                     f"with {i['n_windows']} window(s) at HIGH. A single loud window is not enough to reach HIGH."),
                _sec("model", "Limits", f"Warning limit z > {c['zw']:.1f}; alarm limit z > {c['za']:.1f}. The limits "
                                        f"were calibrated on normal recordings of the machines.")]
    elif kind == "machine":
        secs = [_sec("record", "Affected machine",
                     f"{_machine_line(c)}. Owner: {i['owner_name']} ({i['owner_email']}). Incident {code}.")]
    elif kind == "when":
        end = (f"It closed at {fmt_time(i['closed_utc'])} after the machine returned below the limits."
               if i["closed_utc"] else "It is still open.")
        ack = (f" Acknowledged by {i['ack_by']} at {fmt_time(i['ack_utc'])}." if i["acknowledged"]
               else " It has not been acknowledged.")
        re_ = f" HIGH returned and re-opened it {i['n_reopened']} time(s)." if i["n_reopened"] else ""
        secs = [_sec("record", "Timeline", f"{code} opened at {fmt_time(i['opened_utc'])} (PKT). {end}{re_}{ack}")]
    elif kind == "notified":
        if c["alerts"]:
            lines = "; ".join(f"{ALERT_KIND.get(a['kind'], a['kind'])}: {alert_text(a)}" for a in c["alerts"])
            secs = [_sec("record", "Notification record", lines + ".")]
        else:
            secs = [_sec("record", "Notification record", "No notification was recorded for this incident.")]
    elif kind == "ack":
        txt = (f"Acknowledged by {i['ack_by']} at {fmt_time(i['ack_utc'])}."
               + (f" Note: {i['ack_note']}" if i["ack_note"] else "")) if i["acknowledged"] else "Not acknowledged yet."
        if c["notes"]:
            txt += " Operator notes: " + " | ".join(f"{n['author']}: {n['note']}" for n in c["notes"])
        secs = [_sec("record", "Acknowledgement", txt)]
    elif kind == "evidence":
        secs = [_detected(c), _evidence(c)]
    elif kind == "source":
        secs = [_source(c)]
    elif kind == "action":
        secs = [_detected(c), _guidance(), _action(c)]
    elif kind == "detect":
        secs = [_detected(c), _evidence(c), _not_fault(), _action(c)]
    else:
        secs = [_sec("not_determined", NOT_DETERMINED[0].upper() + NOT_DETERMINED[1:],
                     "This question cannot be answered from the model results or the incident records. Questions that "
                     "can be answered: " + "; ".join(SUGGESTED_QUESTIONS) + ".")]
    return {"incident_id": i["id"], "question": question, "intent": kind, "sections": secs}


def answer(conn, incident_id: int, question: str, thresholds: dict, rule: dict) -> dict:
    """Load the incident's records and answer. Raises LookupError if the incident does not exist."""
    from smart_listener import repo
    inc = repo.get_incident(conn, incident_id)
    if inc is None:
        raise LookupError("incident not found")
    opening = repo.get_analysis(conn, inc["opened_analysis_id"]) if inc["opened_analysis_id"] else None
    c = _context(inc, repo.list_severity_events(conn, inc["machine_id"], 200), repo.list_alerts(conn, incident_id=incident_id),
                 repo.list_incident_notes(conn, incident_id), opening, thresholds, rule)
    return answer_from_context(c, question)
