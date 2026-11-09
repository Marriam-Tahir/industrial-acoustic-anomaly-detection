"""Reports built from the database: CSV exports and PDF summaries (matplotlib's PDF backend; no extra dependency).

Reports record what the system measured and what happened (incidents, acknowledgements, notifications). They state
plainly that the model does not diagnose a physical fault, predict failure time or give a failure probability.
"""
import csv
import io
import json
from pathlib import Path
from typing import Optional

from smart_listener import repo
from smart_listener.dashboard_ui import fmt_time, inc_code, machine_type_of

PRODUCT = "Smart Listener - Industrial Acoustic Anomaly Detection"
LIMITATION = ("This report records model measurements and system events. The model measures how unlike its learned "
              "normal sound a machine is. It does not diagnose the physical fault (fault diagnosis: not determined), "
              "does not predict failure time and does not give a probability of failure.")

INCIDENT_COLUMNS = ["incident", "machine_id", "machine_name", "machine_type", "status", "opened_utc", "closed_utc",
                    "peak_z", "peak_score", "high_windows", "times_reopened", "acknowledged", "ack_by", "ack_utc", "ack_note"]
ANALYSIS_COLUMNS = ["analysis_id", "machine_id", "machine_name", "source", "source_ref", "window_utc", "model", "score",
                    "z", "z_warn", "z_alarm", "clip_level", "severity_state", "session_id"]


def _csv(header, rows) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
    return buf.getvalue()


def incidents_csv(incidents: list) -> str:
    return _csv(INCIDENT_COLUMNS, [[inc_code(i["id"]), i["model_group"], i["machine_name"], i["machine_type"], i["status"],
                                    i["opened_utc"], i["closed_utc"] or "", f"{i['peak_z']:.3f}", f"{i['peak_score']:.6f}",
                                    i["n_windows"], i["n_reopened"], "yes" if i["acknowledged"] else "no",
                                    i["ack_by"] or "", i["ack_utc"] or "", i["ack_note"]] for i in incidents])


def analyses_csv(rows: list) -> str:
    return _csv(ANALYSIS_COLUMNS, [[a["id"], a["model_group"], a["machine_name"], a["source"], a["source_ref"],
                                    a["window_utc"], a["model_kind"], f"{a['score']:.6f}", f"{a['z']:.3f}",
                                    f"{a['z_warn']:.2f}", f"{a['z_alarm']:.2f}", a["clip_level"], a["state"] or "",
                                    a["session_id"] or ""] for a in rows])


# ----------------------------------------------------------------------------- PDF
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


class _Pdf:
    """A4 pages with a simple top-to-bottom text flow and embedded PNG images."""
    W, H = 8.27, 11.69

    def __init__(self, title: str):
        from matplotlib.backends.backend_pdf import PdfPages
        self.plt = _plt()
        self.buf = io.BytesIO()
        self.pdf = PdfPages(self.buf, metadata={"Title": title, "Creator": PRODUCT})
        self.fig, self.y = None, 0
        self.new_page()

    def new_page(self):
        if self.fig is not None:
            self.pdf.savefig(self.fig)
            self.plt.close(self.fig)
        self.fig = self.plt.figure(figsize=(self.W, self.H))
        self.fig.text(0.07, 0.965, PRODUCT, fontsize=8.5, color="#4a6a86")
        self.y = 0.93

    def _need(self, h):
        if self.y - h < 0.05:
            self.new_page()

    def text(self, s, size=9.5, bold=False, color="#1d2733", gap=0.006, wrap=105):
        import textwrap
        lines = textwrap.wrap(str(s), wrap) or [""]
        h = len(lines) * size * 1.45 / 72 / self.H
        self._need(h)
        self.fig.text(0.07, self.y, "\n".join(lines), fontsize=size, va="top", color=color,
                      fontweight="bold" if bold else "normal", linespacing=1.45)
        self.y -= h + gap

    def rows(self, pairs):
        for k, v in pairs:
            h = 9.5 * 1.6 / 72 / self.H
            self._need(h)
            self.fig.text(0.07, self.y, k, fontsize=9, va="top", color="#5b6b7b")
            self.fig.text(0.34, self.y, str(v), fontsize=9.5, va="top", color="#1d2733")
            self.y -= h

    def image(self, png: bytes, width=0.86):
        import matplotlib.image as mpimg
        img = mpimg.imread(io.BytesIO(png), format="png")
        h = width * self.W * img.shape[0] / img.shape[1] / self.H
        self._need(h + 0.01)
        ax = self.fig.add_axes([0.07, self.y - h, width, h])
        ax.imshow(img)
        ax.axis("off")
        self.y -= h + 0.015

    def rule(self):
        self._need(0.02)
        self.fig.add_artist(self.plt.Line2D([0.07, 0.93], [self.y, self.y], color="#c9d3dd", lw=0.8))
        self.y -= 0.015

    def bytes(self) -> bytes:
        self.pdf.savefig(self.fig)
        self.plt.close(self.fig)
        self.pdf.close()
        return self.buf.getvalue()


def _evidence_png(evidence_dir, analysis: Optional[dict]) -> Optional[bytes]:
    if not analysis or not analysis.get("evidence_json") or evidence_dir is None:
        return None
    try:
        fig = json.loads(analysis["evidence_json"]).get("figure")
    except ValueError:
        return None
    p = Path(evidence_dir) / f"m{analysis['machine_id']}" / fig if fig else None
    return p.read_bytes() if p and p.exists() else None


def _source_text(source):
    return {"simulated": "SIMULATED MONITORING (replayed MIMII dataset recordings, not a live machine)",
            "upload": "Uploaded recording"}.get(source, source or "not recorded")


def incident_pdf(conn, incident_id: int, evidence_dir=None, include_evidence: bool = True) -> bytes:
    from smart_listener import evidence
    i = repo.get_incident(conn, incident_id)
    if i is None:
        raise LookupError("incident not found")
    opening = repo.get_analysis(conn, i["opened_analysis_id"]) if i["opened_analysis_id"] else None
    events = [e for e in repo.list_severity_events(conn, i["machine_id"], 500)
              if e["event_utc"] >= i["opened_utc"][:19] and (not i["closed_utc"] or e["event_utc"] <= i["closed_utc"])]
    alerts = repo.list_alerts(conn, incident_id=incident_id)
    notes = repo.list_incident_notes(conn, incident_id)
    d = _Pdf(f"Incident report {inc_code(i['id'])}")
    d.text(f"Incident report  {inc_code(i['id'])}", size=17, bold=True, gap=0.012)
    d.rows([("Machine", f"{i['machine_name']} ({i['model_group']})"), ("Machine type", machine_type_of(i["model_group"])),
            ("Highest severity", "HIGH"), ("Status", i["status"].upper()),
            ("Start time (UTC)", fmt_time(i["opened_utc"])), ("End time (UTC)", fmt_time(i["closed_utc"]) or "ongoing"),
            ("Peak anomaly level", f"z = {i['peak_z']:.2f}  (score {i['peak_score']:.4f})"),
            ("Alarm / warning limit", f"z > {opening['z_alarm']:.2f} / z > {opening['z_warn']:.2f}" if opening else "not recorded"),
            ("Windows at HIGH", i["n_windows"]), ("Times re-opened", i["n_reopened"]),
            ("Data source", _source_text(opening["source"] if opening else None)),
            ("Acknowledged", f"yes, by {i['ack_by']} at {fmt_time(i['ack_utc'])}" if i["acknowledged"] else "no"),
            ("Owner", f"{i['owner_name']} <{i['owner_email']}>"), ("Fault diagnosis", "not determined")])
    if i["ack_note"]:
        d.text(f"Acknowledgement note: {i['ack_note']}")
    d.rule()
    d.text("Severity changes during the incident", bold=True)
    d.rows([(fmt_time(e["event_utc"]), f"{e['from_state']} -> {e['to_state']}   {e['reason']}") for e in reversed(events)]
           or [("-", "none recorded")])
    d.text("Notifications", bold=True)
    d.rows([(a["recipient"], a["status"].upper() + ("  (e-mail sending is not connected; recorded only)"
                                                     if a["status"] == "pending" else "")) for a in alerts]
           or [("-", "none recorded")])
    if notes:
        d.text("Operator notes", bold=True)
        d.rows([(f"{fmt_time(n['created_utc'])}  {n['author']}", n["note"]) for n in notes])
    if include_evidence:
        rows = [r for r in repo.list_analyses_between(conn, i["machine_id"], None, None)
                if r["state"] and r["window_utc"] >= i["opened_utc"][:10]]
        if rows:
            d.text("Anomaly level over time", bold=True)
            zw, za = rows[0]["z_warn"], rows[0]["z_alarm"]
            d.image(evidence.history_figure(rows[-400:], zw, za, f"{i['machine_name']} - anomaly level", theme="light"))
        png = _evidence_png(evidence_dir, opening)
        if png:
            d.text("Evidence for the window that opened the incident", bold=True)
            d.image(png)
            if opening and opening.get("evidence_json"):
                d.text(json.loads(opening["evidence_json"]).get("sentence", ""), size=8.5)
    d.rule()
    d.text(LIMITATION, size=8.5, color="#5b6b7b")
    return d.bytes()


def machine_pdf(conn, machine_id: int, date_from: Optional[str], date_to: Optional[str], evidence_dir=None,
                include_evidence: bool = True) -> bytes:
    from smart_listener import evidence
    m = repo.get_machine(conn, machine_id)
    if m is None:
        raise LookupError("machine not found")
    rows = repo.list_analyses_between(conn, machine_id, date_from, date_to)
    incs = [i for i in repo.list_incidents(conn, None, machine_id, 1000)
            if (not date_from or i["opened_utc"][:10] >= date_from) and (not date_to or i["opened_utc"][:10] <= date_to)]
    period = f"{date_from or 'first record'} to {date_to or 'latest record'}"
    d = _Pdf(f"Machine report {m['model_group']}")
    d.text(f"Machine report  {m['name']}", size=17, bold=True, gap=0.012)
    sim = [r for r in rows if r["source"] == "simulated"]
    up = [r for r in rows if r["source"] == "upload"]
    d.rows([("Machine ID", m["model_group"]), ("Machine type", machine_type_of(m["model_group"])),
            ("Location", m["location"] or "-"), ("Owner", f"{m['owner_name']} <{m['owner_email']}>"),
            ("Period (UTC)", period), ("Windows analysed", f"{len(rows)}  ({len(sim)} simulated monitoring, {len(up)} uploaded)"),
            ("Highest anomaly level", f"z = {max(r['z'] for r in rows):.2f}" if rows else "no data"),
            ("Limits", f"warning z > {rows[0]['z_warn']:.2f}, alarm z > {rows[0]['z_alarm']:.2f}" if rows else "-"),
            ("Windows above alarm limit", sum(r["clip_level"] == "above_alarm_limit" for r in rows)),
            ("Incidents", f"{len(incs)} ({sum(i['status'] == 'open' for i in incs)} open)"),
            ("Fault diagnosis", "not determined")])
    if sim:
        d.text("Data source: SIMULATED MONITORING - replayed MIMII dataset recordings, not a live machine.", size=8.5,
               color="#9a6a00")
    d.rule()
    d.text("Incidents", bold=True)
    d.rows([(inc_code(i["id"]), f"{i['status'].upper()}  opened {fmt_time(i['opened_utc'])}  peak z {i['peak_z']:.1f}  "
                                f"{'acknowledged' if i['acknowledged'] else 'not acknowledged'}") for i in incs]
           or [("-", "none in this period")])
    if include_evidence and sim:
        d.text("Anomaly level over time (simulated monitoring)", bold=True)
        d.image(evidence.history_figure([r for r in sim if r["state"]][-400:], sim[0]["z_warn"], sim[0]["z_alarm"],
                                        f"{m['name']} - anomaly level", theme="light"))
    if include_evidence and rows:
        top = max(rows, key=lambda r: r["z"])
        png = _evidence_png(evidence_dir, top)
        if png:
            d.text(f"Evidence for the highest window ({fmt_time(top['window_utc'])}, z = {top['z']:.1f})", bold=True)
            d.image(png)
    d.rule()
    d.text(LIMITATION, size=8.5, color="#5b6b7b")
    return d.bytes()
