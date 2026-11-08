"""Phase 5: continuous monitoring (SIMULATED from dataset recordings) and the shared per-window pipeline.

IMPORTANT, and shown to users everywhere: simulated monitoring replays recordings from the MIMII test data one
10-second clip at a time, as if they arrived live from a microphone. It is a demonstration of the monitoring pipeline,
NOT a connection to a real machine. The replayed sequence is built by placing separate recordings one after another
(for example 12 normal ones, then 24 abnormal ones); the dataset has no continuous long recordings or fault onsets.

Real microphone later: write a class like ReplaySource that yields (waveform, meta) every 10 seconds. Nothing else
in this file needs to change - ReplaySource and MicrophoneSource (stub below) are the two sources.

Per window the pipeline does:  score -> z -> severity state -> save analysis -> state change event ->
incident (open / continue / close) -> alert RECORD for a new incident (Phase 7 sends it) -> evidence picture
(for windows above the warning limit).

Every monitoring thread has its OWN database connection and its own severity engine and incident tracker.
"""
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd

from smart_listener import alerts, audio, db, evidence, repo, severity
from smart_listener.config import resolve_path

SCENARIOS = {
    "normal": "only normal recordings (should stay NORMAL)",
    "fault": "normal recordings, then abnormal recordings (the dataset's fault recordings)",
    "fault_recovery": "normal, then abnormal, then normal again (shows the incident closing)",
}
WINDOW_SECONDS = 10
EVIDENCE_PER_SESSION = 100      # most evidence pictures saved per monitoring session


class ReplayError(ValueError):
    """The requested replay cannot be built (for example not enough recordings). Message is user-readable."""


# ----------------------------------------------------------------------------- sources
def window_times(n: int, end: Optional[datetime] = None) -> list:
    """Simulated clock: the LAST window ends 'now' and earlier windows lie 10 seconds apart before it, so no
    timestamp is ever in the future and they increase with the replay order."""
    end = end or datetime.now(timezone.utc).replace(microsecond=0)
    return [(end - timedelta(seconds=WINDOW_SECONDS * (n - 1 - i))).isoformat(timespec="seconds") for i in range(n)]


class ReplaySource:
    """Replays dataset clips of one machine group in recording order. Iterating yields (waveform, meta)."""

    kind = "simulated"

    def __init__(self, cfg: dict, group: str, scenario: str = "fault", split: str = "test", n_normal_before: int = 12,
                 n_fault: int = 24, n_normal_after: int = 12, n_windows: int = 30,
                 manifest: Optional[pd.DataFrame] = None, mono_dir=None):
        if scenario not in SCENARIOS:
            raise ReplayError(f"unknown scenario '{scenario}'. Use one of {sorted(SCENARIOS)}")
        if split not in ("val", "test"):
            raise ReplayError("split must be 'val' or 'test' (training clips are never replayed as 'new' sound)")
        for name, v in (("n_normal_before", n_normal_before), ("n_fault", n_fault), ("n_normal_after", n_normal_after),
                        ("n_windows", n_windows)):
            if not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > 500:
                raise ReplayError(f"{name} must be a whole number from 0 to 500")
        self.cfg, self.group, self.scenario, self.split = cfg, group, scenario, split
        self.mono_dir = Path(mono_dir) if mono_dir else resolve_path(cfg, "mono_dir")
        if manifest is None:
            manifest = pd.read_csv(resolve_path(cfg, "manifest_dir") / "manifest.csv", keep_default_na=False)
        m = manifest[(manifest["group"] == group) & (manifest["split_ad"] == split) & (manifest["exclude_reason"] == "")]
        self._normal = m[m["label_name"] == "normal"].sort_values("index_in_label")
        self._abnormal = m[m["label_name"] == "abnormal"].sort_values("index_in_label")
        if scenario == "normal":
            plan = [("normal", n_windows)]
        elif scenario == "fault":
            plan = [("normal", n_normal_before), ("abnormal", n_fault)]
        else:
            plan = [("normal", n_normal_before), ("abnormal", n_fault), ("normal", n_normal_after)]
        need_n = sum(c for k, c in plan if k == "normal")
        need_a = sum(c for k, c in plan if k == "abnormal")
        if need_n > len(self._normal) or need_a > len(self._abnormal):
            raise ReplayError(f"{group} has {len(self._normal)} normal and {len(self._abnormal)} abnormal {split} clips; "
                              f"this scenario needs {need_n} and {need_a}. Ask for fewer windows.")
        if need_n + need_a == 0:
            raise ReplayError("the scenario contains no windows")
        pools = {"normal": iter(self._normal.to_dict("records")), "abnormal": iter(self._abnormal.to_dict("records"))}
        self.plan = [next(pools[k]) | {"replay_label": k} for k, c in plan for _ in range(c)]

    def __len__(self) -> int:
        return len(self.plan)

    def __iter__(self) -> Iterator:
        n = audio.expected_samples(self.cfg)
        for i, row in enumerate(self.plan):
            y = audio.load_mono_wav(self.mono_dir / row["path"], self.cfg)
            y, _ = audio.fix_length(y, n)
            yield y, {"index": i, "file_id": row["file_id"], "replay_label": row["replay_label"]}


class MicrophoneSource:
    """Extension point for a real microphone: it would yield (waveform, meta) for each 10 seconds of live sound.
    Not implemented in this project: no live machine or microphone is connected."""

    kind = "live"

    def __iter__(self):
        raise NotImplementedError("live microphone input is not implemented; use ReplaySource")


# ----------------------------------------------------------------------------- the per-window pipeline
@dataclass
class SessionState:
    machine: dict
    session_id: int
    source: str
    engine: severity.SeverityEngine
    tracker: severity.IncidentTracker
    settings: dict
    evidence_dir: Optional[Path] = None
    evidence_left: int = EVIDENCE_PER_SESSION
    incident_id: Optional[int] = None
    prev_state: str = "NORMAL"
    last_window_utc: Optional[str] = None
    counts: dict = field(default_factory=lambda: {"windows": 0, "incidents_opened": 0})


def new_state(machine: dict, session_id: int, source: str, analyzer, settings: dict, evidence_dir=None) -> SessionState:
    t = analyzer.thresholds
    rule = analyzer.rule
    return SessionState(machine, session_id, source, severity.SeverityEngine(t["z_warn"], t["z_alarm"], rule),
                        severity.IncidentTracker(rule.get("cooldown", severity.DEFAULT_RULE["cooldown"])),
                        settings, Path(evidence_dir) if evidence_dir else None)


def process_window(conn, analyzer, st: SessionState, y: np.ndarray, window_utc: str, source_ref: str = "",
                   meta: Optional[dict] = None) -> dict:
    """Run ONE 10-second window through the whole pipeline and return a summary dict (also used for the API)."""
    group = st.machine["model_group"]
    res = analyzer.analyze_waveform(group, y)
    step = st.engine.update(res.z)
    state = step["state"]
    ev_json = dict(meta or {})
    aid = repo.add_analysis(conn, machine_id=st.machine["id"], source=st.source, window_utc=window_utc,
                            model_kind=res.model_kind, score=res.score, z=res.z, z_warn=res.z_warn, z_alarm=res.z_alarm,
                            clip_level=res.clip_level, state=state, reason=step["reason"], session_id=st.session_id,
                            source_ref=source_ref, evidence=ev_json)
    if state != st.prev_state:
        repo.add_severity_event(conn, machine_id=st.machine["id"], from_state=st.prev_state, to_state=state,
                                event_utc=window_utc, analysis_id=aid, reason=step["reason"])
    st.prev_state = state

    sentence = ""
    if res.clip_level != "within_normal" and st.evidence_dir is not None and st.evidence_left > 0:
        try:
            ev = evidence.compute_evidence(analyzer, group, y)
            st.evidence_dir.mkdir(parents=True, exist_ok=True)
            png = evidence.clip_figure(y, ev, analyzer.cfg, f"{st.machine['name']} - {window_utc}")
            fname = f"a{aid}.png"
            (st.evidence_dir / fname).write_bytes(png)
            ev_json.update({"summary": ev["summary"], "sentence": ev["sentence"], "figure": fname})
            repo.set_analysis_evidence(conn, aid, ev_json)
            sentence = ev["sentence"]
            st.evidence_left -= 1
        except Exception:  # a failed picture must never stop monitoring
            ev_json["evidence_error"] = traceback.format_exc(limit=1).strip().splitlines()[-1][:200]
            repo.set_analysis_evidence(conn, aid, ev_json)

    event = st.tracker.update(state)
    incident_event = None
    if event == "opened":
        st.incident_id = repo.open_incident(conn, machine_id=st.machine["id"], opened_utc=window_utc, analysis_id=aid,
                                            z=res.z, score=res.score, summary={"model_group": group, "source": st.source})
        alerts.record_high_alerts(conn, st.machine, st.incident_id, window_utc, res.z, st.settings, sentence)
        st.counts["incidents_opened"] += 1
        incident_event = "opened"
    elif event == "reopened" and st.incident_id:
        repo.reopen_incident(conn, st.incident_id)
        repo.update_incident_progress(conn, st.incident_id, res.z, res.score)
        incident_event = "reopened"
    elif event == "closed" and st.incident_id:
        repo.close_incident(conn, st.incident_id, window_utc)
        incident_event = "closed"
    elif state == "HIGH" and st.incident_id:
        repo.update_incident_progress(conn, st.incident_id, res.z, res.score)
    st.last_window_utc = window_utc
    st.counts["windows"] += 1
    return {"analysis_id": aid, "window_utc": window_utc, "state": state, "reason": step["reason"],
            "incident_event": incident_event, "incident_id": st.incident_id, **res.as_dict()}


def close_open_incident(conn, st: SessionState) -> None:
    """When a session ends, an incident that is still open is closed at the time of the last window."""
    if st.tracker.is_open and st.incident_id:
        repo.close_incident(conn, st.incident_id, st.last_window_utc or repo.now_utc())


# ----------------------------------------------------------------------------- the monitoring manager
class MonitorError(RuntimeError):
    """Monitoring cannot start. `code` is an HTTP-style status the API passes on."""

    def __init__(self, message: str, code: int = 409):
        super().__init__(message)
        self.code = code


class MonitorManager:
    """Starts, stops and reports simulated monitoring sessions. One session per machine at a time."""

    def __init__(self, db_path, analyzer, cfg: dict, evidence_dir=None, max_sessions: int = 4, source_factory=None):
        self.db_path, self.analyzer, self.cfg = db_path, analyzer, cfg
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        self.max_sessions = max_sessions
        self.source_factory = source_factory or (lambda machine, opts: ReplaySource(cfg, machine["model_group"], **opts))
        self._lock = threading.Lock()
        self._threads, self._stops, self._progress = {}, {}, {}

    # ---- control
    def start(self, machine: dict, options: Optional[dict] = None, background: bool = True, delay: Optional[float] = None) -> dict:
        options = dict(options or {})
        if not machine.get("active", 1):
            raise MonitorError("this machine is inactive; activate it first")
        ok, why = self.analyzer.available(machine["model_group"])
        if not ok:
            raise MonitorError(f"cannot monitor {machine['name']}: {why}", 409)
        try:
            source = self.source_factory(machine, options)
        except ReplayError as e:
            raise MonitorError(str(e), 422) from None
        conn = db.connect(self.db_path)
        try:
            with self._lock:
                running = [m for m, _ in self._progress.items() if self._progress[m].get("running")]
                if any(self._progress[s]["machine_id"] == machine["id"] for s in running):
                    raise MonitorError(f"{machine['name']} is already being monitored; stop that session first")
                if len(running) >= self.max_sessions:
                    raise MonitorError(f"at most {self.max_sessions} machines can be monitored at the same time")
                settings = repo.get_settings(conn)
                sid = repo.create_session_row(conn, machine["id"], "simulated",
                                              f"SIMULATED replay: {options.get('scenario', 'fault')}, {len(source)} clips "
                                              f"from the {options.get('split', 'test')} data")
                self._progress[sid] = {"running": True, "machine_id": machine["id"], "done": 0, "total": len(source),
                                       "latest": None, "error": None}
                self._stops[sid] = threading.Event()
        finally:
            conn.close()
        if delay is None:
            delay = 0.0 if not background else float(settings["replay_seconds_per_window"])
        if background:
            th = threading.Thread(target=self._run, args=(sid, machine, source, settings, delay), daemon=True,
                                  name=f"monitor-{sid}")
            self._threads[sid] = th
            th.start()
        else:
            self._run(sid, machine, source, settings, delay)
        return self.status(sid)

    def stop(self, session_id: int) -> bool:
        ev = self._stops.get(session_id)
        if ev is None or not self._progress.get(session_id, {}).get("running"):
            return False
        ev.set()
        th = self._threads.get(session_id)
        if th is not None:
            th.join(timeout=30)
        return True

    def status(self, session_id: int) -> Optional[dict]:
        conn = db.connect(self.db_path)
        try:
            row = repo.get_session_row(conn, session_id)
        finally:
            conn.close()
        if row is None:
            return None
        prog = self._progress.get(session_id)
        row["progress"] = ({k: prog[k] for k in ("done", "total", "latest", "error")} if prog else None)
        row["simulated"] = row["source"] == "simulated"
        row["label"] = "SIMULATED: replayed dataset recordings, not a live machine" if row["simulated"] else ""
        return row

    def active_session_ids(self) -> list:
        return [s for s, p in self._progress.items() if p.get("running")]

    def cleanup_stale(self) -> int:
        conn = db.connect(self.db_path)
        try:
            return repo.fail_stale_sessions(conn)
        finally:
            conn.close()

    def shutdown(self, timeout: float = 15.0) -> None:
        for sid, ev in list(self._stops.items()):
            ev.set()
        for th in list(self._threads.values()):
            th.join(timeout=timeout)

    # ---- the worker
    def _run(self, sid: int, machine: dict, source, settings: dict, delay: float) -> None:
        conn = db.connect(self.db_path)
        prog, stop = self._progress[sid], self._stops[sid]
        status, note = "finished", ""
        st = None
        try:
            st = new_state(machine, sid, "simulated", self.analyzer, settings,
                           (self.evidence_dir / f"m{machine['id']}") if self.evidence_dir else None)
            times = window_times(len(source))
            for i, (y, meta) in enumerate(source):
                if stop.is_set():
                    status, note = "stopped", "stopped by an administrator"
                    break
                out = process_window(conn, self.analyzer, st, y, times[i], meta.get("file_id", ""),
                                     {"replay_label": meta.get("replay_label"), "simulated": True})
                prog["done"], prog["latest"] = i + 1, {k: out[k] for k in ("window_utc", "state", "z", "clip_level", "incident_event")}
                if delay > 0 and stop.wait(delay):
                    status, note = "stopped", "stopped by an administrator"
                    break
        except Exception as e:
            status, note = "error", f"{type(e).__name__}: {e}"[:300]
            prog["error"] = note
        finally:
            try:
                if st is not None:
                    close_open_incident(conn, st)
                repo.finish_session_row(conn, sid, status, note)
            finally:
                prog["running"] = False
                conn.close()


# ----------------------------------------------------------------------------- uploads
def analyze_upload(conn, analyzer, machine: dict, prepared, filename: str, evidence_dir=None) -> dict:
    """Analyse the windows of an uploaded file. Every window is saved (source 'upload') with its clip level.

    An upload never changes the machine's current alert state, never opens an incident and never creates an alert:
    it is a manual check of a recording, not live monitoring. The severity engine is run over the file's windows only
    to tell the user what alert level THIS FILE alone would show (`file_state`)."""
    group = machine["model_group"]
    t = analyzer.thresholds
    engine = severity.SeverityEngine(t["z_warn"], t["z_alarm"], analyzer.rule)
    sid = repo.create_session_row(conn, machine["id"], "upload", f"upload: {Path(filename).name[:100]}")
    times = window_times(len(prepared.windows))
    rows, states = [], []
    try:
        for i, y in enumerate(prepared.windows):
            res = analyzer.analyze_waveform(group, y)
            step = engine.update(res.z)
            states.append(step["state"])
            aid = repo.add_analysis(conn, machine_id=machine["id"], source="upload", window_utc=times[i],
                                    model_kind=res.model_kind, score=res.score, z=res.z, z_warn=res.z_warn,
                                    z_alarm=res.z_alarm, clip_level=res.clip_level, state=None, session_id=sid,
                                    source_ref=Path(filename).name[:120],
                                    evidence={"offset_s": prepared.offsets_s[i]})
            rows.append({"analysis_id": aid, "offset_s": prepared.offsets_s[i], "window_utc": times[i], **res.as_dict()})
        worst = int(np.argmax([r["z"] for r in rows]))
        figure = None
        if evidence_dir is not None:
            ev = evidence.compute_evidence(analyzer, group, prepared.windows[worst])
            d = Path(evidence_dir) / f"m{machine['id']}"
            d.mkdir(parents=True, exist_ok=True)
            figure = f"a{rows[worst]['analysis_id']}.png"
            (d / figure).write_bytes(evidence.clip_figure(prepared.windows[worst], ev, analyzer.cfg,
                                                          f"{machine['name']} - {Path(filename).name} (window {worst + 1})"))
            repo.set_analysis_evidence(conn, rows[worst]["analysis_id"],
                                       {"offset_s": prepared.offsets_s[worst], "summary": ev["summary"],
                                        "sentence": ev["sentence"], "figure": figure})
            rows[worst]["evidence"] = {"summary": ev["summary"], "sentence": ev["sentence"], "figure": figure}
        repo.finish_session_row(conn, sid, "finished")
    except Exception as e:
        repo.finish_session_row(conn, sid, "error", f"{type(e).__name__}: {e}")
        raise
    n = len(rows)
    return {"session_id": sid, "windows": rows, "worst_window": worst, "evidence_figure": figure,
            "summary": {"n_windows": n,
                        "n_above_warning_limit": sum(r["clip_level"] != "within_normal" for r in rows),
                        "n_above_alarm_limit": sum(r["clip_level"] == "above_alarm_limit" for r in rows),
                        "max_z": max(r["z"] for r in rows), "median_z": float(np.median([r["z"] for r in rows])),
                        "file_state": states[-1],
                        "file_state_note": ("The alert level is judged over several consecutive 10-second windows. "
                                            "This file has only %d, so use the per-window levels." % n) if n < engine.rule["n_window"] else ""}}
