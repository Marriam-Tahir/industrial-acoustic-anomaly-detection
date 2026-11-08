"""Operational severity: turns a stream of anomaly scores into NORMAL / WATCH / WARNING / HIGH and into incidents.

What this is NOT: a probability of failure, a remaining-life estimate, or a fault diagnosis. The data has no failure
times or severity labels, so none of those can be estimated. What it IS: an alert level with measured behaviour (how
often it fires on normal sound, how fast it fires on abnormal sound), built from three things:

  1. Calibration  - each machine's anomaly score is converted to a z-value: how far above THAT machine's own normal
                    level it is (in units of that machine's normal spread). Alert limits are z-values.
  2. Persistence  - one odd clip is not an incident: it takes several clips above a limit.
  3. Hysteresis   - the level only goes down after several clean clips, so it does not flicker.

One window = one 10-second clip.
"""
import json
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from smart_listener.config import project_root

STATES = ("NORMAL", "WATCH", "WARNING", "HIGH")
RANK = {s: i for i, s in enumerate(STATES)}

DEFAULT_RULE = {
    "n_window": 6,                  # look back this many windows for warning-level counts
    "k_warn": 3,                    # WARNING if at least k_warn of the last n_window windows exceed the warning limit
    "m_high": 5,                    # look back this many windows for HIGH
    "k_high": 3,                    # HIGH if at least k_high of the last m_high windows exceed the alarm limit
    "recover": 5,                   # clean windows (below the warning limit) needed before the level may go down
    "cooldown": 180,                # windows (30 min): a HIGH that returns within this time re-opens the same incident
    "single_alarm_warning": True,   # a single alarm-level window inside the last n_window windows already means WARNING
}


# ----------------------------------------------------------------------------- calibration
def calibration_path() -> Path:
    return project_root() / "config" / "severity_calibration.json"


def fit_center_scale(normal_scores):
    """Robust centre and spread of a machine's normal scores in log space (median and 1.4826 * MAD)."""
    x = np.log(np.maximum(np.asarray(normal_scores, dtype=float), 1e-12))
    center = float(np.median(x))
    scale = float(1.4826 * np.median(np.abs(x - center)))
    if scale < 1e-6:
        scale = float(np.std(x)) or 1.0
    return center, scale


def to_z(scores, center: float, scale: float):
    return (np.log(np.maximum(np.asarray(scores, dtype=float), 1e-12)) - center) / scale


def load_calibration(path=None) -> dict:
    p = Path(path) if path else calibration_path()
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Run scripts/calibrate_severity.py first.")
    return json.loads(p.read_text(encoding="utf-8"))


def save_calibration(cal: dict, path=None) -> Path:
    p = Path(path) if path else calibration_path()
    cal["updated_utc"] = datetime.now(timezone.utc).isoformat()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal, indent=2), encoding="utf-8")
    return p


def load_selected_scores(rep_dir: Path, score_col: str = "score_mean") -> pd.DataFrame:
    """Score table of the model selected per machine in Phase 2 (selected_models.csv).
    Adds `order` = recording number of the clip (from its file id), so streams can follow the real recording order."""
    sel = pd.read_csv(rep_dir / "selected_models.csv")
    parts = []
    for _, r in sel.iterrows():
        f = rep_dir / f"scores_{r['selected_model']}_{r['group']}.csv"
        d = pd.read_csv(f)
        d["score"] = d[score_col]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["order"] = df["file_id"].str.extract(r"_(\d+)$")[0].astype(int)
    return df


# ----------------------------------------------------------------------------- the state machine
class SeverityEngine:
    """Feed it one z-value per window with update(). Pure Python, no files, no clock: easy to test."""

    def __init__(self, z_warn: float, z_alarm: float, rule: dict = None):
        if z_alarm < z_warn:
            raise ValueError("the alarm limit must not be below the warning limit")
        r = {**DEFAULT_RULE, **(rule or {})}
        self.z_warn, self.z_alarm, self.rule = float(z_warn), float(z_alarm), r
        self.reset()

    def reset(self):
        r = self.rule
        self._warn = deque(maxlen=r["n_window"])
        self._alarm = deque(maxlen=max(r["n_window"], r["m_high"]))
        self.state, self.clean_run, self.t = "NORMAL", 0, -1

    def update(self, z: float) -> dict:
        r = self.rule
        self.t += 1
        w, a = z > self.z_warn, z > self.z_alarm
        self._warn.append(w)
        self._alarm.append(a)
        alarm_hist = list(self._alarm)
        n_warn = sum(self._warn)
        n_alarm_n = sum(alarm_hist[-r["n_window"]:])
        n_alarm_m = sum(alarm_hist[-r["m_high"]:])
        self.clean_run = 0 if w else self.clean_run + 1

        if n_alarm_m >= r["k_high"]:
            cand, why = "HIGH", f"{n_alarm_m} of the last {min(r['m_high'], self.t + 1)} windows above the alarm limit"
        elif n_warn >= r["k_warn"] or (r["single_alarm_warning"] and n_alarm_n >= 1):
            cand, why = "WARNING", f"{n_warn} of the last {min(r['n_window'], self.t + 1)} windows above the warning limit"
        elif n_warn >= 1:
            cand, why = "WATCH", "isolated window above the warning limit"
        else:
            cand, why = "NORMAL", "within this machine's normal range"

        if RANK[cand] >= RANK[self.state]:
            self.state, reason = cand, why
        elif self.clean_run >= r["recover"]:
            self.state, reason = cand, f"{self.clean_run} clean windows in a row; level lowered"
        else:
            reason = f"holding {self.state}: only {self.clean_run} of {r['recover']} clean windows so far"
        return {"state": self.state, "reason": reason, "z": float(z), "n_warn": n_warn,
                "n_alarm_high": n_alarm_m, "window": self.t}


class IncidentTracker:
    """Turns the state sequence into incidents. A new incident opens when the state first becomes HIGH.
    If HIGH returns within `cooldown` windows of the last close, the SAME incident is re-opened (no new alert)."""

    def __init__(self, cooldown: int = DEFAULT_RULE["cooldown"]):
        self.cooldown, self.t, self.is_open, self.last_close, self.count = cooldown, -1, False, None, 0

    def update(self, state: str):
        """Returns 'opened', 'reopened', 'closed' or None."""
        self.t += 1
        if state == "HIGH" and not self.is_open:
            self.is_open = True
            if self.last_close is not None and self.t - self.last_close <= self.cooldown:
                return "reopened"
            self.count += 1
            return "opened"
        if state != "HIGH" and self.is_open:
            self.is_open, self.last_close = False, self.t
            return "closed"
        return None


def run_stream(z_values, z_warn: float, z_alarm: float, rule: dict = None) -> dict:
    """Run a whole sequence of z-values. Returns states, number of incidents, and first WARNING / HIGH window."""
    eng = SeverityEngine(z_warn, z_alarm, rule)
    trk = IncidentTracker((rule or {}).get("cooldown", DEFAULT_RULE["cooldown"]))
    states = []
    for z in z_values:
        s = eng.update(float(z))["state"]
        trk.update(s)
        states.append(s)
    first = lambda name: next((i for i, s in enumerate(states) if RANK[s] >= RANK[name]), None)  # noqa: E731
    return {"states": states, "incidents": trk.count, "first_warning": first("WARNING"), "first_high": first("HIGH")}
