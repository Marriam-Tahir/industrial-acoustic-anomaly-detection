"""Suggest which registered machine an uploaded recording came from - ONLY when validation supports it.

The 8 autoencoders are anomaly detectors, not machine classifiers. A clip is scored by every machine's model; the
machine whose model finds it most familiar (lowest calibrated z) is a candidate. That is unreliable in two known ways:
  - an ABNORMAL recording is unfamiliar to its own model too, so the lowest z can point at the wrong machine;
  - two similar machines can be almost equally familiar.
So a suggestion is only made when
  1. scripts/investigate_identification.py has been run on this installation's models and saved
     reports/identification/validation.json, and the rule reached MIN_ACCURACY on held-out TEST clips
     (the acceptance margin is chosen on VALIDATION clips only), and
  2. for this file the best model's z is within the alarm limit and the gap to the second-best machine is at least
     the validated margin.
Otherwise the user selects the machine. The suggestion never replaces that choice: the user always confirms it.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

MIN_ACCURACY = 0.98          # share of answered held-out clips that must be identified correctly
MIN_ANSWERED_VAL = 0.20      # a margin that answers almost nothing is not useful
MIN_TEST_ANSWERED = 50       # at least this many answered test clips before the accuracy is trusted
MARGINS = [0, 1, 2, 3, 5, 7.5, 10, 15, 20, 30, 50, 100]
VALIDATION_FILE = Path("reports") / "identification" / "validation.json"
RULE_TEXT = ("lowest calibrated anomaly level across all machine models; answered only when that level is within "
             "the alarm limit and the next machine is at least the validated margin higher")


# ----------------------------------------------------------------------------- validation (used by the script)
def _answer(Z: np.ndarray, M: float, z_alarm: float):
    srt = np.sort(Z, axis=1)
    return Z.argmin(1), (srt[:, 1] - srt[:, 0] >= M) & (srt[:, 0] <= z_alarm)


def _stats(pred, answered, truth, labels) -> dict:
    ok = pred == truth
    n_ans = int(answered.sum())
    out = {"n": int(len(truth)), "answered": n_ans, "answered_rate": float(answered.mean()) if len(truth) else 0.0,
           "correct": int((ok & answered).sum()), "wrong": int((~ok & answered).sum()),
           "accuracy_answered": float(ok[answered].mean()) if n_ans else None}
    for lab in ("normal", "abnormal"):
        m = labels == lab
        out[f"answered_rate_{lab}"] = float(answered[m].mean()) if m.any() else None
        out[f"accuracy_answered_{lab}"] = float(ok[answered & m].mean()) if (answered & m).any() else None
    return out


def evaluate(Z: np.ndarray, truth: np.ndarray, split: np.ndarray, labels: np.ndarray, groups: list, z_alarm: float) -> dict:
    """Choose the margin on validation clips, then measure it once on test clips."""
    Z, truth, split, labels = np.asarray(Z, float), np.asarray(truth), np.asarray(split), np.asarray(labels)
    v, t = split == "val", split == "test"
    table, chosen = [], None
    for M in MARGINS:
        pv, av = _answer(Z[v], M, z_alarm)
        sv = _stats(pv, av, truth[v], labels[v])
        table.append({"margin": M, "val": sv})
        if chosen is None and sv["accuracy_answered"] is not None and sv["accuracy_answered"] >= MIN_ACCURACY \
                and sv["answered_rate"] >= MIN_ANSWERED_VAL:
            chosen = M
    result = {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "groups": list(groups), "rule": RULE_TEXT, "z_alarm": float(z_alarm), "min_accuracy": MIN_ACCURACY,
              "margins_on_validation": table, "chosen_margin": chosen, "test": None, "unregistered_wrong_answer_rate": None}
    if chosen is None:
        result.update(enabled=False, reason=f"No margin reached {MIN_ACCURACY:.0%} correct on validation clips.")
        return result
    pt, at = _answer(Z[t], chosen, z_alarm)
    st = _stats(pt, at, truth[t], labels[t])
    result["test"] = st
    # a clip from a machine that is NOT registered: drop its own model and see how often a wrong machine is named
    wrong = []
    for i in range(len(groups)):
        rows = Z[t & (truth == i)]
        if len(rows):
            _, a = _answer(np.delete(rows, i, axis=1), chosen, z_alarm)
            wrong.append(a.mean())
    result["unregistered_wrong_answer_rate"] = float(np.mean(wrong)) if wrong else None
    if st["accuracy_answered"] is None or st["answered"] < MIN_TEST_ANSWERED:
        result.update(enabled=False, reason="Too few held-out test clips were answered to trust the accuracy.")
    elif st["accuracy_answered"] < MIN_ACCURACY:
        result.update(enabled=False, reason=f"Only {st['accuracy_answered']:.1%} of answered test clips were "
                                            f"identified correctly (needed {MIN_ACCURACY:.0%}).")
    else:
        result.update(enabled=True, reason="")
    return result


# ----------------------------------------------------------------------------- use in the application
def status(root: Path, groups_now: list) -> dict:
    """Whether suggestions may be shown, and the measured numbers behind that decision."""
    p = Path(root) / VALIDATION_FILE
    base = {"enabled": False, "validated": False, "rule": RULE_TEXT, "min_accuracy": MIN_ACCURACY}
    if not p.exists():
        return {**base, "reason": "Automatic identification has not been validated for these models. Run "
                                  "scripts/investigate_identification.py; until then select the machine yourself."}
    try:
        v = json.loads(p.read_text(encoding="utf-8"))
    except ValueError:
        return {**base, "reason": "The identification validation file is damaged; run the investigation again."}
    if sorted(v.get("groups", [])) != sorted(groups_now):
        return {**base, "reason": "The trained models changed since identification was validated; run the "
                                  "investigation again."}
    return {**base, "validated": True, "enabled": bool(v.get("enabled")), "reason": v.get("reason", ""),
            "margin": v.get("chosen_margin"), "test": v.get("test"), "created_utc": v.get("created_utc"),
            "unregistered_wrong_answer_rate": v.get("unregistered_wrong_answer_rate")}


def rank(analyzer, windows: list, groups: list, max_windows: int = 6) -> list:
    """Median calibrated z of the file under every machine model, most familiar first."""
    use = windows[:max_windows]
    specs = [analyzer.spectrogram(y) for y in use]
    out = []
    for g in groups:
        zs = [analyzer.result_from_score(g, analyzer.score_spectrogram(g, s)).z for s in specs]
        out.append({"group": g, "z": float(np.median(zs))})
    return sorted(out, key=lambda r: r["z"])


def suggest(ranking: list, st: dict, z_alarm: float) -> dict:
    """{'group': ... or None, 'reason': ...}. Never a guess when validation does not support one."""
    if not st.get("enabled"):
        return {"group": None, "reason": st.get("reason") or "Identification is not validated."}
    if len(ranking) < 2:
        return {"group": None, "reason": "Fewer than two machine models are available."}
    best, second = ranking[0], ranking[1]
    if best["z"] > z_alarm:
        return {"group": None, "reason": "The recording is unlike the normal sound of every registered machine, so "
                                         "it cannot be matched. Select the machine it came from."}
    if second["z"] - best["z"] < float(st["margin"]):
        return {"group": None, "reason": f"{best['group']} and {second['group']} fit this recording almost equally "
                                         f"well. Select the machine it came from."}
    acc = (st.get("test") or {}).get("accuracy_answered")
    return {"group": best["group"], "reason": f"Matched by sound. In validation on held-out recordings, "
                                              f"{acc:.1%} of answered matches were correct." if acc is not None else ""}
