"""READ-ONLY investigation: can the 8 existing autoencoders identify WHICH machine a recording came from?

Nothing is trained, downloaded or changed. For every validation and test clip of all 8 machines (normal AND abnormal),
the clip is scored by ALL 8 models, exactly as the application scores an upload (same features, each model's own
normalisation, each machine's own calibration to z). Then several simple identification rules are compared against the
known truth from the manifest.

    python scripts/investigate_identification.py            # all clips of the val and test splits
    python scripts/investigate_identification.py --max-per-group 120      # quicker look

Outputs (reports/identification/): zmatrix.csv (one row per clip, one z per model), summary.txt (the numbers below),
validation.json (read by the application: suggestions are switched on only if this validation passes).
Training clips are NOT used (the models have seen those, so they would flatter the result).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from smart_listener import anomaly, identification
from smart_listener.config import load_config, project_root
from smart_listener.inference import Analyzer

MARGINS = [0, 2, 5, 10, 20, 50, 100]


def pct(x):
    return f"{100 * x:5.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-per-group", type=int, default=0, help="limit clips per machine (0 = all val+test clips)")
    ap.add_argument("--out", default="reports/identification")
    args = ap.parse_args()

    cfg = load_config()
    an = Analyzer(cfg)
    groups = [g for g in an.calibration["machines"] if an.available(g)[0]]
    if len(groups) < 2:
        print("Need at least two trained models.")
        return 1
    zw, za = an.thresholds["z_warn"], an.thresholds["z_alarm"]
    out_dir = project_root() / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Models: {groups}\nScoring every val/test clip with every model (this takes a few minutes)...")

    rows, raw = [], []
    centroid = {g: np.asarray(anomaly.load_group(cfg, g)["mean"], dtype=np.float64) for g in groups}
    for gi, g in enumerate(groups):
        data = anomaly.load_group(cfg, g)
        r = data["rows"]
        idx = np.where(r["split_ad"].isin(["val", "test"]).to_numpy())[0]
        if args.max_per_group and len(idx) > args.max_per_group:
            idx = np.random.default_rng(0).choice(idx, args.max_per_group, replace=False)
            idx.sort()
        for n, i in enumerate(idx):
            spec = np.asarray(data["spec"][i])
            scores = {m: an.score_spectrogram(m, spec) for m in groups}
            z = {m: an.result_from_score(m, scores[m]).z for m in groups}
            clipmean = spec.mean(axis=1).astype(np.float64)          # mean log-mel spectrum of the clip (64 bands)
            cdist = {m: float(np.linalg.norm(clipmean - centroid[m])) for m in groups}
            rows.append({"true": g, "split": r.loc[i, "split_ad"], "label": r.loc[i, "label_name"], "file_id": r.loc[i, "file_id"],
                         **{f"z_{m}": z[m] for m in groups}, **{f"s_{m}": scores[m] for m in groups},
                         **{f"c_{m}": cdist[m] for m in groups}})
            if (n + 1) % 100 == 0:
                print(f"  {g}: {n + 1}/{len(idx)}", end="\r")
        print(f"  [{gi + 1}/{len(groups)}] {g}: {len(idx)} clips done      ")
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "zmatrix.csv", index=False)

    Z = df[[f"z_{m}" for m in groups]].to_numpy()
    S = df[[f"s_{m}" for m in groups]].to_numpy()
    C = df[[f"c_{m}" for m in groups]].to_numpy()
    truth = df["true"].map({g: i for i, g in enumerate(groups)}).to_numpy()
    typ = np.array([g.split("_")[0] for g in groups])

    lines = []
    P = lambda s="": (lines.append(s), print(s))
    P(f"IDENTIFICATION INVESTIGATION - {len(df)} clips (val+test, normal and abnormal), {len(groups)} models")
    P(f"calibrated limits: warning z>{zw:.2f}, alarm z>{za:.2f}\n")

    rules = {"A  lowest calibrated z": Z.argmin(1), "B  lowest RAW score (not comparable across models)": S.argmin(1),
             "C  nearest mean-spectrum centroid (no model needed)": C.argmin(1)}
    P("1. Top-1 accuracy of each rule (random guessing = 12.5%)")
    P(f"{'rule':58s}{'all':>8s}{'normal':>8s}{'abnormal':>9s}{'type only':>10s}")
    for name, pred in rules.items():
        ok = pred == truth
        tok = typ[pred] == typ[truth]
        P(f"{name:58s}{pct(ok.mean()):>8s}{pct(ok[df.label == 'normal'].mean()):>8s}{pct(ok[df.label == 'abnormal'].mean()):>9s}{pct(tok.mean()):>10s}")

    pred = rules["A  lowest calibrated z"]
    P("\n2. Rule A per machine (rows: true machine; columns: how often each model won)")
    conf = pd.crosstab(pd.Categorical(df["true"], groups), pd.Categorical([groups[p] for p in pred], groups), dropna=False)
    P(conf.to_string())
    per = pd.DataFrame({"acc_normal": [(pred == truth)[(truth == i) & (df.label == "normal").to_numpy()].mean() for i in range(len(groups))],
                        "acc_abnormal": [(pred == truth)[(truth == i) & (df.label == "abnormal").to_numpy()].mean() for i in range(len(groups))]}, index=groups)
    P("\n   accuracy by machine and label:\n" + (per * 100).round(1).to_string())

    P("\n3. Is the right model's z lower than the best WRONG model's z? (this is what Rule A needs)")
    own = Z[np.arange(len(df)), truth]
    Zw = Z.copy()
    Zw[np.arange(len(df)), truth] = np.inf
    best_wrong = Zw.min(1)
    P(f"   own-model z:        median {np.median(own):8.1f}   95th pct {np.percentile(own, 95):8.1f}")
    P(f"   best wrong-model z: median {np.median(best_wrong):8.1f}   5th pct  {np.percentile(best_wrong, 5):8.1f}")
    for lab in ("normal", "abnormal"):
        m = (df.label == lab).to_numpy()
        P(f"   {lab:9s}: own z median {np.median(own[m]):8.1f}; best wrong z median {np.median(best_wrong[m]):8.1f}; "
          f"own z < best wrong z in {pct((own[m] < best_wrong[m]).mean())}")

    P("\n4. Wrong-model leakage: how often does ANOTHER machine's model call the clip 'within normal range' (z <= warning limit)?")
    leak = pd.DataFrame(index=groups, columns=groups, dtype=float)
    for i, g in enumerate(groups):
        for j, m in enumerate(groups):
            leak.loc[g, m] = np.nan if i == j else (Z[truth == i, j] <= zw).mean() * 100
    P("   rows: true machine, columns: wrong model, value: % of clips that look NORMAL to that wrong model\n" + leak.round(1).to_string(na_rep="-"))
    attractors = leak.max(axis=0).sort_values(ascending=False)
    P(f"\n   most 'accepting' wrong models: " + ", ".join(f"{k} ({v:.1f}%)" for k, v in attractors.head(3).items()))

    P("\n5. Accept only confident answers (Rule A). Two acceptance tests are compared:")
    P("   'margin only'   : best-vs-second z gap >= M   (also accepts clips that are abnormal for their own machine)")
    P("   'margin+normal' : margin >= M AND best z <= alarm limit   (refuses clips that look abnormal even to the best model)")
    srt = np.sort(Z, axis=1)
    margin = srt[:, 1] - srt[:, 0]
    best_z = srt[:, 0]
    ok = (pred == truth)
    P(f"{'M':>6s} | {'answered':>9s}{'correct':>9s}{'wrong':>7s} | {'answered':>9s}{'correct':>9s}{'wrong':>7s}")
    for M in MARGINS:
        a1 = margin >= M
        a2 = a1 & (best_z <= za)
        f = lambda m: f"{pct(m.mean()):>9s}{(pct(ok[m].mean()) if m.any() else '      n/a'):>9s}{int((~ok[m]).sum()):>7d}"
        P(f"{M:>6d} | {f(a1)} | {f(a2)}")

    P("\n6. Unregistered-machine test: remove the true machine's model; would the system wrongly accept the clip?")
    P(f"{'true machine (model removed)':30s}{'accepted as normal (z<=warn)':>30s}{'accepted (z<=alarm)':>22s}")
    for i, g in enumerate(groups):
        others = np.delete(Z[truth == i], i, axis=1).min(1)
        P(f"{g:30s}{pct((others <= zw).mean()):>30s}{pct((others <= za).mean()):>22s}")

    P("\n7. Own-model sanity: share of each machine's NORMAL clips that its own model calls normal, and abnormal clips above the alarm limit")
    for i, g in enumerate(groups):
        n = (truth == i) & (df.label == "normal").to_numpy()
        a = (truth == i) & (df.label == "abnormal").to_numpy()
        P(f"   {g}: normal within limit {pct((Z[n, i] <= zw).mean())}; abnormal above alarm {pct((Z[a, i] > za).mean())}")
    P("\n8. Validation used by the application (margin chosen on VALIDATION clips, measured once on TEST clips)")
    v = identification.evaluate(Z, truth, df["split"].to_numpy(), df["label"].to_numpy(), groups, za)
    if v["chosen_margin"] is None:
        P(f"   no margin reached {identification.MIN_ACCURACY:.0%} on validation clips -> suggestions stay OFF")
    else:
        t = v["test"]
        P(f"   chosen margin {v['chosen_margin']}; test clips answered {t['answered']}/{t['n']} ({pct(t['answered_rate'])}), "
          f"correct {t['correct']}, wrong {t['wrong']}")
        P(f"   answered on test: normal {pct(t['answered_rate_normal'] or 0)}, abnormal {pct(t['answered_rate_abnormal'] or 0)}")
        P(f"   unregistered machine named wrongly: {pct(v['unregistered_wrong_answer_rate'] or 0)}")
    P(f"   application suggestions: {'ON' if v['enabled'] else 'OFF'} {v['reason']}")
    import json as _json
    (out_dir / "validation.json").write_text(_json.dumps(v, indent=2), encoding="utf-8")
    (out_dir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    P(f"\nSaved {out_dir / 'summary.txt'} and {out_dir / 'zmatrix.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
