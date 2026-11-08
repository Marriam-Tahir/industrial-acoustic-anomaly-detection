"""Stream-level evaluation of the severity rules, with a small, honest tuning step.

    python scripts/evaluate_streams.py

Step 1 (TUNING, validation data only): try a small grid of rules (how many windows must exceed which limit) and pick
        the one with the lowest average detection delay among those with no false HIGH incident on normal validation
        streams (natural order) and at most --max-false-per-1000 on shuffled validation streams.
Step 2 (FINAL, test data, once): report false HIGH incidents and detection delay of the chosen rule.
        The chosen rule and limits are saved into config/severity_calibration.json.

Streams (one window = one 10-second clip):
  normal stream : the split's normal clips in recording order ("natural"), and 200 random shuffles of them.
                  Natural order keeps the real similarity between neighbouring recordings. Shuffling does not, so
                  shuffled numbers are optimistic. Both are reported.
  onset stream  : 20 random normal clips followed by up to 40 abnormal clips in recording order, started at every
                  possible abnormal clip. Delay = windows from the first abnormal clip until HIGH (or WARNING).
                  A fault that never reaches the level within 40 windows counts as 40 for the median (not detected).
These streams are REPLAYS of benchmark clips, not real machine histories. Abnormal clips from different recording
sessions are joined together, so this measures how fast the rules react to abnormal sound, not how a real fault
develops. The test set has few clips per machine, so a rate of 0 is reported with its upper bound, not as "never".
"""
import argparse
import itertools
import sys
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smart_listener import severity as sv
from smart_listener.config import project_root

PREFIX, RUN, N_SHUFFLE = 20, 40, 200
WINDOW_SECONDS = 10
COL = {"NORMAL": "#d9f0d9", "WATCH": "#fff3c4", "WARNING": "#ffd9a8", "HIGH": "#f7b4b4"}


def seed_for(base: int, group: str) -> int:
    return (base + zlib.crc32(group.encode())) % (2 ** 31)


def split_arrays(df: pd.DataFrame, split: str) -> dict:
    out = {}
    for g, d in df[df["split"] == split].groupby("group"):
        out[g] = (d[d["label"] == 0].sort_values("order")["z"].to_numpy(),
                  d[d["label"] == 1].sort_values("order")["z"].to_numpy())
    return out


def normal_eval(arrs, z_warn, z_alarm, rule, seed):
    rows = []
    for g, (n, _) in arrs.items():
        nat = sv.run_stream(n, z_warn, z_alarm, rule)
        st = np.array(nat["states"])
        rng = np.random.default_rng(seed_for(seed, g))
        inc = sum(sv.run_stream(rng.permutation(n), z_warn, z_alarm, rule)["incidents"] for _ in range(N_SHUFFLE))
        rows.append({"group": g, "windows": len(n), "natural_incidents": nat["incidents"],
                     "shuffled_incidents": inc, "shuffled_windows": N_SHUFFLE * len(n),
                     "share_watch": float((st == "WATCH").mean()), "share_warning": float((st == "WARNING").mean()),
                     "share_high": float((st == "HIGH").mean())})
    return pd.DataFrame(rows)


def onset_eval(arrs, z_warn, z_alarm, rule, seed):
    rows = []
    for g, (n, a) in arrs.items():
        rng = np.random.default_rng(seed_for(seed + 1, g))
        run = min(RUN, len(a))
        d_high, d_warn = [], []
        for i in range(len(a)):
            z = np.r_[rng.choice(n, PREFIX, replace=True), a[(np.arange(run) + i) % len(a)]]
            states = sv.run_stream(z, z_warn, z_alarm, rule)["states"]
            d_high.append(next((j - PREFIX for j in range(PREFIX, len(z)) if states[j] == "HIGH"), None))
            d_warn.append(next((j - PREFIX for j in range(PREFIX, len(z)) if sv.RANK[states[j]] >= 2), None))
        cens = lambda L: np.array([run if x is None else x for x in L])  # noqa: E731
        det = lambda L: np.array([x for x in L if x is not None])  # noqa: E731
        rows.append({"group": g, "n_onsets": len(a), "detect_rate_high": float(np.mean([x is not None for x in d_high])),
                     "median_delay_high": float(np.median(cens(d_high))),
                     "p90_delay_high_detected": float(np.percentile(det(d_high), 90)) if len(det(d_high)) else np.nan,
                     "detect_rate_warning": float(np.mean([x is not None for x in d_warn])),
                     "median_delay_warning": float(np.median(cens(d_warn)))})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Tune and evaluate severity rules on replayed streams")
    ap.add_argument("--max-false-per-1000", type=float, default=1.0,
                    help="largest accepted rate of false HIGH incidents per 1000 shuffled validation windows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    root = project_root()
    rep2, rep3, figs = root / "reports" / "phase2", root / "reports" / "phase3", root / "reports" / "figures"
    rep3.mkdir(parents=True, exist_ok=True)
    cal = sv.load_calibration()
    df = sv.load_selected_scores(rep2, cal["score_column"])
    df["z"] = np.nan
    for g, d in df.groupby("group"):
        m = cal["machines"][g]
        df.loc[d.index, "z"] = sv.to_z(d["score"], m["center"], m["scale"])
    val, test = split_arrays(df, "val"), split_arrays(df, "test")
    pooled = df[(df["split"] == "val") & (df["label"] == 0)]["z"].to_numpy()
    z_warn = cal["thresholds"]["z_warn"]

    # ---------------- step 1: tuning on validation streams
    grid = []
    for qa, (kh, mh), kw, sa in itertools.product((0.99, 0.995), ((2, 3), (3, 5), (4, 6)), (2, 3), (True, False)):
        grid.append({"q_alarm": qa, "k_high": kh, "m_high": mh, "k_warn": kw, "single_alarm_warning": sa})
    res = []
    print(f"Tuning {len(grid)} rules on validation streams (this takes about a minute)...")
    for combo in grid:
        z_alarm = float(np.quantile(pooled, combo["q_alarm"]))
        rule = {**sv.DEFAULT_RULE, **{k: v for k, v in combo.items() if k != "q_alarm"}}
        nm, on = normal_eval(val, z_warn, z_alarm, rule, args.seed), onset_eval(val, z_warn, z_alarm, rule, args.seed)
        res.append({**combo, "z_alarm": z_alarm, "val_natural_false_high": int(nm["natural_incidents"].sum()),
                    "val_shuffled_false_high_per_1000": float(nm["shuffled_incidents"].sum() / nm["shuffled_windows"].sum() * 1000),
                    "val_share_warning_normal": float(nm["share_warning"].mean()),
                    "val_mean_median_delay_high": float(on["median_delay_high"].mean()),
                    "val_mean_detect_rate_high": float(on["detect_rate_high"].mean())})
    g = pd.DataFrame(res)
    ok = g[(g["val_natural_false_high"] == 0) & (g["val_shuffled_false_high_per_1000"] <= args.max_false_per_1000)]
    if len(ok):
        best = ok.sort_values(["val_mean_median_delay_high", "val_shuffled_false_high_per_1000"]).iloc[0]
        how = "lowest mean detection delay among rules with no natural-order false HIGH and shuffled false-HIGH rate within the limit"
    else:
        best = g.sort_values(["val_shuffled_false_high_per_1000", "val_mean_median_delay_high"]).iloc[0]
        how = "NO rule met the false-alarm limit on validation; this is the one with the fewest false HIGH incidents"
        print("WARNING: " + how)
    g.sort_values("val_mean_median_delay_high").to_csv(rep3 / "stream_grid_val.csv", index=False)
    pd.set_option("display.width", 230)
    print("\nRules that met the false-alarm limit on validation, fastest first (full grid in reports/phase3/stream_grid_val.csv):"
          if len(ok) else "\nNo rule met the limit; the 8 with the fewest false HIGH incidents:")
    shown = ok.sort_values(["val_mean_median_delay_high", "val_shuffled_false_high_per_1000"]) if len(ok) else g.sort_values(["val_shuffled_false_high_per_1000", "val_mean_median_delay_high"])
    print(shown.head(8).round(3).to_string(index=False))
    rule = {**sv.DEFAULT_RULE, "k_high": int(best["k_high"]), "m_high": int(best["m_high"]), "k_warn": int(best["k_warn"]),
            "single_alarm_warning": bool(best["single_alarm_warning"])}
    z_alarm = float(best["z_alarm"])
    print(f"\nCHOSEN RULE ({how}):\n  alarm limit z={z_alarm:.2f} (q={best['q_alarm']}), warning limit z={z_warn:.2f}\n  {rule}")

    # ---------------- step 2: final numbers on validation (for reference) and test (the real result)
    for name, arrs in (("val", val), ("test", test)):
        nm, on = normal_eval(arrs, z_warn, z_alarm, rule, args.seed), onset_eval(arrs, z_warn, z_alarm, rule, args.seed)
        out = nm.merge(on, on="group")
        out["shuffled_false_high_per_1000"] = out["shuffled_incidents"] / out["shuffled_windows"] * 1000
        out.to_csv(rep3 / f"stream_metrics_{name}.csv", index=False)
        label = "TEST (final result, used once)" if name == "test" else "VALIDATION (used for tuning, optimistic)"
        print(f"\n=== {label} ===")
        cols = ["group", "windows", "natural_incidents", "shuffled_false_high_per_1000", "share_warning", "share_high",
                "n_onsets", "detect_rate_high", "median_delay_high", "p90_delay_high_detected", "detect_rate_warning", "median_delay_warning"]
        print(out[cols].round(3).to_string(index=False))
        tw, ti = int(nm["windows"].sum()), int(nm["natural_incidents"].sum())
        print(f"Normal streams in natural order: {ti} false HIGH incidents in {tw} windows ({tw * WINDOW_SECONDS / 3600:.1f} hours of audio)."
              + (f" With 0 incidents the 95% upper bound is about {3 / tw * 1000:.1f} per 1000 windows." if ti == 0 else ""))
        print(f"Shuffled (optimistic): {nm['shuffled_incidents'].sum() / nm['shuffled_windows'].sum() * 1000:.2f} false HIGH incidents per 1000 windows.")
        print(f"Onset: mean detection rate for HIGH {on['detect_rate_high'].mean():.2f}; delay is in windows, 1 window = {WINDOW_SECONDS} s; "
              f"mean of per-machine median delay {on['median_delay_high'].mean():.1f} windows ({on['median_delay_high'].mean() * WINDOW_SECONDS:.0f} s).")

    cal["thresholds"].update({"q_alarm": float(best["q_alarm"]), "z_alarm": z_alarm})
    cal["rule"] = rule
    cal["tuning"] = {"selection": how, "max_false_per_1000": args.max_false_per_1000, "grid_size": len(grid),
                     "prefix_windows": PREFIX, "run_windows": RUN, "n_shuffles": N_SHUFFLE, "seed": args.seed}
    sv.save_calibration(cal)
    print("\nSaved the chosen limits and rule into config/severity_calibration.json")

    # ---------------- figure: one example onset stream per machine (test)
    groups = sorted(test)
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), squeeze=False)
    rng = np.random.default_rng(args.seed)
    for ax, gname in zip(axes.ravel(), groups):
        n, a = test[gname]
        run = min(RUN, len(a))
        z = np.r_[rng.choice(n, PREFIX, replace=True), a[(np.arange(run) + len(a) // 2) % len(a)]]
        states = sv.run_stream(z, z_warn, z_alarm, rule)["states"]
        for i, s in enumerate(states):
            ax.axvspan(i - 0.5, i + 0.5, color=COL[s], lw=0)
        ax.plot(z, color="k", lw=1)
        ax.axhline(z_warn, color="orange", ls="--", lw=1)
        ax.axhline(z_alarm, color="red", ls="--", lw=1)
        ax.axvline(PREFIX - 0.5, color="grey", ls=":")
        ax.set_title(gname, fontsize=9)
        ax.set_xlabel("window (1 window = 10 s)", fontsize=8)
    for ax in axes.ravel()[len(groups):]:
        ax.axis("off")
    fig.suptitle("Replayed test stream: normal clips, then abnormal clips from the dotted line. Background = severity state (green NORMAL, yellow WATCH, orange WARNING, red HIGH)")
    fig.tight_layout()
    fig.savefig(figs / "phase3_example_streams.png", dpi=120)
    plt.close(fig)
    print("Saved reports/figures/phase3_example_streams.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
