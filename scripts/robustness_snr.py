"""Phase 8 robustness test: score MIMII 0 dB and -6 dB TEST clips with the existing models and alert limits.

Download from https://zenodo.org/records/3384388 (any of these; each run handles one file):
    0_dB_fan.zip (10.4 GB)   0_dB_pump.zip (7.9 GB)   -6_dB_fan.zip (10.9 GB)   -6_dB_pump.zip (8.2 GB)
Nothing is unpacked: clips are read from the zip. After a zip has been scored you may delete it.

    python scripts/robustness_snr.py --baseline                      # the same test clips at 6 dB (once, ~minutes)
    python scripts/robustness_snr.py --source D:/MIMII/0_dB_fan.zip  # repeat for each downloaded zip
    python scripts/robustness_snr.py --summary                       # rebuild the summary from all saved scores

Writes reports/robustness/: scores_<snr>dB_<type>.csv (one row per clip), metrics.csv, summary.txt,
auc_by_snr.png. No model is retrained and no limit is changed; only TEST-split clips are used.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from smart_listener import robustness, severity
from smart_listener.config import load_config, project_root, resolve_path
from smart_listener.inference import Analyzer


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_summary(out: Path, z_warn: float, z_alarm: float) -> int:
    files = sorted(out.glob("scores_*.csv"))
    if not files:
        print("No score files yet.")
        return 1
    scores = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = robustness.summarize(scores, z_warn, z_alarm)
    df.to_csv(out / "metrics.csv", index=False)
    lines = ["ROBUSTNESS TO FACTORY NOISE - MIMII test clips scored with the 6 dB models and the SAME alert limits",
             f"limits: warning z > {z_warn:.2f}, alarm z > {z_alarm:.2f}   (nothing retrained or re-tuned)", "",
             f"{'machine':<11}{'SNR':>5}{'normal':>8}{'abnorm':>8}{'ROC-AUC':>9}{'vs 6dB':>8}"
             f"{'normal>warn':>13}{'normal>alarm':>14}{'abnormal>alarm':>16}"]
    for _, r in df.iterrows():
        d = "" if pd.isna(r["auc_change_vs_6db"]) or r["snr_db"] == 6 else f"{r['auc_change_vs_6db']:+.3f}"
        lines.append(f"{r['group']:<11}{int(r['snr_db']):>5}{int(r['n_normal']):>8}{int(r['n_abnormal']):>8}"
                     f"{r['auc']:>9.3f}{d:>8}{r['normal_above_warning_pct']:>12.1f}%{r['normal_above_alarm_pct']:>13.1f}%"
                     f"{r['abnormal_above_alarm_pct']:>15.1f}%")
    lines += ["", "How to read it:",
              "  ROC-AUC: how well normal and abnormal clips are separated (0.5 = chance, 1.0 = perfect).",
              "  normal>warn / normal>alarm: share of NORMAL clips that would cross a limit (false warnings / alarms).",
              "  abnormal>alarm: share of ABNORMAL clips above the alarm limit (detections, per 10-second window).",
              "  A machine is set to HIGH only after several windows are above the alarm limit, so per-window figures",
              "  are not the same as incident rates."]
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    groups = sorted(df["group"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for g in groups:
        h = df[df["group"] == g].sort_values("snr_db")
        ax.plot(h["snr_db"], h["auc"], marker="o", label=g)
    ax.set_xlabel("Signal-to-noise ratio (dB) - lower means louder background noise")
    ax.set_ylabel("ROC-AUC (test clips)")
    ax.set_xticks(sorted(df["snr_db"].unique()))
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.legend(fontsize=8, ncol=2)
    ax.set_title("Detector performance as factory noise increases (models trained at 6 dB)")
    fig.tight_layout()
    fig.savefig(out / "auc_by_snr.png", dpi=130)
    plt.close(fig)
    print(f"\nSaved {out / 'summary.txt'}, metrics.csv and auc_by_snr.png")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Score MIMII 0 dB / -6 dB test clips with the existing models")
    ap.add_argument("--source", help="a MIMII zip (e.g. 0_dB_fan.zip) or a folder extracted from one")
    ap.add_argument("--snr", type=int, help="SNR in dB if it cannot be read from the file name")
    ap.add_argument("--baseline", action="store_true", help="score the same test clips at 6 dB (from Phase 1 audio)")
    ap.add_argument("--summary", action="store_true", help="only rebuild the summary from saved score files")
    ap.add_argument("--verify-md5", action="store_true", help="check the zip against the official checksum first")
    ap.add_argument("--groups", nargs="*", help="limit to these machine IDs, e.g. fan_id_06 pump_id_04")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    root = project_root()
    out = Path(args.out) if args.out else root / "reports" / "robustness"
    out.mkdir(parents=True, exist_ok=True)
    cal = severity.load_calibration()
    zw, za = float(cal["thresholds"]["z_warn"]), float(cal["thresholds"]["z_alarm"])
    if args.summary:
        return write_summary(out, zw, za)
    if not args.source and not args.baseline:
        ap.error("give --source <zip>, --baseline or --summary")

    manifest = pd.read_csv(resolve_path(cfg, "manifest_dir") / "manifest.csv", keep_default_na=False)
    analyzer = Analyzer(cfg)
    groups = [g for g in (args.groups or sorted(cal["machines"])) if analyzer.available(g)[0]]
    clips = robustness.test_clips(manifest, groups)
    if clips.empty:
        print("No test clips for these machines in the manifest.")
        return 1
    t0 = time.time()
    progress = lambda k: print(f"  {k} clips scored ({time.time() - t0:.0f} s)", flush=True)

    if args.baseline:
        mono = resolve_path(cfg, "mono_dir")
        from smart_listener import audio
        n = audio.expected_samples(cfg)
        items = ((r, audio.fix_length(audio.load_mono_wav(mono / r["path"], cfg), n)[0]) for r in clips.to_dict("records"))
        print(f"Scoring {len(clips)} test clips at 6 dB ...")
        robustness.score_rows(analyzer, items, cfg, 6, progress).to_csv(out / "scores_6dB_all.csv", index=False)
        return write_summary(out, zw, za)

    src = Path(args.source)
    if not src.exists():
        print(f"Not found: {src}")
        return 1
    snr = args.snr if args.snr is not None else robustness.snr_from_name(src.name)
    if snr is None:
        ap.error("could not read the SNR from the file name; add --snr 0 or --snr -6")
    if snr == 6:
        print("This is the 6 dB release the models were trained on; use --baseline instead.")
        return 1
    if args.verify_md5 and src.is_file():
        want = cfg["dataset"]["checksums"].get(src.name)
        got = md5(src)
        print(f"md5 {got} " + ("OK" if got == want else f"DOES NOT MATCH the official {want}"))
        if want and got != want:
            return 1
    names = robustness.source_names(src)
    have, expect = robustness.folder_counts(names), robustness.manifest_counts(manifest)
    present = sorted({k[0] for k in have})
    clips = clips[clips["machine_type"].isin(present)]
    mism = [(k, have.get(k, 0), v) for k, v in expect.items() if k[0] in present and k[1] in {g.split("_", 1)[1] for g in groups}
            and have.get(k, 0) != v]
    if mism:
        print("WARNING: the number of files differs from the 6 dB release, so file numbers may not refer to the same "
              "recordings. Results for these folders are NOT comparable:")
        for k, h, v in mism:
            print(f"   {'/'.join(k)}: {h} here, {v} at 6 dB")
        bad = {f"{t}_{i}" for (t, i, _), _, _ in mism}
        clips = clips[~clips["group"].isin(bad)]
    wanted = {r["key"]: r for r in clips.to_dict("records")}
    print(f"{src.name}: SNR {snr} dB, {len(wanted)} test clips for {', '.join(sorted(clips['group'].unique()))}")
    df = robustness.score_rows(analyzer, robustness.iter_source(src, wanted), cfg, snr, progress)
    missing = len(wanted) - len(df)
    if missing:
        print(f"WARNING: {missing} expected test clips were not found in {src.name}")
    kind = "_".join(present)
    df.to_csv(out / f"scores_{snr}dB_{kind}.csv", index=False)
    print(f"Scored {len(df)} clips in {time.time() - t0:.0f} s")
    return write_summary(out, zw, za)


if __name__ == "__main__":
    sys.exit(main())
