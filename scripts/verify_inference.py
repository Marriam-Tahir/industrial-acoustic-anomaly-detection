"""Check that the application scores clips exactly like the Phase 2 evaluation did.

    python scripts/verify_inference.py              # 12 test clips per machine
    python scripts/verify_inference.py --n 30

For each machine that has a trained model, it takes clips from the Phase 2 score tables, scores the original WAV files
again with the application code (the same code an upload goes through) and compares. The two numbers must agree to
within rounding. If they do not, the application would give different answers from the ones you evaluated in Phase 2,
and Phase 5 is not complete.
"""
import argparse
import sys

import numpy as np
import pandas as pd

from smart_listener import audio, severity
from smart_listener.config import load_config, project_root, resolve_path
from smart_listener.inference import Analyzer


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare application scores with Phase 2 scores")
    ap.add_argument("--n", type=int, default=12, help="clips per machine")
    ap.add_argument("--tolerance", type=float, default=1e-3, help="largest allowed relative difference")
    args = ap.parse_args()

    cfg = load_config()
    an = Analyzer(cfg)
    rep = project_root() / "reports" / "phase2"
    manifest = pd.read_csv(resolve_path(cfg, "manifest_dir") / "manifest.csv", keep_default_na=False).set_index("file_id")
    mono = resolve_path(cfg, "mono_dir")
    n_samples = audio.expected_samples(cfg)
    rows, bad = [], 0
    for g, m in an.calibration["machines"].items():
        ok, why = an.available(g)
        score_file = rep / f"scores_{m['model']}_{g}.csv"
        if not ok or not score_file.exists():
            print(f"[skip] {g}: {why or str(score_file) + ' not found'}")
            continue
        s = pd.read_csv(score_file)
        s = s[s["split"] == "test"] if "split" in s.columns else s
        s = s.sample(min(args.n, len(s)), random_state=0)
        diffs = []
        for _, r in s.iterrows():
            y, _ = audio.fix_length(audio.load_mono_wav(mono / manifest.loc[r["file_id"], "path"], cfg), n_samples)
            new = an.analyze_waveform(g, y).score
            diffs.append(abs(new - r["score_mean"]) / max(abs(r["score_mean"]), 1e-12))
        worst = float(np.max(diffs))
        status = "OK " if worst <= args.tolerance else "BAD"
        bad += worst > args.tolerance
        rows.append((g, len(diffs), worst))
        print(f"[{status}] {g}: {len(diffs)} clips, largest relative difference {worst:.2e}")
    if not rows:
        print("Nothing could be checked (no model files or score tables found).")
        return 1
    print("\nRESULT:", "application scores match Phase 2" if not bad else f"{bad} machine(s) DIFFER - do not continue")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
