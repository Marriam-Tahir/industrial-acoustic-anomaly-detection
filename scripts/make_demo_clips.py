"""Copy ready-to-upload test recordings into demo_audio/ for trying the dashboard.

    python scripts/make_demo_clips.py                      # FAN-06 and PUMP-04 (the clearest models)
    python scripts/make_demo_clips.py --groups all         # all 8 machines
    python scripts/make_demo_clips.py --n 3 --groups pump_id_00 fan_id_02

For each machine it copies TEST-split clips only (never training clips: a model has already seen those, so they
would look normal for the wrong reason):
    <machine>_normal_1.wav ...      single 10-second normal clips
    <machine>_abnormal_1.wav ...    single 10-second abnormal clips
    <machine>_normal_then_abnormal_60s.wav   3 normal + 3 abnormal clips joined (shows severity over a recording)
The machine ID is at the start of each name, so Analyze Audio pre-selects the right machine.
demo_audio/ holds MIMII data: do not commit it (it is added to .gitignore by this script).
"""
import argparse
import shutil
import sys

import numpy as np
import pandas as pd
import soundfile as sf

from smart_listener import audio, profiles
from smart_listener.config import load_config, project_root, resolve_path

DEFAULT = ["fan_id_06", "pump_id_04"]


def main() -> int:
    ap = argparse.ArgumentParser(description="Copy test recordings for trying the dashboard")
    ap.add_argument("--groups", nargs="*", default=DEFAULT, help="machine IDs, or 'all'")
    ap.add_argument("--n", type=int, default=2, help="single clips per label and machine")
    ap.add_argument("--out", default="demo_audio")
    args = ap.parse_args()
    cfg = load_config()
    root = project_root()
    groups = profiles.model_groups(cfg) if args.groups == ["all"] else args.groups
    man = pd.read_csv(resolve_path(cfg, "manifest_dir") / "manifest.csv", keep_default_na=False)
    mono = resolve_path(cfg, "mono_dir")
    out = root / args.out
    out.mkdir(exist_ok=True)
    n_samples = audio.expected_samples(cfg)
    print(f"Writing to {out}\n")
    for g in groups:
        m = man[(man["group"] == g) & (man["split_ad"] == "test") & (man["exclude_reason"] == "")]
        if m.empty:
            print(f"{g}: no test clips found (is it a valid machine ID?)")
            continue
        picked = {}
        for lab in ("normal", "abnormal"):
            rows = m[m["label_name"] == lab].sort_values("index_in_label")
            # spread the picks over the whole test split instead of taking neighbouring files
            idx = np.linspace(0, len(rows) - 1, num=min(max(args.n, 3), len(rows))).round().astype(int)
            picked[lab] = rows.iloc[sorted(set(idx))]
            for k, path in enumerate(picked[lab]["path"].head(args.n), 1):
                shutil.copyfile(mono / path, out / f"{g}_{lab}_{k}.wav")
        parts = [audio.fix_length(audio.load_mono_wav(mono / p, cfg), n_samples)[0]
                 for lab in ("normal", "abnormal") for p in picked[lab]["path"].head(3)]
        sf.write(out / f"{g}_normal_then_abnormal_60s.wav", np.concatenate(parts), cfg["audio"]["sample_rate"])
        print(f"{g}: {min(args.n, len(picked['normal']))} normal, {min(args.n, len(picked['abnormal']))} abnormal, "
              f"1 joined file ({len(parts) * 10} s)")
    gi = root / ".gitignore"
    line = f"{args.out}/"
    if not gi.exists() or line not in gi.read_text(encoding="utf-8", errors="replace").splitlines():
        with open(gi, "a", encoding="utf-8") as f:
            f.write(f"\n{line}\n")
        print(f"\nAdded {line} to .gitignore")
    return 0


if __name__ == "__main__":
    sys.exit(main())
