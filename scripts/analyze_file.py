"""Analyse one WAV file from the command line (no database, no web server).

    python scripts/analyze_file.py path/to/clip.wav --group fan_id_06
    python scripts/analyze_file.py path/to/clip.wav --group pump_id_04 --png reports/phase5/example.png

--group is the machine whose model should be used. You choose it; it is never detected from the sound.
Prints the anomaly level (z) of every 10-second window against the warning and alarm limits, and can save the
evidence picture of the highest window. Anomaly levels are not failure probabilities and do not name a fault.
"""
import argparse
import sys
from pathlib import Path

from smart_listener import evidence, inference
from smart_listener.config import load_config


def main() -> int:
    ap = argparse.ArgumentParser(description="Analyse one WAV file")
    ap.add_argument("wav")
    ap.add_argument("--group", required=True, help="model to use, for example fan_id_06 or pump_id_04")
    ap.add_argument("--png", help="save the evidence picture of the highest window here")
    args = ap.parse_args()

    cfg = load_config()
    try:
        prepared = inference.prepare_upload(args.wav, cfg)
    except inference.AudioRejected as e:
        print(f"Cannot analyse this file: {e}")
        return 1
    an = inference.Analyzer(cfg)
    ok, why = an.available(args.group)
    if not ok:
        print(f"Cannot use {args.group}: {why}")
        return 1
    hint = inference.check_hint(inference.filename_hint(args.wav), args.group)
    for w in prepared.warnings + ([hint] if hint else []):
        print(f"NOTE: {w}")
    t = an.thresholds
    print(f"{Path(args.wav).name}: {prepared.info['duration_s']} s, {len(prepared.windows)} window(s). "
          f"Limits: warning z>{t['z_warn']:.2f}, alarm z>{t['z_alarm']:.2f}")
    results = [an.analyze_waveform(args.group, y) for y in prepared.windows]
    for off, r in zip(prepared.offsets_s, results):
        print(f"  {off:7.1f} s   score {r.score:.4f}   z {r.z:6.2f}   {evidence.CLIP_TEXT[r.clip_level]}")
    if args.png:
        worst = max(range(len(results)), key=lambda i: results[i].z)
        ev = evidence.compute_evidence(an, args.group, prepared.windows[worst])
        out = Path(args.png)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(evidence.clip_figure(prepared.windows[worst], ev, cfg, f"{Path(args.wav).name} (window {worst + 1})"))
        print(f"Saved {out}\n{ev['sentence']}")
    print("Anomaly levels show how unlike this machine's normal sound a recording is; they are not failure probabilities.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
