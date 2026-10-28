"""Turn the downloaded ZIPs into mono 16 kHz WAV files (microphone 1 only), one machine at a time.

    python scripts/prepare_mono.py          # all 8 machines from Dataset/ into data/interim/mono16k/

Safe to stop and run again: finished files are skipped. The original ZIPs are never changed.
"""
import argparse
import subprocess
import sys
from pathlib import Path

from smart_listener.config import load_config, resolve_path, resolve_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract mono WAVs from the MIMII ZIPs")
    ap.add_argument("--zips", default=None, help="folder with the ZIPs (default: Dataset)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ids", nargs="+", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    zdir = Path(args.zips) if args.zips else resolve_path(cfg, "raw_zip_dir")
    out = Path(args.out) if args.out else resolve_path(cfg, "mono_dir")
    tool = Path(__file__).with_name("mimii_tool.py")
    for mtype in cfg["dataset"]["machine_types"]:
        zpath = zdir / cfg["dataset"]["zips"][mtype]["file"]
        if not zpath.exists():
            print(f"Missing {zpath}. Run scripts/download_mimii.py first.")
            return 1
        for mid in args.ids or cfg["dataset"]["machine_ids"]:
            cmd = [sys.executable, str(tool), "extract", "--local-zip", str(zpath), "--machine-type", mtype,
                   "--machine-id", mid, "--out", str(out), "--channel-index", str(cfg["audio"]["channel_index"]),
                   "--snr-db", str(cfg["dataset"]["snr_db"])]
            print("\n$", " ".join(cmd), flush=True)
            if subprocess.run(cmd).returncode:
                print(f"FAILED for {mtype} {mid}. Send me the output above.")
                return 1
    print("\nAll machines extracted. Checking the files ...")
    rc = subprocess.run([sys.executable, str(tool), "verify", "--dir", str(out)]).returncode
    if rc:
        print("\nNOTE: the list above shows unusual files. That is fine: scripts/audit_audio.py decides what to exclude.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
