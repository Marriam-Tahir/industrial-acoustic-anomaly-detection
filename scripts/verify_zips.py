"""Check the downloaded MIMII ZIP files against the OFFICIAL md5 checksums from Zenodo.

    python scripts/verify_zips.py                  # checks every .zip in the Dataset folder
    python scripts/verify_zips.py --dir "E:\\path"  # another folder

For each ZIP it (1) computes the md5 and compares it with the official value, (2) opens the ZIP index and counts
the WAV files, and (3) writes the evidence to data/manifests/zip_checksums.csv. A wrong md5 means a damaged
or different file: delete it and download it again. Nothing is ever modified.
"""
import argparse
import csv
import hashlib
import sys
import time
import zipfile
from pathlib import Path

from smart_listener.config import load_config, resolve_path

CHUNK = 8 * 1024 * 1024


def md5_of(path: Path) -> str:
    h, done, total, last = hashlib.md5(), 0, path.stat().st_size, time.time()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
            done += len(block)
            if time.time() - last > 15:
                print(f"     ... {100 * done / total:4.0f}%", flush=True)
                last = time.time()
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify MIMII ZIPs against official md5 checksums")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    folder = Path(args.dir) if args.dir else resolve_path(cfg, "raw_zip_dir")
    known = cfg["dataset"]["checksums"]
    by_md5 = {v: k for k, v in known.items()}
    needed = [cfg["dataset"]["zips"][t]["file"] for t in cfg["dataset"]["machine_types"]]
    zips = sorted(folder.rglob("*.zip")) if folder.exists() else []
    if not zips:
        print(f"No .zip files found in {folder}. Check the folder name.")
        return 1

    rows, bad = [], 0
    for z in zips:
        print(f"\n== {z.name}  ({z.stat().st_size / 1e9:.2f} GB) ==  computing md5 ...", flush=True)
        got = md5_of(z)
        status, note = "OK", ""
        if z.name in known and known[z.name] == got:
            note = "matches the official checksum"
        elif got in by_md5:
            status, note = "RENAME", f"content is {by_md5[got]}; rename the file to that"
        elif z.name in known:
            status, note = "BAD", f"expected {known[z.name]}; damaged download, delete and download again"
        else:
            status, note = "UNKNOWN", "not one of the 12 official files"
        n_wav = n_ch = -1
        try:
            with zipfile.ZipFile(z) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".wav") and "__MACOSX" not in n]
                n_wav = len(names)
        except Exception as exc:  # noqa: BLE001
            status, note = "BAD", f"cannot open as ZIP: {exc!r}"
        print(f"   {status}: md5 {got}\n   {note}\n   WAV files inside: {n_wav}")
        bad += status not in ("OK",)
        rows.append({"file": z.name, "size_bytes": z.stat().st_size, "md5": got, "status": status, "wav_files": n_wav})

    out = resolve_path(cfg, "manifest_dir") / "zip_checksums.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ok_names = {r["file"] for r in rows if r["status"] == "OK"}
    missing = [n for n in needed if n not in ok_names]
    print(f"\nSaved {out}")
    if missing:
        print("STILL NEEDED and not verified:", missing)
    print("RESULT:", "ALL FILES VERIFIED" if not bad and not missing else "PROBLEMS FOUND - see above")
    return 0 if not bad and not missing else 1


if __name__ == "__main__":
    sys.exit(main())
