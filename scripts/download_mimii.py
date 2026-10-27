"""Download the MIMII 6 dB ZIP files from the official Zenodo record, safely.

    python scripts/download_mimii.py                 # fan and pump into Dataset/
    python scripts/download_mimii.py --verify-only   # only re-check files that are already there

Safe means: it can resume after a broken connection (run the same command again), it writes to a
.part file and renames it only when finished, and it checks the md5 checksum from Zenodo's page.
A file with the wrong checksum is deleted, never used.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import requests

from smart_listener.config import load_config, resolve_path

CHUNK = 4 * 1024 * 1024


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, part: Path, tries: int = 30) -> None:
    """Resumable download into `part`. Raises if it cannot finish."""
    for attempt in range(1, tries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code == 416:                       # we already have every byte
                    return
                r.raise_for_status()
                if have and r.status_code == 200:              # server ignored the range: start again
                    have = 0
                mode = "ab" if have and r.status_code == 206 else "wb"
                total = have + int(r.headers.get("Content-Length", 0))
                got, last = have, time.time()
                with open(part, mode) as fh:
                    for block in r.iter_content(CHUNK):
                        fh.write(block)
                        got += len(block)
                        if time.time() - last > 5:
                            pct = f"{100 * got / total:5.1f}%" if total else ""
                            print(f"   {got / 1e9:6.2f} GB {pct}", flush=True)
                            last = time.time()
                if total and got < total:
                    raise IOError(f"connection closed early at {got} of {total} bytes")
                return
        except (requests.RequestException, IOError) as exc:
            wait = min(60, 5 * attempt)
            print(f"   problem ({exc!r}); retry {attempt}/{tries} in {wait} s. Progress is kept.", flush=True)
            time.sleep(wait)
    raise RuntimeError("download did not finish; run the same command again later")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download MIMII ZIPs with resume and md5 check")
    ap.add_argument("--types", nargs="+", default=None, help="fan pump (default: from config)")
    ap.add_argument("--dest", default=None, help="folder for the ZIPs (default: Dataset)")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dest = Path(args.dest) if args.dest else resolve_path(cfg, "raw_zip_dir")
    dest.mkdir(parents=True, exist_ok=True)
    base = args.base_url or f"https://zenodo.org/records/{cfg['dataset']['zenodo_record']}/files"
    bad = 0
    for mtype in args.types or cfg["dataset"]["machine_types"]:
        z = cfg["dataset"]["zips"][mtype]
        final, part = dest / z["file"], dest / (z["file"] + ".part")
        print(f"\n== {z['file']} ==")
        if not final.exists() and not args.verify_only:
            print("   downloading (safe to stop with Ctrl+C and run again)")
            download(f"{base}/{z['file']}?download=1", part)
            part.rename(final)
        if not final.exists():
            print("   MISSING")
            bad += 1
            continue
        print("   checking md5 (a few minutes for a big file) ...", flush=True)
        got = md5_of(final)
        if got == z["md5"]:
            print(f"   OK  md5 {got}  size {final.stat().st_size / 1e9:.2f} GB")
        else:
            print(f"   BAD md5 {got}, expected {z['md5']}. Deleting the file; run again.")
            final.unlink()
            bad += 1
    print("\nALL ZIPS VERIFIED" if not bad else f"\n{bad} file(s) not usable")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
