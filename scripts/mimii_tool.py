#!/usr/bin/env python3
"""MIMII helper tool. One file, no project imports, so it also runs on Google Colab.

Commands
  probe    Does a ZIP link support partial ("range") downloads? What is inside the ZIP?
  extract  Copy ONE machine's recordings out of a ZIP as mono 16-bit WAV files + an inventory CSV.
  verify   Re-check a folder of extracted WAV files (sample rate, channels, length).

Examples
  python mimii_tool.py probe   --url "https://zenodo.org/records/3384388/files/6_dB_fan.zip?download=1"
  python mimii_tool.py extract --url "<same link>" --machine-type fan --machine-id id_00 --out work/mono --limit-per-label 20
  python mimii_tool.py extract --local-zip /content/zips/6_dB_fan.zip --machine-type fan --machine-id id_00 --out work/mono
  python mimii_tool.py verify  --dir work/mono
"""
import argparse
import csv
import io
import re
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import soundfile as sf

ID_RE = re.compile(r"^id_\d{2}$")
LABELS = ("normal", "abnormal")
KNOWN_TYPES = ("fan", "pump", "slider", "valve")
EXPECTED_SR = 16000
EXPECTED_CHANNELS = 8
EXPECTED_SECONDS = 10.0
SECONDS_TOLERANCE = 0.5
INVENTORY_COLUMNS = ["source_member", "machine_type", "machine_id", "snr_db", "label", "sr", "channels",
                     "frames", "duration_s", "peak", "rms", "out_path", "issues"]


# ----------------------------------------------------------------------------- helpers
def parse_member(name):
    """Work out machine id and label from a path inside the ZIP. Returns None for non-audio/junk entries."""
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    if not parts or not parts[-1].lower().endswith(".wav"):
        return None
    if parts[0] == "__MACOSX" or parts[-1].startswith("._"):
        return None
    folders = parts[:-1]
    machine_id = next((p for p in folders if ID_RE.match(p)), None)
    label = next((p for p in folders if p in LABELS), None)
    machine_type = next((p for p in folders if p in KNOWN_TYPES), None)
    if machine_id is None or label is None:
        return None
    return {"machine_type": machine_type, "machine_id": machine_id, "label": label, "basename": parts[-1]}


def open_archive(url=None, local_zip=None):
    """Open a ZIP from disk, or from a web link by reading only the parts we ask for."""
    if local_zip:
        return zipfile.ZipFile(local_zip)
    if not url:
        sys.exit("ERROR: give either --url or --local-zip")
    from remotezip import RemoteZip  # imported here so local-only use does not need it
    return RemoteZip(url)


def human(n_bytes):
    n = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024


# ----------------------------------------------------------------------------- probe
def summarize_archive(infos):
    wavs = [i for i in infos if i.filename.lower().endswith(".wav")]
    parsed = Counter()
    unparsed = []
    for info in wavs:
        meta = parse_member(info.filename)
        if meta is None:
            unparsed.append(info.filename)
        else:
            parsed[(meta["machine_type"] or "?", meta["machine_id"], meta["label"])] += 1
    print(f"Entries in ZIP: {len(infos)}  |  .wav files: {len(wavs)}  |  other entries: {len(infos) - len(wavs)}")
    print(f"Total size  compressed: {human(sum(i.compress_size for i in infos))}   "
          f"uncompressed: {human(sum(i.file_size for i in infos))}")
    print("First 5 names (this shows the real folder layout):")
    for info in infos[:5]:
        print("   ", info.filename)
    print("Recordings found, by machine id and label:")
    for (mtype, mid, label), n in sorted(parsed.items()):
        print(f"    type={mtype:<6} {mid}  {label:<9} {n}")
    if unparsed:
        print(f"WARNING: {len(unparsed)} .wav files did not match the id_XX/normal|abnormal pattern, e.g.:")
        for name in unparsed[:5]:
            print("   ", name)
    if not parsed:
        print("NOTE: no recordings matched id_XX/normal|abnormal. Copy the 'First 5 names' above and send them to me.")


def cmd_probe(args):
    if args.local_zip:
        print(f"Local ZIP: {args.local_zip}")
        with open_archive(local_zip=args.local_zip) as z:
            summarize_archive(z.infolist())
        return 0

    if not args.url:
        sys.exit("ERROR: probe needs --url (or --local-zip)")
    import requests

    print(f"Testing: {args.url}")
    all_ok = True
    for what, rng, want in (("start of file", "bytes=0-99", 100), ("end of file  ", "bytes=-1024", 1024)):
        try:
            with requests.get(args.url, headers={"Range": rng}, stream=True, timeout=60, allow_redirects=True) as r:
                data = b""
                for part in r.iter_content(chunk_size=4096):
                    data += part
                    if len(data) >= want:
                        break
                content_range = r.headers.get("Content-Range")
                ok = r.status_code == 206 and len(data) >= want and bool(content_range)
                print(f"  {what}: asked {rng:<12} -> HTTP {r.status_code}, got {min(len(data), want)} bytes, "
                      f"Content-Range={content_range}, Accept-Ranges={r.headers.get('Accept-Ranges')}, "
                      f"served by {urlparse(r.url).netloc}  => {'OK' if ok else 'NOT SUPPORTED'}")
                all_ok = all_ok and ok
        except Exception as exc:  # noqa: BLE001
            print(f"  {what}: request failed: {exc!r}")
            all_ok = False

    if not all_ok:
        print("\nRESULT: partial downloads are NOT confirmed for this link. Reading single files remotely will not work.")
        print("        Use PLAN B (download the ZIP once to the temporary Colab disk).")
        return 2

    print("\nPartial downloads work. Reading the ZIP table of contents (small download)...")
    t0 = time.time()
    try:
        with open_archive(url=args.url) as z:
            infos = z.infolist()
    except Exception as exc:  # noqa: BLE001
        print("RESULT: could not read the ZIP table of contents remotely:", repr(exc))
        print("        Use PLAN B (download the ZIP once to the temporary Colab disk).")
        return 2
    print(f"Done in {time.time() - t0:.1f} s.\n")
    summarize_archive(infos)
    print("\nRESULT: remote reading works. Use PLAN A (extract only the files we need).")
    return 0


# ----------------------------------------------------------------------------- extract
def select_members(infos, machine_id, limit_per_label):
    chosen = {"normal": [], "abnormal": []}
    for info in infos:
        meta = parse_member(info.filename)
        if meta and meta["machine_id"] == machine_id:
            chosen[meta["label"]].append((info, meta))
    selected = []
    for label in LABELS:
        items = sorted(chosen[label], key=lambda t: t[0].filename)
        if limit_per_label > 0:
            items = items[:limit_per_label]
        selected += items
    return selected


def convert_one(data, out_path, channel_index):
    """Read WAV bytes, keep one microphone channel, save as mono 16-bit WAV. Returns facts about the file."""
    issues = []
    with sf.SoundFile(io.BytesIO(data)) as f:
        sr, channels, frames = f.samplerate, f.channels, len(f)
        audio = f.read(dtype="int16", always_2d=True)
    duration = frames / sr if sr else 0.0
    if sr != EXPECTED_SR:
        issues.append(f"sample_rate={sr}")
    if channels != EXPECTED_CHANNELS:
        issues.append(f"channels={channels}")
    if abs(duration - EXPECTED_SECONDS) > SECONDS_TOLERANCE:
        issues.append(f"duration={duration:.2f}s")
    if channel_index >= channels:
        raise ValueError(f"file has only {channels} channels, cannot take channel {channel_index}")
    mono = audio[:, channel_index]
    peak = int(abs(mono.astype("int32")).max()) if len(mono) else 0
    rms = float((mono.astype("float64") ** 2).mean() ** 0.5) if len(mono) else 0.0
    if peak == 0:
        issues.append("silent")
    if peak >= 32767:
        issues.append("clipped")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.parent / (out_path.name + ".part")  # write to a temp name first, so a crash leaves no half file
    sf.write(str(tmp), mono, sr, subtype="PCM_16", format="WAV")
    tmp.replace(out_path)
    return {"sr": sr, "channels": channels, "frames": frames, "duration_s": round(duration, 3),
            "peak": peak, "rms": round(rms, 2), "issues": ";".join(issues)}


def read_with_retry(zf, info, reopen, tries=5):
    """Read one file from the ZIP. On a network problem wait, reconnect and try again."""
    for attempt in range(1, tries + 1):
        try:
            return zf.read(info), zf  # zipfile also checks the CRC32 checksum here
        except Exception as exc:  # noqa: BLE001
            if attempt == tries:
                raise
            wait = min(60, 2 ** attempt)
            print(f"  problem reading {info.filename}: {exc!r} -> retry {attempt}/{tries - 1} in {wait}s")
            time.sleep(wait)
            try:
                zf.close()
            except Exception:  # noqa: BLE001
                pass
            zf = reopen()
    raise RuntimeError("unreachable")


def cmd_extract(args):
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    inv_path = out_dir / f"{args.machine_type}_{args.machine_id}_inventory.csv"

    def reopen():
        return open_archive(args.url, args.local_zip)

    zf = reopen()
    selected = select_members(zf.infolist(), args.machine_id, args.limit_per_label)
    if not selected:
        ids = sorted({m["machine_id"] for i in zf.infolist() for m in [parse_member(i.filename)] if m})
        sys.exit(f"ERROR: no recordings for {args.machine_id} in this ZIP. IDs found: {ids or 'none'}. "
                 f"Run the 'probe' command and read the folder layout.")

    done = set()
    if inv_path.exists():
        with open(inv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if (out_dir / row["out_path"]).exists():
                    done.add(row["source_member"])
    todo = [(i, m) for i, m in selected if i.filename not in done]
    print(f"{args.machine_type} {args.machine_id}: {len(selected)} files selected, "
          f"{len(done)} already done, {len(todo)} to do.")

    new_file = not inv_path.exists()
    counts, problems, errors = Counter(), Counter(), 0
    t0 = time.time()
    with open(inv_path, "a", newline="", encoding="utf-8") as inv:
        writer = csv.DictWriter(inv, fieldnames=INVENTORY_COLUMNS)
        if new_file:
            writer.writeheader()
        for n, (info, meta) in enumerate(todo, 1):
            rel = Path(args.machine_type) / args.machine_id / meta["label"] / meta["basename"]
            row = {"source_member": info.filename, "machine_type": args.machine_type, "machine_id": args.machine_id,
                   "snr_db": args.snr_db, "label": meta["label"], "out_path": rel.as_posix()}
            try:
                data, zf = read_with_retry(zf, info, reopen)
                row.update(convert_one(data, out_dir / rel, args.channel_index))
            except Exception as exc:  # noqa: BLE001  one bad file must not stop the whole run
                errors += 1
                print(f"  ERROR on {info.filename}: {exc!r}")
                continue
            writer.writerow(row)
            inv.flush()
            counts[meta["label"]] += 1
            for issue in filter(None, row["issues"].split(";")):
                problems[issue.split("=")[0]] += 1
            if n % 25 == 0 or n == len(todo):
                rate = n / (time.time() - t0)
                print(f"  {n}/{len(todo)} done  ({rate:.1f} files/s, about {(len(todo) - n) / rate / 60:.1f} min left)")
    try:
        zf.close()
    except Exception:  # noqa: BLE001
        pass

    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min. New files: normal={counts['normal']}, abnormal={counts['abnormal']}. "
          f"Errors: {errors}.")
    print("Files with unusual properties:", dict(problems) if problems else "none")
    print(f"Inventory: {inv_path}")
    return 1 if errors else 0


# ----------------------------------------------------------------------------- verify
def cmd_verify(args):
    root = Path(args.dir)
    files = sorted(root.rglob("*.wav"))
    if not files:
        print(f"No .wav files found under {root}")
        return 1
    groups, bad = Counter(), []
    for p in files:
        try:
            info = sf.info(str(p))
        except Exception as exc:  # noqa: BLE001
            bad.append((p, f"unreadable: {exc!r}"))
            continue
        groups[p.parent.relative_to(root).as_posix()] += 1
        dur = info.frames / info.samplerate
        if info.channels != 1:
            bad.append((p, f"channels={info.channels}"))
        if info.samplerate != EXPECTED_SR:
            bad.append((p, f"sample_rate={info.samplerate}"))
        if info.subtype != "PCM_16":
            bad.append((p, f"subtype={info.subtype}"))
        if abs(dur - EXPECTED_SECONDS) > SECONDS_TOLERANCE:
            bad.append((p, f"duration={dur:.2f}s"))
    print(f"Checked {len(files)} files under {root}")
    for folder, n in sorted(groups.items()):
        print(f"  {folder:<30} {n} files")
    if bad:
        print(f"PROBLEMS in {len(bad)} check(s):")
        for p, why in bad[:20]:
            print("  ", p, why)
        return 1
    print("RESULT: all files are mono, 16000 Hz, 16-bit, about 10 seconds. OK")
    return 0


# ----------------------------------------------------------------------------- main
def build_parser():
    ap = argparse.ArgumentParser(description="MIMII ZIP helper")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("probe", help="check range support and list ZIP contents")
    p.add_argument("--url")
    p.add_argument("--local-zip")
    p.set_defaults(func=cmd_probe)

    e = sub.add_parser("extract", help="extract one machine as mono 16-bit WAV")
    e.add_argument("--url")
    e.add_argument("--local-zip")
    e.add_argument("--machine-type", required=True, choices=["fan", "pump"])
    e.add_argument("--machine-id", required=True, help="for example id_00")
    e.add_argument("--out", required=True)
    e.add_argument("--limit-per-label", type=int, default=0, help="0 = all files")
    e.add_argument("--channel-index", type=int, default=0, help="0 = microphone No. 1")
    e.add_argument("--snr-db", type=int, default=6)
    e.set_defaults(func=cmd_extract)

    v = sub.add_parser("verify", help="re-check extracted WAV files")
    v.add_argument("--dir", required=True)
    v.set_defaults(func=cmd_verify)
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(args.func(args))
