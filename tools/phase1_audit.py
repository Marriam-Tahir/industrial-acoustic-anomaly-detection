"""
Phase 1 audit, version 2 (MODIFIED: replaces the earlier tools/phase1_audit.py).
READ-ONLY: never changes your data, manifests, features or code.
It only writes a report into <project>/reports/.

Run from the project root, with the smartlistener conda environment active:
    python tools/phase1_audit.py
"""
import argparse, json, random, re, subprocess, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

EXPECTED = {  # (normal, abnormal) per machine/ID, MIMII paper table
    ("pump", "00"): (1006, 143), ("pump", "02"): (1005, 111), ("pump", "04"): (702, 100), ("pump", "06"): (1036, 102),
    ("fan", "00"): (1011, 407), ("fan", "02"): (1016, 359), ("fan", "04"): (1033, 348), ("fan", "06"): (1015, 361),
}
SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "venv", ".venv", "env", "node_modules", "reports", ".pytest_cache", "Dataset"}
LINES, FAILS, WARNS = [], [], []


def out(s=""):
    print(s); LINES.append(str(s))


def fail(s): FAILS.append(s); out("  [FAIL] " + s)
def warn(s): WARNS.append(s); out("  [WARN] " + s)
def ok(s): out("  [ OK ] " + s)
def info(s): out("  [info] " + s)


def head(s):
    out("\n" + "=" * 70 + "\n" + s + "\n" + "=" * 70)


def parse_path(p, assume_db):
    s = str(p).replace("\\", "/")
    low = s.lower()
    m = re.search(r"(?<![\d])([+-]?\d+)_?\s?db", low)
    db = int(m.group(1)) if m else assume_db
    m = re.search(r"(?:^|[/_])(fan|pump)(?:[/_]|$)", low)
    machine = m.group(1) if m else None
    m = re.search(r"id_(\d\d)", low)
    mid = m.group(1) if m else None
    m = re.search(r"/(normal|abnormal)/", low)
    label = m.group(1) if m else None
    return db, machine, mid, label, Path(s).name


def canon_split(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unassigned"
    s = str(v).strip().lower()
    if s.startswith("train"): return "train"
    if s.startswith("val"): return "val"
    if s.startswith("test"): return "test"
    return "other:" + s


def step_env(root):
    head("STEP 0  Environment and project structure")
    out(f"  python: {sys.executable}")
    if "smartlistener" not in sys.executable.lower():
        warn("This is NOT the smartlistener conda environment. Run: conda activate smartlistener")
    try:
        import smart_listener  # noqa
        ok(f"smart_listener importable from {Path(smart_listener.__file__).parent}")
    except Exception as e:
        fail(f"smart_listener not importable ({e}). Activate smartlistener, then run: pip install -e .")
    for p in sorted(Path(root).iterdir()):
        if p.name in SKIP_DIRS or p.name.startswith(".git"):
            continue
        if p.is_dir():
            out(f"  {p.name}/  ->  {', '.join(k.name for k in sorted(p.iterdir())[:12])}")
        else:
            out(f"  {p.name}")


def step_inventory(mono, assume_db):
    head("STEP 1  Extracted WAV inventory")
    inv = defaultdict(list)
    for p in Path(mono).rglob("*.wav"):
        db, mach, mid, lab, _ = parse_path(p.relative_to(mono), assume_db)
        if None in (mach, mid, lab):
            warn(f"cannot parse {p}"); continue
        inv[(mach, mid, lab)].append(p)
    if not inv:
        fail(f"No WAV files under {mono}"); return inv
    rows = [{"machine": k[0], "id": k[1], "label": k[2], "n": len(v)} for k, v in sorted(inv.items())]
    out(pd.DataFrame(rows).pivot_table(index=["machine", "id"], columns="label", values="n", aggfunc="sum").to_string())
    bad = 0
    for (mach, mid), (en, ea) in EXPECTED.items():
        gn, ga = len(inv.get((mach, mid, "normal"), [])), len(inv.get((mach, mid, "abnormal"), []))
        if (gn, ga) != (en, ea):
            bad += 1; warn(f"{mach} id_{mid}: found {gn}/{ga} normal/abnormal, paper table says {en}/{ea}")
    if not bad:
        ok("All 8 machine/ID counts equal the MIMII paper table (nothing missing)")
    return inv


def step_integrity(inv, deep_n):
    head("STEP 2  WAV integrity (every header + 200 decoded)")
    files = [p for v in inv.values() for p in v]
    sr_c, ch_c, bad = defaultdict(int), defaultdict(int), []
    t0 = time.time()
    for p in files:
        try:
            i = sf.info(str(p)); sr_c[i.samplerate] += 1; ch_c[i.channels] += 1
            d = i.frames / i.samplerate
            if not 9.5 <= d <= 10.5: bad.append((p.name, f"duration {d:.2f}s"))
        except Exception as e:
            bad.append((p.name, f"unreadable {e}"))
    out(f"  {len(files)} files in {time.time() - t0:.0f}s | sample rates {dict(sr_c)} | channels {dict(ch_c)}")
    (fail if bad else ok)(f"{len(bad)} odd/unreadable files {bad[:3]}" if bad else "All readable, about 10 s each")
    if set(sr_c) != {16000}: warn(f"sample rates not all 16000: {dict(sr_c)}")
    if set(ch_c) != {1}: warn(f"extracted files are not all mono: {dict(ch_c)}")
    random.seed(0); n = 0
    for p in random.sample(files, min(deep_n, len(files))):
        x, _ = sf.read(str(p), always_2d=True)
        if not np.isfinite(x).all() or np.abs(x).max() == 0: n += 1; warn(f"silent/NaN: {p.name}")
    if not n: ok("Random decoded sample: finite and not silent")


def step_manifest(root, mono, assume_db):
    head("STEP 3  Manifest, splits and leakage")
    cands = [p for p in (Path(root) / "data").rglob("*.csv") if "manifest" in p.name.lower() or "split" in p.name.lower()]
    man = None
    for p in cands:
        df = pd.read_csv(p)
        out(f"  {p.relative_to(root)}  rows={len(df)}  columns={list(df.columns)}")
        if "path" in df.columns and man is None:
            man = df
        elif "path" not in df.columns:
            info(f"{p.name} has no path column: treated as a report table, not a manifest")
    if man is None:
        fail("No manifest with a 'path' column found"); return None
    df = man.copy()
    parsed = df["path"].astype(str).map(lambda x: parse_path(x, assume_db))
    df["_m"] = [x[1] for x in parsed]; df["_id"] = [x[2] for x in parsed]; df["_lab"] = [x[3] for x in parsed]; df["_name"] = [x[4] for x in parsed]
    if df["_m"].isna().any() or df["_lab"].isna().any():
        warn("some manifest paths have no machine/id/label in the path")
    if "label_name" in df.columns:
        mism = (df["_lab"] != df["label_name"].astype(str).str.lower()).sum()
        (fail if mism else ok)(f"{mism} rows where folder label differs from label_name column")
    dup = df.duplicated(subset=["path"]).sum()
    (fail if dup else ok)(f"{dup} duplicated paths in manifest")
    if "sha256" in df.columns:
        d = df.duplicated(subset=["sha256"], keep=False).sum()
        (warn if d else ok)(f"{d} rows share identical audio content (sha256)")
    if "exclude_reason" in df.columns:
        ex = df["exclude_reason"].notna() & (df["exclude_reason"].astype(str).str.strip() != "")
        info(f"rows with an exclude_reason: {int(ex.sum())}")
        if ex.any(): out(df.loc[ex, "exclude_reason"].value_counts().to_string())
    miss = [p for p in df["path"].astype(str) if not (Path(mono) / p).exists()]
    (fail if miss else ok)(f"{len(miss)} manifest paths missing on disk" + (f", e.g. {miss[0]}" if miss else ""))
    df["_key"] = df["_m"].astype(str) + "|" + df["_id"].astype(str) + "|" + df["_lab"].astype(str) + "|" + df["_name"]
    gcol = "group" if "group" in df.columns else None
    for sc in [c for c in df.columns if c.lower().startswith("split")]:
        out(f"\n  --- split column '{sc}' ---")
        df["_s"] = df[sc].map(canon_split)
        out("  values: " + str(df[sc].value_counts(dropna=False).to_dict()))
        out(pd.crosstab([df[gcol] if gcol else df["_m"] + "_id_" + df["_id"], df["_s"]], df["_lab"]).to_string())
        used = df[df["_s"].isin(["train", "val", "test"])]
        n = used.groupby("_key")["_s"].nunique()
        (fail if (n > 1).any() else ok)(f"[{sc}] clips in more than one split: {int((n > 1).sum())}")
        if "sha256" in df.columns:
            n2 = used.groupby("sha256")["_s"].nunique()
            (fail if (n2 > 1).any() else ok)(f"[{sc}] identical audio content in more than one split: {int((n2 > 1).sum())}")
        if sc.lower() != "split_ad":
            info(f"[{sc}] is not the anomaly-detection split (probably the fan-vs-pump split); only the overlap checks above apply")
            continue
        tr_ab = ((used["_s"] == "train") & (used["_lab"] == "abnormal")).sum()
        (fail if tr_ab else ok)(f"[{sc}] abnormal clips in train: {int(tr_ab)}")
        for s in ("val", "test"):
            sub = used[used["_s"] == s]
            if len(sub) and (sub["_lab"] == "abnormal").sum() == 0: warn(f"[{sc}] {s} has no abnormal clips")
        if "index_in_label" in df.columns and sc.lower() == "split_ad":
            out("  closest recording-order distance between a train clip and a val/test clip of the same label (bigger = safer):")
            for g, gd in used.groupby(gcol if gcol else "_m"):
                for lab in ("normal", "abnormal"):
                    t = gd[(gd["_s"] == "train") & (gd["_lab"] == lab)]["index_in_label"].values
                    o = gd[(gd["_s"] != "train") & (gd["_lab"] == lab)]["index_in_label"].values
                    if len(t) and len(o):
                        out(f"    {g:12s} {lab:9s} min gap = {int(np.abs(t[:, None] - o[None, :]).min())}")
    return df


def step_features(root, man):
    head("STEP 4  Feature cache")
    d = Path(root) / "data" / "processed"
    if not d.exists():
        fail("data/processed missing"); return
    for p in sorted(d.rglob("*")):
        if p.is_file():
            out(f"  {p.relative_to(root)}  {p.stat().st_size / 1e6:.1f} MB")
    gcol = "group" if man is not None and "group" in man.columns else None
    for p in sorted(d.rglob("*.npy")):
        a = np.load(p, mmap_mode="r")
        n_exp = int((man[gcol] == p.stem).sum()) if gcol is not None else None
        good = True
        for i in range(0, a.shape[0], 200):
            if not np.isfinite(np.asarray(a[i:i + 200])).all(): good = False; break
        msg = f"{p.name}: shape {a.shape} {a.dtype}, finite={good}, manifest rows for this group={n_exp}"
        (ok if good and (n_exp in (None, a.shape[0])) else fail)(msg)
    for p in sorted(d.rglob("*.json"))[:4]:
        try:
            js = json.loads(p.read_text())
            info(f"{p.name} keys: {list(js)[:12] if isinstance(js, dict) else type(js)}")
        except Exception as e:
            warn(f"cannot read {p.name}: {e}")


def step_tests(root):
    head("STEP 5  pytest")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header"], cwd=root, capture_output=True, text=True, timeout=900)
    out("\n".join((r.stdout + r.stderr).strip().splitlines()[-10:]))
    (ok if r.returncode == 0 else fail)(f"pytest exit code {r.returncode}")


def step_handoff(root):
    head("STEP 6  Handoff dump for Phase 2 (files printed so Claude can fit Phase 2 to your code)")
    names = ["config/audio_config.yaml", "src/smart_listener/audio.py", "scripts/extract_features.py", "scripts/build_manifest.py"]
    names += [str(p.relative_to(root)).replace("\\", "/") for p in sorted((Path(root) / "src" / "smart_listener").glob("*.py")) if p.name not in ("audio.py",)]
    seen = set()
    for n in names:
        if n in seen or not (Path(root) / n).exists(): continue
        seen.add(n)
        out(f"\n----- FILE: {n} -----")
        out((Path(root) / n).read_text(encoding="utf-8", errors="ignore")[:9000])
    out("\n----- first 3 manifest rows -----")
    mp = Path(root) / "data" / "manifests" / "manifest.csv"
    if mp.exists(): out(pd.read_csv(mp).head(3).T.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--mono-dir", default="data/interim/mono16k")
    ap.add_argument("--assume-db", type=int, default=6, help="dB level of the extracted WAVs when the folder name has none")
    ap.add_argument("--deep", type=int, default=200)
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--skip-integrity", action="store_true")
    a = ap.parse_args()
    root = Path(a.root).resolve(); mono = root / a.mono_dir
    out(f"Phase 1 audit v2 | root={root} | wav folder={mono} | assumed dB={a.assume_db}")
    step_env(root)
    inv = step_inventory(mono, a.assume_db)
    if inv and not a.skip_integrity: step_integrity(inv, a.deep)
    man = step_manifest(root, mono, a.assume_db)
    step_features(root, man)
    if not a.skip_tests: step_tests(root)
    step_handoff(root)
    head("SUMMARY")
    out(f"  FAIL: {len(FAILS)}   WARN: {len(WARNS)}")
    for f in FAILS: out("   FAIL: " + f)
    for w in WARNS: out("   WARN: " + w)
    rep = root / "reports"; rep.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (rep / f"phase1_audit_v2_{stamp}.txt").write_text("\n".join(LINES), encoding="utf-8")
    out(f"\nReport saved: {rep / ('phase1_audit_v2_' + stamp + '.txt')}")


if __name__ == "__main__":
    main()
